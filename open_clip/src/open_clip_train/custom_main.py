import mgap
from open_clip_train.scheduler import cosine_lr_custom
from mgap import config
import argparse
import time
from datetime import timedelta
from mgap import utils
from mgap.utils import  get_latest_checkpoint, write_to_csv
from mgap import utils, filtering_architectures, filtering_algorithms
from open_clip_train.file_utils import pt_load, remote_sync
from open_clip_train.custom_train import train_one_epoch, evaluate, evaluate_coco_controlled, plot_histogram_to_compare_if_sae_bias_is_null_vector, evaluate_mad


from open_clip_train.params import parse_args
from open_clip_train.logger import setup_logging
from open_clip_train.distributed import is_master, init_distributed_device, broadcast_object
from open_clip_train.data import get_data, get_cc12m_dataset
from open_clip import create_model_and_transforms, get_tokenizer, create_loss
import logging
import os
import sys
from open_clip_train.precision import get_autocast


try:
    import smartclip
except ImportError:  # only needed for --use-smartclip-model
    smartclip = None
import torch
from torch import optim
import os.path as osp
import socket
import copy


try:
    import flair
except ImportError:  # only needed for FLAIR models and the retrieval benchmarks
    flair = None
try:
    import align_clip
except ImportError:  # only needed for AlignCLIP / SharedCLIP checkpoints
    align_clip = None


try:
    import wandb
except ImportError:
    wandb = None

tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


LATEST_CHECKPOINT_NAME = "epoch_latest.pt"
BEST_CHECKPOINT_NAME = "epoch_best.pt"


def set_train_order(args, num_train_pieces):
    train_order = None
    if is_master(args):
        train_order = torch.randperm(num_train_pieces)
        logging.info(f"Train order created: {train_order}.")
    broadcast_object(args, train_order)
    train_piece_idx = 0
    logging.info(f"Train order Received: {train_order}.")
    return train_order, train_piece_idx


def get_clip_parameters(model, args): 
    def exclude(
        n, p): return p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n

    def include(n, p): return not exclude(n, p)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [
        p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(
        n, p) and p.requires_grad]


    param_list = [{"params": gain_or_bias_params, "weight_decay": 0., "lr": args.lr, "betas": (args.beta1, args.beta2), "eps": args.eps, "name": "clip"},
                {"params": rest_params, "weight_decay": args.wd, "lr": args.lr, "betas": (
                    args.beta1, args.beta2), "eps": args.eps, "name": "clip"},
                ]
    return param_list


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk-ranking", type=int, default=None)
    parser.add_argument("--rerank-eval", action="store_true", default=False,
                        help="Two-stage retrieval: original CLIP does the coarse retrieval and gives "
                             "the top-K candidate pool, then our conditioned scores rerank that pool. "
                             "Both matrices are computed on the fly in the same eval run (no dumps).")
    parser.add_argument("--rerank-k-sweep", type=int, nargs="+", default=[1, 5, 10, 20, 50, 100, 200],
                        help="Candidate pool sizes K to sweep for --rerank-eval.")
    parser.add_argument("--rerank-fast", action="store_true", default=False,
                        help="TRUE two-stage retrieval: coarse scores from the ORIGINAL (pre-finetuning) "
                             "CLIP pick a top-K candidate pool, then TEVI conditioning runs ONLY on that "
                             "pool. Never builds the full conditioned matrix, so it is much faster than "
                             "--rerank-eval, but reports no full-conditioned / mgap numbers. "
                             "Pool size is max(--rerank-k-sweep).")
    parser.add_argument("--save-all-embeddings", action="store_true", default=False)
    parser.add_argument("--save-umap-plots", action="store_true", default=False)
    parser.add_argument("--pr-zero-shot", action="store_true", default=False)
    parser.add_argument("--compare-cossims-with-sae-bias", action="store_true", default=False)

    args = parse_args(args=None, parser=parser)

    # fully initialize distributed device environment
    device = init_distributed_device(args)
    assert (args.finetune_text_encoder)
    assert (args.train_mask_batch_size == args.batch_size)
    utils.common_init(args)

    args.log_path = None
    log_base_path = args.log_base_path

    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    if (os.path.exists(os.path.join(args.checkpoint_path, f"checkpoint_{args.train_mask_epochs}.pt")) or os.path.exists(os.path.join(args.checkpoint_path, f"epoch_{args.train_mask_epochs}.pt"))) and not args.resume_combined_training: # for CC12M, it should be actual_epochs = 10* train_mask_epochs
        logging.info(f"Checkpoint already exists at {args.checkpoint_path}")
        sys.exit(0)

    resume_latest = args.resume == 'latest'

    args.autocast = get_autocast(
        args.precision, device_type=torch.device(args.device).type)

    if is_master(args):
        logging.info(
            f"Training CLIP: {args.train_clip}; Finetuning Text Encoder: {args.finetune_text_encoder}; Training SAE: {args.train_sae}; Training Mask: {args.train_mask}")
        logging.info(
            f"Pretrained CLIP: {args.use_pretrained_clip}; Pretrained SAE: {args.use_pretrained_sae}; Pretrained Mask: {args.use_pretrained_mask}\n\n")

        logging.info(f"Log base path: {args.log_base_path}")
        logging.info(f"Path to load SAE from: {args.load_sae_checkpoint_path}")
        logging.info(
            f"Path to load CLIP model from: {args.load_clip_checkpoint_path}\n\n")

        logging.info(f"CLIP Load Config Name: {args.load_name}")
        logging.info(f"SAE Load Config Name: {args.load_sae_config_name}")

        logging.info(f"CLIP Config Name: {args.name}")
        logging.info(f"SAE Config Name: {args.sae_config_name}")
        logging.info(f"Mask Config Name: {args.mask_config_name}\n\n")

    job_id = os.getenv('SLURM_JOB_ID')
    step_gpu_name = os.getenv('SLURM_STEP_GPUS')
    job_gpu_name = os.getenv('SLURM_JOB_GPUS')
    slurm_host = os.getenv("SLURMD_NODENAME")
    host_name = slurm_host if slurm_host else socket.gethostname()
    if is_master(args):
        logging.info(
            f"Job ID: {job_id}, Step GPU Name: {step_gpu_name}, Job GPU Name: {job_gpu_name}")
        logging.info(f"Running on host: {host_name}")
    args.slurm_host_name = host_name

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    if is_master(args):
        args.tensorboard_path = os.path.join(
            log_base_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            # otherwise, list checkpoint dir contents and pick the newest checkpoint
            resume_from = get_latest_checkpoint(
                checkpoint_path, remote=args.remote_sync is not None)
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from
    else:
        if is_master(args):
            resume_from = os.path.join(args.checkpoint_path, f"checkpoint_{args.resume}.pt")
            if not osp.exists(resume_from):
                resume_from = os.path.join(args.checkpoint_path, f"epoch_{args.resume}.pt")

            if os.path.exists(resume_from):
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(
                    f'No latest resume checkpoint found in {resume_from}.')
                resume_from = None
            

    if args.distributed:
        # sync found checkpoint path to all ranks
        resume_from = broadcast_object(args, resume_from)
    args.resume = resume_from


    # start the sync proces if remote-sync is not None
    remote_sync_process = None


    if args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    dist_model = None
    args.distill = args.distill_model is not None and args.distill_pretrained is not None
    if args.distill:
        assert args.accum_freq == 1
        assert 'coca' not in args.model.lower()

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]
    model_kwargs = {}

        
    model_load_function = create_model_and_transforms


    # Different calls use different arguemnets in the function
    model, preprocess_train, preprocess_val = model_load_function(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,  # only effective for inference
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image,
        output_dict=True,
        cache_dir=args.cache_dir,
        **model_kwargs,
    )
    args.preprocess_val = preprocess_val
    original_model = copy.deepcopy(model)
    # For loading the original model
    if args.use_pretrained_clip and not args.use_smartclip_model:
        assert not args.use_real_clip and not args.use_alignclip_checkpoint and not args.use_openai_model
        original_clip_checkpoint_dict = pt_load(args.load_clip_checkpoint_path)
        sd = original_clip_checkpoint_dict["state_dict"]
        original_model.load_state_dict(sd)

    if args.distill:
        dist_model, _, _ = create_model_and_transforms(
            args.distill_model,
            args.distill_pretrained,
            device=device,
            precision=args.precision,
            output_dict=True,
            cache_dir=args.cache_dir,
        )

    # already done inside common_init


    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(log_base_path, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    ddp_args = {}
    if args.distributed and not args.horovod:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], **ddp_args)

        if args.distill:
            dist_model = torch.nn.parallel.DistributedDataParallel(
                dist_model, device_ids=[device], **ddp_args)

    if args.distributed:
        val_process_group = torch.distributed.new_group(ranks=[0])
    else:
        val_process_group = None

    if args.filter_algo_name !="Identity":
        filter_algo = mgap.filtering_algorithms.get_filtering_algorithm(
            args, device, **ddp_args, embed_dim=config.embed_dim_dict[args.model], hidden_size=config.unpooled_embed_dim_dict[args.model], num_heads=args.num_heads)
    else:
        filter_algo=None
    if args.filter_arch_name !="Identity":
        filter_obj = mgap.filtering_architectures.get_filtering_architecture(
            args, device, **ddp_args)
    else:
        filter_obj = None
    if args.filter_algo_name !="Identity":
        filter_algo.set_filtering_architecture(filter_obj)
        args.filter_algo = filter_algo

    if args.finetune_text_encoder and not args.finetune_vision_encoder:
        # freeze tehvisual component only 
        for param in model.visual.parameters():
            param.requires_grad = False
        if args.filter_algo_name != "Identity":
            filter_obj.freeze_parameters()


    # create optimizer and scaler
    optimizer = None
    scaler = None
    
    if not args.eval_only and (args.train_data or args.dataset_type == "synthetic" or args.train_clip or args.finetune_text_encoder):
        assert not args.trace, 'Cannot train with traced model'

        param_list = get_clip_parameters(model, args)

        if filter_algo is not None:
            param_list += filter_algo.get_optimizer_param_list()


        optimizer = optim.AdamW(param_list)


        scaler = None
        try:
            scaler = torch.amp.GradScaler(device=device)
        except (AttributeError, TypeError) as e:
            scaler = torch.cuda.amp.GradScaler()

    # optionally resume from a checkpoint
    start_epoch = 0
    step = 0
    
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        logging.info(f"Using checkpoint {args.resume}")
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
        elif 'step' in checkpoint and not args.cyclic_training:
            step = checkpoint["step"]
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            if is_master(args):
                logging.info(
                    f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            if not args.eval_only:
                if args.dataset_type != "mad" and args.dataset_type != "cocomad":
                    raise ValueError("Checkpoint does not contain epoch information.")
    else:
        checkpoint = None
        logging.info(f"No checkpoint found!!")


    if checkpoint is not None:
        if "epoch" in checkpoint:
            args.epoch_to_save = checkpoint["epoch"]
        else: 
            args.epoch_to_save = args.resume.split(".pt")[0].split("_")[-1]


        if (args.train_clip or args.finetune_text_encoder):
            clip_state_dict = checkpoint["clip_model"]
            if not args.distributed and next(iter(clip_state_dict.items()))[0].startswith('module'):
                clip_state_dict = {
                    k[len('module.'):]: v for k, v in clip_state_dict.items()}
            model.load_state_dict(clip_state_dict)

    elif args.use_pretrained_clip:
        if args.use_pretrained_clip:
            assert not args.use_real_clip and not args.use_alignclip_checkpoint and not args.use_openai_model
            clip_checkpoint_dict = pt_load(args.load_clip_checkpoint_path)
            clip_state_dict = clip_checkpoint_dict["state_dict"]
            if not args.distributed and next(iter(clip_state_dict.items()))[0].startswith('module'):
                clip_state_dict = {
                    k[len('module.'):]: v for k, v in clip_state_dict.items()}
            model.load_state_dict(clip_state_dict)

    
    # initialize datasets
    tokenizer = get_tokenizer(args.model, cache_dir=args.cache_dir)


    if (args.finetune_text_encoder or args.train_clip) and args.filter_algo_name != "Identity":
        filter_algo.set_model(model, tokenizer, device)

    # Path is accordingly set in common_init in utils
    mask_checkpoint_dict = None
    if resume_from is not None and args.train_mask:
        mask_checkpoint_dict = checkpoint
        if is_master(args):
            logging.info(f"Resumed mask checkpoint from {resume_from}")
    if mask_checkpoint_dict is not None:
        if not args.distributed and next(iter(mask_checkpoint_dict.items()))[0].startswith('module'):
            mask_checkpoint_dict = {
                k[len('module.'):]: v for k, v in mask_checkpoint_dict.items()}
        
        filter_algo.load_checkpoint(mask_checkpoint_dict)

    sae_checkpoint_dict = None
    if resume_from is not None and args.train_sae:
        sae_checkpoint_dict = checkpoint
        if is_master(args):
            logging.info(f"Resumed SAE checkpoint from {resume_from}")
    elif args.use_pretrained_sae and args.load_sae_checkpoint_path is not None and not args.random_sae:
        sae_checkpoint_dict = pt_load(args.load_sae_checkpoint_path)
        if is_master(args):
            logging.info(
                f"Loaded pretrained SAE checkpoint from {args.load_sae_checkpoint_path}")
    if sae_checkpoint_dict is not None:
        if not args.distributed and next(iter(sae_checkpoint_dict.items()))[0].startswith('module'):
            sae_checkpoint_dict = {
                k[len('module.'):]: v for k, v in sae_checkpoint_dict.items()}
        filter_algo.filter_obj.load_checkpoint(sae_checkpoint_dict)
        if "mse_scaler" in sae_checkpoint_dict:
            filter_algo.filter_obj.mse_scale = sae_checkpoint_dict["mse_scaler"]
        else:
            filter_algo.filter_obj.set_mse_scale_from_data()


    if args.filter_algo_name != "Identity":
        filter_algo.init_distributed()


    ################################ Getting data
    
    train_fnames = None
    num_train_pieces = 1

    args.actual_epochs = args.train_mask_epochs * num_train_pieces
    if not args.eval_only:
        if args.finetune_vision_encoder and not args.dataset_type == "mad" and not args.dataset_type == "cocomad":
            data = get_data(args, (preprocess_train, preprocess_val), epoch=start_epoch, tokenizer=tokenizer)
            total_steps = data["train"].dataloader.num_batches * args.actual_epochs
            if args.warmup_mask_epochs is not None:
                warmup_steps = args.warmup_mask_epochs * (total_steps // args.actual_epochs)
            else:
                warmup_steps = args.warmup
            

        #######################################################################################


    elif (args.dataset_name == 'cc12m' and (args.finetune_text_encoder and not args.finetune_vision_encoder)) or (args.filter_arch_name == "Identity" and args.filter_algo_name == "Identity"):
        print("\n")
        print("===================================>")
        print(f"Getting CC12M loaders for evaluation during training or oinference: ")
        data={} 
        data['val']= get_cc12m_dataset(args, is_train=False, epoch=start_epoch, tokenizer=tokenizer, val_split = 'val')
    else:
        preprocess_fns = (preprocess_train, preprocess_val)
        data = get_data(args, preprocess_fns, epoch=start_epoch, tokenizer=tokenizer)

    print(f"Compare cossims {args.compare_cossims_with_sae_bias}=======================================================> \n \n")
    if args.compare_cossims_with_sae_bias:
        print("Compare cossims  Inside=======================================================> \n \n")
        plot_histogram_to_compare_if_sae_bias_is_null_vector(original_model, args, tokenizer, filter_algo, data, is_train=False)
        sys.exit()


    assert len(data), 'At least one train or eval dataset must be specified.'

    if not args.eval_only:

        base_lr_dict = {}
        if (args.train_clip or args.finetune_text_encoder) and not args.frozen_clip:
            base_lr_dict['clip'] = args.lr
        if args.train_sae:
            base_lr_dict["filter_arch"] = args.train_sae_lr
        if args.train_mask:
            base_lr_dict["filter_algo"] = args.train_mask_lr


        scheduler = cosine_lr_custom(
            optimizer, base_lr_dict, warmup_steps, total_steps)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.wandb_name = utils.get_wandb_name(args)
        print(f"Actual wandb run name: {args.wandb_name}")
        wandb_log_dir = args.log_base_path
        os.environ["WANDB__SERVICE_WAIT"] = "500"
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project_name,
            name=args.wandb_name,
            notes=args.wandb_notes,
            tags=[],
            config=vars(args),
            id=args.wandb_name,
            resume='must' if (resume_from is not None and not args.debug) else "allow",
            mode=args.wandb_mode,
        )
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')
    else:
        logging.info('wandb not enabled.')

    # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    uncompiled_model = model

    collator_get_w = None

    args.orig_textorig_zero_shot_results = None
    args.orig_textorig_retrieval = None
    

    if is_master(args):
        if  (args.eval_only and args.resume is not None) or args.eval_before_train:
            logging.info(
                "No training data specified. Only performing evaluation.")
            # If using int8, convert to inference mode.
            # Evaluate.

            with torch.no_grad():

                if args.dataset_name == 'MAD':


                    val_loader = utils.get_mad_dataloader(args.load_clip_path, is_split='val', model=model, tokenizer=tokenizer, args=args, return_dataset=False, use_full_conditioning=False)


                    # manually_edit_sae_latent = True for fdw, when we do not want to use the learntmask but edit the sae latents by their true value

                    eval_logs, all_img_embeddings, all_textorig_embeddings, all_textft_embeddings, all_pr_embeddings = evaluate_mad(args, filter_algo, val_loader, tokenizer, model, original_model, dump_cossims=True)
                    print(eval_logs)


                else:
                    eval_logs, _, _, _, _ = evaluate_coco_controlled(data, 'coco-controlled-val', args, model, start_epoch, tokenizer, original_model, filter_algo)


                if resume_from is not None:
                    if "epoch" in checkpoint:
                        epoch_to_save = checkpoint["epoch"]
                    else: 
                        print("Could not find epoch information in checkpoint dict. Logging default epoch 25")
                        epoch_to_save = 25
                    eval_logs.update({'epoch_resume': int(epoch_to_save)})


    if not args.eval_only:
        logging.info("Starting training.")

        loss = create_loss(args)


        if args.finetune_text_encoder and not args.finetune_vision_encoder:
            if resume_from is not None:
                train_order = checkpoint['train_order']
                train_piece_idx = checkpoint['train_piece_idx']
                logging.info(f"Train order fetched from checkpoint dict: {train_order} and the index is: {train_piece_idx}")
            elif num_train_pieces > 1:
                train_order, train_piece_idx = set_train_order(args, num_train_pieces)


        best_config_train_loss = float('inf')

        for epoch in range(start_epoch, args.actual_epochs):
            if is_master(args):
                logging.info(
                    f"Current_epoch/actual_epochs : {epoch}/{args.actual_epochs} \n")

                start_time = time.time()

                    
            completed_epoch = epoch + 1
            args.completed_epoch = completed_epoch

            if is_master(args):
                logging.info(
                    f"Epoch: {(epoch//num_train_pieces)+1}, Actual epoch: {completed_epoch}, Num Samples: {data['train'].dataloader.num_samples}")

                # num_train_pieces * args.val_frequency)) == 0 or completed_epoch == args.actual_epochs))
            evaluate_this_epoch = (args.val_frequency != 0 and ((completed_epoch % args.val_frequency) == 0 or completed_epoch == args.actual_epochs))


            flair_data = None

                    
            step, total_train_loss_across_batches = train_one_epoch(model, data, loss, completed_epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=writer, tokenizer=tokenizer, filter_algo=filter_algo, evaluate_this_epoch=evaluate_this_epoch, train_mask_directly=args.train_mask_directly, step=step, collator_get_w=collator_get_w, flair_data=flair_data)


            if evaluate_this_epoch and is_master(args) and not args.dataset_name == "cocomad" and not args.dataset_name == "MAD" and not args.dataset_name == "cc12m":
                with torch.no_grad():
                    eval_logs, all_img_embeddings, all_textorig_embeddings, all_textft_embeddings, all_pr_embeddings = evaluate(data, 'val', args, model, completed_epoch, tokenizer, original_model, filter_algo, step, process_group=val_process_group)

                    if args.save_all_embeddings and completed_epoch == args.actual_epochs:
                        save_path = osp.join(
                            args.checkpoint_path, "..", "all_embeddings")
                        os.makedirs(save_path, exist_ok=True)
                        embeddings_dict = {"all_img_embeddings": all_img_embeddings, "all_textorig_embeddings": all_textorig_embeddings, "all_textft_embeddings": all_textft_embeddings, "all_pr_embeddings": all_pr_embeddings, "completed_epoch": completed_epoch}
                        torch.save(embeddings_dict, osp.join(
                            save_path, f"all_embeddings_dict_epo{completed_epoch}.pt"))
                        logging.info(f'All embeddings saved to {save_path}')
                    del all_img_embeddings, all_textorig_embeddings, all_textft_embeddings, all_pr_embeddings

                    
            # Saving checkpoints.
            if args.save_logs:
                checkpoint_dict = {
                    "epoch": completed_epoch,
                    "name": args.name,
                    "optimizer": optimizer.state_dict(),
                    "mask_name": args.mask_config_name,
                    "sae_name": args.sae_config_name,
                    "step": step,
                }
                if filter_algo is not None:
                    checkpoint_dict.update(filter_algo.get_checkpoint_dict())

                if args.finetune_text_encoder or args.train_clip:
                    checkpoint_dict["clip_model"] = uncompiled_model.state_dict()

                if scaler is not None:
                    checkpoint_dict["scaler"] = scaler.state_dict()

                if completed_epoch == args.actual_epochs or (args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0):

                    torch.save(checkpoint_dict, os.path.join(args.checkpoint_path, f"checkpoint_{completed_epoch}.pt"),)
                    logging.info(
                        f"Saved checkpoint at epoch {completed_epoch} at: {osp.join(args.checkpoint_path, f'checkpoint_{completed_epoch}.pt')}")


                if total_train_loss_across_batches < best_config_train_loss: 
                    # try not to corrupt the latest checkpoint if save fails
                    tmp_save_path = os.path.join(args.checkpoint_path, "tmp_best.pt")
                    latest_save_path = os.path.join(args.checkpoint_path, "epoch_best.pt")
                    torch.save(checkpoint_dict, tmp_save_path)
                    os.replace(tmp_save_path, latest_save_path)
                    logging.info(f"Saved the best checkpoint at: {latest_save_path}")
                    best_config_train_loss = total_train_loss_across_batches


            stop_now = False
            if is_master(args):
                end_time = time.time()
                delta = timedelta(seconds=end_time - start_time)
                if args.wandb:
                    wandb.save(args.log_path)
                if utils.check_and_terminate(threshold_seconds=delta.total_seconds()):
                    stop_now = True
            broadcast_object(args, stop_now)
            if stop_now:
                logging.info("Stopping training.")
                break

    if args.wandb and is_master(args):
        wandb.finish()

    # run a final sync.
    if remote_sync_process is not None:
        logging.info('Final remote sync.')
        remote_sync_process.terminate()
        result = remote_sync(
            os.path.join(args.log_base_path),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol
        )
        if result:
            logging.info('Final remote sync successful.')
        else:
            logging.info('Final remote sync failed.')

    if args.eval_only:
        write_to_csv(eval_logs, args)
        print("saved to csv ===========================>")


    if not args.eval_only:
        if completed_epoch == args.actual_epochs:
            previous_checkpoint = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
            if os.path.exists(previous_checkpoint):
                os.remove(previous_checkpoint)


if __name__ == "__main__":
    main(sys.argv[1:])

