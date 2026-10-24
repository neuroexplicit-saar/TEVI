import time
from datetime import timedelta
from mgap import utils
from mgap.filtering_algorithms import get_filtering_algorithm
from mgap.filtering_architectures import get_filtering_architecture
from mgap.utils import random_seed, get_latest_checkpoint, write_to_csv_original
from mgap import utils
from open_clip_train.file_utils import pt_load, remote_sync
from open_clip_train.train import train_one_epoch, evaluate

from open_clip_train.scheduler import cosine_lr
from open_clip_train.params import parse_args
from open_clip_train.logger import setup_logging
from open_clip_train.distributed import is_master, init_distributed_device, broadcast_object
from open_clip_train.data import get_data
from open_clip import create_model_and_transforms, get_tokenizer, create_loss
import logging
import os
import sys
from open_clip_train.precision import get_autocast


import torch
from torch import optim
import os.path as osp


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


def main(args, parser=None):
    args = parse_args(args, parser=parser)
    # Both should use pretrained models simultaneously
    if not args.use_pretrained_clip:
        print("WARNING-----------------> Not using pretrained CLIP model!")
    if not args.use_pretrained_sae:
        print("WARNING-----------------> Not using pretrained filter model!")


    args.autocast = get_autocast(args.precision, device_type=torch.device(args.device).type)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    utils.common_init(args)

    resume_latest = args.resume == 'latest'

    log_base_path = args.log_base_path
    filter_algo = get_filtering_algorithm(args)
    filter_obj = get_filtering_architecture(args)
    filter_algo.set_filtering_architecture(filter_obj)


    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
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
            resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
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
    random_seed(args.seed, 0)
    model_kwargs = {}


    model, preprocess_train, preprocess_val = create_model_and_transforms(
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
    if args.distill:
        dist_model, _, _ = create_model_and_transforms(
            args.distill_model,
            args.distill_pretrained,
            device=device,
            precision=args.precision,
            output_dict=True,
            cache_dir=args.cache_dir,
        )

    random_seed(args.seed, args.rank)


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

    if args.distributed and not args.horovod:
        ddp_args = {}
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)

        if args.distill:
            dist_model = torch.nn.parallel.DistributedDataParallel(dist_model, device_ids=[device], **ddp_args)

    if args.distributed:
        val_process_group = torch.distributed.new_group(ranks=[0])
    else:
        val_process_group = None

    # create optimizer and scaler
    optimizer = None
    scaler = None
    if args.train_data or args.dataset_type == "synthetic":
        assert not args.trace, 'Cannot train with traced model'

        def exclude(n, p): return p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        def include(n, p): return not exclude(n, p)

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]


        param_list = [{"params": gain_or_bias_params, "weight_decay": 0., "lr": args.lr, "betas": (args.beta1, args.beta2), "eps": args.eps},
                      {"params": rest_params, "weight_decay": args.wd, "lr": args.lr, "betas": (args.beta1, args.beta2), "eps": args.eps},
                      ]
        param_list += filter_algo.get_optimizer_param_list()


        optimizer = optim.AdamW(param_list)


        scaler = None
        try:
            scaler = torch.amp.GradScaler(device=device)
        except (AttributeError, TypeError) as e:
            scaler = torch.cuda.amp.GradScaler()

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")


    # load from pretrained model if not resuming and using pretrained models
    #         # sae_checkpoint_dict = pt_load(args.load_sae_checkpoint_path)
    #         # sae_state_dict = sae_checkpoint_dict["sae_model"]
    #         # args.ae.load_state_dict(sae_state_dict)


    # Added later so. that I can use this script to dump imagenet features

    # initialize datasets
    tokenizer = get_tokenizer(args.model, cache_dir=args.cache_dir)
    filter_algo.set_model(model, tokenizer, device)


    data = get_data(
        args,
        (preprocess_train, preprocess_val),
        epoch=start_epoch,
        tokenizer=tokenizer,
    )
    assert len(data), 'At least one train or eval dataset must be specified.'
    print(f"Datasets present in dataloader{data.keys()} =============================================================>")

    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
        if args.warmup_epochs is not None:
            warmup_steps = args.warmup_epochs * (total_steps // args.epochs)
        else:
            warmup_steps = args.warmup
        scheduler = cosine_lr(optimizer, args.lr, warmup_steps, total_steps)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        if args.train_data:
            args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project_name,
            name=args.wandb_name,
            notes=args.wandb_notes,
            tags=[],
            config=vars(args),
            id=args.wandb_name,
            resume='must' if args.resume == "latest" else "allow",
        )
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    original_model = model

    if 'train' not in data:
        logging.info("No training data specified. Only performing evaluation.")
        # If using int8, convert to inference mode.
        # Evaluate.
        eval_logs = evaluate(model, data, start_epoch, args, tb_writer=writer, tokenizer=tokenizer, process_group=val_process_group)
        eval_logs['epoch'] = start_epoch
        if args.save_eval_csv:
            write_to_csv_original(eval_logs, args)
            print("saved to csv ===========================>") 
        return

    logging.info("Starting training.")

    loss = create_loss(args)

    if args.eval_before_train and any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
        evaluate(model, data, 0, args, tb_writer=writer, tokenizer=tokenizer, process_group=val_process_group, filter_algo=filter_algo)

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=writer, tokenizer=tokenizer, filter_algo=filter_algo)
        completed_epoch = epoch + 1

        if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
            evaluate(model, data, completed_epoch, args, tb_writer=writer, tokenizer=tokenizer, process_group=val_process_group, filter_algo=filter_algo)

        # Saving checkpoints.
        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": original_model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            checkpoint_dict.update(filter_algo.get_checkpoint_dict())

            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )


        end_time = time.time()
        delta = timedelta(seconds=end_time - start_time)
        if utils.check_and_terminate(threshold_seconds=delta.total_seconds()):
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


if __name__ == "__main__":
    main(sys.argv[1:])
