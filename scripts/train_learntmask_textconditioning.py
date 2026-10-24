import torch
import mgap
import mgap.filtering_algorithms
import mgap.filtering_architectures
from open_clip_train.params import parse_args
from open_clip_train.scheduler import cosine_lr_custom
import sys
from torch.utils.data import DataLoader
import os
import os.path as osp
import wandb
import argparse
from mgap import utils
from open_clip_train.distributed import init_distributed_device
from tqdm import tqdm
import copy
from mgap import config, metrics, utils
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip_train.file_utils import pt_load
import socket

from open_clip_train.precision import get_autocast


from mgap import config
from mgap.datasets import LearntMaskTextConditioningEvalDataset, LearntMaskTextConditioningEvalCollator 
from open_clip import get_input_dtype


decoder_name_cum_len_list_of_cat = config.decoder_name_cum_len_list_of_cat
mad_decoder_name_list_cat_wise_literal = config.mad_decoder_name_list_cat_wise_literal


def train_one_epoch(args, filter_algo, optimizer, scheduler, train_loader, loss, epoch, step, evaluate_this_epoch, train_mask_directly=False, model=None):
    total_loss = 0.0
    total_contra_loss = 0.0
    total_inv_loss = 0.0
    num_samples = 0

    dims= 26

    if args.use_fixed_decoder_weights or args.use_fixed_latent_meanings: 
        all_mask_vals = torch.zeros(dims, dtype=torch.float32).to(args.device)
    else:
        all_mask_vals = torch.zeros(args.embed_dim*args.expansion_factor, dtype=torch.float32).to(args.device)

    if train_mask_directly:
        print("Training mask directly")
    if args.finetune_text_encoder:
        model.train()
        print("Finetuning text encoder")
    mask_accuracy = metrics.MADMaskPredictionTextAccuracy().to(args.device)
    input_dtype = get_input_dtype(args.precision) 

    for batch_idx, loader_dict in enumerate(tqdm(train_loader)):
        img_features = loader_dict["img_features"]
        text_features = loader_dict["text_features"]
        img_mask = loader_dict["img_mask"]
        text_mask = loader_dict["text_mask"]
        img_attributes = loader_dict["img_attributes"]
        img_mask_attribute_idxs = loader_dict["img_mask_attribute_idxs"]
        
        img_features = img_features.to(args.device)
        text_features = text_features.to(args.device)
        img_attributes = img_attributes.to(args.device)
        img_mask = img_mask.to(args.device)
        text_mask = text_mask.to(args.device)
        img_mask_attribute_idxs = img_mask_attribute_idxs.to(args.device)
        
        optimizer.zero_grad()
        
        if args.finetune_text_encoder:
            # The inputs here are actually text tokens when we finetune the text encoder
            # Text features when we directly load them instead of finetuning
            text_features = model.encode_text(text_features)
        if args.clip_logit_scale_fixed is not None:
            logit_scale = torch.tensor(args.clip_logit_scale_fixed)
        else: 
            logit_scale = model.logit_scale.exp()
        pred_features, sae_outputs_dict = filter_algo.get_filtered_embeddings(None, None, img_features, text_features, custom_inds=img_mask_attribute_idxs, text_mask=text_mask)
        
        img_features_norm = img_features / img_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        pred_features_norm = pred_features / pred_features.norm(dim=-1, keepdim=True)

        inv_loss = torch.tensor(0.0)
        
        if args.filter_arch_name != "Identity":
            mask_vals = sae_outputs_dict["info_updated"]["mask"]
            all_mask_vals += mask_vals.sum(dim=0)
            inds = sae_outputs_dict["info"]["inds"]
        
        if not train_mask_directly:
            if args.loss_name == "mse":
                loss_val = loss(pred_features, text_features)
            elif args.loss_name == "contra":
                loss_val = utils.get_clip_loss(pred_features_norm, text_features_norm, logit_scale, img_features.shape[0], args.device)
        else:
            pred_mask_selected, gt_mask_selected = mask_accuracy.get_mask_gt_from_gt_attrs(mask_vals, img_mask, text_mask, inds)
            if args.loss_name == "mse":
                loss_val = loss(pred_mask_selected, gt_mask_selected)
            else:
                raise NotImplementedError
        comb_loss = loss_val
        comb_loss.backward()
        optimizer.step()
        lr_dict = scheduler(step)
        step +=1
        total_loss += comb_loss.item() * img_features.shape[0]
        total_inv_loss += inv_loss.item() * img_features.shape[0]
        total_contra_loss += loss_val.item() * img_features.shape[0]
        num_samples += img_features.shape[0]

    total_loss /= num_samples
    total_inv_loss /= num_samples
    total_contra_loss /= num_samples
    all_mask_vals /= num_samples
    

    print(f"Epoch: {epoch}, Train Loss: {total_loss}, Train Contra Loss: {total_contra_loss}, Train Inv Loss: {total_inv_loss}")
    # Mask accuracy: {mask_accuracy_logging_dict}")
    print(f"Epoch: {epoch}, Mask vals: {all_mask_vals}")
    wandb_log_dict = {}
    wandb_log_dict["train/train_loss"] = total_loss
    wandb_log_dict["train/train_contra_loss"] = total_contra_loss
    wandb_log_dict["train/train_inv_loss"] = total_inv_loss
    for key, lr in lr_dict.items():
        wandb_log_dict[f"train/{key}_lr"] = lr
    if evaluate_this_epoch and args.wandb:
        wandb.log(wandb_log_dict, commit=False)
    elif args.wandb:
        wandb.log(wandb_log_dict, commit=True)

    return step


def evaluate(args, filter_algo, val_loader, loss, epoch, metric_list, load_path, model=None, original_model=None):
    total_loss = 0.0
    total_contra_loss = 0.0
    total_inv_loss = 0.0
    num_samples = 0
    orig_textorig_captioncossims = metrics.CaptionCosSimsSimple().to(args.device)
    orig_textft_captioncossims = metrics.CaptionCosSimsSimple().to(args.device)
    pr_textft_captioncossims = metrics.CaptionCosSimsSimple().to(args.device)
    mask_accuracy = metrics.MADMaskPredictionTextAccuracy().to(args.device)
    orig_textorig_mgap = metrics.ModalityGapLiang(embed_dim=args.embed_dim).to(args.device)
    orig_textorig_rmg = metrics.RMGSchrodi().to(args.device)
    orig_textft_mgap = metrics.ModalityGapLiang(embed_dim=args.embed_dim).to(args.device)
    orig_textft_rmg = metrics.RMGSchrodi().to(args.device)
    pr_textft_mgap = metrics.ModalityGapLiang(embed_dim=args.embed_dim).to(args.device)
    pr_textft_rmg = metrics.RMGSchrodi().to(args.device)
    all_img_embeddings = []
    all_textorig_embeddings = []
    all_textft_embeddings = []
    all_pr_embeddings = []
    zs_original_dict = {} # to store the zs on clip embeddings
    zs_dict = {}  # to store the zs on the filterred embeddings
    if args.finetune_text_encoder:
        model.eval()
        original_model.eval()
    # [zs_dict[cat_name]:{[cat_name]: {}} for cat_name in mad_decoder_name_list_cat_wise_literal[:-1]]

    zeroshot_weight_dict = utils.get_mad_zeroshot_eval_embeddings(model, tokenizer, device, args) 
    for cat_name in mad_decoder_name_list_cat_wise_literal[:-1]: 
        zeroshot_weight = zeroshot_weight_dict[cat_name]
        zs_original_dict[cat_name]=   metrics.MADZeroShot(cat_name, zeroshot_weight).to(args.device)

    for cat_name in mad_decoder_name_list_cat_wise_literal[:-1]: 
        zs_dict[cat_name]= {}      
        
        for cat_name_later in mad_decoder_name_list_cat_wise_literal:
            zeroshot_weight = zeroshot_weight_dict[cat_name_later]
            zs_dict[cat_name][cat_name_later]= {}
            zs_dict[cat_name][cat_name_later]['positive'] = metrics.MADZeroShot(cat_name_later, zeroshot_weight).to(args.device)
            zs_dict[cat_name][cat_name_later]['negative'] = metrics.MADZeroShot(cat_name_later, zeroshot_weight).to(args.device)
            

    for batch_idx, loader_dict in enumerate(tqdm(val_loader)):

        img_features = loader_dict["img_features"]
        load_text_features = loader_dict["text_features"]
        img_mask = loader_dict["img_mask"] # attributes that actually are present in the image
        text_mask = loader_dict["text_mask"] # attributes that are present in teh partial cpation
        all_templates = loader_dict["templates"]
        img_attributes = loader_dict["img_attributes"]
        img_mask_attribute_idxs = loader_dict["img_mask_attribute_idxs"]
        
        img_features = img_features.to(args.device)
        load_text_features = load_text_features.to(args.device)
        img_attributes = img_attributes.to(args.device)
        img_mask = img_mask.to(args.device)
        text_mask = text_mask.to(args.device)
        img_mask_attribute_idxs = img_mask_attribute_idxs.to(args.device)
        for cat_name, v in all_templates.items():
            for type_of_template, v1 in v.items():
                if 'features' in type_of_template:
                    all_templates[cat_name][type_of_template] = v1.to(args.device) # type_of_template : 'positive', 'negative'
        if args.finetune_text_encoder:
            text_features = model.encode_text(load_text_features)
            original_text_features = original_model.encode_text(load_text_features)
        if args.clip_logit_scale_fixed is not None:
            logit_scale = torch.tensor(args.clip_logit_scale_fixed)
        else: 
            logit_scale = model.logit_scale.exp()
        pred_features, sae_outputs_dict = filter_algo.get_filtered_embeddings(None, None, img_features, text_features, custom_inds=img_mask_attribute_idxs, text_mask=text_mask)
        
        img_features_norm = img_features / img_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        pred_features_norm = pred_features / pred_features.norm(dim=-1, keepdim=True)
        original_text_features_norm = original_text_features / original_text_features.norm(dim=-1, keepdim=True)
        all_img_embeddings.append(img_features.detach().cpu())
        all_textorig_embeddings.append(original_text_features.detach().cpu())
        all_textft_embeddings.append(text_features.detach().cpu())
        all_pr_embeddings.append(pred_features.detach().cpu())

        inv_loss = torch.tensor(0.0)

        if args.loss_name == "mse":
            loss_val = loss(pred_features, text_features)
        elif args.loss_name == "contra":
            loss_val = utils.get_clip_loss(pred_features_norm, text_features_norm, logit_scale, img_features.shape[0], args.device)


        for cat_idx, cat_name in enumerate(mad_decoder_name_list_cat_wise_literal[:-1]): 
            zs_original_dict[cat_name].update(img_features, img_attributes[:, cat_idx])
        

        for cat_name in mad_decoder_name_list_cat_wise_literal[:-1]: 
            positive_template_features = all_templates[cat_name]["positive_template_features"]
            negative_template_features = all_templates[cat_name]["negative_template_features"]
            positive_texts = all_templates[cat_name]["positive_template_texts"]
            negative_texts = all_templates[cat_name]["negative_template_texts"]
            positive_mask = utils.generate_attributes_from_texts(positive_texts).to(args.device)
            negative_mask = utils.generate_attributes_from_texts(negative_texts).to(args.device)
            positive_pred_features, _ = filter_algo.get_filtered_embeddings(None, None, img_features, positive_template_features, custom_inds=img_mask_attribute_idxs, text_mask=positive_mask)
            negative_pred_features, _ = filter_algo.get_filtered_embeddings(None, None, img_features,negative_template_features, custom_inds=img_mask_attribute_idxs, text_mask=negative_mask)

            for cat_idx, cat_name_later in enumerate(mad_decoder_name_list_cat_wise_literal):
                zs_dict[cat_name][cat_name_later]['positive'].update(positive_pred_features, img_attributes[:, cat_idx])
                zs_dict[cat_name][cat_name_later]['negative'].update(negative_pred_features, img_attributes[:, cat_idx])
        
        #     # positive_templates = all_templates[cat_name]["positive_template_features"]
        #     # negative_templates = all_templates[cat_name]["negative_template_features"]


        comb_loss = loss_val
        total_loss += comb_loss.item() * img_features.shape[0]
        total_inv_loss += inv_loss.item() * img_features.shape[0]
        total_contra_loss += loss_val.item() * img_features.shape[0]
        num_samples += img_features.shape[0]

        
        orig_textorig_cossims = img_features_norm @ original_text_features_norm.T
        orig_textft_cossims = img_features_norm @ text_features_norm.T
        pr_textft_cossims = pred_features_norm @ text_features_norm.T

        orig_textorig_captioncossims.update(orig_textorig_cossims)
        orig_textft_captioncossims.update(orig_textft_cossims)
        pr_textft_captioncossims.update(pr_textft_cossims)
        orig_textorig_mgap.update(img_features_norm, original_text_features_norm)
        orig_textorig_rmg.update(img_features_norm, original_text_features_norm)
        orig_textft_mgap.update(img_features_norm, text_features_norm)
        orig_textft_rmg.update(img_features_norm, text_features_norm)
        pr_textft_mgap.update(pred_features_norm, text_features_norm)
        pr_textft_rmg.update(pred_features_norm, text_features_norm)

        if args.filter_arch_name != "Identity":
            mask_vals = sae_outputs_dict["info_updated"]["mask"]
            inds = sae_outputs_dict["info"]["inds"]
            mask_accuracy.update(mask_vals, img_mask, text_mask, inds)
    total_loss /= num_samples
    total_inv_loss /= num_samples
    total_contra_loss /= num_samples

    all_img_embeddings = torch.cat(all_img_embeddings, dim=0)
    all_textorig_embeddings = torch.cat(all_textorig_embeddings, dim=0)
    all_textft_embeddings = torch.cat(all_textft_embeddings, dim=0)
    all_pr_embeddings = torch.cat(all_pr_embeddings, dim=0)

    orig_textorig_captioncossims_vals, _ = orig_textorig_captioncossims.get_logging_dict(panel_name="val", prefix="orig_textorig")
    orig_textft_captioncossims_vals, _ = orig_textft_captioncossims.get_logging_dict(panel_name="val", prefix="orig_textft")
    pr_textft_captioncossims_vals, _ = pr_textft_captioncossims.get_logging_dict(panel_name="val", prefix="pr_textft")
    all_mgap_rmg_vals = orig_textorig_mgap.get_logging_dict(panel_name="val", prefix="orig_textorig")
    all_mgap_rmg_vals.update(orig_textorig_rmg.get_logging_dict(panel_name="val", prefix="orig_textorig"))
    all_mgap_rmg_vals.update(orig_textft_mgap.get_logging_dict(panel_name="val", prefix="orig_textft"))
    all_mgap_rmg_vals.update(orig_textft_rmg.get_logging_dict(panel_name="val", prefix="orig_textft"))
    all_mgap_rmg_vals.update(pr_textft_mgap.get_logging_dict(panel_name="val", prefix="pr_textft"))
    all_mgap_rmg_vals.update(pr_textft_rmg.get_logging_dict(panel_name="val", prefix="pr_textft"))
    orig_mgap_vals = {k: v for k, v in all_mgap_rmg_vals.items() if "orig" in k}
    pr_mgap_vals = {k: v for k, v in all_mgap_rmg_vals.items() if "pr" in k}


    wandb_log_dict = {}
    zs_vals = {}
    zs_vals["pr_mean_positive"] = 0
    zs_vals["pr_mean_negative"] = 0
    zs_vals["pr_mean_difference"] = 0
    for cat_name in mad_decoder_name_list_cat_wise_literal[:-1]: 
        zs_vals[cat_name] = {}
        for cat_idx, cat_name_later in enumerate(mad_decoder_name_list_cat_wise_literal):
            zs_vals[cat_name][cat_name_later] = {}
            zs_vals[cat_name][cat_name_later]['positive'] = zs_dict[cat_name][cat_name_later]['positive'].compute()["accuracy"]
            zs_vals[cat_name][cat_name_later]['negative'] = zs_dict[cat_name][cat_name_later]['negative'].compute()["accuracy"]
            if cat_name == cat_name_later:
                zs_vals["pr_mean_positive"] += zs_vals[cat_name][cat_name_later]['positive']
                zs_vals["pr_mean_negative"] += zs_vals[cat_name][cat_name_later]['negative']
                zs_vals["pr_mean_difference"] += (zs_vals[cat_name][cat_name_later]['positive'] - zs_vals[cat_name][cat_name_later]['negative'])
            wandb_log_dict[f"{cat_name}-/{cat_name_later}/positive"]= zs_vals[cat_name][cat_name_later]['positive']
            wandb_log_dict[f"{cat_name}-/{cat_name_later}/negative"]= zs_vals[cat_name][cat_name_later]['negative']
            wandb_log_dict[f"{cat_name}-/{cat_name_later}/difference"]= zs_vals[cat_name][cat_name_later]['positive'] - zs_vals[cat_name][cat_name_later]['negative']
    zs_vals["pr_mean_positive"] /= len(mad_decoder_name_list_cat_wise_literal[:-1])
    zs_vals["pr_mean_negative"] /= len(mad_decoder_name_list_cat_wise_literal[:-1])
    zs_vals["pr_mean_difference"] /= len(mad_decoder_name_list_cat_wise_literal[:-1])
    wandb_log_dict["mean/pr_mean_positive"]= zs_vals["pr_mean_positive"]
    wandb_log_dict["mean/pr_mean_negative"]= zs_vals["pr_mean_negative"]
    wandb_log_dict["mean/pr_mean_difference"]= zs_vals["pr_mean_difference"]
            

    zs_original_vals= {}
    zs_original_vals["mean"] = 0
    for cat_idx, cat_name in enumerate(mad_decoder_name_list_cat_wise_literal[:-1]): 
        zs_original_vals[cat_name]= zs_original_dict[cat_name].compute()["accuracy"]
        zs_original_vals["mean"] += zs_original_vals[cat_name]
        wandb_log_dict[f"val_original_zeroshot/{cat_name}"]= zs_original_vals[cat_name]
    zs_original_vals["mean"] /= len(mad_decoder_name_list_cat_wise_literal[:-1])
    wandb_log_dict["mean/original"]= zs_original_vals["mean"]
    
    if args.filter_arch_name != "Identity":
        mask_accuracy_logging_dict = mask_accuracy.get_logging_dict(split="val")
    else:
        mask_accuracy_logging_dict = {}
    print(f"Epoch: {epoch}, Val Loss: {total_loss}, Val Contra Loss: {total_contra_loss}, Val Inv Loss: {total_inv_loss}, Orig caption cossims: {orig_textorig_captioncossims_vals}, Pr caption cossims: {pr_textft_captioncossims_vals}, Mask accuracy: {mask_accuracy_logging_dict}, val_original_zeroshot: {zs_original_vals}")
    
    print("Zero shot: ")
    for cat_name in mad_decoder_name_list_cat_wise_literal[:-1]:
        print(f"{cat_name}:")
        for cat_name_later in mad_decoder_name_list_cat_wise_literal:
            print(f"{cat_name_later}: {zs_vals[cat_name][cat_name_later]}")
        print("\n")

    
    wandb_log_dict["val/val_loss"] = total_loss
    wandb_log_dict["val/val_inv_loss"] = total_inv_loss
    wandb_log_dict["val/val_contra_loss"] = total_contra_loss
    for k, v in mask_accuracy_logging_dict.items():
        wandb_log_dict[f"val/mask_accuracy.{k}"] = v
    for k, v in orig_textorig_captioncossims_vals.items():
        wandb_log_dict[f"val/{k}"] = v
    for k, v in orig_textft_captioncossims_vals.items():
        wandb_log_dict[f"val/{k}"] = v
    for k, v in pr_textft_captioncossims_vals.items():
        wandb_log_dict[f"val/{k}"] = v
    for k, v in orig_mgap_vals.items():
        wandb_log_dict[f"val/original.{k}"] = v
    for k, v in pr_mgap_vals.items():
        wandb_log_dict[f"val/pr.{k}"] = v
    if args.wandb:
        wandb.log(wandb_log_dict, commit=True)
    return wandb_log_dict, all_img_embeddings, all_textorig_embeddings, all_textft_embeddings, all_pr_embeddings
        

parser = argparse.ArgumentParser()
# --save-eval-csv is defined in open_clip_train/params.py; re-adding it here raises an argparse conflict
parser.add_argument("--save-all-embeddings", action="store_true", default=False)
parser.add_argument("--save-umap-plots", action="store_true", default=False)
args = parse_args(args=None, parser=parser)
device = init_distributed_device(args)

utils.common_init(args, mask_config_suffix="-textconditioning")
job_id = os.getenv('SLURM_JOB_ID')
step_gpu_name = os.getenv('SLURM_STEP_GPUS')
job_gpu_name = os.getenv('SLURM_JOB_GPUS')
print(f"Job ID: {job_id}, Step GPU Name: {step_gpu_name}, Job GPU Name: {job_gpu_name}")
slurm_host = os.getenv("SLURMD_NODENAME")
host_name = slurm_host if slurm_host else socket.gethostname()
print(f"Running on host: {host_name}")
args.slurm_host_name = host_name

print(f"\n CLIP config Name: {args.name}")
print("-------------------------------------------------------")

print(f"\n SAE Config Name: {args.sae_config_name}")
print("-------------------------------------------------------")

print(f"\n Mask Config Name: {args.mask_config_name}")
print("-------------------------------------------------------")

if os.path.exists(os.path.join(args.checkpoint_path, f"{args.mask_config_name}.pt")) and not args.resume_combined_training:
    print(f"Checkpoint already exists at {args.checkpoint_path}")
    sys.exit(0)


if args.finetune_text_encoder:
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
    )
    original_model = copy.deepcopy(model)
    original_load_path = os.path.join(args.load_clip_path, "checkpoints", f"epoch_{args.pretrained_specific_checkpoint_path}.pt")
    original_clip_checkpoint_dict = pt_load(original_load_path)
    original_model.load_state_dict(original_clip_checkpoint_dict["state_dict"])
    
    if args.use_pretrained_clip:
        clip_checkpoint_dict = pt_load(args.load_clip_checkpoint_path)
        clip_state_dict = clip_checkpoint_dict["state_dict"]
        model.load_state_dict(clip_state_dict)
        tokenizer = get_tokenizer(args.model, cache_dir=args.cache_dir)
    
else:
    model = None
    tokenizer = None


if args.wandb:
    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project_name,
        name=args.wandb_name,
        notes=args.wandb_notes,
        tags=[],
        config=vars(args),
    )


filter_algo = mgap.filtering_algorithms.get_filtering_algorithm(args)
filter_obj = mgap.filtering_architectures.get_filtering_architecture(args)
filter_algo.set_filtering_architecture(filter_obj)


if args.finetune_text_encoder:
    filter_algo.set_model(model, tokenizer, device)

if not args.eval_only:
    optimizer_param_list = filter_algo.get_optimizer_param_list()
    

    if args.finetune_text_encoder:
        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]


        optimizer_param_list += [{"params": gain_or_bias_params, "weight_decay": 0., "lr":args.lr, "betas":(args.beta1, args.beta2),"eps":args.eps, 'name':'clip'},
                            {"params": rest_params, "weight_decay": args.wd, "lr":args.lr, "betas":(args.beta1, args.beta2),"eps":args.eps, 'name':'clip'},
                ]
    optimizer = torch.optim.AdamW(optimizer_param_list)
    args.optimizer_name = "AdamW"


    train_img_features_attributes = torch.load(osp.join(args.load_clip_path, 'train_features_and_attributes.pt'), map_location="cpu")
    train_img_features = train_img_features_attributes['features']
    train_img_attributes = train_img_features_attributes['attribute_lists']

    train_text_features_attributes = torch.load(osp.join(args.load_clip_path, 'train_text_features_and_multihotGT.pt'), map_location="cpu")
    train_text_features = train_text_features_attributes['features']
    train_img_mask = train_text_features_attributes['multihotGT']
    train_text_mask = train_text_features_attributes['multihot_partial_caption']

    train_texts = train_text_features_attributes['partial_caption_list']
    if args.finetune_text_encoder:
        
        train_tokenized_texts = tokenizer(train_texts)
        train_dataset = LearntMaskTextConditioningEvalDataset(train_img_features, train_tokenized_texts, train_img_mask, train_text_mask, positive_negative_templates=None, img_attributes = train_img_attributes, texts= train_texts)
    else:
        train_dataset = LearntMaskTextConditioningEvalDataset(train_img_features, train_text_features, train_img_mask, train_text_mask, positive_negative_templates=None, img_attributes= train_img_attributes, texts=train_texts)
    train_loader = DataLoader(train_dataset, batch_size=args.train_mask_batch_size, shuffle=True, collate_fn=LearntMaskTextConditioningEvalCollator()) #, num_workers = 4, pin_memory=True

    total_steps= (len(train_dataset) // args.train_mask_batch_size) * args.train_mask_epochs

    warmup_steps = args.warmup_mask_epochs * (total_steps // args.train_mask_epochs)


    base_lr_dict= {}
    if args.train_clip or args.finetune_text_encoder: 
        base_lr_dict['clip'] = args.lr
    if args.train_mask: 
        base_lr_dict["filter_algo"] = args.train_mask_lr

    scheduler = cosine_lr_custom(optimizer, base_lr_dict, warmup_steps, total_steps)


val_img_features_attributes = torch.load(osp.join(args.load_clip_path, 'val_features_and_attributes.pt'), map_location="cpu")
val_img_features = val_img_features_attributes['features']
val_img_attributes = val_img_features_attributes['attribute_lists']
val_text_features_attributes = torch.load(osp.join(args.load_clip_path, 'val_text_features_and_multihotGT.pt'), map_location="cpu")

val_text_features = val_text_features_attributes['features']

val_img_mask = val_text_features_attributes['multihotGT']
val_text_mask = val_text_features_attributes['multihot_partial_caption']

# Used to get the saved embeddings of the templates, but since we are finetuning text encoder, we need to genearte on teh go


# preparing this before training  to evaluate 
val_texts = val_text_features_attributes['partial_caption_list']
if args.finetune_text_encoder:
    val_tokenized_texts = tokenizer(val_texts)
    val_dataset = LearntMaskTextConditioningEvalDataset(val_img_features, val_tokenized_texts, val_img_mask, val_text_mask, positive_negative_templates=None, img_attributes = val_img_attributes, texts= val_texts)
else:
    val_dataset = LearntMaskTextConditioningEvalDataset(val_img_features, val_text_features, val_img_mask, val_text_mask, positive_negative_templates=None, img_attributes = val_img_attributes, texts= val_texts)

val_loader = DataLoader(val_dataset, batch_size=args.train_mask_batch_size, shuffle=False, collate_fn=LearntMaskTextConditioningEvalCollator())


if args.loss_name == "mse":
    loss = torch.nn.MSELoss()
elif args.loss_name == "contra": 
    loss = utils.get_clip_loss

args.autocast = get_autocast(
        args.precision, device_type=torch.device(args.device).type)
metric_list = []
step = 0


if not args.eval_only:
    for e in range(args.train_mask_epochs):
        evaluate_this_epoch = True
        step = train_one_epoch(args, filter_algo, optimizer, scheduler, train_loader, loss, e, step, evaluate_this_epoch, train_mask_directly=args.train_mask_directly, model=model)
        

        checkpoint = filter_algo.get_checkpoint_dict()
        if args.finetune_text_encoder:
            checkpoint["clip_model"] = model.state_dict()
        checkpoint['epoch'] = e+1
        os.makedirs(args.checkpoint_path, exist_ok=True)

        if e % args.save_frequency == 0: 
            torch.save(checkpoint, osp.join(args.checkpoint_path, f"{args.mask_config_name}_epoch_{e+1}.pt"))

    
    with open(osp.join(args.checkpoint_path, "args.txt"), "w") as f:
        f.write(str(args))
    torch.save(checkpoint, osp.join(args.checkpoint_path, f"{args.mask_config_name}.pt"))

