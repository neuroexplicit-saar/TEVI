from seaborn import heatmap
from random import shuffle
from random import shuffle
import json
import logging
import math
import os
import time


import numpy as np
import torch
import torch.distributed
from mgap import utils, metrics
from tqdm import tqdm
import os.path as osp
import matplotlib.pyplot as plt
from open_clip import OPENAI_DATASET_INVERSE_MEAN, OPENAI_DATASET_INVERSE_STD
# , IMAGENET_INVERSE_MEAN, IMAGENET_INVERSE_STD
import textwrap
try:
    import flair
except ImportError:  # only needed for the retrieval benchmarks
    flair = None

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype
from open_clip import COCO_CLASSNAMES
from mgap.config import CAPTION_LIST
from open_clip_train.distributed import is_master
from open_clip.loss import COCOControlledLoss


from open_clip_train.train import MetricWrapperFinetune, EmbeddingsWrapper, RetrievalWrapper, ZeroShotWrapper, RetrievalWrapperCocoControlled
from mgap import config
from mgap.utils import get_cossims_for_correlation, get_correlations
from mgap.utils import plot_correlation, plot_topk_mask
from mgap.utils import gpu_mem
from open_clip_train.train import AverageMeter, unwrap_model, backward, update_log_data_for_wandb, update_cumulative_losses, normalize_cumulative_losses, MetricWrapper, maybe_compute_generative_loss, get_clip_loss

import seaborn
from open_clip import COCO_CLASSNAMES
import matplotlib.pyplot as plt
import numpy as np
import random


decoder_name_cum_len_list_of_cat = config.decoder_name_cum_len_list_of_cat
mad_decoder_name_list_cat_wise_literal = config.mad_decoder_name_list_cat_wise_literal


import torch
import matplotlib.pyplot as plt


def evaluate_mad(args, filter_algo, val_loader, tokenizer, model=None, original_model=None, dump_cossims=False, manually_edit_sae_latent=False):

    if args.finetune_text_encoder and args.train_clip:
        model.eval()
        original_model.eval()
    device = args.device
   
    orig_textorig_captioncossims = metrics.CaptionCosSimsSimple(return_pos_cossims = True).to(args.device)
    orig_textft_captioncossims = metrics.CaptionCosSimsSimple(return_pos_cossims = True).to(args.device)
    pr_textft_captioncossims = metrics.CaptionCosSimsSimple(return_pos_cossims = True).to(args.device)
    
    mask_accuracy = metrics.MADMaskPredictionTextAccuracy().to(args.device)
    
    orig_textorig_mgap = metrics.ModalityGapLiang(embed_dim=args.embed_dim).to(args.device)
    orig_textft_mgap = metrics.ModalityGapLiang(embed_dim=args.embed_dim).to(args.device)
    pr_textft_mgap = metrics.ModalityGapLiang(embed_dim=args.embed_dim).to(args.device)

    orig_textorig_rmg = metrics.RMGSchrodi().to(args.device)
    orig_textft_rmg = metrics.RMGSchrodi().to(args.device)
    pr_textft_rmg = metrics.RMGSchrodi().to(args.device)

    all_img_embeddings = []
    all_textorig_embeddings = []
    all_textft_embeddings = []
    all_pr_embeddings = []
    wandb_log_dict = {}

    zs_original_dict = {}  # to store the zs on clip embeddings
    zs_dict = {}  # to store the zs on the filterred embeddings
    zeroshot_weight_dict = utils.get_mad_zeroshot_eval_embeddings(model, tokenizer, device, args)
    zeroshot_original_dict = utils.get_mad_zeroshot_eval_embeddings(original_model, tokenizer, device, args)
    for cat_name in mad_decoder_name_list_cat_wise_literal:
        zeroshot_original_weight = zeroshot_original_dict[cat_name].to(args.device)
        zs_original_dict[cat_name] = metrics.MADZeroShot(cat_name, zeroshot_original_weight).to(args.device)

    bucket_types = ['present_in_both', 'wrong_thing_present_in_caption', 'caption_does_not_contain_it']
    total_counts_cat_bucket_wise = {cat_name: {bucket_type: 0.0 for bucket_type in bucket_types} for cat_name in mad_decoder_name_list_cat_wise_literal}
    correct_counts_cat_bucket_wise = {cat_name: {bucket_type: 0.0 for bucket_type in bucket_types} for cat_name in mad_decoder_name_list_cat_wise_literal}

    list_idxs_start = decoder_name_cum_len_list_of_cat
    list_idxs_end = decoder_name_cum_len_list_of_cat[1:] + [26]
    

    for batch_idx, loader_dict in enumerate(tqdm(val_loader)):

        img_features = loader_dict["img_features"]
        img_attributes = loader_dict["img_attributes"]
        img_mask = loader_dict["img_mask"]  # attributes that actually are present in the image
        text_mask = loader_dict["text_mask"]  # attributes that are present in teh partial cpation
        img_mask_attribute_idxs = loader_dict["img_mask_attribute_idxs"]
        texts = loader_dict["texts"]

        img_features = img_features.to(args.device)
        img_attributes = img_attributes.to(args.device)
        img_mask = img_mask.to(args.device)
        text_mask = text_mask.to(args.device)
        img_mask_attribute_idxs = img_mask_attribute_idxs.to(args.device)


        for cat_idx, cat_name in enumerate(mad_decoder_name_list_cat_wise_literal):
            zs_original_dict[cat_name].update(img_features, img_attributes[:, cat_idx])

        tokenized_texts = tokenizer(texts).to(device)
        if args.finetune_text_encoder and args.train_clip:
            text_features = model.encode_text(tokenized_texts)
            original_text_features = original_model.encode_text(tokenized_texts)
        
        
        if manually_edit_sae_latent: 
            info = filter_algo.filter_obj.get_latents(img_features)
            latents = torch.relu(info['latents_pre_act'])
            inds = info['inds']
            vals = info['vals']
            updated_latents = latents * text_mask
            updated_vals = torch.gather(updated_latents, dim=1, index=inds)
            info_updated = {'inds': inds, 'vals': updated_vals}
            pred_features = filter_algo.filter_obj.get_reconstructed_features(
            img_features, info_updated)
        else:
            pred_features, sae_outputs_dict = filter_algo.get_filtered_embeddings(None, None, img_features, text_features, text_mask=text_mask)


        img_features_norm = img_features / img_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        pred_features_norm = pred_features / pred_features.norm(dim=-1, keepdim=True)
        original_text_features_norm = original_text_features / original_text_features.norm(dim=-1, keepdim=True)

        all_img_embeddings.append(img_features.detach().cpu())
        all_textorig_embeddings.append(original_text_features.detach().cpu())
        all_textft_embeddings.append(text_features.detach().cpu())
        all_pr_embeddings.append(pred_features.detach().cpu())

        inv_loss = torch.tensor(0.0).to(args.device)

        cossim_args = {}

         
        orig_textorig_cossims = img_features_norm @ original_text_features_norm.T
        orig_textft_cossims = img_features_norm @ text_features_norm.T
        pr_textft_cossims = pred_features_norm @ text_features_norm.T


        orig_textorig_captioncossims.update(orig_textorig_cossims)
        orig_textft_captioncossims.update(orig_textft_cossims)
        pr_textft_captioncossims.update(pr_textft_cossims)

        orig_textorig_mgap.update(img_features_norm, original_text_features_norm)
        orig_textft_mgap.update(img_features_norm, text_features_norm)
        pr_textft_mgap.update(pred_features_norm, text_features_norm)


        if args.filter_arch_name != "Identity" and args.expansion_factor==1:
            mask_vals = sae_outputs_dict["info_updated"]["mask"]
            inds = sae_outputs_dict["info"]["inds"]
            mask_accuracy.update(mask_vals, img_mask, text_mask, inds)
    
        for cat_idx, cat_name in enumerate(mad_decoder_name_list_cat_wise_literal):
            col_start = list_idxs_start[cat_idx]
            col_end = list_idxs_end[cat_idx]


            zeroshot_weight = zeroshot_weight_dict[cat_name].to(args.device)
            hits = utils.get_mad_zeroshot_full(pred_features, img_attributes[:, cat_idx], zeroshot_weight, cat_name, use_extra_ortho_dim=args.use_extra_ortho_dim)
            A = img_mask[:, col_start:col_end]
            B = text_mask[:, col_start:col_end]
            A_pos = A.argmax(dim=1)
            B_pos = B.argmax(dim=1)

            # Mask where B has a 1 (since B_pos=0 could mean column 0 or no '1')
            B_has_1 = B.any(dim=1)


            present_in_both = (B_has_1 & (A_pos == B_pos)).int()  # 1 if same column has 1 in A and B
            wrong_thing_present_in_caption = (B_has_1 & (A_pos != B_pos)).int()  # 1 if A and B have 1s in different columns
            caption_does_not_contain_it = (~B_has_1).int()


            total_counts_cat_bucket_wise[cat_name]['present_in_both'] += present_in_both.sum().item()
            total_counts_cat_bucket_wise[cat_name]['wrong_thing_present_in_caption'] += wrong_thing_present_in_caption.sum().item()
            total_counts_cat_bucket_wise[cat_name]['caption_does_not_contain_it'] += caption_does_not_contain_it.sum().item()
            correct_counts_cat_bucket_wise[cat_name]['present_in_both'] += (present_in_both * hits).sum().item()
            correct_counts_cat_bucket_wise[cat_name]['wrong_thing_present_in_caption'] += (wrong_thing_present_in_caption * hits).sum().item()
            correct_counts_cat_bucket_wise[cat_name]['caption_does_not_contain_it'] += (caption_does_not_contain_it * hits).sum().item()


    orig_acc_tensor = torch.full((len(mad_decoder_name_list_cat_wise_literal),), -1.0)
    zs_original_vals = {}
    zs_original_vals["mean"] = 0
    for cat_idx, cat_name in enumerate(mad_decoder_name_list_cat_wise_literal):
        zs_original_vals[cat_name] = zs_original_dict[cat_name].compute()["accuracy"]
        zs_original_vals["mean"] += zs_original_vals[cat_name]
        orig_acc_tensor[cat_idx] = zs_original_vals[cat_name]
    zs_original_vals["mean"] /= len(mad_decoder_name_list_cat_wise_literal)
    print(f"\n")
    print(f"Category names: {mad_decoder_name_list_cat_wise_literal}")
    print(f"Accuracy tensor: {orig_acc_tensor.numpy()}") 


    all_img_embeddings = torch.cat(all_img_embeddings, dim=0)
    all_textorig_embeddings = torch.cat(all_textorig_embeddings, dim=0)
    all_textft_embeddings = torch.cat(all_textft_embeddings, dim=0)
    all_pr_embeddings = torch.cat(all_pr_embeddings, dim=0)

    N= 5000
    indices = torch.randperm(len(all_img_embeddings))[:N]  # Random permutation of indices

    orig_textorig_rmg.update(all_img_embeddings[indices], all_textorig_embeddings[indices], normalized = False)
    orig_textft_rmg.update(all_img_embeddings[indices], all_textft_embeddings[indices], normalized = False)
    pr_textft_rmg.update(all_pr_embeddings[indices], all_textft_embeddings[indices], normalized = False)


    orig_textorig_captioncossims_vals, val_orig_textorig_pos_cossims = orig_textorig_captioncossims.get_logging_dict(panel_name="val", prefix="orig_textorig")
    orig_textft_captioncossims_vals, val_orig_textft_pos_cossims = orig_textft_captioncossims.get_logging_dict(panel_name="val", prefix="orig_textft")
    pr_textft_captioncossims_vals, val_pr_textft_pos_cossims = pr_textft_captioncossims.get_logging_dict(panel_name="val", prefix="pr_textft")

    if val_orig_textorig_pos_cossims is not None: 
        if dump_cossims: 
            cossims_dir = osp.join(args.log_base_path, 'cossims')
            os.makedirs(cossims_dir, exist_ok=True)
            torch.save(val_orig_textorig_pos_cossims, osp.join(cossims_dir, f"val_orig_textorig_pos_cossims.pt"))
            torch.save(val_orig_textft_pos_cossims, osp.join(cossims_dir, f"val_orig_textft_pos_cossims.pt"))
            torch.save(val_pr_textft_pos_cossims, osp.join(cossims_dir, f"val_pr_textft_pos_cossims.pt"))


    all_mgap_rmg_vals= {}
    all_mgap_rmg_vals.update(orig_textorig_mgap.get_logging_dict(panel_name="val", prefix="orig_textorig"))
    all_mgap_rmg_vals.update(orig_textft_mgap.get_logging_dict(panel_name="val", prefix="orig_textft"))
    all_mgap_rmg_vals.update(pr_textft_mgap.get_logging_dict(panel_name="val", prefix="pr_textft"))

    all_mgap_rmg_vals.update(orig_textorig_rmg.get_logging_dict(panel_name="val", prefix="orig_textorig"))
    all_mgap_rmg_vals.update(orig_textft_rmg.get_logging_dict(panel_name="val", prefix="orig_textft"))
    all_mgap_rmg_vals.update(pr_textft_rmg.get_logging_dict(panel_name="val", prefix="pr_textft"))
   
    orig_mgap_vals = {k: v for k, v in all_mgap_rmg_vals.items() if "orig" in k}
    pr_mgap_vals = {k: v for k, v in all_mgap_rmg_vals.items() if "pr" in k}


    acc_tensor = torch.full((len(mad_decoder_name_list_cat_wise_literal), len(bucket_types)), -1.0)
    
    for cat_id, cat_name in enumerate(mad_decoder_name_list_cat_wise_literal):
        for bucket_id, bucket_type in enumerate(bucket_types):
            correct_counts = correct_counts_cat_bucket_wise[cat_name][bucket_type] 
            total_counts = total_counts_cat_bucket_wise[cat_name][bucket_type]
            acc = correct_counts / total_counts if total_counts > 0 else -1
            acc_tensor[cat_id, bucket_id] = acc
            wandb_log_dict[f"pr_zs_accuracy.{cat_name}.{bucket_type}"] = acc

        
    print(f"Rows: Bucket types: {bucket_types}")
    print(f"Columns: Category names: {mad_decoder_name_list_cat_wise_literal}")
    print(acc_tensor.T.numpy())

    if args.filter_arch_name != "Identity" and args.expansion_factor==1:
        mask_accuracy_logging_dict = mask_accuracy.get_logging_dict(split="val")
    else:
        mask_accuracy_logging_dict = {}
    

    for k,v in zs_original_vals.items():
        wandb_log_dict[f"orig_zs_accuracy.{k}"] = v
        print(f"{k}: {v}")
        print(f"\n")

    for k, v in orig_textorig_captioncossims_vals.items():
        wandb_log_dict[f"{k}"] = v
        print(f"{k}: {v}")
        print(f"\n")

    for k, v in orig_textft_captioncossims_vals.items():
        wandb_log_dict[f"{k}"] = v
        print(f"{k}: {v}")
        print(f"\n")

    for k, v in pr_textft_captioncossims_vals.items():
        wandb_log_dict[f"{k}"] = v
        print(f"{k}: {v}")
        print(f"\n")

    for k, v in orig_mgap_vals.items():
        wandb_log_dict[f"original.{k}"] = v
        print(f"{k}: {v}")
        print(f"\n")

    for k, v in pr_mgap_vals.items():
        wandb_log_dict[f"pr.{k}"] = v
        print(f"{k}: {v}")
        print(f"\n")


    return wandb_log_dict, all_img_embeddings, all_textorig_embeddings, all_textft_embeddings, all_pr_embeddings


# Original function which gave good results


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None, tokenizer=None, filter_algo=None, evaluate_this_epoch=None, train_mask_directly=False, step=None, collator_get_w=None, flair_data=None):
    gpu_mem(prefix=f"Epoch: {epoch} at the start of training epoch: ")
    if is_master(args):
        logging.info(
            f"==============================> Starting training epoch: {epoch}; Step: {step}")
    device = args.device
    input_dtype = get_input_dtype(args.precision)

    # Since SimpleMLP and Cross Attention were only started with coco_controlled loss


    if args.dataset_type == "mad" and args.use_fixed_decoder_weights:
        all_mask_vals = torch.zeros(26, dtype=torch.float32)
    else:
        all_mask_vals = torch.zeros(args.embed_dim * args.expansion_factor, dtype=torch.float32)

    if (args.finetune_text_encoder or args.finetune_vision_encoder) and args.train_clip:
        model.train()
        if args.finetune_text_encoder and is_master(args):
            logging.info("Finetuning text encoder")
        else:
            model.eval()
            logging.info("CLIP is not being trained.")
    elif not args.train_clip: 
        model.eval()


    model.train()

    if args.filter_algo_name != "Identity":
        filter_algo.train()

    if args.distill:
        dist_model.eval()


    # set epoch in process safe manner via sampler or shared_epoch
    data['train'].set_epoch(epoch)
    dataloader = data['train'].dataloader


    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))


    losses_m = {}
    losses = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    logit_bias = unwrap_model(model).logit_bias
    assert logit_bias is None, "Logit bias is not supported in our code yet"

    all_texts = []
    uncommitted_step = None
    count =0 
    total_loss_across_batches =0.0
    for i, batch in enumerate(dataloader):
        if i % 10 == 0 and is_master(args) and i < 1000:
            print(f"Epoch: {epoch}, Step: {i}, Batch size: {len(batch[0])}, Total batches: {dataloader.num_batches}, Total samples: {dataloader.num_samples}", flush=True)
        i_accum = i // args.accum_freq

        step += 1

        lr_dict = scheduler(step)

        if args.finetune_vision_encoder or args.finetune_text_encoder:
            assert args.train_clip, "If finetuning vision or text encoder, train_clip must be True"
            images, texts = batch
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            tokenized_texts = tokenizer(texts).to(device=args.device, non_blocking=True)
                    
        
        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        return_unpooled = False

        total_loss = 0.0

        if args.finetune_vision_encoder or args.finetune_text_encoder:


            text_features = model.encode_text(tokenized_texts, normalize=False)


            if args.clip_logit_scale_fixed is not None: 
                logit_scale = torch.tensor(args.clip_logit_scale_fixed)
            else: 
                logit_scale = model.logit_scale.exp()
                logit_bias = None

        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)


        model_out = {"image_features": image_features_norm,
                     "text_features": text_features_norm, "logit_scale": logit_scale}
        if logit_bias is not None:
            model_out["logit_bias"] = logit_bias

        # Get pred_features for SAE only
        if args.filter_arch_name in ("SAE", "IdentityMaskable"):

            # Variation1: image was being conditioned on corresponding positive text
            pred_features, sae_outputs_dict = filter_algo.get_filtered_embeddings(None, None, image_features, text_features, which_type = args.which_type_conditioning)
            mask_loss_with_lambda = torch.tensor(0.0).to(args.device)

            pred_features_norm = pred_features / pred_features.norm(dim=-1, keepdim=True)
            if args.filter_arch_name != "Identity" and args.filter_algo_name != "Identity":
                mask_vals = sae_outputs_dict["info_updated"]["mask"]
                # Detach and move to CPU to prevent GPU memory accumulation
                all_mask_vals += mask_vals.sum(dim=0).detach().cpu()


        if args.filter_algo_name != "Identity":
            model_out_filtered = {"image_features": pred_features_norm, "text_features": text_features_norm, "logit_scale": logit_scale}


        if args.distill:
            raise NotImplementedError
            with torch.no_grad():
                dist_model_out = dist_model(images, texts)
            model_out.update(
                {f'dist_{k}': v for k, v in dist_model_out.items()})


        if args.filter_algo_name != 'Identity' or args.filter_arch_name != 'Identity':
            with torch.no_grad():
                loss_unfiltered = utils.get_clip_loss(image_features_norm, text_features_norm, logit_scale, image_features_norm.shape[0], args.device)
            losses["loss/unfiltered_contrastive_loss"] = loss_unfiltered.item()


            filtered_contrastive_loss  = loss(**model_out_filtered,  output_dict=False)

            losses["loss/filtered_contrastive_loss"] = filtered_contrastive_loss .item()


            # CODE FOR INVERTED CONDITIONING #######################################################################################
            inv_loss = torch.tensor(0.0).to(args.device)
            inv_loss_with_lambda = inv_loss * args.inv_loss_lambda
            losses["loss/inv_loss"] = inv_loss.item()
            losses["loss/inv_loss_with_lambda"] = inv_loss_with_lambda.item()
            # CODE FOR INVERTED CONDITIONING #######################################################################################


            total_loss = filtered_contrastive_loss + inv_loss_with_lambda + mask_loss_with_lambda
            gpu_mem(prefix=f"Epoch: {epoch} before getting recons from SAE: ")
            if args.train_sae and filter_algo.filter_obj.ae.encoder.weight.requires_grad:
                filter_loss_with_lambda, filter_loss_log = filter_algo.get_loss(
                    images, texts, image_features, text_features, sae_outputs_dict)
                losses.update({"loss/" + filter_algo.get_wandb_name() + "." + name: val for name, val in filter_loss_log.items()})
            else:
                filter_loss_with_lambda = torch.tensor(
                    0.0).to(args.device)
            gpu_mem(prefix=f"Epoch: {epoch} before getting recons from SAE: ")
            losses["loss/sae_loss_with_lambda"] = filter_loss_with_lambda.item()
            total_loss += filter_loss_with_lambda

            losses["loss/total_loss"] = total_loss 

        else:
            total_loss = loss(**model_out, output_dict=True)
            if args.filter_algo_name == "Identity" and args.filter_arch_name == "Identity":
                total_loss = total_loss['contrastive_loss']

            losses["loss/contrastive_loss"] = total_loss


        backward(total_loss, scaler)
        # Detach and move to CPU to prevent GPU memory accumulation
        total_loss_across_batches += total_loss.detach().cpu().item()

        del pred_features
        del model_out
        del model_out_filtered
        del pred_features_norm
        del text_features
        if args.train_sae:
            filter_algo.post_backward_operation()

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # reset gradient accum, if enabled

        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))
            

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        args.log_every_n_steps = 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            if args.wandb and is_master(args) and uncommitted_step is not None:
                logging.info(f"Committing train data at step {uncommitted_step}")
                wandb.log(log_data, step=uncommitted_step, commit=True)
                uncommitted_step = None
            batch_size = len(image_features)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if isinstance(val, torch.Tensor):
                    val = val.item()
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                
                losses_m[key].update(val, batch_size)

            logit_scale_scalar = logit_scale.item()
            grad_scale_scalar = scaler.get_scale()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * \
                args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Step: {step} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )
            if scaler is not None:
                logging.info(f"Grad Scaler Scale: {grad_scale_scalar:.3f}")

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                # "lr": optimizer.param_groups[0]["lr"]
            }
            if scaler is not None:
                log_data["grad_scale"] = grad_scale_scalar
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = update_log_data_for_wandb(log_data)

            for key, lr in lr_dict.items():
                log_data[f"train/{key}_lr"] = lr

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            log_data['step'] = step
            log_data['epoch'] = epoch
            if args.wandb:
                uncommitted_step = step
            
            batch_time_m.reset()
            data_time_m.reset()


    if args.wandb and is_master(args) and uncommitted_step is not None:
        if not evaluate_this_epoch:
            logging.info(f"Not evaluating this epoch at epoch {epoch}, committing at step {uncommitted_step}")
            wandb.log(log_data, step=step, commit=True)
            print(f"wandb logged: step: {step}, epoch: {epoch}.")
        else:
            logging.info(f"Evaluating this epoch at epoch {epoch}, not committing at step {uncommitted_step}")
            wandb.log(log_data, step=step, commit=False)

    gpu_mem(prefix=f"Epoch: {epoch} at the end of training epoch: ")
    return step, total_loss_across_batches


def get_cossims_after(image_features, coco_attribute_list, model, tokenizer, device, filter_algo, use_single_template=True, get_all_combination=False, cossims_with_null=False, null_vec=None, args=None, unpooled_image_features=None):
    '''
    get_all_combination: If True, condition on something and get cossim with all attributes

    '''
    template_texts = []
    for attr in coco_attribute_list:
        if use_single_template:
            template_texts.append(f"a photo of a {attr}.")
        else:
            template_texts.append(attr)
    text_tokenized = tokenizer(template_texts).to(device=device)

    text_features = model.encode_text(text_tokenized, normalize=False, return_unpooled=False)
    text_features_unpooled = None


    text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)  # NAttr x 768

    if not get_all_combination:
        cossims_after = torch.zeros((image_features.shape[0], text_features.shape[0]))  # B x NAttr
        if cossims_with_null:
            assert null_vec is not None
            null_vec = null_vec.to(device=device)
            cossims_after_with_null = torch.zeros((image_features.shape[0], text_features.shape[0]))  # B x NAttr
        else: 
            cossims_after_with_null = None
    else:
        cossims_after = torch.zeros((image_features.shape[0], text_features.shape[0], text_features.shape[0]))  # B x NAttr


    image_features_to_be_processed = image_features

    for i in range(len(image_features_to_be_processed)):

        this_image_feature = image_features_to_be_processed[i:i + 1].to(device=device)

        part_recon_image_features, _ = filter_algo.get_filtered_embeddings_single_image_multiple_texts(None, None, this_image_feature, text_features, unpooled_image_embeddings=this_image_feature, unpooled_text_embeddings=text_features_unpooled)
        part_recon_image_features = part_recon_image_features / part_recon_image_features.norm(dim=-1, keepdim=True)


        # NAttr x 768
        part_recon_image_features = part_recon_image_features.float()
        text_features_norm = text_features_norm.float()
        if not get_all_combination:
            cossims_after[i] = (part_recon_image_features * text_features_norm).sum(dim=-1)
            if cossims_with_null:
                cossims_after_with_null[i] = (part_recon_image_features * null_vec).sum(dim=-1)
        else:
            cossims_after[i] = (part_recon_image_features @ text_features_norm.T)


    return cossims_after, cossims_after_with_null


def get_cossims_before(original_image_features, coco_attribute_list, original_model, tokenizer, device, use_single_template=True):
    original_image_features_norm = original_image_features / original_image_features.norm(dim=-1, keepdim=True)  # B x 768
    template_texts = []
    for attr in coco_attribute_list:
        if use_single_template:
            template_texts.append(f"a photo of a {attr}.")
        else:
            template_texts.append(attr)
    text_tokenized = tokenizer(template_texts).to(device=device)
    original_text_features = original_model.encode_text(text_tokenized, normalize=False).cpu()
    original_text_features_norm = original_text_features / original_text_features.norm(dim=-1, keepdim=True)  # NAttr x 768
    cossims_before = original_image_features_norm.float() @ original_text_features_norm.float().t()  # B x NAttr
    return cossims_before


def plot_cossims_before_after_attribute_wise(save_path, image_features, original_image_features, coco_attribute_list, model, original_model, tokenizer, filter_algo, filter_obj, select_one_in_five=False, args=None, if_plot=False, if_save=False, get_all_combination=False, use_single_template=True):
    '''
        x-axis cossim before, y-axis cossim after
        2 plots: one with images containing teh attribute, the other with images not containing teh attribute
    '''

    cossims_after = get_cossims_after(image_features, coco_attribute_list, model, tokenizer, args.device, filter_algo, use_single_template=use_single_template, get_all_combination=get_all_combination, args=args)
    cossims_before = get_cossims_before(original_image_features, coco_attribute_list, original_model, tokenizer, args.device, use_single_template=use_single_template)
    img_multihot_attributes = torch.load(config.VAL_COUNT_ATTR_CAPTION_DICT)["img_multihot_attributes"]


    if if_plot:
        for aidx, attr in enumerate(coco_attribute_list):
            chosen_img_idxs = torch.where(img_multihot_attributes[:, aidx] == 1)
            # complement img_ids
            not_chosen_img_idxs = torch.where(img_multihot_attributes[:, aidx] == 0)

            fig, ax = plt.subplots(1, 2, figsize=(10, 6))

            ax[0, 0].scatter(cossims_before[chosen_img_idxs, aidx].cpu().numpy(), cossims_after[chosen_img_idxs, aidx].cpu().numpy())
            ax[0, 0].set_xlabel('Cossim Before')
            ax[0, 0].set_ylabel('Cossim After')
            ax[0, 0].set_title(f"{attr}_imgs_containing_attribute")
            ax[0, 0].plot([0, 1], [0, 1], 'r--')

            ax[0, 1].scatter(cossims_before[not_chosen_img_idxs, aidx].cpu().numpy(), cossims_after[not_chosen_img_idxs, aidx].cpu().numpy())
            ax[0, 1].set_xlabel('Cossim Before')
            ax[0, 1].set_ylabel('Cossim After')
            ax[0, 1].set_title(f"{attr}_imgs_not_containing_attribute")
            ax[0, 1].plot([0, 1], [0, 1], 'r--')


            plt.savefig(osp.join(save_path, f"{attr}"))
            plt.close(fig)

    if if_save:
        if not osp.exists(osp.join(save_path, f'cossims_before.pt')):
            torch.save(cossims_before, osp.join(save_path, f'cossims_before.pt'))
        torch.save(cossims_after, osp.join(save_path, f'cossims_after.pt'))

    return cossims_before, cossims_after


def get_caption(word_list):
    shuffle(word_list)
    return ', '.join(word_list)


def shuffle_random(lst):
    shuffle(lst)
    return lst


def get_caption_from_idx_matrix(all_category_names, idxs_matrix, num_subsample=False):
    if not num_subsample:
        caption_list = [
            get_caption([all_category_names[idx] for idx in torch.nonzero(row).flatten().tolist()]) for row in idxs_matrix]
    else:
        len(num_subsample) == len(idxs_matrix)
        caption_list = [
            get_caption([all_category_names[idx] for idx in shuffle_random(torch.nonzero(row).flatten().tolist())[0:end_idx]]) for row, end_idx in zip(idxs_matrix, num_subsample)]

    return caption_list

################################################################################################
################################################################################################
################################################################################################


def plot_cossims_before_and_after_for_all_caption_setups(bucket_list, caption_setups, sim_dict_after, sim_dict_before, save_path, fname, title="CosSims Single Attribute Before After Per Bucket"):
    '''
    plots for both: with single attribute and with full caption, x-axis cossim before, y-axis cossim after
    bucket_name : corresponds to column in the plot
    caption_setups: corrsponds to each row in the plot
    '''

    fig, axs = plt.subplots(5, 7, figsize=(27, 20))
    fig.suptitle(title)
    for key_id, key in enumerate(caption_setups):
        if key_id == 0:
            for ax_idx in range(len(axs)):
                axs[ax_idx, key_id].set_ylabel(f'{bucket_list[ax_idx]}')

        axs[0, key_id].set_title(key)
        sim_list = sim_dict_after[key]
        sim_list_before = sim_dict_before[key]
        for ax, bucket_name, cossim, cossim_before in zip(axs[:, key_id], bucket_list, sim_list, sim_list_before):
            ax.scatter(cossim_before, cossim, alpha=0.5, s=0.5)
            ax.set_xlim(-1.0, 1.0)
            ax.set_ylim(-1.0, 1.0)
            ax.plot([-1, 1], [-1, 1], 'r--')
            ax.set_xlabel('Before Cossims')
    fig.savefig(osp.join(save_path, fname))
    plt.close(fig)


def plot_cossims_after_for_all_caption_setups(bucket_list, caption_setups, sim_dict_after, save_path, fname, bins=50, title="CosSims Single Attribute Histogram Per Bucket", cossims_dict_with_null=None, bucket_list_with_null=None):
    '''
    plots for both: with single attribute and with full caption, x-axis cossim after, y-axis frequency
    bucket_name : corresponds to column in the plot
    caption_setups: corrsponds to each row in the plot
    '''

    n_rows = 5
    if cossims_dict_with_null is not None and bucket_list_with_null is not None:
        n_rows = 8
        bucket_list.extend(bucket_list_with_null)

    fig, axs = plt.subplots(n_rows, 7, figsize=(27, n_rows * 4))
    fig.suptitle(title)
    for key_id, key in enumerate(caption_setups):
        if key_id == 0:
            for ax_idx in range(len(axs)):
                axs[ax_idx, key_id].set_ylabel(f'{bucket_list[ax_idx]}')
        axs[0, key_id].set_title(key)
        sim_list = sim_dict_after[key]
        if cossims_dict_with_null is not None:
            sim_list.extend(cossims_dict_with_null[key])

        for ax, bucket_name, cossim in zip(axs[:, key_id], bucket_list, sim_list):
            ax.hist(cossim, bins=bins, alpha=0.5)
            ax.set_xlim(-1.0, 1.0)
            ax.set_xlabel('Cossims')

    fig.savefig(osp.join(save_path, fname))
    plt.close(fig)


def plot_cossims_attribute_wise_coco_controlled(embeddings_dict, model, original_model, tokenizer, args=None, idxs_per_caption_type=None, epoch=None, cossims_dict_with_null=None, bucket_list_with_null=None, use_single_template=True, which_attr_setup=None):
    '''
        Plot1: compare cossim before and after, attribute wise
        Plot2: histogram of cossims after, attribute wise
    '''
    suffix = ''
    if not use_single_template:
        suffix += "_without_template"
    if which_attr_setup is not None:
        suffix += f'_{which_attr_setup}'

    multihot_attributes = torch.load(config.VAL_COUNT_ATTR_CAPTION_DICT)["multihot_attributes"]
    original_image_features = embeddings_dict["all_imgorig_embeddings"].to(torch.float32)
    pred_features = embeddings_dict["all_pr_embeddings"].to(torch.float32)

    original_image_features = original_image_features / original_image_features.norm(dim=-1, keepdim=True)  # B x 768
    pred_features = pred_features / pred_features.norm(dim=-1, keepdim=True)  # B x 768

    all_category_names = COCO_CLASSNAMES

    caption_setups = list(idxs_per_caption_type.keys())

    template_texts = []
    for attr in all_category_names:
        if use_single_template:
            template_texts.append(f"a photo of a {attr}.")
        else:
            template_texts.append(attr)

    text_tokenized = tokenizer(template_texts).to(device=args.device)  # len(COCO_CLASSNAMES) x 77
    text_features = model.encode_text(text_tokenized, normalize=True).cpu()
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # NAttr x 768

    cossims_after_all = (pred_features @ text_features.T)

    cossims_after_dict = {}

    original_image_features_chosen = original_image_features[idxs_per_caption_type[caption_setups[0]]]

    cossims_before = get_cossims_before(original_image_features_chosen, all_category_names, original_model, tokenizer, device=args.device, use_single_template=use_single_template)

    for key in caption_setups:
        cossims_after_dict[key] = cossims_after_all[idxs_per_caption_type[key]]

    GT_correct_ids = multihot_attributes['full_correct']


    save_path = osp.join(args.log_base_path, 'plots')
    os.makedirs(save_path, exist_ok=True)
    sim_dict_after = {}
    sim_dict_before = {}
    results_dict = {}
    bucket_list = ['in_img_in_caption', 'in_caption_not_in_img', 'in_img_not_in_caption', 'union', 'random']
    for key_id, key in enumerate(caption_setups):

        captions_ids = multihot_attributes[key]
        cossims_after = cossims_after_dict[key]


        in_img_in_caption_ids_filtered, _, in_caption_not_in_img_ids_filtered, _, in_img_not_in_caption_ids_filtered, _, union_ids_filtered, _, random_ids_filtered, _, num_subsample = get_filtered_ids(GT_correct_ids, captions_ids)

        sim_list = [cossims_after[torch.where(in_img_in_caption_ids_filtered == 1)], cossims_after[torch.where(in_caption_not_in_img_ids_filtered == 1)], cossims_after[torch.where(in_img_not_in_caption_ids_filtered == 1)], cossims_after[torch.where(union_ids_filtered == 1)], cossims_after[torch.where(random_ids_filtered == 1)]]

        sim_list_before = [cossims_before[torch.where(in_img_in_caption_ids_filtered == 1)], cossims_before[torch.where(in_caption_not_in_img_ids_filtered == 1)], cossims_before[torch.where(in_img_not_in_caption_ids_filtered == 1)], cossims_before[torch.where(union_ids_filtered == 1)], cossims_before[torch.where(random_ids_filtered == 1)]]
        sim_dict_before[key] = sim_list_before
        sim_dict_after[key] = sim_list

        for bucket_name, sim_before, sim_after in zip(bucket_list, sim_list_before, sim_list):
            assert len(sim_before) == len(sim_after)
            if len(sim_before) == 0:
                results_dict[f"{key}_before_{bucket_name}_attr"] = 'NA'
                results_dict[f"{key}_after_{bucket_name}_attr"] = 'NA'
            else:
                results_dict[f"{key}_before_{bucket_name}_attr"] = sim_before.mean().item()
                results_dict[f"{key}_after_{bucket_name}_attr"] = sim_after.mean().item()


    plot_cossims_before_and_after_for_all_caption_setups(bucket_list, caption_setups, sim_dict_after, sim_dict_before, save_path, f'cossims_before_and_after_for_all_caption_setups_attr_wise_epo{epoch}{suffix}.png', title="CosSims Single Attribute Before After Per Bucket")
    plot_cossims_after_for_all_caption_setups(bucket_list, caption_setups, sim_dict_after, save_path, f'cossims_after_for_all_caption_setups_attr_wise_epo{epoch}{suffix}.png', bins=50, title="CosSims Single Attribute Histogram Per Bucket", cossims_dict_with_null=cossims_dict_with_null, bucket_list_with_null=bucket_list_with_null)
    return results_dict


def plot_cossims_whole_caption_coco_controlled(embeddings_dict, idxs_per_caption_type, model, original_model, tokenizer, args, epoch, cossims_dict_with_null, bucket_list_with_null=None, which_attr_setup=None):

    suffix = ''
    if which_attr_setup is not None:
        suffix += f'_{which_attr_setup}'

    pred_features = embeddings_dict['all_pr_embeddings'].to(torch.float32)
    original_image_features = embeddings_dict['all_imgorig_embeddings'].to(torch.float32)
    original_image_features_norm = original_image_features / original_image_features.norm(dim=-1, keepdim=True)  # B x 768
    pred_features_norm = pred_features / pred_features.norm(dim=-1, keepdim=True)  # B x 768
    multihot_attributes = torch.load(config.VAL_COUNT_ATTR_CAPTION_DICT)["multihot_attributes"]


    GT_correct_ids = multihot_attributes['full_correct']

    sim_dict_after = {}
    sim_dict_before = {}
    bucket_list = ['in_img_in_caption', 'in_caption_not_in_img', 'in_img_not_in_caption', 'union', 'random']
    caption_setups = list(idxs_per_caption_type.keys())
    results_dict = {}
    for key in idxs_per_caption_type.keys():
        captions_ids = multihot_attributes[key]
        edited_image_embedding = pred_features_norm[idxs_per_caption_type[key]]  # (5k, embedding_dim)
        original_image_embedding = original_image_features_norm[idxs_per_caption_type[key]]  # (5k, embedding_dim)

        device = args.device
        all_category_names = COCO_CLASSNAMES

        in_img_in_caption_ids_filtered, in_img_in_caption_ids_keep_rows, in_caption_not_in_img_ids_filtered, in_caption_not_in_img_ids_keep_rows, in_img_not_in_caption_ids_filtered, in_img_not_in_caption_ids_keep_rows, union_ids_filtered, union_ids_keep_rows, random_ids_filtered, random_ids_keep_rows, num_subsample = get_filtered_ids(GT_correct_ids, captions_ids)

        results_dict, in_img_in_caption_cossims_with_edited, in_img_in_caption_cossims_with_original = gather_metrics(in_img_in_caption_ids_filtered, in_img_in_caption_ids_keep_rows, tokenizer, model, original_model, all_category_names, edited_image_embedding, original_image_embedding, results_dict, device, key, bucket_list[0], args)

        results_dict, in_caption_not_in_img_cossims_with_edited, in_caption_not_in_img_cossims_with_original = gather_metrics(in_caption_not_in_img_ids_filtered, in_caption_not_in_img_ids_keep_rows, tokenizer, model, original_model, all_category_names, edited_image_embedding, original_image_embedding, results_dict, device, key, bucket_list[1], args)

        results_dict, in_img_not_in_caption_cossims_with_edited, in_img_not_in_caption_cossims_with_original = gather_metrics(in_img_not_in_caption_ids_filtered, in_img_not_in_caption_ids_keep_rows, tokenizer, model, original_model, all_category_names, edited_image_embedding, original_image_embedding, results_dict, device, key, bucket_list[2], args)

        results_dict, union_with_edited, union_with_original = gather_metrics(union_ids_filtered, union_ids_keep_rows, tokenizer, model, original_model, all_category_names, edited_image_embedding, original_image_embedding, results_dict, device, key, bucket_list[3], args)

        results_dict, random_with_edited, random_with_original = gather_metrics(random_ids_filtered, random_ids_keep_rows, tokenizer, model, original_model, all_category_names, edited_image_embedding, original_image_embedding, results_dict, device, key, bucket_list[4], args, num_subsample=num_subsample)


        assert (len(in_img_in_caption_cossims_with_edited) == len(in_img_in_caption_cossims_with_original))
        sim_list = [in_img_in_caption_cossims_with_edited, in_caption_not_in_img_cossims_with_edited, in_img_not_in_caption_cossims_with_edited, union_with_edited, random_with_edited]
        sim_list_before = [in_img_in_caption_cossims_with_original, in_caption_not_in_img_cossims_with_original, in_img_not_in_caption_cossims_with_original, union_with_original, random_with_original]
        sim_dict_before[key] = sim_list_before
        sim_dict_after[key] = sim_list

    save_path = osp.join(args.log_base_path, 'plots')
    os.makedirs(save_path, exist_ok=True)
    # _more_than_attr1
    # _more_than_attr1_without_template
    plot_cossims_before_and_after_for_all_caption_setups(bucket_list, caption_setups, sim_dict_after, sim_dict_before, save_path, f'cossims_before_and_after_for_all_caption_setups_whole_caption_epo{epoch}{suffix}.png', title="CosSims Whole Caption Before After Per Bucket")
    plot_cossims_after_for_all_caption_setups(bucket_list, caption_setups, sim_dict_after, save_path, f'cossims_after_for_all_caption_setups_whole_caption_epo{epoch}{suffix}.png', bins=50, title="CosSims Whole Caption Histogram Per Bucket", cossims_dict_with_null=cossims_dict_with_null, bucket_list_with_null=bucket_list_with_null)

    return results_dict
################################################################################################
################################################################################################
################################################################################################


def get_filtered_ids(GT_correct_ids, captions_ids):
    in_img_in_caption_ids = GT_correct_ids * captions_ids
    in_caption_not_in_img_ids = (captions_ids - in_img_in_caption_ids).clamp(min=0)
    in_img_not_in_caption_ids = (GT_correct_ids - in_img_in_caption_ids).clamp(min=0)
    union_ids = (captions_ids + GT_correct_ids).clamp(max=1)
    random_ids = (1 - union_ids).clamp(min=0)


    in_img_in_caption_ids_keep_rows = torch.any(in_img_in_caption_ids == 1, dim=1)
    in_img_in_caption_ids_filtered = in_img_in_caption_ids[in_img_in_caption_ids_keep_rows]  # shape: (M, 80), where M ≤ N

    in_caption_not_in_img_ids_keep_rows = torch.any(in_caption_not_in_img_ids == 1, dim=1)
    in_caption_not_in_img_ids_filtered = in_caption_not_in_img_ids[in_caption_not_in_img_ids_keep_rows]  # shape: (M, 80), where M ≤ N

    in_img_not_in_caption_ids_keep_rows = torch.any(in_img_not_in_caption_ids == 1, dim=1)
    in_img_not_in_caption_ids_filtered = in_img_not_in_caption_ids[in_img_not_in_caption_ids_keep_rows]  # shape: (M, 80), where M ≤ N

    union_ids_keep_rows = torch.any(union_ids == 1, dim=1)
    union_ids_filtered = union_ids[union_ids_keep_rows]  # shape: (M, 80), where M ≤ N

    random_ids_keep_rows = torch.any(random_ids == 1, dim=1)
    random_ids_filtered = random_ids[random_ids_keep_rows]  # shape: (M, 80), where M ≤ N

    GT_correct_ids_filetered = GT_correct_ids[random_ids_keep_rows]  # shape: (M, 80), where M ≤ N
    num_subsample = GT_correct_ids_filetered.sum(dim=1)
    num_subsample = [torch.randint(1, single_subsample + 1, (1,)).item() for single_subsample in num_subsample]

    return in_img_in_caption_ids_filtered, in_img_in_caption_ids_keep_rows, in_caption_not_in_img_ids_filtered, in_caption_not_in_img_ids_keep_rows, in_img_not_in_caption_ids_filtered, in_img_not_in_caption_ids_keep_rows, union_ids_filtered, union_ids_keep_rows, random_ids_filtered, random_ids_keep_rows, num_subsample


def gather_metrics(union_ids_filtered, union_ids_keep_rows, tokenizer, model, original_model, all_category_names, edited_image_embedding, original_image_embedding, results_dict, device, key_name, bucket_name, args, num_subsample=None):

    if len(union_ids_filtered) != 0:
        if num_subsample is not None:
            img_union_caption = get_caption_from_idx_matrix(all_category_names, union_ids_filtered, num_subsample=num_subsample)
        else:
            img_union_caption = get_caption_from_idx_matrix(all_category_names, union_ids_filtered)

        img_union_caption_tokenized = tokenizer(img_union_caption).to(device=device, non_blocking=True)

        img_union_caption_features = model.encode_text(img_union_caption_tokenized, normalize=True).cpu().float()

        original_img_union_caption_features = original_model.encode_text(img_union_caption_tokenized, normalize=True).cpu().float()

        union_with_edited = (edited_image_embedding[union_ids_keep_rows] * img_union_caption_features).sum(dim=-1).cpu()

        union_with_original = (original_image_embedding[union_ids_keep_rows] * original_img_union_caption_features).sum(dim=-1).cpu()


        union_with_edited_mean = union_with_edited.mean().item()
        union_with_original_mean = union_with_original.mean().item()
        results_dict[f"{key_name}_before_{bucket_name}_whole"] = union_with_original_mean
        results_dict[f"{key_name}_after_{bucket_name}_whole"] = union_with_edited_mean
        img_union_caption_tokenized = img_union_caption_tokenized.cpu()

        del img_union_caption_tokenized

    else:
        union_with_edited_mean = 'NA'
        union_with_original_mean = 'NA'
        results_dict[f"{key_name}_before_{bucket_name}_whole"] = union_with_original_mean
        results_dict[f"{key_name}_after_{bucket_name}_whole"] = union_with_edited_mean
        union_with_edited = torch.tensor([])
        union_with_original = torch.tensor([])


    torch.cuda.empty_cache()

    return results_dict, union_with_edited, union_with_original


# condition_image_with_multiple_texts
def all_possible_cossims_edited_with_attr_present_heatmap(images, embeddings_dict, coco_attribute_list, model, original_model, tokenizer, filter_algo, args, use_single_template, idxs_per_caption_type=None, num_to_plot=None):
    '''
       image: i, t1, t2, t3 attr present in image, and t4, t5 are random attr not present in image
        i|t1 with t1,t2,t3
        i|t2 with t1 t2 t3
        i|t3 with t1 t2 t3
        i|t4 with t1 t2 t3
        i|t5 with t1 t2 t3
    '''
    suffix = ""
    if not use_single_template:
        suffix += '_without_template'


    chosen_img_idxs = idxs_per_caption_type['full_correct']
    images = images[chosen_img_idxs][0:num_to_plot]
    images = (images - OPENAI_DATASET_INVERSE_MEAN) / OPENAI_DATASET_INVERSE_STD

    image_features = embeddings_dict["all_img_embeddings"][chosen_img_idxs][0:num_to_plot]
    original_image_features = embeddings_dict["all_imgorig_embeddings"][chosen_img_idxs][0:num_to_plot]
    unpooled_image_features = embeddings_dict["all_unpooled_img_embeddings"][chosen_img_idxs][0:num_to_plot]


    image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # B x 768
    original_image_features = original_image_features / original_image_features.norm(dim=-1, keepdim=True)  # B x 768
    save_path = osp.join(args.log_base_path, 'img_vis_for_single_attr_conditioning_heatmap')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    multihot_attributes = torch.load(config.VAL_COUNT_ATTR_CAPTION_DICT)["multihot_attributes"]
    GT_correct_ids = multihot_attributes['full_correct'][0:num_to_plot]

    assert (len(GT_correct_ids) == len(images) == len(image_features) == len(original_image_features))

    cossims_after = get_cossims_after(image_features, coco_attribute_list, model, tokenizer, args.device, filter_algo, use_single_template=use_single_template, get_all_combination=True, args=args, unpooled_image_features=unpooled_image_features)
    cossims_before = get_cossims_before(original_image_features, coco_attribute_list, original_model, tokenizer, args.device, use_single_template=use_single_template)


    for image_idx, GT_correct_id in zip(range(len(image_features)), GT_correct_ids):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        heatmap_array = []
        
        exact_idxs = torch.where(GT_correct_id == 1)[0]
        extra_idxs = torch.where(GT_correct_id == 0)[0][0:2]

        category_names_single = [coco_attribute_list[idx.item()] for idx in exact_idxs]
        extra_category_names_single = [f"rand_{coco_attribute_list[idx.item()]}" for idx in extra_idxs]

        category_names_single.extend(extra_category_names_single)
        exact_idxs = torch.cat((exact_idxs, extra_idxs), dim=0)


        assert (len(category_names_single) == len(exact_idxs))
        y_ticklabels = ["orig"]
        for name_single in category_names_single:
            y_ticklabels.append(f"i|{name_single}")
        cossim_before = (cossims_before[image_idx][exact_idxs]).cpu()
        heatmap_array.append(cossim_before)

        for first_exact_idx in exact_idxs:
            cossim_after = []
            for second_exact_idx in exact_idxs:
                cossim_after.append(cossims_after[image_idx][first_exact_idx][second_exact_idx])
            cossim_after = torch.stack(cossim_after).cpu()
            heatmap_array.append(cossim_after)

        heatmap_array = torch.stack(heatmap_array).numpy()
        ax[0].imshow(images[image_idx])
        ax[0].axis('off')
        

        seaborn.heatmap(heatmap_array, ax=ax[1], annot=True, fmt=".2f", cmap="YlGnBu", cbar=False)
        ax[1].set_xticklabels(category_names_single, rotation=45)
        ax[1].set_yticklabels(y_ticklabels, rotation=45)
        ax[1].set_title("Cosine Similarity")
        plt.savefig(os.path.join(save_path, f"heatmap_{image_idx}{suffix}.png"))


def get_correct_ids_with_or_wihtout_attr(GT_correct_ids, attr_idx):
    GT_correct_id_across_images_for_single_attr = GT_correct_ids[:, attr_idx]
    GT_correct_id_across_images_with_single_attr = torch.where(GT_correct_id_across_images_for_single_attr == 1)[0]
    GT_correct_id_across_images_without_single_attr = torch.where(GT_correct_id_across_images_for_single_attr == 0)[0]
    assert (len(GT_correct_id_across_images_with_single_attr) + len(GT_correct_id_across_images_without_single_attr) == len(GT_correct_id_across_images_for_single_attr))
    return GT_correct_id_across_images_with_single_attr, GT_correct_id_across_images_without_single_attr


def get_cossims_with_or_wihtout_attr(cossims_all, GT_correct_id_across_images_with_single_attr, GT_correct_id_across_images_without_single_attr):
    cossims_with_single_attr = cossims_all[GT_correct_id_across_images_with_single_attr]
    cossims_without_single_attr = cossims_all[GT_correct_id_across_images_without_single_attr]
    return cossims_with_single_attr, cossims_without_single_attr


def get_cossims_null_with_attr(null_vec, model, tokenizer, device, coco_attribute_list, use_single_template, args):
    '''
    cossim of null_vector with all the 80 coco attributes
    '''
    template_texts = []
    null_vec = null_vec.to(device)
    for attr in coco_attribute_list:
        if use_single_template:
            template_texts.append(f"a photo of a {attr}.")
        else:
            template_texts.append(attr)
    text_tokenized = tokenizer(template_texts).to(device=device)
    text_features = model.encode_text(text_tokenized, normalize=False)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # NAttr x 768


    cossims_null_with_attr = (text_features * null_vec).sum(dim=-1)
    return cossims_null_with_attr


def cossim_histogram_before_after_and_comparison(embeddings_dict, coco_attribute_list, epoch, model, original_model, tokenizer, filter_algo, args=None, use_single_template=True, null_vec=None, idxs_per_caption_type=None, which_attr_setup=None, cossims_with_null=False, plot_before_after=False, GT_correct_ids=None, suffix=None):
    '''
    80*6 plot, each row is for a particular attribute t1,
    1st column: cossims(i,t1) with images containing i1
    2nd column: cossims(i,t1) with images not containing t1
    3rd column: cossims(i|t1,t1) with iamges containing t1
    4th column: cossims(i|t1,t1) with iamges not containing t1
    5th column : compare cossims before and after of images containing t1
    6th column : compare cossims before and after of images not containing t1
    '''
    if not use_single_template:
        suffix += "_without_template"
    if which_attr_setup is not None:
        suffix += f'_{which_attr_setup}'

    results_dict = {}

    image_features, original_image_features = embeddings_dict["all_img_embeddings"], embeddings_dict["all_imgorig_embeddings"]


    unpooled_image_features = None


    save_path = osp.join(args.log_base_path, 'plots')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    multihot_attributes = torch.load(config.VAL_COUNT_ATTR_CAPTION_DICT)["multihot_attributes"]

    if GT_correct_ids is None:
        GT_correct_ids = multihot_attributes['full_correct']
    assert(GT_correct_ids.shape == (image_features.shape[0], len(coco_attribute_list)))

 
    cossims_after, cossims_after_with_null = get_cossims_after(image_features, coco_attribute_list, model, tokenizer, args.device, filter_algo, use_single_template=use_single_template, get_all_combination=False, cossims_with_null=cossims_with_null, null_vec=null_vec, args=args, unpooled_image_features=unpooled_image_features)
    if cossims_after_with_null is not None:
        assert (cossims_after.shape == cossims_after_with_null.shape)
        cossims_null_with_attr = get_cossims_null_with_attr(null_vec, model, tokenizer, args.device, coco_attribute_list, use_single_template, args=args)


    image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # B x 768
    original_image_features = original_image_features / original_image_features.norm(dim=-1, keepdim=True)  # B x 768
    cossims_before = get_cossims_before(original_image_features, coco_attribute_list, original_model, tokenizer, args.device, use_single_template=use_single_template)

    
    assert (cossims_after.shape == cossims_before.shape)

    
    results_per_image  = []
    for image_idx, GT_correct_id in zip(range(len(image_features)), GT_correct_ids):
        before = cossims_before[image_idx]
        after = cossims_after[image_idx]
        with_attr = torch.where(GT_correct_ids[image_idx]==1)[0]
        without_attr = torch.where(GT_correct_ids[image_idx]==0)[0]

        before_with_attr = before[with_attr]
        before_without_attr = before[without_attr]
        after_with_attr = after[with_attr]
        after_without_attr = after[without_attr]
        results_per_image.append(torch.tensor([before_with_attr.mean().item()- before_without_attr.mean().item(), after_with_attr.mean().item()- after_without_attr.mean().item()]))
    
    
    results_per_image = torch.stack(results_per_image)
    tensor1 = results_per_image[:, 0].numpy()
    tensor2 = results_per_image[:, 1].numpy()

    x = np.arange(len(tensor1))  # X-axis indices

    bar_width = 0.4

    plt.figure(figsize=(12, 6))
    plt.bar(x - bar_width/2, tensor1, width=bar_width, label='Tensor 1', color='skyblue')
    plt.bar(x + bar_width/2, tensor2, width=bar_width, label='Tensor 2', color='orange')

    plt.xlabel("img_id")
    plt.ylabel("delta (with - without attr) cossim before vs after")
    plt.title("Cossim delta comparison before vs after across images")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"histogram_cossims_across_images_cmp_with_without_attr_epo{epoch}{suffix}.png"))


    n_cols = 9
    fig, ax = plt.subplots(len(coco_attribute_list), n_cols, figsize=(n_cols * 4, len(coco_attribute_list) * 4))
    all_cossims_with_attr_with_null = []
    all_cossims_with_null_without_attr = []
    all_cossims_with_attr = []
    all_cossims_without_attr = []

    # This loops through the attributes 
    for attr_name, attr_idx in zip(coco_attribute_list, range(len(coco_attribute_list))):
        if attr_idx % 10 == 0:
            print(f"attr: {attr_name}: {attr_idx}/{len(coco_attribute_list)}")

        GT_correct_id_across_images_with_single_attr, GT_correct_id_across_images_without_single_attr = get_correct_ids_with_or_wihtout_attr(GT_correct_ids, attr_idx)

        cossims_before_with_single_attr, cossims_before_without_single_attr = get_cossims_with_or_wihtout_attr(cossims_before[:, attr_idx], GT_correct_id_across_images_with_single_attr, GT_correct_id_across_images_without_single_attr)

        cossims_after_with_single_attr, cossims_after_without_single_attr = get_cossims_with_or_wihtout_attr(cossims_after[:, attr_idx], GT_correct_id_across_images_with_single_attr, GT_correct_id_across_images_without_single_attr)


        if cossims_after_with_null is not None:
            cossims_after_with_null_with_single_attr, cossims_after_with_null_without_single_attr = get_cossims_with_or_wihtout_attr(cossims_after_with_null[:, attr_idx], GT_correct_id_across_images_with_single_attr, GT_correct_id_across_images_without_single_attr)
            
            all_cossims_with_attr_with_null.append(cossims_after_with_null_with_single_attr)
            all_cossims_with_null_without_attr.append(cossims_after_with_null_without_single_attr)
            mean_cossims_after_with_null_with_single_attr = cossims_after_with_null_with_single_attr.mean().item()
            mean_cossims_after_with_null_without_single_attr = cossims_after_with_null_without_single_attr.mean().item()

            delta_with_null_with_vs_without = (mean_cossims_after_with_null_with_single_attr - mean_cossims_after_with_null_without_single_attr)
            ax[attr_idx, 7].hist(cossims_after_with_null_with_single_attr, alpha=0.5, label=f"{mean_cossims_after_with_null_with_single_attr:.3f}_diff{delta_with_null_with_vs_without:.3f}")
            ax[attr_idx, 8].hist(cossims_after_with_null_without_single_attr, alpha=0.5, label=f"{mean_cossims_after_with_null_without_single_attr:.3f}")
            ax[attr_idx, 7].set_title(f"sim_after_withAttr_with_null")
            ax[attr_idx, 8].set_title(f"sim_after_woAttr_with_null")


        all_cossims_with_attr.append(cossims_after_with_single_attr)
        all_cossims_without_attr.append(cossims_after_without_single_attr)
        
        
        mean_cossims_before_with_single_attr = cossims_before_with_single_attr.mean().item()
        mean_cossims_before_without_single_attr = cossims_before_without_single_attr.mean().item()
        
        mean_cossims_after_with_single_attr = cossims_after_with_single_attr.mean().item()
        mean_cossims_after_without_single_attr = cossims_after_without_single_attr.mean().item()
        

        delta_before_with_vs_wihtout = (mean_cossims_before_with_single_attr - mean_cossims_before_without_single_attr)
        ax[attr_idx, 1].hist(cossims_before_with_single_attr, alpha=0.5, label=f"{mean_cossims_before_with_single_attr:.3f}_diff{delta_before_with_vs_wihtout:.3f}")
        ax[attr_idx, 2].hist(cossims_before_without_single_attr, alpha=0.5, label=f"{mean_cossims_before_without_single_attr:.3f}")


        delta_after_with_vs_wihtout = (mean_cossims_after_with_single_attr - mean_cossims_after_without_single_attr)
        ax[attr_idx, 3].hist(cossims_after_with_single_attr, alpha=0.5, label=f"{mean_cossims_after_with_single_attr:.3f}_diff{delta_after_with_vs_wihtout:.3f}")
        ax[attr_idx, 4].hist(cossims_after_without_single_attr, alpha=0.5, label=f"{mean_cossims_after_without_single_attr:.3f}")


        min_ylim = min(cossims_after_with_single_attr.min(), cossims_after_without_single_attr.min())
        max_ylim = max(cossims_after_with_single_attr.max(), cossims_after_without_single_attr.max())
        min_xlim = min(cossims_before_with_single_attr.min(), cossims_before_without_single_attr.min())
        max_xlim = max(cossims_before_with_single_attr.max(), cossims_before_without_single_attr.max())
        
        if plot_before_after:
            ax[attr_idx, 5].scatter(cossims_before_with_single_attr, cossims_after_with_single_attr, alpha=0.5, s=0.5)
            ax[attr_idx, 6].scatter(cossims_before_without_single_attr, cossims_after_without_single_attr, alpha=0.5, s=0.5)
            ax[attr_idx, 5].set_ylim(min_ylim, max_ylim)
            ax[attr_idx, 6].set_ylim(min_ylim, max_ylim)
            ax[attr_idx, 5].set_xlim(min_xlim, max_xlim)
            ax[attr_idx, 6].set_xlim(min_xlim, max_xlim)
            ax[attr_idx, 5].set_title(f"sim_bef_aft_withAttr")
            ax[attr_idx, 6].set_title(f"sim_bef_aft_woAttr")


        ax[attr_idx, 1].set_title(f"sim_bef_withAttr")
        ax[attr_idx, 2].set_title(f"sim_bef_woAttr")
        ax[attr_idx, 3].set_title(f"sim_aft_withAttr")
        ax[attr_idx, 4].set_title(f"sim_aft_woAttr")
        
        
        if cossims_after_with_null is not None:
            min_xlim = min(cossims_before_with_single_attr.min(), cossims_before_without_single_attr.min(), cossims_after_with_single_attr.min(), cossims_after_without_single_attr.min(), cossims_after_with_null_with_single_attr.min(), cossims_after_with_null_without_single_attr.min())
            max_xlim = max(cossims_before_with_single_attr.max(), cossims_before_without_single_attr.max(), cossims_after_with_single_attr.max(), cossims_after_without_single_attr.max(), cossims_after_with_null_with_single_attr.max(), cossims_after_with_null_without_single_attr.max())
        else:
            min_xlim = min(cossims_before_with_single_attr.min(), cossims_before_without_single_attr.min(), cossims_after_with_single_attr.min(), cossims_after_without_single_attr.min())
            max_xlim = max(cossims_before_with_single_attr.max(), cossims_before_without_single_attr.max(), cossims_after_with_single_attr.max(), cossims_after_without_single_attr.max())
        min_xlim -= 0.1
        max_xlim += 0.1


        for ax_id in range(n_cols):
            ax[attr_idx, ax_id].legend()
            ax[attr_idx, ax_id].set_xlim(min_xlim, max_xlim)

        import textwrap

        if cossims_after_with_null is not None:
            long_text = f"{attr_name}-(sim_with_null{cossims_null_with_attr[attr_idx].item():.3f})"
        else: 
            long_text = f"{attr_name}"
        wrapped_text = textwrap.fill(long_text, width=13)


        ax[attr_idx, 0].text(x=0.05, y=0.3, s=wrapped_text, fontsize=16, wrap=True)
        # set ticks off
        ax[attr_idx, 0].set_xticks([])
        ax[attr_idx, 0].set_yticks([])


    if cossims_after_with_null is not None:
        results_dict['cossims_attr_wise_with_null_with_attr'] = torch.cat(all_cossims_with_attr_with_null, dim=0).mean().item()
        results_dict['cossims_attr_wise_with_null_without_attr'] = torch.cat(all_cossims_with_null_without_attr, dim=0).mean().item()
    results_dict['cossims_attr_wise_with_attr'] = torch.cat(all_cossims_with_attr, dim=0).mean().item()
    results_dict['cossims_attr_wise_without_attr'] = torch.cat(all_cossims_without_attr, dim=0).mean().item()
    fig.savefig(os.path.join(save_path, f"histogram_cossims_cmp_with_without_attr_epo{epoch}{suffix}.png"), bbox_inches='tight')
    plt.close(fig)
    return results_dict


def evaluate_coco_controlled(data, split, args, model, epoch, tokenizer, original_model, filter_algo):


    idxs_per_caption_type = {}
    dataloader = data[split].dataloader
    caption_list = config.CAPTION_LIST
    len_caption_list = len(caption_list)


    end_idx = len(dataloader.dataset)
    for key_id, key in enumerate(caption_list):
        start_idx = key_id
        idxs_per_caption_type[key] = torch.arange(start_idx, end_idx, len_caption_list)
        # .to(device=args.device)

    embeddings_wrapper = EmbeddingsWrapper()
    retrieval_metrics = RetrievalWrapperCocoControlled(args, idxs_per_caption_type)

    model.eval()
    original_model.eval()
    filter_algo.eval()
    unwrapped_model = unwrap_model(model)
    logit_scale = unwrapped_model.logit_scale.exp()
    input_dtype = get_input_dtype(args.precision)
    utils.print_total_cpu_memory()
    losses = {}
    eval_logs = {}

    loss = COCOControlledLoss(args)
    all_images = []


    return_unpooled = False

    with torch.no_grad():
        for batch_idx, data_batch in enumerate(tqdm(dataloader)):
            with args.autocast():
                print(batch_idx)
                (images, texts, counts, multihot_attributes, img_multihot_attributes, _) = data_batch
                all_images.append(images)
                images = images.to(device=args.device, dtype=input_dtype, non_blocking=True)


                counts = counts.to(
                    device=args.device, dtype=input_dtype, non_blocking=True)
                multihot_attributes = multihot_attributes.to(
                    device=args.device, dtype=input_dtype, non_blocking=True)
                img_multihot_attributes = img_multihot_attributes.to(
                    device=args.device, dtype=input_dtype, non_blocking=True)

                texts_tokenized = tokenizer(texts).to(device=args.device, non_blocking=True)

                image_features = unwrapped_model.encode_image(images, normalize=False, return_unpooled=return_unpooled)
                text_features = unwrapped_model.encode_text(texts_tokenized, normalize=False, return_unpooled=return_unpooled)
                unpooled_image_features = None
                unpooled_text_features = None

                original_text_features = original_model.encode_text(texts_tokenized, normalize=False)
                original_img_features = original_model.encode_image(images, normalize=False)

                text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
                image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)


                embeddings_wrapper.update(image_features, original_text_features, text_features, pred_features, texts, None, original_img_features, unpooled_image_features, unpooled_text_features)


        utils.print_available_cpu_memory()
        embeddings_dict = embeddings_wrapper.get_dict()
        all_images = torch.cat(all_images, dim=0)

        all_images = all_images.permute(0, 2, 3, 1)
        all_images = all_images.numpy()


    only_vis = False
    use_single_template = False
    which_attr_setup = 'more_than_1attr'


    if not only_vis:
        null_vector_norm = null_vector_norm.to(torch.float32).cpu().unsqueeze(0)
        cossims_dict_with_null, bucket_list_with_null, results_dict_with_null = get_cossims_with_null_vector(embeddings_dict['all_imgorig_embeddings'].to(torch.float32), embeddings_dict['all_textorig_embeddings'].to(torch.float32), embeddings_dict['all_pr_embeddings'].to(torch.float32), null_vector_norm, idxs_per_caption_type, args)
        # multihot_attributes
        with torch.no_grad():
            with args.autocast():
                logging.info(f"Start:   plot cossims whole  caption: ===============================>")
                results_dict_whole_caption = plot_cossims_whole_caption_coco_controlled(embeddings_dict, idxs_per_caption_type, model, original_model, tokenizer, args, epoch, cossims_dict_with_null=cossims_dict_with_null, bucket_list_with_null=bucket_list_with_null, which_attr_setup=which_attr_setup)
                logging.info(f"End:   plot cossims whole  caption: ===============================>")

                logging.info(f"Start: plot cossims attr wise ===============================>")
                results_dict_attr_wise = plot_cossims_attribute_wise_coco_controlled(embeddings_dict, model, original_model, tokenizer, args=args, idxs_per_caption_type=idxs_per_caption_type, epoch=epoch, cossims_dict_with_null=cossims_dict_with_null, bucket_list_with_null=bucket_list_with_null, use_single_template=use_single_template, which_attr_setup=which_attr_setup)
                logging.info(f"End: plot cossims attr wise ===============================>")

                logging.info(f"plot cossims attr wise 80*8 plot ===============================>")
                # WARNING: null_vector should be normalized
                results_dict_attr_wise_with_attr_and_null = cossim_histogram_before_after_and_comparison(embeddings_dict, COCO_CLASSNAMES, epoch, model, original_model, tokenizer, filter_algo, args, use_single_template, null_vec=null_vector_norm, idxs_per_caption_type=idxs_per_caption_type, which_attr_setup=which_attr_setup)
                logging.info(f"plot cossims attr wise 80*8 plot ===============================>")


            eval_logs.update(losses)
            eval_logs.update(results_dict_whole_caption)
            eval_logs.update(results_dict_attr_wise)
            eval_logs.update(results_dict_with_null)
            eval_logs.update(results_dict_attr_wise_with_attr_and_null)
    else:
        with torch.no_grad():
            with args.autocast():
                logging.info(f"Start: Heatmap ===============================>")
                all_possible_cossims_edited_with_attr_present_heatmap(all_images, embeddings_dict, COCO_CLASSNAMES, model, original_model, tokenizer, filter_algo, args, use_single_template, idxs_per_caption_type=idxs_per_caption_type, num_to_plot=200)
                logging.info(f"End: Heatmap ===============================>")
        eval_logs = None

    return eval_logs, None, None, None, None


def get_cossims_with_null_vector(imgorig_embeddings, textorig_embeddings, edited_image_embeddings, null_vector, idxs_per_caption_type, args):
    imgorig_embeddings = imgorig_embeddings / imgorig_embeddings.norm(dim=-1, keepdim=True)
    textorig_embeddings = textorig_embeddings / textorig_embeddings.norm(dim=-1, keepdim=True)
    edited_image_embeddings = edited_image_embeddings / edited_image_embeddings.norm(dim=-1, keepdim=True)
    null_vector = null_vector / null_vector.norm(dim=-1, keepdim=True)
    # get cossims
    cossims_dict_with_null = {}
    results_dict = {}
    for key in idxs_per_caption_type.keys():
        imgorig_embeddings_chosen = imgorig_embeddings[idxs_per_caption_type[key]]
        textorig_embeddings_chosen = textorig_embeddings[idxs_per_caption_type[key]]
        edited_image_embeddings_chosen = edited_image_embeddings[idxs_per_caption_type[key]]

        assert (len(imgorig_embeddings_chosen) == len(textorig_embeddings_chosen))
        assert (len(imgorig_embeddings_chosen) == len(edited_image_embeddings_chosen))
        assert (len(textorig_embeddings_chosen) == len(edited_image_embeddings_chosen))

        imgorig_null_vector_cossims = imgorig_embeddings_chosen @ null_vector.T
        textorig_null_vector_cossims = textorig_embeddings_chosen @ null_vector.T
        edited_image_null_vector_cossims = edited_image_embeddings_chosen @ null_vector.T

        cossims_dict_with_null[key] = [imgorig_null_vector_cossims.squeeze().numpy(),
                                       textorig_null_vector_cossims.squeeze().numpy(),
                                       edited_image_null_vector_cossims.squeeze().numpy()]

        results_dict.update({f"{key}_imgorig_null_vector_cossims": imgorig_null_vector_cossims.mean().item(),
                             f"{key}_textorig_null_vector_cossims": textorig_null_vector_cossims.mean().item(),
                             f"{key}_edited_image_null_vector_cossims": edited_image_null_vector_cossims.mean().item()})

    bucket_list = ["orig_img_null", "orig_text_null", "edit_img_null"]
    return cossims_dict_with_null, bucket_list, results_dict


def plot_histogram_to_compare_if_sae_bias_is_null_vector(model, args, tokenizer, filter_algo, data, is_train):

    if is_train:
        dataloader = data['train'].dataloader
    else:
        dataloader = data['val'].dataloader
    unwrapped_model = unwrap_model(model)
    input_dtype = get_input_dtype(args.precision)

    all_img_img_cossims, all_text_text_cossims, all_img_text_pos_cossims, all_img_text_neg_cossims, all_bias_img_cossims, all_bias_img_cossims = None, None, None, None, None, None
    with torch.no_grad():
        for batch_idx, data_batch in enumerate(tqdm(dataloader)):
            with args.autocast():
                print(batch_idx)
                (images, texts) = data_batch
                images = images.to(device=args.device, dtype=input_dtype, non_blocking=True)

                texts_tokenized = tokenizer(texts).to(device=args.device, non_blocking=True)
                
                image_features = unwrapped_model.encode_image(images, normalize=True)
                text_features = unwrapped_model.encode_text(texts_tokenized, normalize=True)


                img_text_cossims = image_features @ text_features.T
                img_img_cossims = image_features @ image_features.T
                text_text_cossims = text_features @ text_features.T

                pos_cossims = (img_text_cossims).diagonal()
                neg_cossims = _get_off_diagonal_elements(img_text_cossims).flatten()
                img_off_diagonal_elements = _get_off_diagonal_elements(img_img_cossims).flatten()
                text_off_diagonal_elements = _get_off_diagonal_elements(text_text_cossims).flatten()


                bias = filter_algo.filter_obj.ae.pre_bias
                bias = bias / bias.norm(dim=-1, keepdim=True)
                bias = bias.to(device=args.device, dtype=input_dtype, non_blocking=True)
                if bias.ndim == 1:
                    bias = bias.unsqueeze(dim=0)
                bias_img_cossims = image_features @ bias.T
                bias_text_cossims = text_features @ bias.T

            if all_img_img_cossims is None:
                all_img_img_cossims = img_off_diagonal_elements.cpu().numpy()
                all_text_text_cossims = text_off_diagonal_elements.cpu().numpy()
                all_img_text_pos_cossims = pos_cossims.cpu().numpy()
                all_img_text_neg_cossims = neg_cossims.cpu().numpy()
                all_bias_img_cossims = bias_img_cossims.cpu().numpy()
                all_bias_text_cossims = bias_text_cossims.cpu().numpy()
            else:
                all_img_img_cossims = np.concatenate((all_img_img_cossims, img_off_diagonal_elements.cpu().numpy()), axis=0)
                all_text_text_cossims = np.concatenate((all_text_text_cossims, text_off_diagonal_elements.cpu().numpy()), axis=0)
                all_img_text_pos_cossims = np.concatenate((all_img_text_pos_cossims, pos_cossims.cpu().numpy()), axis=0)
                all_img_text_neg_cossims = np.concatenate((all_img_text_neg_cossims, neg_cossims.cpu().numpy()), axis=0)
                all_bias_img_cossims = np.concatenate((all_bias_img_cossims, bias_img_cossims.cpu().numpy()), axis=0)
                all_bias_text_cossims = np.concatenate((all_bias_text_cossims, bias_text_cossims.cpu().numpy()), axis=0)

        fig, ax = plt.subplots(3, 2, figsize=(6, 10))
        ax[0, 0].hist(all_img_img_cossims, bins=100)
        ax[0, 0].set_title('img_img_cossims')
        ax[0, 1].hist(all_text_text_cossims, bins=100)
        ax[0, 1].set_title('text_text_cossims')
        ax[1, 0].hist(all_img_text_pos_cossims, bins=100)
        ax[1, 0].set_title('img_text_pos_cossims')
        ax[1, 1].hist(all_img_text_neg_cossims, bins=100)
        ax[1, 1].set_title('img_text_neg_cossims')
        ax[2, 0].hist(all_bias_img_cossims, bins=100)
        ax[2, 0].set_title('bias_img_cossims')
        ax[2, 1].hist(all_bias_text_cossims, bins=100)
        ax[2, 1].set_title('bias_text_cossims')
        for i, j in zip(range(3), range(2)):
            ax[i, j].set_xlim(-0.3, 0.8)
        os.makedirs(osp.join('plots'), exist_ok=True)
        if is_train:
            plt.savefig(osp.join('plots', 'bias_comparison_cossims_train.png'))
        else:
            plt.savefig(osp.join('plots', 'bias_comparison_cossims_val.png'))


def _get_off_diagonal_elements(M):
    return M[~torch.eye(*M.shape, dtype=torch.bool)]


def get_correlation_mask_metrics(args, wandb_log_dict, embeddings_dict, completed_epoch):
    all_img_embeddings = embeddings_dict['all_img_embeddings'].float()
    all_textorig_embeddings = embeddings_dict['all_textorig_embeddings'].float()
    all_textft_embeddings = embeddings_dict['all_textft_embeddings'].float()
    all_pr_embeddings = embeddings_dict['all_pr_embeddings'].float()
    all_texts = embeddings_dict['all_texts']
    all_topk_mask = embeddings_dict['all_topk_masks'].float()

    image_text_cossim_clip, text_text_cossim_clip, text_text_cossim_lang_model, conditioned_image_to_text_cossim = get_cossims_for_correlation(all_img_embeddings, all_textorig_embeddings, all_texts, all_pr_embeddings, all_textft_embeddings, normalize=True, args=args)
    metrics_by_using_clip_text_enc, metric_by_using_lang_model = get_correlations(image_text_cossim_clip, text_text_cossim_clip, text_text_cossim_lang_model, conditioned_image_to_text_cossim)


    delta_corr_clip_text_enc = np.array(metrics_by_using_clip_text_enc['conditionedIT_vs_TT']) - np.array(metrics_by_using_clip_text_enc['IT_vs_TT'])
    delta_corr_lang_model = np.array(metric_by_using_lang_model['conditionedIT_vs_TT']) - np.array(metric_by_using_lang_model['IT_vs_TT'])

    # how many positive enytries
    delta_corr_clip_text_enc_pos_num = np.sum(delta_corr_clip_text_enc > 0)
    delta_corr_lang_model_pos_num = np.sum(delta_corr_lang_model > 0)
    # sum of positives
    delta_corr_clip_text_enc_pos_mean = np.mean(delta_corr_clip_text_enc[delta_corr_clip_text_enc > 0])
    delta_corr_lang_model_pos_mean = np.mean(delta_corr_lang_model[delta_corr_lang_model > 0])

    # how many positive enytries
    delta_corr_clip_text_enc_neg_num = np.sum(delta_corr_clip_text_enc < 0)
    delta_corr_lang_model_neg_num = np.sum(delta_corr_lang_model < 0)
    # sum of negitives
    delta_corr_clip_text_enc_neg_mean = np.mean(delta_corr_clip_text_enc[delta_corr_clip_text_enc < 0])
    delta_corr_lang_model_neg_mean = np.mean(delta_corr_lang_model[delta_corr_lang_model < 0])

    wandb_log_dict['delta_corr_clip_text_enc_mean'] = delta_corr_clip_text_enc.mean()
    wandb_log_dict['delta_corr_clip_text_enc_pos_num'] = delta_corr_clip_text_enc_pos_num
    wandb_log_dict['delta_corr_clip_text_enc_pos_mean'] = delta_corr_clip_text_enc_pos_mean
    wandb_log_dict['delta_corr_clip_text_enc_neg_num'] = delta_corr_clip_text_enc_neg_num
    wandb_log_dict['delta_corr_clip_text_enc_neg_mean'] = delta_corr_clip_text_enc_neg_mean

    wandb_log_dict['delta_corr_lang_model_mean'] = delta_corr_lang_model.mean()
    wandb_log_dict['delta_corr_lang_model_pos_num'] = delta_corr_lang_model_pos_num
    wandb_log_dict['delta_corr_lang_model_pos_mean'] = delta_corr_lang_model_pos_mean
    wandb_log_dict['delta_corr_lang_model_neg_num'] = delta_corr_lang_model_neg_num
    wandb_log_dict['delta_corr_lang_model_neg_mean'] = delta_corr_lang_model_neg_mean

    logging.info("Plot =====================================================================>")
    plot_correlation(args, metrics_by_using_clip_text_enc,
                     fname=f'text_text_sim_with_clip_text_enc_epo{completed_epoch}')
    plot_correlation(args, metric_by_using_lang_model,
                     fname=f'text_text_sim_with_sbert_epo{completed_epoch}')
    
    plot_topk_mask(all_topk_mask, args, fname=f'topk_mask_epo{completed_epoch}.png')

    return wandb_log_dict


    ####### NOTE


def evaluate(data, split, args, model, epoch, tokenizer, original_model, filter_algo, step, process_group=None):


    if not is_master(args):
        return {}, None, None, None, None

    input_dtype = get_input_dtype(args.precision)
    logging.info(
        f"==============================> Starting Evaluation epoch: {epoch}")
    wandb_log_dict = {}
    dataloader = data[split].dataloader
    num_samples = 0
    samples_per_val = dataloader.num_samples

    cumulative_loss_dict = {}
    orig_textorig_metrics = MetricWrapperFinetune(tag="orig_text_orig", args=args, process_group=process_group)
    orig_textft_metrics = MetricWrapperFinetune(tag="orig_textft", args=args, process_group=process_group)
    pr_textft_metrics = MetricWrapperFinetune(tag="pr_textft", args=args, process_group=process_group)
    retrieval_metrics = RetrievalWrapper(args)
    embeddings_wrapper = EmbeddingsWrapper()
    zs_metrics = ZeroShotWrapper(args)

    model.eval()
    original_model.eval()
    filter_algo.eval()

    unwrapped_model = unwrap_model(model)
    utils.print_total_cpu_memory()


    for batch_idx, data_batch in enumerate(tqdm(dataloader)):
        with args.autocast():
            if args.finetune_text_encoder:
                (img_features, texts_tokenized, texts) = data_batch
                texts_tokenized = texts_tokenized.to(device=args.device, non_blocking=True)
                img_features = img_features.to(device=args.device, dtype=input_dtype, non_blocking=True)

            original_text_features = original_model.encode_text(
                texts_tokenized)
            original_img_features = original_model.encode_image(images)
            if args.finetune_text_encoder and not args.finetune_vision_encoder:
                # The inputs here are actually text tokens when we finetune the text encoder
                # Text features when we directly load them instead of finetuning
                text_features = unwrapped_model.encode_text(texts_tokenized)
            logit_scale = unwrapped_model.logit_scale.exp()

            pred_features, sae_outputs_dict = filter_algo.get_filtered_embeddings(
                None, None, img_features, text_features)
            pred_features = pred_features.to(dtype=img_features.dtype)

            img_features_norm = img_features / img_features.norm(dim=-1, keepdim=True)
            text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
            pred_features_norm = pred_features / pred_features.norm(dim=-1, keepdim=True)
            original_text_features_norm = original_text_features / original_text_features.norm(dim=-1, keepdim=True)

            embeddings_wrapper.update(img_features, original_text_features, text_features, pred_features, texts, sae_outputs_dict['topk_mask'], original_img_features)

            orig_textorig_metrics.update(img_features_norm, original_text_features_norm, logit_scale, use_intra_modal_cossims=True)
            orig_textft_metrics.update(img_features_norm, text_features_norm, logit_scale)
            loss_val = pr_textft_metrics.update(pred_features_norm, text_features_norm, logit_scale, use_intra_modal_cossims=True)

            ################################################################################################################################################

            inv_loss = torch.tensor(0.0).to(args.device)
            inv_loss_with_lambda = inv_loss * args.inv_loss_lambda


            # returns 0 if args.train_sae is True
            sae_loss_with_lambda, sae_loss_log = filter_algo.get_loss(images, texts, img_features, text_features, sae_outputs_dict)


            batch_size = img_features.shape[0]
            num_samples += batch_size

            total_loss = loss_val + inv_loss_with_lambda + sae_loss_with_lambda  
            loss_dict = {"inv_loss": inv_loss, "inv_loss_with_lambda": inv_loss_with_lambda, "sae_loss": sae_loss_log['sae_loss'], "sae_loss_with_lambda": sae_loss_with_lambda, "recon_loss": sae_loss_log['recon_loss'], "maxk_recon_loss": sae_loss_log['maxk_recon_loss'], 'total_loss': total_loss, }
            update_cumulative_losses(loss_dict, cumulative_loss_dict, batch_size)

    mean_loss_dict = normalize_cumulative_losses(cumulative_loss_dict, num_samples)
    embeddings_dict = embeddings_wrapper.get_dict()

    wandb_log_dict.update(get_correlation_mask_metrics(args, wandb_log_dict, embeddings_dict, epoch))
    wandb_log_dict.update(orig_textorig_metrics.get_logging_dict())
    wandb_log_dict.update(orig_textft_metrics.get_logging_dict())
    wandb_log_dict.update(pr_textft_metrics.get_logging_dict())

    if args.eval_only:
        save_path = osp.join(args.log_base_path, f'cossims_attrwise_plots_{epoch}')
        os.makedirs(save_path, exist_ok=True)
        save_path = osp.join(args.log_base_path, f'cossims_attrwise_plots_coco_{epoch}')
        os.makedirs(save_path, exist_ok=True)
        all_img_embeddings = []
        all_imgorig_embeddings = []
        for batch_idx, data_batch in enumerate(tqdm(data["coco-val"].dataloader)):
            (images, texts) = data_batch
            images = images.to(device=args.device, dtype=input_dtype, non_blocking=True)
            img_features = unwrapped_model.encode_image(images)
            original_img_features = original_model.encode_image(images)
            all_img_embeddings.append(img_features)
            all_imgorig_embeddings.append(original_img_features)
        all_img_embeddings = torch.cat(all_img_embeddings, dim=0).to(args.device)
        all_imgorig_embeddings = torch.cat(all_imgorig_embeddings, dim=0).to(args.device)
        plot_cossims_before_after_attribute_wise(save_path, all_img_embeddings, all_imgorig_embeddings, COCO_CLASSNAMES, model, original_model, tokenizer, filter_algo, filter_algo.filter_obj, select_one_in_five=False, args=args)

        plot_matrix_to_check_for_conditioning(args, all_img_embeddings, all_imgorig_embeddings, COCO_CLASSNAMES, model, original_model, tokenizer, filter_algo, epoch, select_one_in_five=False)

    retrieval_dict = retrieval_metrics.get_logging_dict(embeddings_dict, logit_scale, filter_algo, split=split)
    wandb_log_dict.update(retrieval_dict)

    if args.pr_zero_shot:
        zs_dict = zs_metrics.get_logging_dict(model, original_model, data, tokenizer, epoch, tag='imagenet', filter_algo=filter_algo, filter_obj=filter_algo.filter_obj)
        wandb_log_dict.update(zs_dict)

    for k, v in mean_loss_dict.items():
        wandb_log_dict[f"val-loss/{k}"] = v

    for k, v in wandb_log_dict.items():
        logging.info(f"Epoch: {epoch}, Step: {step},  {k}: {v}")

    wandb_log_dict['epoch'] = epoch
    wandb_log_dict['step'] = step

    if args.wandb:
        logging.info(f"Logging val data to wandb at step {step}, committing")
        wandb.log(wandb_log_dict, step=step, commit=True)

    all_img_embeddings, all_textorig_embeddings, all_textft_embeddings, all_pr_embeddings = embeddings_dict['all_img_embeddings'], embeddings_dict['all_textorig_embeddings'], embeddings_dict['all_textft_embeddings'], embeddings_dict['all_pr_embeddings']

    if args.save_logs:
        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(wandb_log_dict, cls=utils.NpEncoder))
            f.write("\n")

    return wandb_log_dict, all_img_embeddings, all_textorig_embeddings, all_textft_embeddings, all_pr_embeddings


def run_val(data, split, device, input_dtype, autocast, args, model, epoch, tokenizer, log_data, process_group=None, filter_algo=None):

    metrics = {}
    dataloader = data[split].dataloader
    num_samples = 0
    samples_per_val = dataloader.num_samples
    unfiltered_metrics = MetricWrapper(
        tag="unfiltered", args=args, process_group=process_group)
    filtered_metrics = MetricWrapper(
        tag="filtered", args=args, process_group=process_group)

    # all_image_features @ all_text_features will blow up memory and compute very quickly

    # Cumulatives for SAE Loss
    cumulative_clip_loss_dict = {}
    cumulative_sae_loss_dict = {}

    cumulative_gen_loss = torch.tensor(0.0).to(device)
    all_image_features, all_text_features = [], []
    part_recon_all_image_features = []

    unwrapped_model = unwrap_model(model)
    with torch.inference_mode():
        for i, batch in enumerate(dataloader):
            images, texts = batch
            images = images.to(
                device=device, dtype=input_dtype, non_blocking=True)
            tokenized_texts = tokenizer(texts)
            tokenized_texts = tokenized_texts.to(
                device=device, non_blocking=True)


            with autocast():
                batch_size = images.shape[0]
                image_features_unnorm = unwrapped_model.encode_image(images)
                text_features_unnorm = unwrapped_model.encode_text(
                    tokenized_texts)

                # To get the CLIP loss on the full reconstruction
                image_features = image_features_unnorm / image_features_unnorm.norm(dim=-1, keepdim=True)
                text_features = text_features_unnorm / text_features_unnorm.norm(dim=-1, keepdim=True)
                logit_scale = unwrapped_model.logit_scale.exp()

                model_out = {"image_features": image_features,
                             "text_features": text_features, "logit_scale": logit_scale}
                total_loss = get_clip_loss(
                    image_features, text_features, logit_scale, batch_size, device)
                clip_loss_dict = {"contrastive_loss": total_loss}


                if args.filter_algo_name != 'Identity' or args.filter_arch_name != 'Identity':
                    part_recon_image_features_unnorm, sae_outputs_dict = filter_algo.get_filtered_embeddings(
                        images, texts, image_features_unnorm, text_features_unnorm)

                    _, sae_loss_dict = filter_algo.get_loss(
                        images, texts, image_features_unnorm, image_features_unnorm, sae_outputs_dict)

                    part_recon_image_features = part_recon_image_features_unnorm / part_recon_image_features_unnorm.norm(
                        dim=-1, keepdim=True)
                    part_recon_image_features = part_recon_image_features.to(
                        device=device, non_blocking=True)
                    part_contra_total_loss = get_clip_loss(
                        part_recon_image_features, text_features, logit_scale, batch_size, device)

                    clip_loss_dict = {"unfiltered_contrastive_loss": total_loss,
                                      "filtered_contrastive_loss": part_contra_total_loss}

                    val_cossims_part_recon = part_recon_image_features @ text_features.T
                    filtered_metrics.update(
                        part_recon_image_features, text_features, val_cossims_part_recon)

                # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                # however, system RAM is easily exceeded and compute time becomes problematic
                # MAD is 2 mililion but CC3M is 8k

                val_cossims = image_features @ text_features.T
                unfiltered_metrics.update(
                    image_features, text_features, val_cossims)


                gen_loss = maybe_compute_generative_loss(model_out)

            update_cumulative_losses(
                clip_loss_dict, cumulative_clip_loss_dict, batch_size)

            if args.filter_algo_name != 'Identity' or args.filter_arch_name != 'Identity':
                update_cumulative_losses(
                    sae_loss_dict, cumulative_sae_loss_dict, batch_size)

            num_samples += batch_size

            if is_master(args) and (i % 100) == 0:
                logging.info(
                    f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t" +
                    "\t".join([f"{k}: {(v.item()/num_samples):#.5g}" for k, v in cumulative_clip_loss_dict.items()]))
                if args.filter_algo_name != 'Identity' or args.filter_arch_name != 'Identity':
                    logging.info("\t".join(
                        [f"{k}: {(v.item()/num_samples):#.5g}" for k, v in cumulative_sae_loss_dict.items()]))

                # f"Split: {split} Clip Loss: {cumulative_all_loss_dict["cumulative_loss"] / num_samples:.6f}\t"
                # f"Split: {split} Part Recon Clip Loss: {cumulative_all_loss_dict["part_contra_cumulative_loss"] / num_samples:.6f}\t"
                # f"SAE Loss: {cumulative_all_loss_dict["cumulative_sae_loss"] / num_samples:.6f}\t"

                if gen_loss is not None:
                    cumulative_gen_loss += gen_loss * batch_size
                    logging.info(
                        f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")


        mgap_metrics, caption_cossims_metrics, contrastive_loss_metrics = unfiltered_metrics.compute(
            split)
        if args.filter_algo_name != 'Identity' or args.filter_arch_name != 'Identity':
            mgap_metrics_filtered, caption_cossims_metrics_filtered, contrastive_loss_metrics_filtered = filtered_metrics.compute(
                split)
            mgap_metrics.update(mgap_metrics_filtered)
            caption_cossims_metrics.update(caption_cossims_metrics_filtered)
            contrastive_loss_metrics.update(contrastive_loss_metrics_filtered)

        logging.info(f"Epoch: {epoch}, Split: {split}, " +
                     ", ".join([f"{k}: {v}" for k, v in mgap_metrics.items()]))
        logging.info(f"Epoch: {epoch}, Split: {split}, " + ", ".join(
            [f"{k}: {v}" for k, v in caption_cossims_metrics.items()]))
        logging.info(f"Epoch: {epoch}, Split: {split}, " + ", ".join(
            [f"{k}: {v}" for k, v in contrastive_loss_metrics.items()]))


        metrics.update(mgap_metrics)
        metrics.update(contrastive_loss_metrics)
        metrics.update(caption_cossims_metrics)

        log_data.update(
            {"val-mgap-metrics/" + name: val for name, val in mgap_metrics.items()})
        log_data.update({"val-contrastive-loss-metrics/" +
                        name: val for name, val in contrastive_loss_metrics.items()})
        log_data.update({"val-cossim-metrics/" + name: val for name,
                        val in caption_cossims_metrics.items()})

        
        cumulative_clip_loss_dict = normalize_cumulative_losses(
            cumulative_clip_loss_dict, num_samples, split=split)
        cumulative_sae_loss_dict = normalize_cumulative_losses(
            cumulative_sae_loss_dict, num_samples, split=split)


        epoch_metrics = {"epoch": epoch, "num_samples": num_samples}

        metrics.update(cumulative_clip_loss_dict)
        metrics.update(cumulative_sae_loss_dict)
        metrics.update(epoch_metrics)

        log_data.update({"val-clip-loss-metrics/" + name: val for name,
                        val in cumulative_clip_loss_dict.items()})
        log_data.update({"val-sae-loss-metrics/" + name: val for name,
                        val in cumulative_sae_loss_dict.items()})
        log_data.update(
            {"val-epoch-metrics/" + name: val for name, val in epoch_metrics.items()})

        if gen_loss is not None:
            gen_loss = cumulative_gen_loss / num_samples
            metrics.update({f"{split}_generative_loss": gen_loss.item()})

    
    return metrics


# which uses saved attribute list


#Evaluating zeroshot

