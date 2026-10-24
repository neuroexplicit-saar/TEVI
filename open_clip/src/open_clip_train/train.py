import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.distributed
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype

from open_clip_train.distributed import is_master
from open_clip_train.zero_shot import zero_shot_eval, zero_shot_eval_mad_schrodi

import mgap.metrics
import mgap.config
import mgap.utils


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def update_log_data_for_wandb(log_data):
    out_dict = {}
    for k, v in log_data.items():
        if "/" in k:
            out_dict["train-" + k] = v
        else:
            out_dict["train/" + k] = v
    return out_dict


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None, tokenizer=None, filter_algo=None):

    device = torch.device(args.device)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    # set epoch in process safe manner via sampler or shared_epoch
    data['train'].set_epoch(epoch)
    dataloader = data['train'].dataloader
    # To visulaize if dataloader is getting right images and captions
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))


    losses_m = {}
    losses = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    logit_bias = unwrap_model(model).logit_bias
    assert logit_bias is None, "Logit bias is not supported in our code yet"

    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        scheduler(step)

        images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts =texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        total_loss = 0.0
        with args.autocast():
            model_out_unnorm = model(
                images, texts, normalize=False)
            image_features = model_out_unnorm["image_features"]
            text_features = model_out_unnorm["text_features"]
            logit_scale = model_out_unnorm["logit_scale"]

            text_features_norm = text_features / \
                text_features.norm(dim=-1, keepdim=True)
            image_features_norm = image_features / \
                image_features.norm(dim=-1, keepdim=True)
            model_out = {"image_features": image_features_norm,
                         "text_features": text_features_norm, "logit_scale": logit_scale}

            if args.filter_algo_name != 'Identity' or args.filter_arch_name != 'Identity':
                filtered_image_features, sae_outputs_dict = filter_algo.get_filtered_embeddings(
                    images, texts, image_features, text_features)
                filtered_image_features_norm = filtered_image_features / \
                    filtered_image_features.norm(dim=-1, keepdim=True)
                model_out_filtered = {"image_features": filtered_image_features_norm,
                                      "text_features": text_features_norm, "logit_scale": logit_scale}

            if args.distill:
                raise NotImplementedError
                with torch.no_grad():
                    dist_model_out = dist_model(images, texts)
                model_out.update(
                    {f'dist_{k}': v for k, v in dist_model_out.items()})

            # Only for logging

            if args.filter_algo_name != 'Identity' or args.filter_arch_name != 'Identity':
                losses_unfiltered = loss(**model_out, output_dict=True)

                # To use for backward pass
                losses = loss(**model_out_filtered, output_dict=True)

                clip_loss = sum(losses.values())
                losses = {"clip_loss/filtered_" +
                          name: val for name, val in losses.items()}
                losses.update(
                    {"clip_loss/unfiltered_" + name: val for name, val in losses_unfiltered.items()})

                filter_loss, filter_loss_log = filter_algo.get_loss(
                    images, texts, image_features, text_features, sae_outputs_dict)
                losses.update({filter_algo.get_wandb_name(
                ) + "/" + name: val for name, val in filter_loss_log.items()})

                total_loss = clip_loss + filter_loss
                losses["clip_loss/total_loss"] = total_loss

            else:
                losses = loss(**model_out, output_dict=True)
                total_loss = sum(losses.values())
                losses = {"clip_loss/" + name: val for name,
                          val in losses.items()}

        backward(total_loss, scaler)

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
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

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
                "lr": optimizer.param_groups[0]["lr"]
            }
            if scaler is not None:
                log_data["grad_scale"] = grad_scale_scalar
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = update_log_data_for_wandb(log_data)

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def update_cumulative_losses(all_loss_dict, cumulative_all_loss_dict, batch_size):
    for k, v in all_loss_dict.items():
        if k in cumulative_all_loss_dict:
            cumulative_all_loss_dict[k] += v * batch_size
        else:
            cumulative_all_loss_dict[k] = v * batch_size


def normalize_cumulative_losses(cumulative_all_loss_dict, num_samples, split=None):
    out_dict = {}
    for k, v in cumulative_all_loss_dict.items():
        if isinstance(v, torch.Tensor):
            v = v.item()
        if split is not None:
            out_dict[f"{split}_{k}"] = v / num_samples
        else:
            out_dict[k] = v / num_samples
    return out_dict


def get_clip_loss(image_features, text_features, logit_scale, batch_size, device):
    logit_scale = logit_scale.mean()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    labels = torch.arange(batch_size, device=device).long()
    contra_total_loss = (
        F.cross_entropy(logits_per_image, labels) +
        F.cross_entropy(logits_per_text, labels)
    ) / 2

    return contra_total_loss


class MetricWrapper:

    def __init__(self, tag, args, **kwargs):
        super().__init__()
        self.tag = tag
        self.args = args
        process_group = kwargs["process_group"]
        self.mgapliang_metric = mgap.metrics.ModalityGapLiang(
            embed_dim=mgap.config.embed_dim_dict[args.model], dist_sync_on_step=True, process_group=process_group).to(args.device)
        self.caption_cossims_metric = mgap.metrics.CaptionCosSims(
            dist_sync_on_step=True, process_group=process_group).to(args.device)
        self.contrastive_loss_metric = mgap.metrics.ContrastiveLossNumeratorDenominator(
            dist_sync_on_step=True, process_group=process_group).to(args.device)
        self.rmg_schrodi_metric = mgap.metrics.RMGSchrodi(
            dist_sync_on_step=True, process_group=process_group).to(args.device)

    def update(self, image_features, text_features, val_cossims):
        self.mgapliang_metric.update(image_features, text_features)
        self.rmg_schrodi_metric.update(image_features, text_features)
        self.caption_cossims_metric.update(val_cossims)
        self.contrastive_loss_metric.update(val_cossims)

    def compute(self, split, prefix=""):
        mgap_norm, _ = self.mgapliang_metric.compute()
        caption_cossims_metrics = self.caption_cossims_metric.get_logging_dict(
            split, prefix=prefix)
        contrastive_loss_metrics = self.contrastive_loss_metric.get_logging_dict(
            split, prefix=prefix)
        rmg_val = self.rmg_schrodi_metric.compute()

        if prefix != "":
            prefix = prefix + "_"
        mgap_metrics = {f"{split}_{prefix}mgap_norm": mgap_norm.item(
        ), f'{split}_{prefix}rmg_schrodi_metric': rmg_val}
        return mgap_metrics, caption_cossims_metrics, contrastive_loss_metrics


class MetricWrapperFinetune:

    def __init__(self, tag, args, **kwargs):
        super().__init__()
        self.tag = tag
        self.args = args
        process_group = kwargs["process_group"]
        self.caption_cossims_metric = mgap.metrics.CaptionCosSimsSimple(
            dist_sync_on_step=True, process_group=process_group).to(args.device)
        self.mgapliang_metric = mgap.metrics.ModalityGapLiang(
            embed_dim=args.embed_dim, dist_sync_on_step=True, process_group=process_group).to(args.device)
        self.rmg_schrodi_metric = mgap.metrics.RMGSchrodi(
            dist_sync_on_step=True, process_group=process_group).to(args.device)
        self.clip_loss_metric = torch.tensor(0.0).to(args.device)
        self.total_num_samples = 0

    def update(self, image_features, text_features, logit_scale, use_intra_modal_cossims=False):
        img_text_cossims = image_features @ text_features.T
        if use_intra_modal_cossims:
            img_img_cossims = image_features @ image_features.T
            text_text_cossims = text_features @ text_features.T
        else:
            img_img_cossims = None
            text_text_cossims = None
        self.mgapliang_metric.update(image_features, text_features)
        self.rmg_schrodi_metric.update(image_features, text_features)
        self.caption_cossims_metric.update(img_text_cossims, img_img_cossims, text_text_cossims)
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        this_clip_loss_metric = mgap.utils.get_clip_loss(
            image_features_norm, text_features_norm, logit_scale, image_features.shape[0], self.args.device)

        self.clip_loss_metric += (this_clip_loss_metric * image_features.shape[0])
        self.total_num_samples += image_features.shape[0]
        return this_clip_loss_metric

    def get_logging_dict(self, panel_prefix=""):
        if panel_prefix != "":
            prefix = f"{prefix}_"
        output = {}
        output.update(self.mgapliang_metric.get_logging_dict(panel_name=f"{panel_prefix}mgap", prefix=self.tag))
        output.update(self.rmg_schrodi_metric.get_logging_dict(panel_name=f"{panel_prefix}mgap", prefix=self.tag))
        output.update(self.caption_cossims_metric.get_logging_dict(panel_name=f"{panel_prefix}cossims", prefix=self.tag))
        output.update({
            f"val-loss/{self.tag}_clip_loss": self.clip_loss_metric.item() / self.total_num_samples,
        })
        return output


class EmbeddingsWrapper:

    def __init__(self):
        super().__init__()
        self.all_img_embeddings = []
        self.all_textorig_embeddings = []
        self.all_textft_embeddings = []
        self.all_pr_embeddings = []
        self.all_texts = []
        self.all_topk_masks = []
        self.all_imgorig_embeddings = []
        self.all_unpooled_img_embeddings = []
        self.all_unpooled_text_embeddings = []

    def update(self, img_embeddings, textorig_embeddings, textft_embeddings, pr_embeddings, texts, topk_masks, imgorig_embeddings, unpooled_img_embeddings=None, unpooled_text_embeddings=None):
        self.all_img_embeddings.append(img_embeddings.detach().cpu())
        self.all_textorig_embeddings.append(textorig_embeddings.detach().cpu())
        self.all_textft_embeddings.append(textft_embeddings.detach().cpu())
        self.all_pr_embeddings.append(pr_embeddings.detach().cpu())
        self.all_texts += texts
        if topk_masks is not None:
            self.all_topk_masks.append(topk_masks.detach().cpu())
        self.all_imgorig_embeddings.append(imgorig_embeddings.detach().cpu())
        if unpooled_img_embeddings is not None:
            self.all_unpooled_img_embeddings.append(
                unpooled_img_embeddings.detach().cpu())
        if unpooled_text_embeddings is not None:
            self.all_unpooled_text_embeddings.append(
                unpooled_text_embeddings.detach().cpu())

    def get_dict(self):
        all_img_embeddings = torch.cat(self.all_img_embeddings, dim=0)
        all_textorig_embeddings = torch.cat(self.all_textorig_embeddings, dim=0)
        all_textft_embeddings = torch.cat(self.all_textft_embeddings, dim=0)
        all_pr_embeddings = torch.cat(self.all_pr_embeddings, dim=0)
        if len(self.all_topk_masks) > 0:
            all_topk_masks = torch.cat(self.all_topk_masks, dim=0)
        else:
            all_topk_masks = None
        if len(self.all_unpooled_img_embeddings) > 0:
            all_unpooled_img_embeddings = torch.cat(self.all_unpooled_img_embeddings, dim=0)
            all_unpooled_text_embeddings = torch.cat(self.all_unpooled_text_embeddings, dim=0)
        else:
            all_unpooled_img_embeddings = None
            all_unpooled_text_embeddings = None
        all_imgorig_embeddings = torch.cat(self.all_imgorig_embeddings, dim=0)

        output_dict = {}
        output_dict["all_img_embeddings"] = all_img_embeddings
        output_dict["all_textorig_embeddings"] = all_textorig_embeddings
        output_dict["all_textft_embeddings"] = all_textft_embeddings
        output_dict["all_pr_embeddings"] = all_pr_embeddings
        output_dict["all_texts"] = self.all_texts
        output_dict["all_topk_masks"] = all_topk_masks
        output_dict["all_imgorig_embeddings"] = all_imgorig_embeddings
        output_dict["all_unpooled_img_embeddings"] = all_unpooled_img_embeddings
        output_dict["all_unpooled_text_embeddings"] = all_unpooled_text_embeddings
        return output_dict


class RetrievalWrapper:

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.orig_textorig_retrieval = None

    def compute(self, embeddings_dict, logit_scale, filter_algo, split="val"):
        all_img_embeddings = embeddings_dict["all_img_embeddings"]
        all_textorig_embeddings = embeddings_dict["all_textorig_embeddings"]
        all_textft_embeddings = embeddings_dict["all_textft_embeddings"]
        if self.orig_textorig_retrieval is None:
            self.orig_textorig_retrieval = get_clip_metrics(
                image_features=all_img_embeddings.float(),
                text_features=all_textorig_embeddings.float(),
                logit_scale=logit_scale.detach().cpu().float(),
                split=split,
                normalize=True,
                suffix="orig_textorig",
            )

        self.orig_textft_retrieval = get_clip_metrics(
            image_features=all_img_embeddings.float(),
            text_features=all_textft_embeddings.float(),
            logit_scale=logit_scale.detach().cpu().float(),
            split=split,
            normalize=True,
            suffix="orig_textft",
        )
        self.pr_textft_retrieval = get_clip_metrics_custom(
            image_features=all_img_embeddings.float(),
            text_features=all_textft_embeddings.float(),
            logit_scale=logit_scale.detach().cpu().float(),
            split=split,
            args=self.args,
            filter_algo=filter_algo,
            filter_obj=filter_algo.filter_obj,
            suffix="pr_textft_custom",
            topk_ranking=self.args.topk_ranking,
        )

    def get_logging_dict(self, embeddings_dict, logit_scale, filter_algo, split="val"):
        self.compute(embeddings_dict, logit_scale, filter_algo, split)
        output = {}
        output.update(self.orig_textorig_retrieval)
        output.update(self.orig_textft_retrieval)
        output.update(self.pr_textft_retrieval)
        split_keys = ["R@10", "R@5", "R@1", "rank"]
        split_output = {}
        for k, v in output.items():
            if "pr_textft" in k and self.args.topk_ranking:
                disp_k = f"topk_{self.args.topk_ranking}_{k}"
            else:
                disp_k = k
            for sk in split_keys:
                if sk in k:
                    split_output[f"retrieval-{sk}/{disp_k}"] = v.item()
                    break
            else:
                split_output[disp_k] = v.item()
        return split_output


class RetrievalWrapperCocoControlled:

    def __init__(self, args, idxs_per_caption_type=None):
        super().__init__()
        self.args = args
        self.idxs_per_caption_type = idxs_per_caption_type
        self.orig_textorig_retrieval_dict = {key: None for key in self.idxs_per_caption_type.keys()}
        self.pr_textft_retrieval_dict = {key: None for key in self.idxs_per_caption_type.keys()}

    def compute(self, embeddings_dict, logit_scale, filter_algo, split="val", all_category_fnames=None):
        from mgap import config
        multihot_attributes = torch.load(config.VAL_COUNT_ATTR_CAPTION_DICT)["multihot_attributes"]


        for key_name, idxs in self.idxs_per_caption_type.items():

            all_img_embeddings = embeddings_dict["all_img_embeddings"][idxs]
            all_textorig_embeddings = embeddings_dict["all_textorig_embeddings"][idxs]
            all_imgorig_embeddings = embeddings_dict["all_imgorig_embeddings"][idxs]
            all_textft_embeddings = embeddings_dict["all_textft_embeddings"][idxs]
            all_unpooled_img_embeddings = embeddings_dict["all_unpooled_img_embeddings"][idxs]
            all_unpooled_text_embeddings = embeddings_dict["all_unpooled_text_embeddings"][idxs]
            if self.orig_textorig_retrieval_dict[key_name] is None:
                self.orig_textorig_retrieval_dict[key_name], rankings = get_clip_metrics(
                    image_features=all_imgorig_embeddings.float(),
                    text_features=all_textorig_embeddings.float(),
                    logit_scale=logit_scale.detach().cpu().float(),
                    split=split,
                    normalize=True,
                    suffix=f"{key_name}_orig_textorig", return_rankings=True
                )

            self.pr_textft_retrieval_dict[key_name] = get_clip_metrics_custom_coco_controlled(
                image_features=all_img_embeddings.float(),
                text_features=all_textft_embeddings.float(),
                logit_scale=logit_scale.detach().cpu().float(),
                split=split,
                args=self.args,
                filter_algo=filter_algo,
                filter_obj=filter_algo.filter_obj,
                suffix=f"{key_name}_pr_textft_custom",
                topk_ranking=self.args.topk_ranking,
                unpooled_image_features=all_unpooled_img_embeddings.float(),
                unpooled_text_features=all_unpooled_text_embeddings.float()

            )

    def get_logging_dict(self, embeddings_dict, logit_scale, filter_algo, split="val"):
        self.compute(embeddings_dict, logit_scale, filter_algo, split)
        output = {}
        for outer_key, outer_val in self.orig_textorig_retrieval_dict.items():
            output.update(outer_val)
        for outer_key, outer_val in self.pr_textft_retrieval_dict.items():
            output.update(outer_val)
        split_keys = ["R@10", "R@5", "R@1", "rank"]
        split_output = {}
        for k, v in output.items():
            if "pr_textft" in k and self.args.topk_ranking:
                disp_k = f"topk_{self.args.topk_ranking}_{k}"
            else:
                disp_k = k
            for sk in split_keys:
                if sk in k:
                    split_output[f"retrieval-{sk}/{disp_k}"] = v.item()
                    break
            else:
                split_output[disp_k] = v.item()
        return split_output


class ZeroShotWrapper:

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.orig_textorig_zeroshot_results = None

    def compute(self, model, original_model, data, tokenizer, epoch, filter_algo=None, filter_obj=None):
        start_time = time.time()
        if self.orig_textorig_zeroshot_results is None:
            self.orig_textorig_zeroshot_results = zero_shot_eval(original_model, data, epoch, self.args, tokenizer, normalize_embeddings=False, average_embeddings=False)
        self.pr_textft_zeroshot_results = zero_shot_eval(model, data, epoch, self.args, tokenizer, use_filtered=True, filter_algo=filter_algo, filter_obj=filter_obj, normalize_embeddings=False, average_embeddings=False)
        self.ft_textft_zeroshot_results = zero_shot_eval(model, data, epoch, self.args, tokenizer, normalize_embeddings=False, average_embeddings=False)
        self.recon_textft_zeroshot_results = zero_shot_eval(model, data, epoch, self.args, tokenizer, use_recons=True, filter_algo=filter_algo, filter_obj=filter_obj, normalize_embeddings=False, average_embeddings=False)

        logging.info(
            f"Time taken for orig_zero_shot and pr_zero_shot for imagenet: {time.time()-start_time}")
        logging.info(
            f"Epoch: {epoch}, Imagenet orig zeroshot: {self.orig_textorig_zeroshot_results}")
        logging.info(f"Imagenet pr zeroshot: {self.pr_textft_zeroshot_results}.")
        logging.info(f"Imagenet recon zeroshot: {self.recon_textft_zeroshot_results}.")
        logging.info(f"Imagenet ft zeroshot: {self.ft_textft_zeroshot_results}.")

    def get_logging_dict(self, model, original_model, data, tokenizer, epoch, tag="imagenet", filter_algo=None, filter_obj=None):
        self.compute(model, original_model, data, tokenizer, epoch, filter_algo, filter_obj)
        output_dict = {}
        for k, v in self.orig_textorig_zeroshot_results.items():
            output_dict[f"{tag}-zeroshot/orig_textorig_{k}"] = v
        for k, v in self.pr_textft_zeroshot_results.items():
            output_dict[f"{tag}-zeroshot/pr_textft_{k}"] = v
        for k, v in self.ft_textft_zeroshot_results.items():
            output_dict[f"{tag}-zeroshot/ft_textft_{k}"] = v
        for k, v in self.recon_textft_zeroshot_results.items():
            output_dict[f"{tag}-zeroshot/recon_textft_{k}"] = v
        return output_dict


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
            texts = texts.to(
                device=device, non_blocking=True)


            with autocast():
                batch_size = images.shape[0]
                image_features_unnorm = unwrapped_model.encode_image(images)
                text_features_unnorm = unwrapped_model.encode_text(texts)

                # To get the CLIP loss on the full reconstruction
                image_features = image_features_unnorm / \
                    image_features_unnorm.norm(dim=-1, keepdim=True)
                text_features = text_features_unnorm / \
                    text_features_unnorm.norm(dim=-1, keepdim=True)
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

                    part_recon_image_features = part_recon_image_features_unnorm / \
                        part_recon_image_features_unnorm.norm(
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


def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None, process_group=None, filter_algo=None):
    metrics = {}
    log_data = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval_mad_across_attributes(
        model, data, epoch, args, tokenizer)
    metrics.update(zero_shot_metrics)
    log_data.update({"val-zeroshot/" + name: val for name,
                    val in zero_shot_metrics.items()})

    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        metrics_val = run_val(data, 'val', device, input_dtype, args.autocast, args,
                              model, epoch, tokenizer, log_data, process_group, filter_algo=filter_algo)
        metrics.update(metrics_val)

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} " + "(at the end)"
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        print("Logging to wandb ===========>")
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics


def zero_shot_eval_mad_across_attributes(model, data, epoch, args, tokenizer=None):
    zero_shot_metrics = {}
    if args.dataset_type == 'mad' and 'val' in data and (args.zeroshot_frequency and ((epoch % args.zeroshot_frequency) == 0 or epoch == args.epochs)):
        thickthinning = ["no thickthinning", "thickening", "thinning"]
        swelling = ["no swelling", "swelling"]
        fracture = ["no fracture", "fracture"]
        scaling = ["large", "small"]
        colors = ["gray", "red", "green", "blue", "cyan", "magenta", "yellow"]
        digits = [str(i) for i in range(10)]
        attribute_name_list = ['thickthinning', 'swelling',
                               'fracture', 'scaling', 'colors', 'digit']
        classnames_list = [thickthinning, swelling,
                           fracture, scaling, colors, digits]

        top1_list = []
        for attribute_name, classnames in zip(attribute_name_list, classnames_list):
            zero_shot_metrics_single = zero_shot_eval_mad_schrodi(
                model, data, epoch, args, classnames=classnames, tokenizer=tokenizer, attribute_name=attribute_name)
            zero_shot_metrics.update(zero_shot_metrics_single)
            top1_list.append(
                zero_shot_metrics_single[f'{attribute_name}-mad-schrodi-zeroshot-val-top1'])

        zero_shot_metrics['mean-attributes-mad-schrodi-zeroshot-val-top1'] = np.mean(
            top1_list[0:-1])

        zero_shot_metrics['mean-mad-schrodi-zeroshot-val-top1'] = np.mean(
            top1_list)

    return zero_shot_metrics


def get_clip_metrics(image_features, text_features, logit_scale, split, normalize=False, suffix=None, return_rankings=False):
    if suffix is not None:
        split = split + "_" + suffix
    if normalize:
        image_features = image_features / \
            image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / \
            text_features.norm(dim=-1, keepdim=True)
    metrics = {}
    logits_per_image = (logit_scale * image_features @
                        text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image,
              "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    rankings = {}
    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        if return_rankings:
            rankings[name] = ranking.detach().cpu().numpy()
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        for k in [1, 5, 10]:
            metrics[f"{split}_{name}_R@{k}"] = np.mean(preds < k)

    if return_rankings:
        return metrics, rankings
    else:
        return metrics, torch.argsort(logits["image_to_text"], descending=True), logits_per_image


def get_clip_metrics_custom(image_features, text_features, logit_scale, split, args, filter_algo, filter_obj, suffix=None, topk_ranking = None, return_cossims = False):
    """
    WARNING: this function expects unnormalized features!
    """

    if suffix is not None:
        split = split + "_" + suffix

    logits_per_image = torch.zeros((image_features.shape[0], text_features.shape[0]))  # .to(args.device)
    

    text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    mask_full = filter_algo.get_mask(text_features)

    if topk_ranking is not None:
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        logits_per_image = (image_features_norm  @ text_features_norm.t()).detach().cpu()

        topk_text_idxs = torch.topk(logits_per_image, k=topk_ranking, dim=1)

    last_time = time.time()
    pos_cossims = []
    neg_cossims = []
    for i, image_feature in enumerate(image_features):
        

        if (i+1) % 4000 == 0:
            logging.info(f"Custom Retrieval-  Processing {i} of {image_features.shape[0]}, time taken: {time.time() - last_time:.2f} seconds")
            last_time = time.time()
        if topk_ranking is not None:
            mask = mask_full[topk_text_idxs.indices[i]]
        else:
            mask = mask_full

        image_feature = image_feature.to(args.device)
        info = filter_obj.get_latents(image_feature)
        latents, inds, vals = filter_obj.unpack_info(info)


        if inds is not None:
            after_mask_vals = vals * mask[:, inds]
            # it repeats the inds to match the number of texts as each image is conditioned on text
            inds = inds.unsqueeze(0)
            inds = inds.expand(mask.shape[0], inds.shape[1])
            inds = inds.contiguous()


            info_updated = {"inds": inds, "vals": after_mask_vals,
                            "latents_pre_act": latents, "mask": mask}
            part_recon_image_features = filter_obj.get_reconstructed_features(
                image_feature, info_updated)


        else:
            part_recon_image_features = latents * mask

        part_recon_image_features = part_recon_image_features / \
            part_recon_image_features.norm(dim=-1, keepdim=True)


        if topk_ranking is not None:
            logits_per_image[i, topk_text_idxs.indices[i]] = ((part_recon_image_features.cpu() * text_features_norm[topk_text_idxs.indices[i]]).sum(dim=1)).detach().cpu()
        else:
            logits_per_image[i, :] = ((part_recon_image_features.cpu() * text_features_norm.cpu()).sum(dim=1)).detach().cpu()
            pos_cossims.append(logits_per_image[i, :][i].item())
            neg_cossims.append(logits_per_image[i, :].mean().item()-logits_per_image[i, :][i].item())


    metrics = {}
    
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image,
              "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        for k in [1, 5, 10]:
            metrics[f"{split}_{name}_R@{k}"] = np.mean(preds < k)
        print(f"done for one! ")
    
    if return_cossims:
        return metrics, torch.argsort(logits["image_to_text"], descending=True), logits_per_image, pos_cossims, neg_cossims
    else:
        return metrics, torch.argsort(logits["image_to_text"], descending=True), logits_per_image


def get_clip_metrics_custom_coco_controlled(image_features, text_features, logit_scale, split, args, filter_algo, filter_obj, suffix=None, topk_ranking=None, unpooled_image_features=None, unpooled_text_features=None):
    """
    WARNING: this function expects unnormalized features!
    Updated to include get_filtered_embeddings_single_image_multiple_texts, to also allow for SimpleMLP
    """

    if suffix is not None:
        split = split + "_" + suffix

    logits_per_image = torch.zeros(
        (image_features.shape[0], text_features.shape[0]))  # .to(args.device)

    text_features = text_features.to(args.device)
    image_features = image_features.to(args.device)
    unpooled_image_features = unpooled_image_features.to(args.device)
    unpooled_text_features = unpooled_text_features.to(args.device)

    text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    if topk_ranking is not None:
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)

        logits_per_image = (logit_scale * image_features_norm  @ text_features_norm.t()).detach().cpu()
        topk_text_idxs = torch.topk(logits_per_image, k=topk_ranking, dim=1)


    image_features_to_be_processed = image_features
    text_features_to_be_processed = text_features


    for i, image_feature in enumerate(image_features_to_be_processed):
        if i % 4000 == 0:
            logging.info(f"Custom Retrieval-  Processing {i} of {image_features.shape[0]}")

        image_feature = image_feature.unsqueeze(0)

        part_recon_image_features, _ = filter_algo.get_filtered_embeddings_single_image_multiple_texts(None, None, image_feature, text_features_to_be_processed)

        part_recon_image_features = part_recon_image_features / \
            part_recon_image_features.norm(dim=-1, keepdim=True)

        if topk_ranking is not None:
            logits_per_image[i, topk_text_idxs.indices[i]] = (
                logit_scale * (part_recon_image_features[topk_text_idxs.indices[i].to(args.device)] * text_features_norm[topk_text_idxs.indices[i].to(args.device)]).sum(dim=1)).detach().cpu()
        else:
            logits_per_image[i, :] = (
                logit_scale * (part_recon_image_features * text_features_norm).sum(dim=1)).detach().cpu()

    metrics = {}
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image,
              "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{split}_{name}_R@{k}"] = np.mean(preds < k)
    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)

