import os.path as osp
import torch
import random
import numpy as np
import os
from mgap import config
import sparse_autoencoder.sae_trainer
from open_clip_train.distributed import broadcast_object
from datetime import datetime
import matplotlib.pyplot as plt


import torch.utils
import json

import subprocess
import re
import glob
import torch.nn.functional as F
from mgap.datasets import CustomTemplateDataset
from tqdm import tqdm
import socket
from datetime import datetime, timedelta

from sentence_transformers import SentenceTransformer as SBERT
import psutil
import logging
from torch.utils.data import DataLoader


from mgap.datasets import LearntMaskTextConditioningEvalDataset, LearntMaskTextConditioningEvalCollator
import fcntl

decoder_name_to_index = config.decoder_name_to_index
mad_decoder_name_list_cat_wise_numerical = config.mad_decoder_name_list_cat_wise_numerical


def generate_text(args, cls_label: int, words, return_idxs=False):
    string_content = [str(cls_label)]
    string_content += random.sample(words, int(args.mad_mode[-1]))
    random.shuffle(string_content)
    indices = [decoder_name_to_index[entry] for entry in string_content]
    caption = "-".join(string_content)

    if return_idxs:
        return caption, indices
    else:
        return caption


def write_to_csv(eval_logs, args):
    if eval_logs is not None:
        title_str = ""
        for k, v in eval_logs.items():
            title_str += f"{k},"
        logging.info(title_str)

        if args.filter_arch_name == "Identity" and args.filter_algo_name == "Identity":
            out_str = "load_clip_config,"+",".join(args.load_name.split("-")) + ",train_clip_lr"+f",{str(args.lr)},"
        
        else:
            # load_sae_config_name / mask_config_name can be None -> fall back to "None".
            load_sae_config_name = args.load_sae_config_name if args.load_sae_config_name is not None else "None"
            mask_config_name = args.mask_config_name if args.mask_config_name is not None else "None"
            out_str = "load_clip_config,"+",".join(args.load_name.split("-")) + ",train_clip_lr"+f",{str(args.lr)}," +"load_sae_config_name,"+ ",".join(
            load_sae_config_name.split("-")) + ",train_sae_config" +f",saelr{str(args.train_sae_lr)},CLL{str(args.contra_loss_lambda)}," + "mask_config_name," + ",".join(mask_config_name.split("-")) + ",topk_ranking" + f",{str(args.topk_ranking)}" + ","

        # the seed selects which trained-mask checkpoint was loaded (-seed{N} in
        # mask_config_name), so rows from a seed sweep are otherwise identical
        out_str += "seed," + f"{str(args.seed)}" + ","


        for k, v in eval_logs.items():
            out_str += f"{v},"
        out_str += "\n"
        csv_path = args.csv_path
        os.makedirs(osp.dirname(csv_path), exist_ok=True)
        with open(csv_path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(out_str)
            fcntl.flock(f, fcntl.LOCK_UN)
        logging.info(f'Eval results saved to {csv_path}')


def write_to_csv_original(eval_logs, args):
    if eval_logs is not None:
        title_str = ""
        for k, v in eval_logs.items():
            title_str += f"{k},"
        logging.info(title_str)

        
        out_str = "train_clip_lr"+f",{str(args.lr)}," + "model" + f",{args.model}," + "siglip" + f",{args.siglip},"
        
        
        for k, v in eval_logs.items():
            out_str += f"{v},"
        out_str += "\n"
        csv_path = args.csv_path
        os.makedirs(osp.dirname(csv_path), exist_ok=True)
        with open(csv_path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(out_str)
            fcntl.flock(f, fcntl.LOCK_UN)
        logging.info(f'Eval results saved to {csv_path}')


def get_cossims_for_correlation(image_features, text_features, texts, conditioned_image_features, ft_text_features, normalize=True, args=None):

    if normalize:
        image_features = image_features / \
            image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / \
            text_features.norm(dim=-1, keepdim=True)
        conditioned_image_features = conditioned_image_features / \
            conditioned_image_features.norm(dim=-1, keepdim=True)
        ft_text_features = ft_text_features / \
            ft_text_features.norm(dim=-1, keepdim=True)

    conditioned_image_to_text_cossim = (
        conditioned_image_features @ ft_text_features.t()).detach().cpu().numpy()
    image_text_cossim_clip = (image_features @
                              text_features.t()).detach().cpu().numpy()
    text_text_cossim_clip = (
        text_features @ text_features.t()).detach().cpu().numpy()

    sbert = SBERT('all-mpnet-base-v2', device=args.device)
    text_embeddings = sbert.encode(texts, convert_to_tensor=True)
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    text_cossim_matrix = text_embeddings @ text_embeddings.t()
    text_text_cossim_lang_model = text_cossim_matrix.detach().cpu().numpy()
    return image_text_cossim_clip, text_text_cossim_clip, text_text_cossim_lang_model, conditioned_image_to_text_cossim


def get_correlations(image_text_cossim_clip, text_text_cossim_clip, text_text_cossim_lang_model, conditioned_image_to_text_cossim):

    from scipy.stats import pearsonr


    metrics_by_using_clip_text_enc = {
        'conditionedIT_vs_TT': [], 'IT_vs_TT': []}

    metric_by_using_lang_model = {'conditionedIT_vs_TT': [], 'IT_vs_TT': []}

    for i in range(text_text_cossim_lang_model.shape[0]):
        metric_by_using_lang_model['conditionedIT_vs_TT'].append(pearsonr(
            conditioned_image_to_text_cossim[i], text_text_cossim_lang_model[i]).statistic)
        metric_by_using_lang_model['IT_vs_TT'].append(
            pearsonr(image_text_cossim_clip[i], text_text_cossim_lang_model[i]).statistic)
        metrics_by_using_clip_text_enc['conditionedIT_vs_TT'].append(pearsonr(
            conditioned_image_to_text_cossim[i], text_text_cossim_clip[i]).statistic)
        metrics_by_using_clip_text_enc['IT_vs_TT'].append(
            pearsonr(image_text_cossim_clip[i], text_text_cossim_clip[i]).statistic)

    return metrics_by_using_clip_text_enc, metric_by_using_lang_model


def plot_correlation(args, correlation_dict, fname):
    list1 = correlation_dict['conditionedIT_vs_TT']
    list2 = correlation_dict['IT_vs_TT']
    array1 = np.array(list1)
    array2 = np.array(list2)

    # get sorted indices of array2
    sorted_indices = np.argsort(array2)
    array1 = array1[sorted_indices]
    array2 = array2[sorted_indices]

    plt.figure(figsize=(10, 6))
    width = 0.35  # Width of bars
    x = np.arange(len(array1))  # X-axis positions


    plt.scatter(x, array1, marker='o', s=1,
                label='conditionedIT_vs_TT', color='blue')
    plt.scatter(x, array2, marker='s', s=1, label='IT_vs_TT', color='red')

    # Customize the plot
    plt.xlabel('correlation')
    plt.ylabel('Scores')
    plt.title(f'Comparing analysis of correlation: {fname} ')
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plot_path = osp.join(args.log_base_path, 'plots')
    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(osp.join(plot_path, f'{fname}.png'))


def plot_topk_mask(all_topk_mask, args, fname):

    # get number of elements where the entries are not -1 in a row
    # sum of the elements per columnwise
    mean_list = []
    std_list = []
    for col_idx in range(all_topk_mask.shape[1]):
        non_dead_idxs = torch.where(all_topk_mask[:, col_idx] != -1)[0]
        num_elems = len(non_dead_idxs)

        if num_elems > 0:
            mean_list.append(all_topk_mask[:, col_idx][non_dead_idxs].mean())
            std_list.append(all_topk_mask[:, col_idx][non_dead_idxs].std())

    mean_topk_mask = torch.tensor(mean_list)
    std_topk_mask = torch.tensor(std_list)


    # get ascending sorted index
    sorted_indices = torch.argsort(mean_topk_mask)

    mean_topk_mask = mean_topk_mask[sorted_indices]
    std_topk_mask = std_topk_mask[sorted_indices]

    upper_bound = mean_topk_mask + std_topk_mask
    lower_bound = mean_topk_mask - std_topk_mask

    mean_topk_mask = mean_topk_mask.numpy()
    x = np.arange(len(mean_topk_mask))
    fig, ax = plt.subplots(1)
    ax.fill_between(x, lower_bound.numpy(), upper_bound.numpy(),
                    facecolor='yellow', alpha=0.5, label='1 sigma range')
    ax.plot(x, mean_topk_mask, marker='o',
            label='mean mask values', color='blue')
    ax.legend(loc='upper left')
    ax.set_xlabel('nodes')
    ax.set_ylabel('mask values')
    ax.grid()
    plot_path = osp.join(args.log_base_path, 'plots')
    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(osp.join(plot_path, fname))


def get_remaining_time():
    job_id = os.getenv('SLURM_JOB_ID')
    if job_id:
        # Query job information
        result = subprocess.run(
            ['scontrol', 'show', 'job', job_id], stdout=subprocess.PIPE, text=True)
        output = result.stdout

        start_time, end_time = None, None

        for line in output.splitlines():
            blocks = line.split()
            for block in blocks:
                if "StartTime" in block:
                    start_time = block.split('=')[1].strip()
                if "EndTime" in block:
                    end_time = block.split('=')[1].strip()

        if start_time and end_time:
            # Parse time strings
            start_dt = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
            end_dt = datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%S")
            remaining = end_dt - datetime.now()
            return remaining

    return None


def check_and_terminate(threshold_seconds):

    remaining_time = get_remaining_time()
    threshold_time = timedelta(seconds=threshold_seconds)
    if remaining_time is None:
        print("SLURM job information not available.")
        return False

    print(f"Remaining time: {remaining_time}")
    if remaining_time < threshold_time:
        print(
            f"Remaining time ({remaining_time}) is below threshold ({threshold_time.total_seconds()/60} minutes). Terminating process.", flush=True)
        return True

    return False


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote: bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(
            ["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1])
                       for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def get_name_config(args):
    # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
    model_name_safe = args.model.replace('/', '-')
    date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    if args.distributed:
        # sync date_str from master to all ranks
        date_str = broadcast_object(args, date_str)

    name = '-'.join([
        f"model_{model_name_safe}",
        f"lr_{args.lr}",
        f"b_{args.batch_size}",
        f"j_{args.workers}",
        f"p_{args.precision}",
    ])
    name += f"-mode_{args.mad_mode}"
    return name


def get_load_name_config(args):
    # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
    model_name_safe = args.model.replace('/', '-')
    date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    if args.distributed:
        # sync date_str from master to all ranks
        date_str = broadcast_object(args, date_str)

    name = '-'.join([
        f"model_{model_name_safe}",
        f"lr_{args.load_lr}",
        f"b_{args.load_batch_size}",
        f"j_{args.load_workers}",
        f"p_{args.precision}",
    ])
    name += f"-mode_{args.mad_mode}"
    return name


def get_sae_config_name(args):
    sae_config_name = '-'.join([
        f"lr_{args.train_sae_lr}",
        f"b_{args.train_sae_batch_size}",
        f"e_{args.train_sae_epochs}",
        f"k_{args.k}",
        f"auxk_{args.auxk}",
        f"ef_{args.expansion_factor}",
    ])
    if not args.dataset_name == "MAD" and args.optimizer_name == "AdamW":
        sae_config_name += f"-AdamW"
    if args.use_fixed_decoder_weights:
        sae_config_name += f"-fdw_{args.decoder_weights_vocab}"

    return sae_config_name


def get_load_sae_config_name(args):
    load_sae_config_name = '-'.join([
        f"lr_{args.load_sae_lr}",
        f"b_{args.load_sae_batch_size}",
        f"e_{args.load_sae_epochs}",
        f"k_{args.load_k}",
        f"auxk_{args.load_auxk}",
        f"ef_{args.expansion_factor}",
    ])
    if not args.dataset_name == "MAD" and args.optimizer_name == "AdamW":
        load_sae_config_name += f"-AdamW"
    
    # Change when we do these experiemnts
    if args.use_fixed_decoder_weights:
        load_sae_config_name += f"-fdw_{args.decoder_weights_vocab}"
    return load_sae_config_name


def get_mask_config_name(args):
    mask_config_name = '-'.join([
        f"masklr{args.train_mask_lr}",
        f"ablt{args.attr_to_ablate}" if args.dataset_name == "MAD" else "",
        f"e{args.train_mask_epochs}",
        f"b{args.train_mask_batch_size}",
        f"warme{args.warmup_mask_epochs}" if args.warmup_mask_epochs > 0 else "",
        f"loss{args.loss_name}",
    ])
    mask_config_name += f"-numlay{args.n_layers}"
    if args.finetune_text_encoder:
        mask_config_name += "-FTtextenc"
    if args.clip_logit_scale_fixed is not None:
        mask_config_name += f"-logitscale{args.clip_logit_scale_fixed}"
    if args.seed !=0: 
        mask_config_name += f"-seed{args.seed}"

        
    return mask_config_name


def get_wandb_name(args):
    mask_config_name = ""
    # CLL is used if we finetune CLIP and SAE together
    mask_config_name += f"lr{args.lr}_bs{args.batch_size}_e{args.train_mask_epochs}_saelr{args.train_sae_lr}_masklr{args.train_mask_lr}_siglip{args.siglip}"


    mask_config_name += f"-nl{args.n_layers}"
    if args.clip_logit_scale_fixed is not None:
        mask_config_name += f"-logitscale{args.clip_logit_scale_fixed}"
    if args.seed !=0: 
        mask_config_name += f"-seed{args.seed}"


    return mask_config_name


def print_total_cpu_memory():
    memory_info = psutil.virtual_memory()
    tot_mem = memory_info.total / (1024**3)
    print(f"===========================>", flush=True)
    print(f"Total CPU memory: {tot_mem:.2f} GB", flush=True)


def print_available_cpu_memory():
    memory_info = psutil.virtual_memory()
    avail_mem = memory_info.available / (1024**3)
    print(f"===========================>", flush=True)
    print(f"Available CPU memory: {avail_mem:.2f} GB", flush=True)
def gpu_mem(prefix=""):
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved  = torch.cuda.memory_reserved() / 1024**2
    print(f"{prefix} | allocated: {allocated:.1f} MB | reserved: {reserved:.1f} MB")

def set_cfg_params(cfg, args, fixed_vocab_dim=None):
    if args.use_pretrained_sae:
        cfg.load_k = args.load_k
        cfg.load_auxk = args.load_auxk
    cfg.k = args.k
    cfg.auxk = args.auxk
    cfg.d_model = args.embed_dim

    if fixed_vocab_dim is not None and not args.hybrid_sae_decoder: 
        cfg.n_dirs = fixed_vocab_dim * args.expansion_factor
    else:
        cfg.n_dirs = cfg.d_model * args.expansion_factor

    return cfg


def get_sae(cfg, args):
    comms = sparse_autoencoder.sae_trainer.make_torch_comms(
        n_op_shards=cfg.n_op_shards, n_replicas=cfg.n_replicas)
    n_dirs_local = cfg.n_dirs // cfg.n_op_shards

    k_used = cfg.k
    auxk_used = cfg.auxk
    if args.use_pretrained_sae:
        k_used = cfg.load_k
        auxk_used = cfg.load_auxk

    ae = sparse_autoencoder.sae_trainer.FastAutoencoder(
        n_dirs_local=n_dirs_local,
        d_model=cfg.d_model,
        k=k_used,
        auxk=auxk_used,
        dead_steps_threshold=cfg.dead_toks_threshold // cfg.bs,
        comms=comms,
        hybrid_sae_decoder = args.hybrid_sae_decoder,
    )
    ae = ae.to(args.device)
    ae.return_inds_vals = True
    return ae, comms


def set_seed(seed=42, rank=0):
    """
    Set seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def sae_loss_fn(ae, flat_acts_batch, recons, info, logger, cfg, mse_scale, split):
    return (
        # MSE
        logger.logkv(f"{split}/recons", mse_scale *
                     sparse_autoencoder.kernels.mse(recons, flat_acts_batch))
        # AuxK
        + logger.logkv(
            f"{split}/maxk_recons",
            cfg.auxk_coef
            * sparse_autoencoder.kernels.normalized_mse(
                ae.decode_sparse(
                    info["auxk_inds"],
                    info["auxk_vals"],
                ),
                flat_acts_batch - recons.detach() + ae.pre_bias.detach(),
            ).nan_to_num(0),
        )
    ),


def save_ckpt(args, save_path, ckpt_tag, sae_model=None, clip_model=None, linear_transform=None, optimizer=None, scaler=None, mse_scaler=None):
    checkpoint = {}
    if sae_model is not None:
        checkpoint["sae_model"] = sae_model.state_dict()
    if mse_scaler is not None:
        checkpoint["mse_scaler"] = mse_scaler
    if clip_model is not None:
        checkpoint["clip_model"] = clip_model.state_dict()
    if linear_transform is not None:
        checkpoint["linear_transform"] = linear_transform.state_dict()
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    checkpoint["args"] = vars(args)
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(
        save_path, f"checkpoint_{ckpt_tag}.pt")
    torch.save(checkpoint, file_path)
    print("Saved checkpoint!")
    print(f"Save path: {file_path}")


def binary_matrix(N):
    numbers = torch.arange(2**N)
    matrix = (numbers[:, None] >> torch.arange(N - 1, -1, -1)) & 1
    return matrix.float()


def common_init(args, return_old_path=False, mask_config_suffix=None):
    args.embed_dim = config.embed_dim_dict[args.model]
    slurm_host = os.getenv("SLURMD_NODENAME")
    host_name = slurm_host if slurm_host else socket.gethostname()
    print(f"Running on host: {host_name}")
    args.slurm_host_name = host_name

    set_seed(args.seed)

    if args.train_clip or args.finetune_text_encoder:
        args.name = get_name_config(args)
    else:
        args.name = None
    if args.train_sae:
        args.sae_config_name = get_sae_config_name(args)
    else:
        args.sae_config_name = None
    if args.train_mask:
        args.mask_config_name = get_mask_config_name(args)
    else:
        args.mask_config_name = None


    if args.use_pretrained_clip:
        args.load_name = get_load_name_config(args)
    else:
        args.load_name = None

    if args.use_pretrained_sae:
        args.load_sae_config_name = get_load_sae_config_name(args)
    else:
        args.load_sae_config_name = None

    args.wandb_name = ""
    if args.mask_config_name is not None:
        args.wandb_name = args.mask_config_name
        if args.sae_config_name is not None:
            args.wandb_name += "-"

    if args.sae_config_name is not None:
        args.wandb_name += args.sae_config_name
        if args.name is not None:
            args.wandb_name += "-"

    if args.name is not None:
        args.wandb_name += args.name

    model_name_for_saving = args.model.replace('/', '')
    if return_old_path:
        parent_directory = args.logs
    else:
        load_dataset_name = args.load_dataset_name
        

        dataset_name = args.dataset_name 


        parent_directory = osp.join(args.logs, model_name_for_saving, load_dataset_name)


        if args.load_clip_logs: 
            parent_directory_load_clip = osp.join(args.load_clip_logs, model_name_for_saving, load_dataset_name)  
            if dataset_name != load_dataset_name: 
                parent_directory_load_clip = osp.join(parent_directory_load_clip, dataset_name)

        else: 
            parent_directory_load_clip = parent_directory

        if args.load_sae_logs:
            parent_directory_load_sae = osp.join(args.load_sae_logs, model_name_for_saving, load_dataset_name)  
            if dataset_name != load_dataset_name: 
                parent_directory_load_sae = osp.join(parent_directory_load_sae, dataset_name)
        else:
            parent_directory_load_sae = parent_directory


    # both train_clip and use_pretrained_clip can not be False together
    if args.use_pretrained_clip:
        log_base_path = osp.join(parent_directory, args.load_name)
        # if we do only eval, we want to store the plots etc in Contents

        if args.train_clip and args.finetune_text_encoder:
            log_base_path = osp.join(log_base_path, "Finetune", args.name)
        else:
            log_base_path = osp.join(log_base_path, "Frozen")
        if not args.train_sae and not args.train_mask and not args.use_pretrained_sae and not args.use_pretrained_mask:
            log_base_path = osp.join(log_base_path, "Contents")

    elif args.train_clip:
        log_base_path = osp.join(parent_directory, args.name)
        if not args.train_sae and not args.train_mask and not args.use_pretrained_sae and not args.use_pretrained_mask:
            log_base_path = osp.join(log_base_path, "Contents")
        else:
            log_base_path = osp.join(log_base_path, "FromScratch")
    else:
        log_base_path = parent_directory

    if args.train_sae or args.use_pretrained_sae:
        log_base_path = osp.join(log_base_path, args.filter_arch_name)

    if args.use_pretrained_sae:
        # For a random SAE, don't bury the mask checkpoint under the trained-SAE
        # hyperparameter folder; use a fixed 'random_sae' segment instead.
        sae_path_segment = "random_sae" if args.random_sae else args.load_sae_config_name
        log_base_path = osp.join(log_base_path, sae_path_segment)
        # if we do only eval, we want to store the plots etc in Contents
        if args.train_sae:
            log_base_path = osp.join(
                log_base_path, "Finetune", args.sae_config_name)
        else:
            log_base_path = osp.join(log_base_path, "Frozen")

    elif args.train_sae:
        log_base_path = osp.join(log_base_path, args.sae_config_name)
        if args.use_pretrained_clip and not args.train_clip and not args.train_mask:
            log_base_path = osp.join(log_base_path, "Contents")
        else:
            log_base_path = osp.join(log_base_path, "FromScratch")

    if args.train_mask or args.use_pretrained_mask:
        log_base_path = osp.join(log_base_path, args.filter_algo_name)


    if args.train_mask:
        log_base_path = osp.join(log_base_path, args.mask_config_name)
        log_base_path = osp.join(log_base_path, "FromScratch")

    args.log_base_path = log_base_path
    args.checkpoint_path = osp.join(log_base_path, 'checkpoints')

    print("\n \n")
    print(f"Log base path: {log_base_path}")


    args.load_clip_path = None
    args.load_clip_checkpoint_path = None
    if args.use_pretrained_clip:
        args.load_clip_path = osp.join(
            parent_directory_load_clip, args.load_name, "Contents")
        args.load_clip_checkpoint_path = osp.join(
            args.load_clip_path, "checkpoints", f"epoch_{args.pretrained_specific_checkpoint_path}.pt")
        print("\n \n")
        print(
            f"Path to load clip model from: {args.load_clip_checkpoint_path}")

    args.load_sae_path = None
    args.load_sae_checkpoint_path = None
    filter_folder_name = args.filter_arch_name
    if args.use_pretrained_sae and args.filter_arch_name != "Identity":
        args.load_sae_path = osp.join(parent_directory_load_sae, args.load_name, "Frozen", filter_folder_name, args.load_sae_config_name, "Contents")
        args.load_sae_checkpoint_path = osp.join(args.load_sae_path, "checkpoints", f"checkpoint_{args.sae_pretrained_specific_checkpoint_path}.pt")
        
        print("\n \n")
        print(
            f"Path to load Filter architecture from: {args.load_sae_checkpoint_path}")


def get_clip_loss(image_features, text_features, logit_scale, batch_size, device, cossim_scaler=None, **kwargs):
    logits_per_image = logit_scale * image_features @ text_features.t()
    if cossim_scaler is not None:
        logits_per_image = cossim_scaler(logits_per_image, **kwargs)
    logits_per_text = logits_per_image.t()

    labels = torch.arange(batch_size, device=device).long()
    contra_total_loss = (
        F.cross_entropy(logits_per_image, labels) +
        F.cross_entropy(logits_per_text, labels)
    ) / 2

    return contra_total_loss


def generate_attributes_from_texts(texts):
    attributes = []
    for idx in range(len(texts)):
        attributes_present = [config.decoder_name_to_index[attr]
                              for attr in texts[idx].split("-")]
        attribute_mask = torch.zeros(len(config.mad_decoder_name_list))
        attribute_mask[attributes_present] = 1
        attributes.append(attribute_mask)
    attributes = torch.stack(attributes)
    return attributes


def get_mad_features(load_clip_path, is_split='train'):
    print("\n")
    print(f"Getting features from {load_clip_path} ========>")
    if is_split == "train":
        img_features_attributes = torch.load(osp.join(load_clip_path, 'train_features_and_attributes.pt'), map_location="cpu")
        img_features = img_features_attributes['features']
        img_attributes = img_features_attributes['attribute_lists']

        text_features_attributes = torch.load(osp.join(load_clip_path, 'train_text_features_and_multihotGT.pt'), map_location="cpu")
        text_features = text_features_attributes['features']
        img_mask = text_features_attributes['multihotGT']
        text_mask = text_features_attributes['multihot_partial_caption']
        texts = text_features_attributes['partial_caption_list']

    else:
        img_features_attributes = torch.load(osp.join(load_clip_path, 'val_features_and_attributes.pt'), map_location="cpu")
        img_features = img_features_attributes['features']  # (N, embed_dim) e.g., (240k, 18)
        img_attributes = img_features_attributes['attribute_lists']  # (N, num category)

        text_features_attributes = torch.load(osp.join(load_clip_path, 'val_text_features_and_multihotGT.pt'), map_location="cpu")
        text_features = text_features_attributes['features']
        img_mask = text_features_attributes['multihotGT']  # (N, num latents in SAE) binary  FT
        text_mask = text_features_attributes['multihot_partial_caption']
        texts = text_features_attributes['partial_caption_list']
    return img_features, text_features, img_mask, text_mask, img_attributes, texts


def get_mad_features_for_full_textconditioning(load_clip_path):
    print("\n")
    print(f"Getting features from {load_clip_path} ========>")
    img_features_attributes = torch.load(osp.join(load_clip_path, 'val_features_and_attributes.pt'), map_location="cpu")
    img_features = img_features_attributes['features']  # (N, embed_dim) e.g., (240k, 18)
    img_attributes = img_features_attributes['attribute_lists']  # (N, num category)

    text_features_attributes = torch.load(osp.join(load_clip_path, 'val_text_features_and_multihotGT.pt'), map_location="cpu")
    img_mask = text_features_attributes['multihotGT']  # (N, num latents in SAE) binary  FT

    noisy_text_features_attributes = torch.load(osp.join(load_clip_path, 'val_text_features_to_condition_and_multihotGT.pt'), map_location="cpu")
    text_mask = noisy_text_features_attributes['multihot_partial_caption']
    texts = noisy_text_features_attributes['partial_caption_list']
    return img_features, None, img_mask, text_mask, img_attributes, texts


def get_mad_zeroshot_full(img_features, labels, zeroshot_weight, attribute_name, use_extra_ortho_dim=False):
    img_features_norm = img_features / \
        img_features.norm(dim=1, keepdim=True)
    if use_extra_ortho_dim:
        img_features_norm = img_features_norm[:, 0:-1]
    cossims = img_features_norm @ zeroshot_weight.T
    preds = cossims.argmax(dim=1)
    hits = (preds == labels).float()
    return hits


def get_mad_dataloader(load_clip_path, is_split, model, tokenizer, args, return_dataset=False, use_full_conditioning=False):
    device = args.device
    batch_size = args.batch_size
    if not use_full_conditioning:
        img_features, text_features, img_mask, text_mask, img_attributes, texts = get_mad_features(load_clip_path, is_split=is_split)
    else:
        img_features, text_features, img_mask, text_mask, img_attributes, texts = get_mad_features_for_full_textconditioning(load_clip_path)

    if is_split == 'val':
        val_positive_negative_templates_saved = torch.load(osp.join(load_clip_path, "val_positive_negative_template_for_zeroshot.pt"), map_location="cpu")
        val_positive_negative_templates = get_embedding_attributewise_from_texts(val_positive_negative_templates_saved, model, tokenizer, device, batch_size)
    else:
        val_positive_negative_templates = None


    if args.finetune_text_encoder:
        tokenized_texts = tokenizer(texts)
        dataset = LearntMaskTextConditioningEvalDataset(img_features, tokenized_texts, img_mask, text_mask, positive_negative_templates=val_positive_negative_templates, img_attributes=img_attributes, texts=texts)
    else:
        dataset = LearntMaskTextConditioningEvalDataset(img_features, text_features, img_mask, text_mask, positive_negative_templates=val_positive_negative_templates, img_attributes=img_attributes, texts=texts)

    shuffle = True if is_split == "train" else False
    loader = DataLoader(dataset, batch_size=args.train_mask_batch_size, shuffle = shuffle, collate_fn=LearntMaskTextConditioningEvalCollator())
    loader.num_samples = len(dataset)
    loader.num_batches = len(loader)
    if return_dataset:
        return loader, dataset
    return loader


def get_embedding_attributewise_from_texts(text_templates, model, tokenizer, device, batch_size):
    '''
    text_templates.keys() : dict_keys(['thickthinning', 'swelling', 'fracture', 'scaling', 'colors'])
    text_templates['swelling'].keys(): dict_keys(['positive_template', 'negative_template', 'positive_template_features', 'negative_template_features'])


    '''


    with torch.no_grad():
        dump_dict = {name: {}
                     for name in config.mad_decoder_name_list_cat_wise_literal[:-1]}
        for id, (key, _) in enumerate(dump_dict.items()):
            print(f"Attribute: {key} {id}/{len(dump_dict)} keys, Getting postive - negative tempaltes and features ")

            positive_template = text_templates[key]['positive_template']
            negative_template = text_templates[key]['negative_template']

            template_dataset = CustomTemplateDataset(
                positive_template, negative_template)  # same order?
            template_dataloader = torch.utils.data.DataLoader(
                template_dataset, batch_size=batch_size, shuffle=False)

            all_positive_features = []
            all_negative_features = []
            for i, (pos_template, neg_template) in enumerate(tqdm(template_dataloader)):

                with torch.no_grad():
                    positive_text_tokens = tokenizer(pos_template)
                    positive_text_tokens = positive_text_tokens.to(device)
                    positive_features = model.encode_text(positive_text_tokens)

                    negative_text_tokens = tokenizer(neg_template)
                    negative_text_tokens = negative_text_tokens.to(device)
                    negative_features = model.encode_text(negative_text_tokens)

                all_positive_features.append(positive_features.cpu())
                all_negative_features.append(negative_features.cpu())
            all_positive_features = torch.cat(all_positive_features, dim=0)
            all_negative_features = torch.cat(all_negative_features, dim=0)

            dump_dict[key]["positive_template_features"] = all_positive_features
            dump_dict[key]["negative_template_features"] = all_negative_features
            dump_dict[key]["positive_template"] = positive_template
            dump_dict[key]["negative_template"] = negative_template

    return dump_dict


def zero_shot_eval_mad_schrodi(args, attribute_name):

    save_path = os.path.join(
        args.load_clip_path, f"{attribute_name}_zero_shot_texts.pt")
    if os.path.exists(save_path):
        print(f"Loading zero shot texts: {attribute_name}")
        classification_texts = torch.load(save_path)
        return classification_texts

    print(
        f"Could not find saved zero shot texts: {attribute_name} at {save_path}")
    print(f'Getting zero shot texts: {attribute_name}')

    attribute_name_dict = {'thickthinning': ["no thickthinning", "thickening", "thinning"],
                           'swelling': ["no swelling", "swelling"],
                           'fracture': ["no fracture", "fracture"],
                           'scaling': ["large", "small"],
                           'colors': ["gray", "red", "green", "blue", "cyan", "magenta", "yellow"],
                           'digit': list(map(str, range(10)))
                           }
    classnames = attribute_name_dict[attribute_name]
    print(f"Classnames: {classnames}")
    remaining_attributes = [attr for attr in attribute_name_dict.keys() if attr not in [
        attribute_name, "digit"]]
    attr_count = int(args.mad_mode[-1])
    if attribute_name == "digit":
        selected_attributes = random.sample(remaining_attributes, attr_count)
    else:
        selected_attributes = ["digit"]
        if attr_count > 1:
            selected_attributes += random.sample(
                remaining_attributes, attr_count - 1)
    print(
        f"Attribute name: {attribute_name}, Selected attributes: {selected_attributes}")
    attr_list = []
    for attr in selected_attributes:
        attr_options = attribute_name_dict[attr]
        sample_attr = random.choice(attr_options)
        attr_list.append(sample_attr)
    class_insert_index = random.randint(0, len(attr_list) - 1)
    classification_texts = []
    for classname in classnames:
        this_class_texts = attr_list[:class_insert_index] + \
            [classname] + attr_list[class_insert_index:]
        classification_texts.append("-".join(this_class_texts))
    print(f"Classification texts: {classification_texts}")
    torch.save(classification_texts, save_path)
    print(f"Saved zero shot texts: {attribute_name} at {save_path}")
    return classification_texts


def get_mad_zeroshot_eval_embeddings(model, tokenizer, device, args):
    embeddings_dict = {}
    with torch.no_grad():
        for attribute_name in ["thickthinning", "swelling", "fracture", "scaling", "colors", "digit"]:
            classification_texts = zero_shot_eval_mad_schrodi(
                args, attribute_name)
            class_tokens = tokenizer(classification_texts)
            embeddings = model.encode_text(class_tokens.to(device))
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            embeddings_dict[attribute_name] = embeddings
    return embeddings_dict


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# https://github.com/ExplainableML/flair/blob/bf0cf64d7cee7295f05a3f6e00cf315a04a3c883/src/flair/data.py#L24C1-L26C82


# model_MGAP-ViT-256-18-lr_0.0005-b_128-j_8-p_amp-mode_object_attribute_3/Finetune/

