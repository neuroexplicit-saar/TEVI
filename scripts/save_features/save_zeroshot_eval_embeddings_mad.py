import glob
import logging
import os
import re
import subprocess
import sys
import random
from datetime import datetime

import numpy as np
import torch


from open_clip import create_model_and_transforms, get_tokenizer
from open_clip_train.data import get_data
from open_clip_train.distributed import is_master, init_distributed_device, broadcast_object
from open_clip_train.logger import setup_logging
from open_clip_train.params import parse_args
from open_clip_train.file_utils import pt_load


LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote : bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def main(args):
    args = parse_args(args)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join([
            f"model_{model_name_safe}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
            f"mode_{args.mad_mode}",
        ])

    resume_latest = args.resume == 'latest'
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'save_features-{args.rank}' if args.log_local else 'save_features.log'
        args.log_path = os.path.join(log_base_path, log_filename)

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)


    checkpoint_path = os.path.join(log_base_path, "checkpoints")
    resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
    if resume_from:
        logging.info(f'Found latest resume checkpoint at {resume_from}.')
    else:
        logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
    args.resume = resume_from


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


    # optionally resume from a checkpoint
    start_epoch = 0
    checkpoint = pt_load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint["state_dict"])
    logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    # initialize datasets
    tokenizer = get_tokenizer(args.model, cache_dir=args.cache_dir)
    data = get_data(
        args,
        (preprocess_train, preprocess_val),
        epoch=start_epoch,
        tokenizer=tokenizer,
    )
    assert len(data), 'At least one train or eval dataset must be specified.'


    # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    original_model = model


    model.eval()

    with torch.no_grad():
        for attribute_name in ["thickthinning", "swelling", "fracture", "scaling", "colors", "digit"]:
            classification_texts = zero_shot_eval_mad_schrodi(args, attribute_name)
            class_tokens = tokenizer(classification_texts)
            embeddings = model.encode_text(class_tokens.to(device))
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            embeddings = embeddings.cpu()
            torch.save(embeddings, os.path.join(log_base_path, f"{attribute_name}_embeddings.pt"))
    

def zero_shot_eval_mad_schrodi(args, attribute_name):

    logging.info(f'Getting zero shot texts: {attribute_name}')

    attribute_name_dict = {'thickthinning': ["no thickthinning", "thickening", "thinning"],
                            'swelling': ["no swelling", "swelling"],
                            'fracture': ["no fracture", "fracture"],
                            'scaling': ["large", "small"],
                            'colors': ["gray", "red", "green", "blue", "cyan", "magenta", "yellow"],
                            'digit': list(map(str, range(10)))
                            }
    classnames = attribute_name_dict[attribute_name]
    logging.info(f"Classnames: {classnames}")
    remaining_attributes = [attr for attr in attribute_name_dict.keys() if attr not in [attribute_name, "digit"]]
    attr_count = int(args.mad_mode[-1])
    if attribute_name == "digit":
        selected_attributes = random.sample(remaining_attributes, attr_count)
    else:
        selected_attributes = ["digit"]
        if attr_count > 1:
            selected_attributes += random.sample(remaining_attributes, attr_count-1)
    logging.info(f"Attribute name: {attribute_name}, Selected attributes: {selected_attributes}")
    attr_list = []
    for attr in selected_attributes:
        attr_options = attribute_name_dict[attr]
        sample_attr = random.choice(attr_options)
        attr_list.append(sample_attr)
    class_insert_index = random.randint(0, len(attr_list)-1)
    classification_texts = []
    for classname in classnames:
        this_class_texts = attr_list[:class_insert_index] + [classname] + attr_list[class_insert_index:]
        classification_texts.append("-".join(this_class_texts))
    logging.info(f"Classification texts: {classification_texts}")
    return classification_texts


if __name__ == "__main__":
    main(sys.argv[1:])