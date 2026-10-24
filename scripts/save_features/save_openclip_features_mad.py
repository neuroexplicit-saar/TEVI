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
from tqdm import tqdm


from open_clip import create_model_and_transforms, get_tokenizer
from open_clip_train.data import get_data
from open_clip_train.distributed import is_master, init_distributed_device, broadcast_object
from open_clip_train.logger import setup_logging
from open_clip_train.params import parse_args
from open_clip_train.file_utils import pt_load


from mgap import config 
decoder_name_to_index = config.decoder_name_to_index


LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def generate_text(args,  cls_label: int, words, return_idxs = False):
    string_content = [str(cls_label)]
    string_content += random.sample(words, int(args.mad_mode[-1]))
    random.shuffle(string_content)
    indices = [decoder_name_to_index[entry] for entry in string_content]
    caption = "-".join(string_content)
    
    if return_idxs:
        return caption, indices 
    else: 
        return caption


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

class CustomTemplateDataset(torch.utils.data.Dataset): 

    def __init__(self, positive_templates, negative_templates): 
        super().__init__()
        self.postive_templates = positive_templates
        self.negative_templates = negative_templates
    
    def __len__(self):
        return len(self.postive_templates)
    
    def __getitem__(self, idx):
        return self.postive_templates[idx], self.negative_templates[idx]


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


def main(args, parser):
    from mgap import config  
    mad_decoder_name_list_cat_wise_literal = config.mad_decoder_name_list_cat_wise_literal

    index_to_decoder_name = config.index_to_decoder_name
    decoder_name_cum_len_list_of_cat = config.decoder_name_cum_len_list_of_cat

 
    args = parse_args(args, parser)

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

    splits = ["train", "val"]
    if args.save_img_features: 
        for split in splits:
            print(f"Split used is {split}!")
            loader = data[split].dataloader
            loader.dataset.follow_json = True
            loader.dataset.return_attribute_list = True
            features = []
            attribute_lists = []
            for i, (img, attribute_list) in enumerate(tqdm(loader)):
                img = img.to(device)
                with torch.no_grad():
                        output = model.encode_image(img)
                features.append(output.cpu())
                attribute_lists.append(attribute_list)
            features = torch.cat(features, dim=0)
            attribute_lists = torch.cat(attribute_lists, dim=0)
            print(features.shape, attribute_lists.shape)
            save_dict = {
                "features": features,
                "attribute_lists": attribute_lists
            }
            torch.save(save_dict, os.path.join(log_base_path, f"{split}_features_and_attributes.pt"))


    elif args.save_text_features:
        for split in splits:
            print(f"Split used is {split}!")
            features = []
            multihotGT = []
            multihot_partial_caption= []

            decoder_name_cum_len_list_of_cat = torch.tensor(decoder_name_cum_len_list_of_cat)
            
            all_features_attribute_ids_list  = torch.load(os.path.join(log_base_path, f"{split}_features_and_attributes.pt"))
            all_attribute_ids_list_dataset = torch.utils.data.TensorDataset(all_features_attribute_ids_list["attribute_lists"])
            all_attribute_ids_list_loader = torch.utils.data.DataLoader(all_attribute_ids_list_dataset, batch_size= args.batch_size, shuffle=False)

            for i, (attribute_ids_list_GT) in enumerate(tqdm(all_attribute_ids_list_loader)):
               
                attribute_ids_list_GT = attribute_ids_list_GT[0] # comes in a tuple, need to extract
                single_multihot = torch.zeros(attribute_ids_list_GT.shape[0], 26)
                single_multihot_partial_caption= torch.zeros(attribute_ids_list_GT.shape[0], 26)


                cum_attribute_ids_list_GT = attribute_ids_list_GT + decoder_name_cum_len_list_of_cat
                row_indices = torch.arange(cum_attribute_ids_list_GT.size(0))
                row_indices = row_indices.unsqueeze(1).expand(-1, cum_attribute_ids_list_GT.size(1))
                single_multihot[row_indices, cum_attribute_ids_list_GT] = 1


                # Getting the category name from the cumulative attribute id 
                caption_words_list_GT= [[index_to_decoder_name[col.item()] for col in cum_attribute_ids_list_GT[row, 0:-1]] for row in range(cum_attribute_ids_list_GT.shape[0])]
                
                partial_caption_list = []
                cum_attribute_ids_list_partial_caption = []
                for (class_label, words) in zip(attribute_ids_list_GT[:, -1], caption_words_list_GT):
                    single_partial_caption, single_indices_partial = generate_text(args, class_label.item(), words, return_idxs= True)
                    partial_caption_list.append(single_partial_caption)
                    cum_attribute_ids_list_partial_caption.append(single_indices_partial)


                # Split the caption and see which attributes are present in the caption and the corresponding multi hot

            
                cum_attribute_ids_list_partial_caption = torch.tensor(cum_attribute_ids_list_partial_caption)
                row_indices = torch.arange(cum_attribute_ids_list_partial_caption.size(0))
                row_indices = row_indices.unsqueeze(1).expand(-1, cum_attribute_ids_list_partial_caption.size(1))
                single_multihot_partial_caption[row_indices, cum_attribute_ids_list_partial_caption] = 1 

                with torch.no_grad():
                    text_tokens = tokenizer(partial_caption_list)
                    text_tokens = text_tokens.to(device)
                    output = model.encode_text(text_tokens)
                features.append(output.cpu())
                multihotGT.append(single_multihot)
                multihot_partial_caption.append(single_multihot_partial_caption)

            features = torch.cat(features, dim=0)
            multihotGT = torch.cat(multihotGT, dim=0)
            multihot_partial_caption = torch.cat(multihot_partial_caption, dim=0)
            print(features.shape, multihotGT.shape)
            save_dict = {
                "features": features,
                "multihotGT": multihotGT.long(),
                "multihot_partial_caption": multihot_partial_caption.long()
            }
            
            torch.save(save_dict, os.path.join(log_base_path, f"{split}_text_features_and_multihotGT.pt"))


    elif args.save_template_for_zeroshot:
        import copy
        print("\n \n ")
        print("Saving template for Zeroshot ==================================>")


        for split in splits:
            print(f"Split used is {split}!")
            multihot_GT_dict  = torch.load(os.path.join(log_base_path, f"{split}_text_features_and_multihotGT.pt"))
            multihot_GT= multihot_GT_dict["multihotGT"]
            dump_dict = {name: {"positive_template":[], "negative_template":[]} for name in mad_decoder_name_list_cat_wise_literal[1:]}

            for id, multihot_GT_single  in enumerate(multihot_GT):
                if id%50000 ==0:
                    print(f"Dumping for image {id}/{len(multihot_GT)}") 

                idxs = torch.where(multihot_GT_single==1)[0] # in sequential order?
                attr_list = [index_to_decoder_name[idx.item()] for idx in idxs]
                
                for attr_cat_name, attr in zip(mad_decoder_name_list_cat_wise_literal[:-1], attr_list[:-1]): 
                    '''
                    Do not change the order of the code
                    '''
                    chosen_attr_list_for_pos_temp =[]
                    chosen_attr_list_for_neg_temp =[]

                    attr_list_copy = copy.deepcopy(attr_list)                    
                    
                    # Remove the digit and the attribute so that the common two elements exculde this
                    attr_list_copy.remove(attr)
                    attr_list_copy.remove(attr_list[-1])

                    # sample 2 elements from teh rest 
                    extra_element_for_both_temp =random.sample(attr_list_copy, 2) #IS seed fixed, always returns same
                    
                    chosen_attr_list_for_pos_temp.extend(extra_element_for_both_temp)
                    chosen_attr_list_for_pos_temp.append(attr)
                    chosen_attr_list_for_pos_temp.append(attr_list[-1])

                    attr_list_copy.remove(extra_element_for_both_temp[0])
                    attr_list_copy.remove(extra_element_for_both_temp[1])


                    extra_element_for_neg_temp =random.sample(attr_list_copy, 1)[0]
                    
                    
                    chosen_attr_list_for_neg_temp.extend(extra_element_for_both_temp)
                    chosen_attr_list_for_neg_temp.append(attr_list[-1])


                    idx = chosen_attr_list_for_pos_temp.index(attr)
                    chosen_attr_list_for_neg_temp.insert(idx, extra_element_for_neg_temp)

                    # shuffle the list
                    chosen_attr_list_for_pos_temp_shuffle = random.sample(chosen_attr_list_for_pos_temp, len(chosen_attr_list_for_pos_temp))
                    chosen_attr_list_for_neg_temp_shuffle = random.sample(chosen_attr_list_for_neg_temp, len(chosen_attr_list_for_neg_temp))
                    
                    
                    positive_template = "-".join(chosen_attr_list_for_pos_temp_shuffle)
                    negative_template = "-".join(chosen_attr_list_for_neg_temp_shuffle)

                    dump_dict[attr_cat_name]["positive_template"].append(positive_template)
                    dump_dict[attr_cat_name]["negative_template"].append(negative_template)
            

            for id, (key, _) in enumerate(dump_dict.items()):
                print(f"Attribute: {key} {id}/{len(dump_dict)} keys ")

                
                positive_template = dump_dict[key]['positive_template']
                negative_template = dump_dict[key]['negative_template']
                
                template_dataset = CustomTemplateDataset(positive_template, negative_template) # same order?
                template_dataloader = torch.utils.data.DataLoader(template_dataset, batch_size= args.batch_size, shuffle=False)


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


                dump_dict[key]["positive_template_features"] =all_positive_features
                dump_dict[key]["negative_template_features"] =all_negative_features
            torch.save(dump_dict, os.path.join(log_base_path, f"{split}_positive_negative_template_for_zeroshot.pt"))


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-text-features", action="store_true")
    parser.add_argument("--save-img-features", action="store_true")
    parser.add_argument("--save-template-for-zeroshot", action="store_true")
    main(sys.argv[1:], parser)
