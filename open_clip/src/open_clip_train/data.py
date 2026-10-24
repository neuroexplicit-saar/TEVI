import ast
import json
import logging
import math
import os
import random
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value

import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

from mgap.cc3m import PartialCaptionTransform
from mgap import config, utils
from pycocotools.coco import COCO

import pandas as pd

import os.path as osp

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import COCO_CLASSNAMES


class COCOControlledDataset(Dataset):

    def __init__(self, root, img_root, split, fname, caption_type_list, transform=None):
        self.root = os.path.join(root, split)
        self.img_path = os.path.join(img_root, split)
        self.annotation_path = os.path.join(img_root, "annotations", f"instances_{split}.json")
        self.split = split
        self.caption_type_list = caption_type_list
        self.transform = transform

        self.coco = COCO(self.annotation_path)

        self.attribute_caption_data = torch.load(os.path.join(
            self.root, f"{fname}.pt"), map_location='cpu')
        self.ids = self.attribute_caption_data["img_ids"]
        assert len(self.ids) == len(self.attribute_caption_data["captions"][caption_type_list[0]])
        assert len(self.ids) == len(self.attribute_caption_data["counts"]["counts"][caption_type_list[0]])
        assert len(self.ids) == len(self.attribute_caption_data["multihot_attributes"][caption_type_list[0]])
        assert len(self.ids) == len(self.attribute_caption_data["img_multihot_attributes"])

    def __len__(self):
        return len(self.ids) * len(self.caption_type_list)

    def __getitem__(self, idx):
        select_img_idx = idx // len(self.caption_type_list)
        select_caption_idx = idx % len(self.caption_type_list)
        coco = self.coco
        img_id = self.ids[select_img_idx]
        # Assert not needed as it is not by construction true
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.img_path, path)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        caption_type = self.caption_type_list[select_caption_idx]

        try:
            return_obj = {"img": img, "caption": self.attribute_caption_data["captions"][caption_type][select_img_idx], "counts": self.attribute_caption_data["counts"]["counts"][caption_type][select_img_idx], "multihot_attributes": self.attribute_caption_data["multihot_attributes"][caption_type][select_img_idx], "img_multihot_attributes": self.attribute_caption_data["img_multihot_attributes"][select_img_idx], "img_ids": self.attribute_caption_data["img_ids"][select_img_idx]}
        except Exception as e:
            logging.error(f"Error in loading caption data for img_id {img_id}: {e}")
        return return_obj


class COCOControlledCollator:

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        images = torch.stack([b["img"] for b in batch], dim=0)
        if self.tokenizer is None:
            texts = [b["caption"] for b in batch]
        else:
            texts = self.tokenizer([b["caption"] for b in batch])
        counts = torch.stack([b["counts"] for b in batch], dim=0)
        multihot_attributes = torch.stack(
            [b["multihot_attributes"] for b in batch], dim=0)
        img_multihot_attributes = torch.stack(
            [b["img_multihot_attributes"] for b in batch], dim=0)
        img_ids = torch.stack(
            [torch.tensor(b["img_ids"]) for b in batch], dim=0)
        return images, texts, counts, multihot_attributes, img_multihot_attributes, img_ids


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist), \
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)])
                         for shard in shards_list])
    elif os.path.exists(len_filename):
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = (
        'png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def filter_no_caption_or_no_image_json(sample):
    has_caption = ('json' in sample)
    has_image = (
        'png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(
                __key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights), \
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, args,  img_features: torch.Tensor, texts: list, split= "train", ft_img_features =None):
        super().__init__()
        if ft_img_features is not None:
            self.ft_img_features = ft_img_features
        self.img_features = img_features
        self.texts = texts
        self.args = args
        self.subcaption_idxs = None
        assert args.which_type_retrieval in ['full', 'global', 'mix', 'random_single']
        
        if split =='val' or split == 'train_val':
            self.subcaption_idxs = args.which_type_retrieval # subcaption_idxs should not be None when split si val 
                # 
                

        # 
        
        print(f"Info on data: --recap-json-caption-key {args.recap_json_caption_key}; --which-type-retrieval {args.which_type_retrieval}")
        assert self.img_features.size(0) == len(self.texts)


    def __len__(self) -> int:
        return len(self.img_features)

    def __getitem__(self, idx: int) -> dict:
        selected_text = self.texts[idx]            

        return self.img_features[idx], selected_text, frac


class CustomDatasetCollator:
    def __init__(self, args, tokenizer=None) -> None:
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, batch):
        img_features, texts_list, frac_sentence_used = zip(*batch)
        tokenized_texts = self.tokenizer(texts_list)
        batched_img_features = torch.stack(img_features, dim=0)
        batched_frac_sentence_used = torch.tensor(frac_sentence_used, dtype=torch.float32)

        return batched_img_features, tokenized_texts, texts_list, batched_frac_sentence_used


def get_data_fnames(args, dirs_only=False, dataset_name = None):
    
    if dataset_name is None: 
        dataset_name = args.dataset_name
    

    if dataset_name== 'cc12m':
        midfix_img="cc12m"
        midfix_text="CC12M"

    elif dataset_name== 'cc3m':
        midfix_img="cc3m"
        midfix_text="CC3M"

    assert  args.which_type_retrieval == 'global'


    save_dir_activations = None
    return img_train_fnames, text_train_fnames, img_fname_dir, text_fname_dir


def get_cc12m_dataset(args, is_train, epoch, tokenizer, order_id=None, val_split=None, dataset_name = None):
    # get_cc12m_dataset(order_id, img_fname_dir, text_fname_dir, img_train_fnames, text_train_fnames, tokenizer, split, batch_size, epoch, args):
    img_train_fnames, text_train_fnames, img_fname_dir, text_fname_dir = get_data_fnames(
        args, dirs_only=not is_train, dataset_name = dataset_name)
    if is_train:
        train_img_features = torch.load(img_train_fnames[order_id], map_location='cpu')  # For MSE scale
        train_texts = torch.load(text_train_fnames[order_id], map_location='cpu')
        
        # 

        dataset = CustomDataset(args, train_img_features, train_texts)


    else:
        val_img_features = torch.load(osp.join(img_fname_dir, f"val_paired"), map_location='cpu')
        val_texts = torch.load(osp.join(text_fname_dir, f"val_paired"), map_location='cpu')
        total_val_points = len(val_img_features)
        train_val_num = int(total_val_points * 0.3)


        #     # if val_split == "train-val":
        #     #     dataset = CustomDataset(args, val_orig_img_features[:train_val_num], val_texts[:train_val_num], split="train_val", ft_img_features = val_img_features[:train_val_num])
        #     # elif val_split =="val":
        #     # if val_split == "train-val":
        #     #     dataset = CustomDataset(args, val_img_features[:train_val_num], val_texts[:train_val_num], split="train_val")
        #     # elif val_split =="val":
        dataset = CustomDataset(args, val_img_features, val_texts, split="val")


    num_samples = len(dataset)
    sampler = DistributedSampler(
        dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    collator = CustomDatasetCollator(args=args, tokenizer=tokenizer)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_mask_batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=collator,)

    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    # create a shared epoch store to sync epoch to dataloader worker proc
    shared_epoch = SharedEpoch(epoch=epoch)

    return DataInfo(dataloader=dataloader, sampler=sampler, shared_epoch=shared_epoch)


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None, split_name=None, specific_partial_caption_fraction=None, return_indices=False):

    if split_name is not None:
        partial_caption_transform = PartialCaptionTransform(preprocess_img, tokenizer, args, is_train=(
            split_name == 'train'), specific_partial_caption_fraction=specific_partial_caption_fraction)
    else:
        partial_caption_transform = PartialCaptionTransform(
            preprocess_img, tokenizer, args, is_train=is_train, specific_partial_caption_fraction=specific_partial_caption_fraction)

    input_shards = args.train_data if is_train else args.val_data
    # 
    
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_shards = None
    # 
    
    if is_train:
        # 
        
        num_samples, num_shards = get_dataset_size(input_shards)
        if not num_samples:
            raise RuntimeError(
                'Currently, the number of dataset samples must be specified for the training dataset. '
                'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0

    # 
    
    # create a shared epoch store to sync epoch to dataloader worker proc
    shared_epoch = SharedEpoch(epoch=epoch)


    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    
    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image) if not args.use_recap_dataset else wds.select(filter_no_caption_or_no_image_json),
        wds.decode("pilrgb", handler=log_and_continue), #, handler=log_and_continue
        wds.rename(image="jpg;png;jpeg;webp", text="txt") if not args.use_recap_dataset else wds.rename(image="jpg;png;jpeg;webp", json="json"),
        wds.map(partial_caption_transform),

    ])


    pipeline.extend([
            wds.to_tuple("image", "tokenized_text", "text","__key__"),
        ])
    pipeline.extend([
        wds.batched(args.batch_size, partial=not is_train)
    ])

    dataset = wds.DataPipeline(*pipeline)


    print(f"==============================> {resampled}")
    if is_train:
        if not resampled:
            # 
            

            num_shards = num_shards or len(expand_urls(input_shards)[0])

            # 
            

            assert num_shards >= args.workers * \
                args.world_size, 'number of shards must be >= total workers'
            # roll over and repeat a few samples to get same number of full batches on each node
       
            round_fn = math.floor if floor else math.ceil
            global_batch_size = args.batch_size * args.world_size
            num_batches = round_fn(num_samples / global_batch_size)
            num_workers = max(1, args.workers)
            num_worker_batches = round_fn(
                num_batches / num_workers)  # per dataloader worker
            num_batches = num_worker_batches * num_workers
            num_samples = num_batches * global_batch_size
            # each worker is iterating over this
            dataset = dataset.with_epoch(num_worker_batches)

    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    collator = identity

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        collate_fn=collator,
    )


    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    dataloader.collator = collator

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def identity(x):
    return x


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(
        dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class SyntheticDataset(Dataset):

    def __init__(
            self,
            transform=None,
            image_size=(224, 224),
            caption="Dummy caption",
            dataset_size=100,
            tokenizer=None,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(
        dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_mad_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None, return_cls_label_with_img=False, debug_visu=False, mode=None, return_indices=False, generate_with_noise=False):
    from mgap import mad_dataset
    # for MAD the args.train_data and args.val_data path is same, is_train decides teh split.
    if mode is None:
        mode = args.mad_mode
    dataset = mad_dataset.MADDataset(
        root=args.val_data, transform=preprocess_fn, train=is_train, mode=mode, tokenizer=tokenizer, return_cls_label_with_img=return_cls_label_with_img, debug_visu=debug_visu, generate_with_noise=generate_with_noise)
    if args.use_mad_train_full_dataset: 
        dataset.follow_json = True
    num_samples = len(dataset)
    sampler = DistributedSampler(
        dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type, dataset_name=None, train_learntmask_conditioning=False):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "mad":
        return get_mad_dataset
    elif dataset_type == "cocomad":
        return get_cocomad_dataset
    elif dataset_type == "coco":
        logging.info("Using COCO dataset")
        return get_mscoco_dataset
    elif dataset_type == "coco_controlled":
        logging.info("Using COCO controlled dataset")
        return get_mscoco_controlled_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


class PromptTokenizeCaption:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, texts):
        if self.tokenizer is None:
            return texts[:5]
        return self.tokenizer(texts[:5])


class COCOCollator:

    def __init__(self):
        pass

    def __call__(self, batch):
        images = torch.stack([b[0] for b in batch], dim=0)
        if isinstance(batch[1][0], torch.Tensor):
            texts = torch.stack([b[1] for b in batch], dim=0)
        else:
            texts = [b[1] for b in batch]
        return images, texts


def get_mscoco_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None, split_name=None, specific_partial_caption_fraction=None, return_indices=False):
    if is_train:
        dataset = get_mscoco(
            config.MSCOCO_TRAIN, config.MSCOCO_TRAIN_ANNOT, preprocess_img, tokenizer)
    else:
        dataset = get_mscoco(
            config.MSCOCO_VAL, config.MSCOCO_VAL_ANNOT, preprocess_img, tokenizer)
    shared_epoch = SharedEpoch(epoch=epoch)
    num_samples = len(dataset)
    sampler = DistributedSampler(
        dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=is_train,
        sampler=sampler,
        shuffle=shuffle,
        collate_fn=COCOCollator(),
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader=dataloader, sampler=sampler, shared_epoch=shared_epoch)


def get_mscoco_controlled_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None, return_indices=False):
    caption_type_list = config.CAPTION_LIST

    dataset = COCOControlledDataset(
        root=config.MSCOCO_ATTR_CAPTION_ROOT,
        img_root=config.MSCOCO_IMG_ROOT,
        split="train2017" if is_train else "val2017",
        fname="count_attr_caption_dict" if is_train else os.path.basename(config.VAL_COUNT_ATTR_CAPTION_DICT)[:-3],
        caption_type_list=caption_type_list,
        transform=preprocess_fn,
    )


    collator = COCOControlledCollator(tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(
        dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=is_train,
        sampler=sampler,
        shuffle=shuffle,
        collate_fn=collator,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    shared_epoch = SharedEpoch(epoch=epoch)
    return DataInfo(dataloader=dataloader, sampler=sampler, shared_epoch=shared_epoch)


def get_mscoco(args, preprocess_fn, tokenizer):

    dataset = datasets.CocoCaptions(
        root=config.ms_coco,
        annFile=config.ms_coco_annot,
        transform=preprocess_fn,
        target_transform=PromptTokenizeCaption(tokenizer)
    )

    return dataset


def get_data_original(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn_original(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    if args.val_data:
        data["val"] = get_dataset_fn_original(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer)


    return data


def get_dataset_fn_original(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset_original
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset_original
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

def get_wds_dataset_original(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_shards = None
    if is_train:
        num_samples, num_shards = get_dataset_size(input_shards)
        if not num_samples:
            raise RuntimeError(
                'Currently, the number of dataset samples must be specified for the training dataset. '
                'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0 

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    
    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train)
    ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )


    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_data(args, preprocess_fns, epoch=0, tokenizer=None, debug_visu=False, return_indices=False, order_id=None, img_fname_dir=None, text_fname_dir=None, img_train_fnames=None, text_train_fnames=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}
    tokenizer_not_none = None

    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_mad_dataset(args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer_not_none, generate_with_noise=args.mad_generate_with_noise)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer_not_none, return_indices=return_indices)
        data["val-full"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer_not_none, mode="object_attribute_5")


    if args.mad_val is not None:
        data["mad-val"] = get_mad_dataset(args, preprocess_val, is_train=False,
                                          tokenizer=tokenizer_not_none, return_cls_label_with_img=True)

    
    # Treat train data like val data, no shuffle and random crops

    
    return data


class COCOMADCollator: 
    def __init__(self): 
        pass
    def __call__(self, batch): 
        images = torch.stack([b[0] for b in batch], dim=0)
        texts = [b[1] for b in batch]
        img_attributes = torch.stack([b[2] for b in batch], dim=0)
        tokenized_texts = torch.stack([b[3] for b in batch], dim=0)
        return images, texts, img_attributes, tokenized_texts


#         # if self.is_train: 
#         #     random_count = random.randint(1, len(attr_list))
#         #     selected_attr = random.sample(attr_list, random_count)
#         #     random.shuffle(selected_attr)
#         # else:
#             # selected_attr = attr_list
            
#         # if self.generate_with_noise:
#         #     noise_count = random.randint(0, random_count)
#         #     incorrect_attrs = list(set(COCO_CLASSNAMES) - set(attr_list))
#         #     noisy_selected_attr = random.sample(incorrect_attrs, noise_count)
#         #     all_attr = selected_attr + noisy_selected_attr
#         #     random.shuffle(all_attr)
#         # else:
#             # all_attr = selected_attr
#         # random.shuffle(all_attr)


class COCOMADStyleDataset(torch.utils.data.Dataset):
    def __init__(self, root, instance_root, transform=None, generate_with_noise=False, return_attribute_list=False, use_saved_texts_with_noise=False, use_saved_texts_without_noise=False, args=None, is_train=None, tokenizer=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.instance_root = instance_root
        self.coco = COCO(instance_root)
        self.ids = list(sorted(self.coco.imgs.keys()))
        selected_ids = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            coco_annotation = self.coco.loadAnns(ann_ids)
            attr_list = list(set([self.coco.cats[annotation["category_id"]]["name"] for annotation in coco_annotation]))
            if len(attr_list) > 0:
                selected_ids.append(img_id)
        self.ids = selected_ids
        self.args = args
        self.tokenizer = tokenizer


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = self.coco.loadImgs(img_id)
        img_without_transform = Image.open(os.path.join(self.root, img[0]['file_name'])).convert('RGB')

        if self.transform is not None:
            img = self.transform(img_without_transform)
        

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        coco_annotation = self.coco.loadAnns(ann_ids)
        attr_list = list(set([self.coco.cats[annotation["category_id"]]["name"] for annotation in coco_annotation]))
       
        img_attr_list = torch.zeros(len(COCO_CLASSNAMES), dtype=torch.long)
        for attr in attr_list:
            img_attr_list[COCO_CLASSNAMES.index(attr)] = 1
        tokenized_attr_list = self.tokenizer("-".join(attr_list))
        
        return img, "-".join(attr_list), img_attr_list, tokenized_attr_list


def get_cocomad_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None, return_indices=False):
    dataset = COCOMADStyleDataset(
        root=os.path.join(config.MSCOCO_IMG_ROOT, "train2017" if is_train else "val2017"),
        instance_root=config.MSCOCO_TRAIN_INSTANCES if is_train else config.MSCOCO_VAL_INSTANCES,
        transform=preprocess_fn,
        args=args, is_train=is_train,
        tokenizer=tokenizer,
    )
    shared_epoch = SharedEpoch(epoch=epoch)
    num_samples = len(dataset)
    sampler = DistributedSampler(
        dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    if is_train:
        collator = None
    else: 
        collator = COCOMADCollator()
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn = collator, 
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader=dataloader, sampler=sampler, shared_epoch=shared_epoch)

