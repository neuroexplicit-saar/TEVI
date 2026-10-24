import os
import webdataset as wds
import clip
import torch


class PartialCaptionTransform:

    def __init__(self, transform, tokenizer, args, is_train, specific_partial_caption_fraction=None):
        self.args = args
        self.tokenizer = tokenizer
        self.transform = transform
        self.is_train = is_train
        split = "train" if is_train else "val"
        if specific_partial_caption_fraction is not None:
            self.partial_caption_fraction = specific_partial_caption_fraction
        else:
            self.partial_caption_fraction = args.partial_caption_fraction
        if self.partial_caption_fraction > 1:
            self.partial_index = torch.load(os.path.join(args.partial_index_path, f"full_data_partial_fraction_idx_{split}_{self.partial_caption_fraction}.pt"))

    def __call__(self, sample):
        idx = sample['__key__']
        if self.transform is not None:
            img = self.transform(sample["image"])
        else:
            img = sample["image"]


        if self.partial_caption_fraction > 1:
            text = sample["text"].split(" ")
            num_words = (len(text) // self.partial_caption_fraction)
            start_idx = self.partial_index[idx] * num_words
            end_idx = (start_idx + 1) * num_words
            mod_text = text[start_idx:end_idx]
            mod_text = " ".join(mod_text)
        else:
            mod_text = sample["text"]

        if self.tokenizer is not None:
            return_text = self.tokenizer(mod_text)[0]
        else:
            return_text = mod_text

        return {"image": img, "text": return_text, "__key__": idx}


#         # if self.tokenizer is not None:
#         #     raise NotImplementedError("Tokenizing here is probably a bug! Please check your code!")
#         #     return_text = self.tokenizer(selected_text)[0]
#         # else:
#         #     return_text = selected_text


class CustomDataCollatorPaired:
    def __init__(self) -> None:
        pass

    def __call__(self, batch):

        imgs = [i["image"] for i in batch]
        idxs = [i['__key__'] for i in batch]
        texts = [i['text'] for i in batch]

        imgs = torch.stack(imgs)

        return imgs, texts, idxs


class CC3MPaired:
    def __init__(self):
        pass

    def filter_no_caption_or_no_image(self, sample):
        has_caption = "txt" in sample
        has_image = "png" in sample or "jpg" in sample or "jpeg" in sample
        return has_caption and has_image

    def get_wds_dataset(self, input_shards, transform, batch_size, collator=None, shuffle=None):
        """
        return a dataset that returns an image, and text
        """
        pipeline = [
            wds.SimpleShardList(input_shards),
            # at this point we have an iterator over all the shards
        ]
        total_shards = int(input_shards.split('..')[1][:5]) - int(input_shards.split('..')[0][-5:])
        if shuffle == True:
            pipeline.extend([wds.shuffle(bufsize=total_shards, initial=total_shards),])

        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(),
            wds.select(self.filter_no_caption_or_no_image),
            wds.decode("pilrgb"),
            wds.rename(image="jpg;png;jpeg", text="txt"),
            wds.map_dict(image=transform, text=lambda text: text),
        ])

        if shuffle == True:
            pipeline.extend([wds.shuffle(100 * batch_size)])

        pipeline.extend(
            [wds.batched(batch_size, partial=False, collation_fn=collator)])
        dataset = wds.DataPipeline(*pipeline)
        return dataset

    # batch_size=None, shuffle=False, num_workers=1
    def get_dataloader(self, dataset, **kwargs):
        loader = torch.utils.data.DataLoader(dataset, **kwargs)
        return loader


#         """
#         return a dataset that returns an image, and text
#         """


#             [wds.batched(batch_size, partial=False, collation_fn=collator)])


if __name__ == '__main__':
    print("sweta\n")
    # Get dataloader  #RN50, 'ViT-B/16'
    clip_name = "RN50"
    device = 'cuda'
    batch_size = 128
    num_workers = 8
    data_root = None
    train_path = "training"
    clip_model, clip_preprocess = clip.load("RN50", device="cuda")

    collator = CustomDataCollatorPaired()
    cc3m_obj = CC3MPaired()

    val_shard = os.path.join(data_root, "CC3M_TAR", "validation", '{00000..00001}.tar')
    val_dtset = cc3m_obj.get_wds_dataset(val_shard, clip_preprocess, batch_size, collator=collator, shuffle=False)
    val_loader = cc3m_obj.get_dataloader(val_dtset, batch_size=None, shuffle=False)


    train_shard = os.path.join(data_root, "CC3M_TAR", "training", '{00000..00299}.tar')
    train_dtset = cc3m_obj.get_wds_dataset(train_shard, clip_preprocess, batch_size, collator=collator, shuffle=True)
    train_loader = cc3m_obj.get_dataloader(train_dtset, batch_size=None, shuffle=False, prefetch_factor=2, num_workers=2)

    from tqdm import tqdm

    total = 0
    for batch_idx, (train_img, train_caption, _) in tqdm(enumerate(train_loader), total=(2100000 // batch_size)):
        total += train_img.shape[0]

