import torch


class LearntMaskTextConditioningEvalDataset(torch.utils.data.Dataset):

    def __init__(self, img_features: torch.Tensor, text_features: torch.Tensor, img_mask: torch.Tensor, text_mask: torch.Tensor, positive_negative_templates: dict, img_attributes: torch.Tensor, texts: list):
        super().__init__()
        self.img_features = img_features
        self.text_features = text_features
        self.img_mask = img_mask
        self.text_mask = text_mask
        self.img_attributes = img_attributes
        self.positive_negative_templates = positive_negative_templates
        self.texts = texts

        if self.text_features is not None:
            assert self.img_features.size(0) == self.text_features.size(0) and self.img_features.size(0) == self.img_mask.size(0) and self.img_features.size(0) == self.text_mask.size(0)
        else:
            assert self.img_features.size(0) == self.img_mask.size(0) and self.img_features.size(0) == self.text_mask.size(0)
        assert self.img_mask.size(1) == self.text_mask.size(1)
        if positive_negative_templates is not None:
            assert self.img_features.size(1) == self.positive_negative_templates['swelling']['positive_template_features'].size(1)
            assert self.positive_negative_templates['swelling']['positive_template_features'].size(0) == self.positive_negative_templates['swelling']['negative_template_features'].size(0)
            assert self.positive_negative_templates['swelling']['positive_template_features'].size(1) == self.positive_negative_templates['swelling']['negative_template_features'].size(1)
        assert self.img_attributes.size(0) == self.img_features.size(0)
        """
        attribute_name
         - positive_template_features: (n_classes, embedding_dim)
         - negative_template_features: (n_classes, embedding_dim)
        """

    def __len__(self) -> int:
        return len(self.img_features)

    def __getitem__(self, idx: int) -> dict:
        return_dict = {}
        return_dict['img_features'] = self.img_features[idx]
        if self.text_features is not None:
            return_dict['text_features'] = self.text_features[idx]
        else:
            return_dict['text_features'] = None
        return_dict['img_mask'] = self.img_mask[idx]
        return_dict['img_mask_attribute_idxs'] = torch.nonzero(self.img_mask[idx]).squeeze(1)
        return_dict['text_mask'] = self.text_mask[idx]
        return_dict['img_attributes'] = self.img_attributes[idx]
        return_dict['texts'] = self.texts[idx]

        if self.positive_negative_templates is not None:
            return_dict['templates'] = {}
            for key in self.positive_negative_templates:
                return_dict['templates'][key] = {}
                return_dict['templates'][key]['positive_template_features'] = self.positive_negative_templates[key]['positive_template_features'][idx]
                return_dict['templates'][key]['negative_template_features'] = self.positive_negative_templates[key]['negative_template_features'][idx]
                return_dict['templates'][key]['positive_template_texts'] = self.positive_negative_templates[key]['positive_template'][idx]
                return_dict['templates'][key]['negative_template_texts'] = self.positive_negative_templates[key]['negative_template'][idx]
        return return_dict


class LearntMaskTextConditioningEvalCollator:
    def __init__(self):
        super().__init__()

    def __call__(self, batch: list) -> dict:
        return_dict = {}
        return_dict['img_features'] = torch.stack([item['img_features'] for item in batch])
        if batch[0]['text_features'] is not None:
            return_dict['text_features'] = torch.stack([item['text_features'] for item in batch])
        else:
            return_dict['text_features'] = None
        return_dict['img_mask'] = torch.stack([item['img_mask'] for item in batch])
        return_dict['img_mask_attribute_idxs'] = torch.stack([item['img_mask_attribute_idxs'] for item in batch])
        return_dict['text_mask'] = torch.stack([item['text_mask'] for item in batch])
        return_dict['img_attributes'] = torch.stack([item['img_attributes'] for item in batch])
        return_dict['texts'] = [item['texts'] for item in batch]
        # if return_dict has key 'templates'
        if 'templates' in batch[0]:
            return_dict['templates'] = {}

            for key in batch[0]['templates']:
                return_dict['templates'][key] = {}
                return_dict['templates'][key]['positive_template_features'] = torch.stack([item['templates'][key]['positive_template_features'] for item in batch])
                return_dict['templates'][key]['negative_template_features'] = torch.stack([item['templates'][key]['negative_template_features'] for item in batch])
                return_dict['templates'][key]['positive_template_texts'] = [item['templates'][key]['positive_template_texts'] for item in batch]
                return_dict['templates'][key]['negative_template_texts'] = [item['templates'][key]['negative_template_texts'] for item in batch]
        return return_dict
    

class CustomTemplateDataset(torch.utils.data.Dataset):

    def __init__(self, positive_templates, negative_templates):
        super().__init__()
        self.postive_templates = positive_templates
        self.negative_templates = negative_templates

    def __len__(self):
        return len(self.postive_templates)

    def __getitem__(self, idx):
        return self.postive_templates[idx], self.negative_templates[idx]
