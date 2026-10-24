import os
import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import random
from skimage import transform
import json
from torch.utils.data import Dataset
from typing import Any
from PIL import Image
import random
import torch
from mgap import config

mean, std = (0.5,), (0.5,)
digit_options = list(map(str, range(10)))  # 10 digits
# 24 morphological transforms
thickthinning_options = ["no thickthinning", "thickening", "thinning"]
swelling_options = ["no swelling", "swelling"]
fracture_options = ["no fracture", "fracture"]
scaling_options = ["large", "small"]
color_options = ["gray", "red", "green", "blue", "cyan", "magenta", "yellow"]  # 7 colors


class MADDataset(Dataset):
    digits = list(map(str, range(10)))
    thickthinning = ["no thickthinning", "thickening", "thinning"]
    swelling = ["no swelling", "swelling"]
    fracture = ["no fracture", "fracture"]
    scaling = ["large", "small"]
    colors = ["gray", "red", "green", "blue", "cyan", "magenta", "yellow"]

    def __init__(self, root, transform, mode: str = "object_only", main_object: str = "digit",
                 return_att_mat: bool = False, train: bool = True,
                 return_att_mat_with_cls: bool = False, debug_visu: bool = False, tokenizer=None, return_cls_label_with_img=False, return_attribute_list=False, return_raw_text=False, return_attribute_list_and_text=False, generate_with_noise=False) -> None:
        super().__init__()

        self.follow_json = False
        if not train:
            self.follow_json = True

        with open(os.path.join(root, "train.json" if train else "test.json"), "r") as f:
            self.json_data = json.load(f)
        self.image_data = np.memmap(os.path.join(root, "train.mmap" if train else "test.mmap"),
                                    dtype=np.uint8, mode="r", shape=(len(self.json_data), 28, 28))

        self.json_data = {int(k): v for k, v in self.json_data.items()}

        # Each block contains all 24 morphological transforms for a single digit
        if train:
            self.block_size = len(self.json_data) // 60000
        else:
            self.block_size = len(self.json_data) // 10000
        self.transform = transform
        self.mode = mode
        self.train = train
        self.return_att_mat = return_att_mat
        self.return_att_mat_with_cls = return_att_mat_with_cls
        self.object = main_object
        self.debug_visu = debug_visu
        self.tokenizer = tokenizer
        self.return_cls_label_with_img = return_cls_label_with_img
        self.attribute_dict = {
            "thickthinning": self.thickthinning,
            "swelling": self.swelling,
            "fracture": self.fracture,
            "scaling": self.scaling,
            "colors": self.colors,
            "digit": self.digits
        }
        self.return_attribute_list = return_attribute_list
        self.color_list = None
        self.caption_list = None
        self.return_raw_text = return_raw_text
        self.return_attribute_list_and_text = return_attribute_list_and_text
        self.generate_with_noise = generate_with_noise


    def set_color_list(self, color_list):
        assert len(color_list) == len(self.json_data)
        self.color_list = color_list

    def set_caption_list(self, caption_list):
        assert len(caption_list) == len(self.json_data)
        self.caption_list = caption_list


    def __getitem__(self, index) -> Any:
        if not self.follow_json:
            rand_transform_index = random.randint(0, self.block_size - 1)
            index = index * self.block_size + rand_transform_index
        pregenerated_dict = self.json_data[index]

        image = np.array(self.image_data[pregenerated_dict["mmap_idx"]])
        cls_label = pregenerated_dict["cls_label"]
        choice1 = pregenerated_dict["thickthinnig"]
        choice2 = pregenerated_dict["swelling"]
        choice3 = pregenerated_dict["fracture"]
        choice4 = pregenerated_dict["scaling"]
        nr_classes = 10

        attribute_list = []
        # thickthinning
        if choice1 == 'no thickthinning':
            attribute_list.append(0)
        if choice1 == "thickening":
            attribute_list.append(1)
        elif choice1 == "thinning":
            attribute_list.append(2)
        if self.object == 'thickthinning':
            drop_idx = 0
            new_cls_label = attribute_list[-1]
            nr_classes = 3
        # swelling
        if choice2 == 'no swelling':
            attribute_list.append(3)
        if choice2 == "swelling":
            attribute_list.append(4)
        if self.object == 'swelling':
            drop_idx = 1
            new_cls_label = attribute_list[-1]
            nr_classes = 2
        # facture
        if choice3 == 'no fracture':
            attribute_list.append(5)
        if choice3 == "fracture":
            attribute_list.append(6)
        if self.object == 'fracture':
            drop_idx = 2
            new_cls_label = attribute_list[-1]
            nr_classes = 2
        # scaling
        if choice4 == "large":
            attribute_list.append(7)
        if choice4 == "small":
            attribute_list.append(8)
        if self.object == 'scaling':
            drop_idx = 3
            new_cls_label = attribute_list[-1]
            nr_classes = 2
        pil_img = Image.fromarray(image).convert("RGB")

        # colorization
        colors = color_options
        # color doesn't seem to have been saved in the JSON, so random each time
        if self.color_list is not None:
            choice5 = colors[self.color_list[index]]
        elif "color" in pregenerated_dict:
            choice5 = pregenerated_dict["color"]
        else:
            choice5 = random.choice(color_options)
        attribute_list.append(np.where([c == choice5 for c in colors])[0][0] + 9)
        if self.object == 'colors':
            drop_idx = 4
            new_cls_label = attribute_list[-1]
            nr_classes = len(colors)

        pil_img = self.colorize(pil_img, choice5)

        if self.transform is not None:
            image = self.transform(pil_img)

        if self.caption_list is not None:
            text = self.caption_list[index]
        else:
            if self.generate_with_noise:
                if self.object == "digit":
                    text, w2, w1, w3 = self.generate_text(cls_label, choice1, choice2, choice3, choice4, choice5)
                elif self.object == "thickthinning":
                    text, w2, w1, w3 = self.generate_text(choice1, str(cls_label), choice2, choice3, choice4, choice5)
                elif self.object == "swelling":
                    text, w2, w1, w3 = self.generate_text(choice2, str(cls_label), choice1, choice3, choice4, choice5)
                elif self.object == "fracture":
                    text, w2, w1, w3 = self.generate_text(choice3, str(cls_label), choice1, choice2, choice4, choice5)
                elif self.object == "scaling":
                    text, w2, w1, w3 = self.generate_text(choice4, str(cls_label), choice1, choice2, choice3, choice5)
                elif self.object == "colors":
                    text, w2, w1, w3 = self.generate_text(choice5, str(cls_label), choice1, choice2, choice3, choice4)
                else:
                    raise NotImplementedError
            else:
                if self.object == "digit":
                    text = self.generate_text(cls_label, choice1, choice2, choice3, choice4, choice5)
                elif self.object == "thickthinning":
                    text = self.generate_text(choice1, str(cls_label), choice2, choice3, choice4, choice5)
                elif self.object == "swelling":
                    text = self.generate_text(choice2, str(cls_label), choice1, choice3, choice4, choice5)
                elif self.object == "fracture":
                    text = self.generate_text(choice3, str(cls_label), choice1, choice2, choice4, choice5)
                elif self.object == "scaling":
                    text = self.generate_text(choice4, str(cls_label), choice1, choice2, choice3, choice5)
                elif self.object == "colors":
                    text = self.generate_text(choice5, str(cls_label), choice1, choice2, choice3, choice4)
                else:
                    raise NotImplementedError


        if self.debug_visu:
            save_text = self.generate_text_save(cls_label, choice1, choice2, choice3, choice4, choice5)
            return pil_img, save_text, text, cls_label, [choice1, choice2, choice3, choice4, choice5]

        raw_text = text
        if self.tokenizer is not None:
            text = self.tokenizer(text).squeeze(0)

        if self.return_att_mat:
            if not self.object == 'digit':
                attribute_list.append(cls_label + 9 + len(colors))  # highest number for attributes, i.e. 9-19 are class labels now
                attribute_list[drop_idx:] = list(np.array(attribute_list[drop_idx:]) - nr_classes)
                attribute_list.pop(drop_idx)
                cls_label = new_cls_label

            if self.return_att_mat_with_cls:
                att_mat, att_mat_bool = self.get_att_mat(attribute_list)
                _, att_mat_bool_with_cls = self.get_att_mat(attribute_list, include_boolean_class_label=True, cls_label=cls_label)

                return image, text, att_mat, att_mat_bool, att_mat_bool_with_cls, cls_label
            elif self.return_att_mat:
                att_mat, att_mat_bool = self.get_att_mat(attribute_list, nr_classes=nr_classes)
                return image, text, att_mat, att_mat_bool, cls_label

        if self.return_cls_label_with_img:
            if self.object == "digit":
                label = cls_label
            elif self.object == "thickthinning":
                label = thickthinning_options.index(choice1)
            elif self.object == "swelling":
                label = swelling_options.index(choice2)
            elif self.object == "fracture":
                label = fracture_options.index(choice3)
            elif self.object == "scaling":
                label = scaling_options.index(choice4)
            elif self.object == "colors":
                label = color_options.index(choice5)
            else:
                raise NotImplementedError
            return image, label

        if self.return_attribute_list:
            my_attribute_list = []
            attribute_vals = [choice1, choice2, choice3, choice4, choice5, str(cls_label)]
            for (k, v), av in zip(self.attribute_dict.items(), attribute_vals):
                my_attribute_list.append(v.index(av))
            return image, torch.tensor(my_attribute_list)

        if self.return_raw_text:
            return image, text, raw_text

        if self.return_attribute_list_and_text:
            my_attribute_list = []
            attribute_vals = [choice1, choice2, choice3, choice4, choice5, str(cls_label)]
            for (k, v), av in zip(self.attribute_dict.items(), attribute_vals):
                my_attribute_list.append(v.index(av))
            return image, text, torch.tensor(my_attribute_list)
        if self.generate_with_noise:
            return image, text, w2, w1, w3
        return image, text


    def __len__(self) -> int:
        if self.follow_json:
            return len(self.json_data)
        return len(self.json_data) // self.block_size

    def get_name_from_index(self, index):
        if self.object == "digit":
            name = str(index)
        elif self.object == "thickthinning":
            name = thickthinning_options[index]
        elif self.object == "swelling":
            name = swelling_options[index]
        elif self.object == "fracture":
            name = fracture_options[index]
        elif self.object == "scaling":
            name = scaling_options[index]
        elif self.object == "colors":
            name = color_options[index]
        else:
            raise NotImplementedError
        return name

    def get_att_mat(self, attribute_list, include_boolean_class_label=False, cls_label=None, nr_classes=0):
        att_mat = np.ones((1, 15)) * -1
        att_mat[:, :len(attribute_list)] = attribute_list
        if not include_boolean_class_label:
            if not self.object == 'digit':
                att_mat_bool = np.zeros((1, 26 - nr_classes))
            else:
                att_mat_bool = np.zeros((1, 16))
            att_mat_bool[:, attribute_list] = 1
        else:
            if not self.object == 'digit':
                raise NotImplementedError
            nr_classes = 10
            nr_rows = 16 + nr_classes
            att_mat_bool = np.zeros((1, nr_rows))
            att_mat_bool[0, cls_label] = 1
            att_mat_bool[:, np.array(attribute_list) + nr_classes] = 1

        return att_mat, att_mat_bool

    def rescale(self, image: np.ndarray) -> np.ndarray:
        new_image = np.zeros_like(image)
        image_rescaled = (transform.rescale(image.astype(np.float32) / 255, 0.75, anti_aliasing=False) * 255).astype(np.uint8)
        y = random.choice([0, 1])
        x = random.choice([0, 1])
        new_image[3 if y == 0 else 4:-3 if y == 1 else -4, 3 if x == 0 else 4:-3 if x == 1 else -4] = image_rescaled
        return new_image

    def colorize(self, image: Image, color: str) -> np.ndarray:
        if color == "gray":
            return image
        image = np.array(image)
        mask = image[..., 0] > 0
        if color == "blue":
            image[mask, 0] = 0
            image[mask, 1] = 0
        elif color == "green":
            image[mask, 0] = 0
            image[mask, 2] = 0
        elif color == "red":
            image[mask, 1] = 0
            image[mask, 2] = 0
        elif color == "yellow":
            image[mask, 0] = 255
            image[mask, 1] = 255
        elif color == "magenta":
            image[mask, 0] = 255
            image[mask, 2] = 255
        elif color == "cyan":
            image[mask, 1] = 255
            image[mask, 2] = 255
        else:
            raise NotImplementedError
        return Image.fromarray(image)

    def generate_text(self, cls_label: int, *args):
        if self.mode == "object_only":
            assert not self.generate_with_noise
            caption = str(cls_label)
        elif "object_attribute_" in self.mode:
            if self.generate_with_noise:
                attributes_to_select = [5] + random.sample(range(5), int(self.mode[-1]))
                random_select = random.randint(0, len(attributes_to_select))
                binary_flags = [1] * random_select + [0] * (len(attributes_to_select) - random_select)
                random.shuffle(binary_flags)
                correct_or_not = np.array(list(zip(attributes_to_select, binary_flags)))
                if correct_or_not[0][1] == 1:
                    string_content = [str(cls_label)]
                else:
                    all_attr_list = config.mad_decoder_name_list_cat_wise_numerical[correct_or_not[0][0]]
                    correct_attr_index = all_attr_list.index(str(cls_label))
                    incorrect_attr_index = random.randint(0, len(all_attr_list) - 2)
                    incorrect_attr_index = incorrect_attr_index if incorrect_attr_index < correct_attr_index else incorrect_attr_index + 1
                    string_content = [all_attr_list[incorrect_attr_index]]
                for i in range(1, len(correct_or_not)):
                    if correct_or_not[i][1] == 1:
                        string_content.append(args[correct_or_not[i][0]])
                    else:
                        all_attr_list = config.mad_decoder_name_list_cat_wise_numerical[correct_or_not[i][0]]
                        correct_attr_index = all_attr_list.index(args[correct_or_not[i][0]])
                        incorrect_attr_index = random.randint(0, len(all_attr_list) - 2)
                        incorrect_attr_index = incorrect_attr_index if incorrect_attr_index < correct_attr_index else incorrect_attr_index + 1
                        string_content.append(all_attr_list[incorrect_attr_index])
                random.shuffle(string_content)
                caption = "-".join(string_content)
                w1 = (correct_or_not[:,1] == 1).sum() / 6
                w2 = (correct_or_not[:,1] == 1).sum() / (int(self.mode[-1]) + 1)
                w3 = 1-((w1+w2)/2.)
            else:
                string_content = [str(cls_label)]
                string_content += random.sample(args, int(self.mode[-1]))
                random.shuffle(string_content)
                caption = "-".join(string_content)
        else:
            raise NotImplementedError
        if self.generate_with_noise:
            return caption, w2, w1, w3
        else:
            return caption

    def generate_text_zs(self, digit_label, remaining_labels):
        if self.mode == "object_only":
            raise NotImplementedError
        elif "object_attribute_" in self.mode:
            rand_attr_count = int(self.mode[-1])
            if self.object == "digit":
                string_content = []
                string_content += random.sample(remaining_labels, int(self.mode[-1]))
            if self.object != "digit":
                rand_attr_count = rand_attr_count - 1
                string_content = [str(digit_label)]
                if rand_attr_count > 0:
                    string_content += random.sample(remaining_labels, rand_attr_count)
            random.shuffle(string_content)
            caption = "-".join(string_content)
        else:
            raise NotImplementedError
        return caption

    def generate_text_save(self, cls_label: int, *args):
        if self.mode == "object_only":
            caption = str(cls_label)
        elif "object_attribute_" in self.mode:
            caption = str(cls_label) + '-' + '-'.join(args)
        else:
            raise NotImplementedError
        return caption


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean, std)
                                    ])
    dataset = MADDataset(root='./MAD/datasets/morphoMNIST_0', transform=transform, mode="object_attribute_1",
                         train=True, debug_visu=True, return_att_mat=True)
    breakpoint()

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
    )

    for batch in tqdm.tqdm(train_loader, total=len(train_loader)):
        continue
