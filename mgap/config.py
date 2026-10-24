IMAGENET_PATH = None
MSCOCO_VAL = None
MSCOCO_VAL_ANNOT = None
MSCOCO_VAL_INSTANCES = None

MSCOCO_TRAIN = None
MSCOCO_TRAIN_ANNOT = None
MSCOCO_TRAIN_INSTANCES = None

MSCOCO_IMG_ROOT = None
MSCOCO_ATTR_CAPTION_ROOT = None
VAL_COUNT_ATTR_CAPTION_DICT = None
# _more_than_1attr

FLICKR_ROOT_DIR = None
DOCCI_ROOT_DIR = None
IIW_ROOT_DIR = None

CAPTION_LIST_GPT = ['fully_correct', 'half_correct_and_half_random', 'random', 'fully_correct_and_random', 'subset_of_correct']

coco_gpt_caption_path = {}
for caption_set_up in CAPTION_LIST_GPT:
    coco_gpt_caption_path[f"retrieval_coco_{caption_set_up}"] = None


CAPTION_LIST = ["full_random", "subset_of_full_correct", "full_correct_plus_random", "full_correct", "part_random_0.25", "part_random_0.5", "part_random_0.75"]

MSCOCO_save_dir = None

GPT_INSTRUCTIONS_PATH = None

FLICKR = None
FLICKR_ANNOT = None
None

embed_dim_dict = {
    "RN50": 1024,
    "ViT-B-16-SharedCLIP": 768,
    "ViT-B-16": 512,
    "ViT-L-14": 768,
    "ViT-B-16-FLAIR": 512,
    "COCO-ViT-B-16": 512,
    "COCO-ViT-B-16-small": 64,
}
embed_dim_dict.update(
    {"MGAP-ViT-256-{}".format(dim): dim for dim in [6, 9, 12, 15, 18]})

unpooled_embed_dim_dict = {
    "ViT-B-16": 768,
    "ViT-B-16-SharedCLIP": 768,  # not sure if it is correct
    "ViT-B-16-FLAIR": 768,
    "ViT-L-14": 768,
    "MGAP-ViT-256-18": 0,
    "COCO-ViT-B-16": 768,
    "COCO-ViT-B-16-small": 768,
}

autoencoder_input_dim_dict = {'RN50_out': 1024,
                              'RN50': 1024,
                              'clip_ViT-B16_out': 512,
                              'clip_ViT-L14_out': 768,
                              'clip_RN50_out': 1024,
                              'ViT-B16_out': 512,
                              'ViT-B-16_out': 512,
                              'ViT-L-14_out': 768,
                              'ViT-B-16-SharedCLIP_out': 768,
                              'COCO-ViT-B-16_out': 512,
                              'COCO-ViT-B-16-small_out': 64}

data_dir_root = None
save_dir_root = './SAE'
probe_cs_save_dir_root = './probe'
vocab_dir = './vocab'


probe_dataset_root_dir_dict = {
    "imagenet": None,
    "cifar100": None,
    "places365": None,
    "cifar10": None,
    "sunattributes": None,
    "waterbirds100": None,
    "flowers_102": None,
    "stanford": "BS/databases34/stanford_cars",
    "food-101": None,
    'imagenet_v2': None,
    'imagenet_sketch': None,
    'imagenet_o': None,
    'imagenet_a': None,
    'imagenet_r': None,
    # 'iamgenet_train':
    # 'iamgenet_val':
}


probe_dataset_nclasses_dict = {"places365": 365,
                               'imagenet': 1000, "cifar10": 10, "cifar100": 100, "waterbirds100": 100, "sunattributes": 0}

mad_vocab_txt_path = None

mad_attribute_pos_dict = {
    "thickthinning": [0, 1, 2],
    "swelling": [3, 4],
    "fracture": [5, 6],
    "scaling": [7, 8],
    "colors": [9, 10, 11, 12, 13, 14, 15],
    "digit": [16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
}


thickthinning = ["no thickthinning", "thickening", "thinning"]
swelling = ["no swelling", "swelling"]
fracture = ["no fracture", "fracture"]
scaling = ["large", "small"]
colors = ["gray", "red", "green", "blue", "cyan", "magenta", "yellow"]
digits = [str(i) for i in range(10)]

mad_decoder_name_list_cat_wise_numerical = [
    thickthinning, swelling, fracture, scaling, colors, digits]
mad_decoder_name_list = thickthinning + \
    swelling + fracture + scaling + colors + digits
decoder_name_to_index = {name: i for i,
                         name in enumerate(mad_decoder_name_list)}
index_to_decoder_name = {i: name for i,
                         name in enumerate(mad_decoder_name_list)}
decoder_name_cum_len_list_of_cat = [0, 3, 5, 7, 9, 16]
mad_decoder_name_list_cat_wise_literal = [
    'thickthinning', 'swelling', 'fracture', 'scaling', 'colors', 'digit']

viz_randperm_path = None

