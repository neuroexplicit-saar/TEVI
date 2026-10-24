OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INCEPTION_MEAN = (0.5, 0.5, 0.5)
INCEPTION_STD = (0.5, 0.5, 0.5)

# Default name for a weights file hosted on the Huggingface Hub.
HF_WEIGHTS_NAME = "open_clip_pytorch_model.bin"  # default pytorch pkl
HF_SAFE_WEIGHTS_NAME = "open_clip_model.safetensors"  # safetensors version
HF_CONFIG_NAME = 'open_clip_config.json'

OPENAI_DATASET_INVERSE_MEAN = tuple(-x / y for x, y in zip(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD))
OPENAI_DATASET_INVERSE_STD = tuple(1 / y for y in OPENAI_DATASET_STD)
IMAGENET_INVERSE_MEAN = tuple(-x / y for x, y in zip(IMAGENET_MEAN, IMAGENET_STD))
IMAGENET_INVERSE_STD = tuple(1 / y for y in IMAGENET_STD)
INCEPTION_INVERSE_MEAN = tuple(-x / y for x, y in zip(INCEPTION_MEAN, INCEPTION_STD))
INCEPTION_INVERSE_STD = tuple(1 / y for y in INCEPTION_STD)
