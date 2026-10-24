import argparse
import ast


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split('=')
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                # fallback to string (avoid need to escape on command line)
                kw[key] = str(value)
        setattr(namespace, self.dest, kw)


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to file(s) with training data. When using webdataset, multiple datasources can be combined using the `::` separator.",
    )
    parser.add_argument(
        "--train-data-upsampling-factors",
        type=str,
        default=None,
        help=(
            "When using multiple data sources with webdataset and sampling with replacement, this can be used to upsample specific data sources. "
            "Similar to --train-data, this should be a string with as many numbers as there are data sources, separated by `::` (e.g. 1::2::0.5) "
            "By default, datapoints are sampled uniformly regardless of the dataset sizes."
        )
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to file(s) with validation data",
    )
    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Required for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--val-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Useful for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["webdataset", "csv", "synthetic", "auto", "mad", "coco", "coco_controlled", "cocomad"],
        default="auto",
        help="Which type of dataset to process."
    )
    parser.add_argument(
        "--train-dataset-type",
        choices=["webdataset", "csv", "synthetic", "auto", "coco", "flickr"],
        default="csv",
        help="Which type of dataset to process."
    )
    parser.add_argument(
        "--dataset-name",
        default="auto",
        help="Which type of dataset to process."
    )
    parser.add_argument(
        "--load-dataset-name",
        default="",
    )
    parser.add_argument(
        "--dataset-resampled",
        default=False,
        action="store_true",
        help="Whether to use sampling with replacement for webdataset shard selection."
    )
    parser.add_argument(
        "--csv-separator",
        type=str,
        default="\t",
        help="For csv-like datasets, which separator to use."
    )
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths."
    )
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="title",
        help="For csv-like datasets, the name of the key for the captions."
    )
    parser.add_argument(
        "--mad-val",
        type=str,
        default=None,
        help="Path to MAD val set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-v2",
        action="store_true",
        default=False,
        help="Path to imagenet v2 for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-sketch",
        action="store_true",
        default=False,
        help="Path to imagenet sketch for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Override system default cache path for model & tokenizer file downloads.",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="log files on local master, otherwise global master only.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--epochs-cooldown", type=int, default=None,
        help="When scheduler w/ cooldown used, perform cooldown from total_epochs - cooldown_epochs onwards."
    )
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate.")
    parser.add_argument("--beta1", type=float,
                        default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float,
                        default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None,
                        help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=10000, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=None, help="Number of epochs to warmup for."
    )
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.")
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default='cosine',
        help="LR scheduler. One of: 'cosine', 'const' (constant), 'const-cooldown' (constant w/ cooldown). Default: cosine",
    )
    parser.add_argument(
        "--lr-cooldown-end", type=float, default=0.0,
        help="End learning rate for cooldown schedule. Default: 0"
    )
    parser.add_argument(
        "--lr-cooldown-power", type=float, default=1.0,
        help="Power for polynomial cooldown schedule. Default: 1.0 (linear decay)"
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--zeroshot-frequency", type=int, default=2, help="How often to run zero shot."
    )
    parser.add_argument(
        "--val-frequency", type=int, default=1, help="How often to run evaluation with val data."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--eval-zeroshot",
        default=False,
        action='store_true',
        help="Lock full image tower by disabling gradients.",
    )
    parser.add_argument(
        "--resume-combined-training",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16",
                 "fp16", "pure_bf16", "pure_fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RN50",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--pretrained",
        default='',
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--pretrained-image",
        default=False,
        action='store_true',
        help="Load imagenet pretrained weights for image tower backbone if available.",
    )
    parser.add_argument(
        "--lock-image",
        default=False,
        action='store_true',
        help="Lock full image tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-image-unlocked-groups",
        type=int,
        default=0,
        help="Leave last n image tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-image-freeze-bn-stats",
        default=False,
        action='store_true',
        help="Freeze BatchNorm running stats in image tower for any locked layers.",
    )
    parser.add_argument(
        '--image-mean', type=float, nargs='+', default=None, metavar='MEAN',
        help='Override default image mean value of dataset')
    parser.add_argument(
        '--image-std', type=float, nargs='+', default=None, metavar='STD',
        help='Override default image std deviation of of dataset')
    parser.add_argument(
        '--image-interpolation',
        default=None, type=str, choices=['bicubic', 'bilinear', 'random'],
        help="Override default image resize interpolation"
    )
    parser.add_argument(
        '--image-resize-mode',
        default=None, type=str, choices=['shortest', 'longest', 'squash'],
        help="Override default image resize (& crop) mode during inference"
    )
    parser.add_argument('--aug-cfg', nargs='*', default={}, action=ParseKwargs)
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action='store_true',
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--local-loss",
        default=False,
        action="store_true",
        help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)"
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="enable full distributed gradient for feature gather"
    )
    parser.add_argument(
        '--force-image-size', type=int, nargs='+', default=None,
        help='Override default image size'
    )
    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action='store_true',
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    parser.add_argument(
        "--force-patch-dropout",
        default=None,
        type=float,
        help="Override the patch dropout during training, for fine tuning with no dropout near the end as in the paper",
    )
    parser.add_argument(
        "--force-custom-text",
        default=False,
        action='store_true',
        help="Force use of CustomTextCLIP model (separate text-tower).",
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action='store_true',
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        "--torchcompile",
        default=False,
        action='store_true',
        help="torch.compile() the model, requires pytorch 2.0 or later.",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action='store_true',
        help="torch.jit.trace the model for inference / eval only",
    )
    parser.add_argument(
        "--accum-freq", type=int, default=1, help="Update the model every --acum-freq steps."
    )
    parser.add_argument(
        "--device", default="cuda", type=str, help="Accelerator to use."
    )
    # arguments for distributed training
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend",
        default=None,
        type=str,
        help="distributed backend. \"nccl\" for GPU, \"hccl\" for Ascend NPU"
    )
    parser.add_argument(
        "--report-to",
        default='',
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False)
    parser.add_argument(
        "--wandb-notes",
        default='',
        type=str,
        help="Notes if logging with wandb"
    )
    parser.add_argument(
        "--wandb-mode",
        default='online',
        type=str,
        help="which mode", choices=['online', 'offline', 'disabled']
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default='open-clip',
        help="Name of the project if logging with wandb.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log directory, and execute from there."
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training."
    )
    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action='store_true',
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )
    parser.add_argument(
        "--grad-clip-norm", type=float, default=None, help="Gradient clip."
    )
    parser.add_argument(
        "--lock-text",
        default=False,
        action='store_true',
        help="Lock full text tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-text-unlocked-layers",
        type=int,
        default=0,
        help="Leave last n text tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-text-freeze-layer-norm",
        default=False,
        action='store_true',
        help="Freeze LayerNorm running stats in text tower for any locked layers.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=100,
        help="Log every n steps to tensorboard/console/wandb.",
    )
    parser.add_argument(
        "--coca-caption-loss-weight",
        type=float,
        default=2.0,
        help="Weight assigned to caption loss in CoCa."
    )
    parser.add_argument(
        "--coca-contrastive-loss-weight",
        type=float,
        default=1.0,
        help="Weight assigned to contrastive loss when training CoCa."
    )
    parser.add_argument(
        "--remote-sync",
        type=str,
        default=None,
        help="Optinoally sync with a remote path specified by this arg",
    )
    parser.add_argument(
        "--remote-sync-frequency",
        type=int,
        default=300,
        help="How frequently to sync to a remote directly if --remote-sync is not None.",
    )
    parser.add_argument(
        "--remote-sync-protocol",
        choices=["s3", "fsspec"],
        default="s3",
        help="How to do the remote sync backup if --remote-sync is not None.",
    )
    parser.add_argument(
        "--delete-previous-checkpoint",
        default=False,
        action="store_true",
        help="If true, delete previous checkpoint after storing a new one."
    )
    parser.add_argument(
        "--distill-model",
        default=None,
        help='Which model arch to distill from, if any.'
    )
    parser.add_argument(
        "--distill-pretrained",
        default=None,
        help='Which pre-trained weights to distill from, if any.'
    )
    parser.add_argument(
        "--use-bnb-linear",
        default=None,
        help='Replace the network linear layers from the bitsandbytes library. '
        'Allows int8 training/inference, etc.'
    )
    parser.add_argument(
        "--siglip",
        default=False,
        action="store_true",
        help='Use SigLip (sigmoid) loss.'
    )

    parser.add_argument(
        "--mad-mode",
        default="object_only",
        type=str,
        help="Mode for MAD training."
    )

    parser.add_argument(
        "--wandb-entity",
        default=None,
        type=str,
        help="Entity for wandb logging."
    )

    parser.add_argument(
        "--embed-dim",
        default=256,
        type=int,
        help="Embedding dimension for MAD training."
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        type=str,
        help="Path to csv file for MAD training."
    )
    parser.add_argument(
        "--partial-caption-fraction",
        default=1,
        type=int,
        help="Fraction of text used for training."
    )
    parser.add_argument(
        "--partial-index-path",
        default=None,
        type=str,
        help="Path to partial index file."
    )
    parser.add_argument(
        "--write-path-dummy",
        default=None,
        type=str,
        help="Path to write dummy data."
    )
    parser.add_argument(
        "--specific-checkpoint-path",
        default=None,
        type=str,
        help="Specific checkpoint to resume from."
    )
    parser.add_argument(
        "--pretrained-specific-checkpoint-path",
        default=None,
        type=str,
        help="Specific checkpoint to resume from."
    )
    
    parser.add_argument(
        "--given-load-clip-path",
        default=None,
        type=str,
        help="Specific checkpoint to resume from."
    )

    parser.add_argument(
        "--save-data",
        default=None,
        type=str,
        help="Path for data for saving features."
    )
    parser.add_argument(
        "--split-name",
        default=None,
        type=str,
        help="Split name for saving features."
    )
    parser.add_argument(
        "--eval-val-full",
        default=False,
        action="store_true",
        help="Evaluate on full caption validation set."
    )
    parser.add_argument(
        "--eval-before-train",
        default=False,
        action="store_true",
        help="Evaluate once before training."
    )

    parser.add_argument(
        "--eval-during-train",
        default=False,
        action="store_true",
        help="Evaluate during training."
    )

    parser.add_argument(
        "--return-indices",
        default=False,
        action="store_true",
        help="Return indices for CC3M/CC12M tar files."
    )

    # arguments for SAE
    parser.add_argument("--train-sae", action="store_true", default=False)
    parser.add_argument("--use-pretrained-sae",
                        action="store_true", default=False)
    parser.add_argument("--random-sae", action="store_true", default=False,
                        help="Keep the randomly-initialized SAE instead of loading the trained SAE checkpoint. "
                             "Paths/architecture are still derived from --use-pretrained-sae, only the weights are left random.")
    parser.add_argument("--random-sae-checkpoint-path", type=str, default=None,
                        help="Where the random SAE is dumped to (first run) and loaded from (subsequent runs), "
                             "so the same random SAE is reused across runs/array-tasks. Required with --random-sae.")
    parser.add_argument("--sae-specific-checkpoint-path", default=None,
                        type=str, help="Specific checkpoint to resume from.")
    parser.add_argument("--sae-pretrained-specific-checkpoint-path", default=None,
                        type=str, help="Specific checkpoint to load the pretrained SAE.")
    parser.add_argument("--optimizer-name", default="Adam", type=str, help="Optimizer name for SAE training (eithe rAdam or AdamW)")
    parser.add_argument("--sae-optimizer-name", default="Adam", type=str, help="Optimizer name for SAE training (eithe rAdam or AdamW)")


    parser.add_argument("--eval-only", action="store_true", default=False, help="to evaluate only for learnt mask textconditioning")

    parser.add_argument("--train-sae-lr", type=float, default=1e-4)
    parser.add_argument("--sae-config-name", type=str, default=None)
    parser.add_argument("--train-sae-batch-size", type=int, default=131072)
    parser.add_argument("--train-sae-epochs", type=int,
                        default=32, help="Number of epochs to train for.")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--auxk", type=int, default=256)
    parser.add_argument("--expansion-factor", type=int, default=32)
    parser.add_argument("--matching-loss-lambda", type=float, default=1.0)
    parser.add_argument("--use-matching-loss",
                        action="store_true", default=False)
    parser.add_argument("--use-contra-loss",
                        action="store_true", default=False)
    parser.add_argument("--contra-loss-lambda", type=float, default=0.0)
    parser.add_argument("--use-partcontra-loss",
                        action="store_true", default=False)
    parser.add_argument("--use-reconcontra-loss",
                        action="store_true", default=False)
    parser.add_argument("--num-dicvec-per-word", type=int, default=3)
    parser.add_argument("--use-fixed-decoder-weights",
                        action="store_true", default=False)
    parser.add_argument("--use-fixed-latent-meanings",
                        action="store_true", default=False)
    parser.add_argument("--bce-on-unselected-indices",
                        action="store_true", default=False)
    parser.add_argument("--latent-zero-loss-lambda", type=float, default=1.0)
    parser.add_argument("--decoder-weights-vocab", type=str, default=None)
    parser.add_argument("--use-fixed-concept-strengths",
                        action="store_true", default=False)
    parser.add_argument("--scale-negative-contra-loss",
                        action="store_true", default=False)
    parser.add_argument("--scale-negative-contra-loss-lambda",
                        type=float, default=1.0)

    parser.add_argument(
        "--save-clip-features-text", default=False, action="store_true", help="loads the clip model and saves the img features, so that SAEs can be trained")
    parser.add_argument("--save-clip-features-img", default=False, action="store_true",
                        help="loads the clip model and saves the text features, so that SAEs can be trained")
    parser.add_argument(
        "--save-clip-features-img-imagenet", default=False, action="store_true", help="loads the clip model and saves the img features, so that SAEs can be evaluate on imagenet val set")
    
    parser.add_argument(
        "--shard", type=str, help="loads the clip model and saves the text features, so that SAEs can be trained, mention the shard")
    parser.add_argument("--shard-name", type=str,
                        help="loads the clip model and saves the text features, so that SAEs can be trained, mention the shard name to be used for file naming")

    parser.add_argument(
        "--recap-json-caption-key",
        type=str,
        default="caption",
        help="For recap json-like datasets, the name of the key for the captions."
    )

    
    # arguments to load clip
    parser.add_argument("--use-pretrained-clip",
                        action="store_true", default=False)
    parser.add_argument("--load-lr", type=float,
                        default=None, help="Learning rate.")
    parser.add_argument("--load-workers", type=int, default=4,
                        help="Number of dataloader workers per GPU.")
    parser.add_argument("--load-batch-size", type=int,
                        default=64, help="Batch size per GPU.")

    # arguments to load SAE
    parser.add_argument("--load-sae-lr", type=float, default=1e-4)
    parser.add_argument("--load-sae-batch-size", type=int, default=131072)
    parser.add_argument("--load-sae-epochs", type=int,
                        default=32, help="Number of epochs to train for.")
    parser.add_argument("--load-k", type=int, default=32)
    parser.add_argument("--load-auxk", type=int, default=256)

    # extra arguments for clip
    parser.add_argument("--train-clip", action="store_true", default=False)
    parser.add_argument("--finetune-text-encoder",
                        action="store_true", default=False)

    parser.add_argument("--run-reconcontra-full-baseline",
                        action="store_true", default=False)

    parser.add_argument("--filter-algo-name", type=str, default=None)
    parser.add_argument("--filter-arch-name", type=str, default=None)

    parser.add_argument("--train-mask", action="store_true", default=False)
    parser.add_argument("--use-pretrained-mask",
                        action="store_true", default=False)
    parser.add_argument("--train-mask-lr", type=float, default=1e-4)
    parser.add_argument("--n-layers", default=1, type=int)
    parser.add_argument("--train-mask-epochs", type=int, default=50)
    parser.add_argument("--loss-name", default="mse", type=str,
                        choices=['contra', 'mse', 'partcontra', 'cosine'])
    # set warmup_epochs
    parser.add_argument("--warmup-mask-epochs", type=int, default=0)
    parser.add_argument("--train-mask-batch-size", type=int, default=4096)

    parser.add_argument("--use-inverted-conditioning",action="store_true", default=False)
    parser.add_argument("--use-extra-loss-term",action="store_true", default=False)
    parser.add_argument("--augment-existing-loss",action="store_true", default=False)
    parser.add_argument("--inv-loss-lambda", type=float, default=1.0)

    parser.add_argument("--use-subcaption-loss",action="store_true", default=False)
    parser.add_argument("--use-l1-penalty-on-mask",action="store_true", default=False)
    parser.add_argument("--use-l2-penalty-on-mask",action="store_true", default=False)
    parser.add_argument("--mask-loss-lambda", type=float, default=1.0)


    parser.add_argument("--apply-l2-regularizer-between-conditioned-img-img", action="store_true", default=False)
    parser.add_argument("--apply-contrastive-loss-between-conditioned-img-img", action="store_true", default=False)
    parser.add_argument("--regularizer-loss-lambda", type=float, default=0.0)
    parser.add_argument("--double_constrained_loss", action="store_true", default=False)
    parser.add_argument("--double_constrained_loss_with_orthogonality", action="store_true", default=False)
    parser.add_argument("--double_constrained_loss_lambda", type=float, default=0.0)

    parser.add_argument("--attr-to-ablate", type=str, default=None)
    parser.add_argument("--use-random-masks",
                        action="store_true", default=False)
    parser.add_argument("--train-mask-directly",
                        action="store_true", default=False)
    parser.add_argument("--sigmoid-bias-fill", type=float, default=0.0)
    parser.add_argument("--mask-layer-dims", nargs='+', type=int, default=None)
    parser.add_argument("--load-clip-logs", type=str, default=None)
    parser.add_argument("--load-sae-logs", type=str, default=None)
    # Root dir to write dumped features to. If set, features are saved under this root
    # mirroring the load_clip_path sub-structure, instead of next to the loaded checkpoint.
    parser.add_argument("--features-save-logs", type=str, default=None)
    parser.add_argument("--finetune-vision-encoder",
                        action="store_true", default=False)
    parser.add_argument("--frozen-clip",
                        action="store_true", default=False)  # overrides finetune-vision-encoder and finetune-text-encoder
    parser.add_argument("--only-coco-from-embeddings", action="store_true", default=False)

    parser.add_argument("--coco-controlled", action="store_true", default=False)
    parser.add_argument("--coco-controlled-null-lambda", type=float, default=0.0)
    parser.add_argument("--coco-controlled-log-version", action="store_true", default=False)
    parser.add_argument("--coco-controlled-only-positive", action="store_true", default=False)

    parser.add_argument("--use-real-clip", action="store_true", default=False)

    parser.add_argument("--use-extra-ortho-dim", action="store_true", default=False)
    parser.add_argument("--null-vector-ortho-loss-lambda", type=float, default=0.0)
    parser.add_argument("--null-vector-ortho-with-abs", action="store_true", default=False)

    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-CAlayers", type=int, default=1)
    parser.add_argument("--attn-type", type=str, default='attn-i-t', help='attention type for cross attention idea')
    parser.add_argument("--use-pooled-text-for-ca", action="store_true", default=False)

    parser.add_argument("--use-multimodal-sae", action="store_true", default=False)
    parser.add_argument("--use-recap-dataset", action="store_true", default=False)
    parser.add_argument("--use-wds-dataset", action="store_true", default=False)
    parser.add_argument("--use-wds-dataset-for-sae", action="store_true", default=False)

    parser.add_argument("--inference-with-flair", action="store_true", default=False)
    parser.add_argument("--inference-with-ours", action="store_true", default=False)

    parser.add_argument("--use-flair-evaluation-codebase", action="store_true", default=False)
    parser.add_argument("--use-flair-model", action="store_true", default=False)


    parser.add_argument(
        "--huggingface-model-name",
        type=str,
        default="",
        help="Name of the huggingface model."
    )
    parser.add_argument(
        "--huggingface-repo-name",
        type=str,
        default="",
        help="Name of the huggingface repo."
    )

    parser.add_argument("--which-type-retrieval", type=str, default='global', choices=['full', 'global', 'mix', 'random_single'])
    parser.add_argument("--retrieval-coco", action="store_true", default=False)
    parser.add_argument(
        "--retrieval-dci",
        default=False,
        action="store_true",
        help="If true, then we enable the dci retrieval task.")

    parser.add_argument(
        "--retrieval-iiw",
        default=False,
        action="store_true",
        help="If true, then we enable the iiw retrieval task.")

    parser.add_argument(
        "--use-finegrained-iiw",
        default=True,
        action="store_true",
        help="If set to true, under the condition that we enable iiw, we further use the fine-grained iiw mode.")

    parser.add_argument(
        "--retrieval-sharegpt4v-1k",
        default=False,
        action="store_true",
        help="If true, then we enable the sharegpt4v retrieval task with 1k data size.")

    parser.add_argument(
        "--retrieval-sharegpt4v-10k",
        default=False,
        action="store_true",
        help="If true, then we enable the sharegpt4v retrieval task with 10k data size.")

    parser.add_argument(
        "--retrieval-flickr",
        default=False,
        action="store_true",
        help="If true, then we enable the flickr retrieval task.")
    parser.add_argument(
        "--flickr-val-or-test",
        type=str,
        default='val',
        choices=['val', 'testing'],
        help="Which dataset to be used for inference, default choices are val or test.",
    )
    parser.add_argument(
        "--retrieval-urban-1k",
        default=False,
        action="store_true",
        help="If true, then we enable the urban-1k retrieval task.")
    parser.add_argument(
        "--retrieval-data-cc3m-train",
        default=False,
        action="store_true",
        help="If true, then we enable the cc3m retrieval task.")
    parser.add_argument(
        "--retrieval-docci",
        default=False,
        action="store_true",
        help="If true, then we enable the DOCCI retrieval task.")

    parser.add_argument(
        "--plot-vis-retrieval-cmp",
        default=False,
        action="store_true",
        help="to compare where laion and FLair do good and ours is worse")

    parser.add_argument(
        "--plot-t2i_i2t",
        default=False,
        action="store_true",
        help="to compare where laion and FLair do good and ours is worse")


    parser.add_argument(
        "--which-type-conditioning", type=str, default='', help="conditioning image on text, there are different variations to conid=sider for loss, as computationally expensive to do all terms",)
    parser.add_argument("--siglip-conditioned", default = False, action="store_true", help="if we want the new siglip loss which uses the image ocndiitoned text")
    parser.add_argument("--contrastive-conditioned", default = False, action="store_true", help="if we want the new siglip loss which uses the image ocndiitoned text")


    parser.add_argument("--initialize-with-identity", action="store_true", default=False)
    parser.add_argument("--plot-recap-stats", action="store_true", default=False)
    parser.add_argument("--cc-w-mode", type=str, default="fraction")
    parser.add_argument("--cc-subtract-mean-w", action="store_true", default=False)
    parser.add_argument("--cc-tail-prob", type=float, default=0.8)
    parser.add_argument("--cc-use-nonuniform-probs", action="store_true", default=False)
    parser.add_argument("--cc-clamp-selected-total-len-by-gt", action="store_true", default=False)
    parser.add_argument("--cc-sigmoid-loss", action="store_true", default=False)
    parser.add_argument("--cc-sigmoid-tau", type=float, default=100.0)

    parser.add_argument("--eval-coco-multilabel", action="store_true", default=False)

    parser.add_argument("--use-recap-siglip", action="store_true", default=False)
    parser.add_argument("--fcr-fraction", type=float, default=0.0)
    parser.add_argument("--sfc-fraction", type=float, default=0.0)

    parser.add_argument("--dont-use-saved-data", action="store_true", default=False)
    parser.add_argument("--not-retrieval-coco-our-setup", action="store_true", default=False)
    parser.add_argument("--retrieval-noisy", action="store_true", default=False)
    parser.add_argument("--use-retrieval-loss", action="store_true", default=False)

    parser.add_argument("--mad-generate-with-noise", action="store_true", default=False)
    parser.add_argument("--save-templates-for-conditioning", action="store_true", default=False)
    parser.add_argument("--evaluate-text-conditioning-with-noisy-captions", action="store_true", default=False)
    

    parser.add_argument("--use-fixed-vocab-dim", action="store_true", default=False)
    parser.add_argument("--clip-logit-scale-fixed", type=float, default=None)
    parser.add_argument("--use-mad-train-full-dataset", action="store_true", default=False)

    parser.add_argument("--cocomad-with-noise", action="store_true", default=False)


    parser.add_argument(
        "--cifar10",
        default=False,
        action="store_true",
        help="If true, then we enable the flickr retrieval task.")

    parser.add_argument(
        "--cifar100",
        default=False,
        action="store_true",
        help="If true, then we enable the flickr retrieval task.")
    
    parser.add_argument(
        "--flowers-102",
        default=False,
        action="store_true",
        help="If true, then we enable the flickr retrieval task.")
    
    parser.add_argument(
        "--imagenet-val",
        default=False,
        action="store_true",
        help="If true, then we enable the flickr retrieval task.")
    
    parser.add_argument(
        "--imagenet-a",
        default=False,
        action="store_true",
        help="If true, then we enable the imagenet-a retrieval task.")
    
    parser.add_argument(
        "--imagenet-r",
        default=False,
        action="store_true",
        help="If true, then we enable the imagenet-r retrieval task.")
    
    parser.add_argument(
        "--imagenet-o",
        default=False,
        action="store_true",
        help="If true, then we enable the imagenet-o retrieval task.")
    parser.add_argument(
        "--imagenet-c",
        default=False,
        action="store_true",
        help="If true, then we enable the imagenet-c retrieval task.")

    parser.add_argument(
        "--stanford",
        default=False,
        action="store_true",
        help="If true, then we enable the flickr retrieval task.")
    
    parser.add_argument(
        "--food-101",
        default=False,
        action="store_true",
        help="If true, then we enable the flickr retrieval task.")


    parser.add_argument("--use-alignclip-checkpoint",
                        action="store_true", default=False)
    parser.add_argument("--use-sharedclip-checkpoint",
                        action="store_true", default=False)
    parser.add_argument("--use-original-sharedclip-checkpoint",
                        action="store_true", default=False)
    parser.add_argument(
        "--use-best-model",
        default=False,
        action="store_true",
        help="If true, evaluates using best model checkpoint.")

    parser.add_argument("--save-eval-csv", action="store_true", default=False)

    parser.add_argument(
        "--dump-batch-size", type=int, default=256, help="Batch size per GPU."
    )
    
    parser.add_argument("--dump-cc3m-features", action="store_true", default=False)
    parser.add_argument("--cc3m-val-data", action="store_true", default=False)
    parser.add_argument("--zeroshot-using-sae-latents-directly", action="store_true", default=False)
    parser.add_argument(
        "--n-sae-decoder", type=int, default=16, help="Batch size per GPU."
    )
    
    parser.add_argument(
        "--annot-fname",
        default=None,
        type=str,
        help="path for annotation file for robust retrieval evaluation"
    )

    parser.add_argument(
        "--robust-coco-img-dir",
        default=None,
        type=str,
        help="path for img files for robust retrieval evaluation"
    )

    parser.add_argument("--retrieval-coco-robust", action="store_true", default=False)
    parser.add_argument("--calculate-mgap-metrics", action="store_true", default=False)
    parser.add_argument("--save-sae-latents-stats", action="store_true", default=False)
    parser.add_argument("--save-clipdissect-embeddings", action="store_true", default=False)
    parser.add_argument("--viz-masking", action="store_true", default=False)

    parser.add_argument("--get-additional-coco-robust", action="store_true", default=False)
    parser.add_argument("--get-additional-coco-robust-img", action="store_true", default=False)

    parser.add_argument("--dump-cossims", action="store_true", default=False)
    parser.add_argument(
        "--save-dir",
        default=None,
        type=str,
        help="path to save stuff"
    )

    parser.add_argument("--dump-retrieval-stats", action="store_true", default=False)
    parser.add_argument("--use-unpooled-features", action="store_true", default=False)
    parser.add_argument("--dump-in-float16", action="store_true", default=False)
    
    parser.add_argument("--return-after-layer-11", action="store_true", default=False)
    parser.add_argument("--num-subsample-tokens", type=int, default=48,
                        help="For unpooled features: number of patch tokens randomly kept per image "
                             "(in addition to the always-kept CLS token) when dumping.")

    parser.add_argument("--save-all-cossim-buckets", action="store_true", default=False)


    parser.add_argument(
        "--cossims-dir-for-original-clip", type=str, default="")


    parser.add_argument("--use-openai-model", action="store_true", default=False)

    parser.add_argument("--cyclic-training", action="store_true", default=False)

    parser.add_argument("--resume-from-this-checkpoint", type=str, default=False)

    parser.add_argument("--hybrid-sae-decoder", action="store_true", default=False)

    parser.add_argument("--use-smartclip-model", action="store_true", default=False)


    return parser


def parse_args(args=None, parser=None):
    # Allow parser to be created outside
    parser = get_parser(parser)
    args = parser.parse_args(args)

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args
