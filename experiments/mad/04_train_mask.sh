#!/bin/bash
# Train the text-conditioned mask on a trained SAE.
#   ARM=cgsae  mask on the CG-SAE
#   ARM=tevi   TEVI, the mask on the SAE with a learnt decoder
#
# Usage: bash experiments/mad/04_train_mask.sh ARM [SEED] [extra flags]
#   SEED is 0, 324, 1278, 765, 8743, or all (default).
#
# Reads MAD_DATA, RUN_ROOT and, optionally, LOAD_ROOT (default: RUN_ROOT) from
# configs/paths.env, or from the environment. With DRY_RUN=1, the commands are printed instead
# of run.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PATHS_ENV="${PATHS_ENV:-$REPO/configs/paths.env}"
if [ -f "$PATHS_ENV" ]; then
    source "$PATHS_ENV"
fi
source "$REPO/experiments/mad/_names.sh"

ARM="${1:?usage: $0 cgsae|tevi [SEED|all] [extra flags]}"
SEED_ARG="${2:-all}"
shift $(( $# < 2 ? $# : 2 ))

: "${MAD_DATA:?set MAD_DATA to the MAD dataset folder}"
: "${RUN_ROOT:?set RUN_ROOT to the run tree}"
LOAD_ROOT="${LOAD_ROOT:-$RUN_ROOT}"

case "$ARM" in
    cgsae)
        MASK_LR=1e-2
        N_LAYERS=6
        ARCH_ARGS=(--use-fixed-decoder-weights --decoder-weights-vocab mad)
        ;;
    tevi)
        MASK_LR=1e-4
        N_LAYERS=3
        ARCH_ARGS=()
        ;;
    *)
        echo "ARM must be cgsae or tevi, got '$ARM'" >&2
        exit 1
        ;;
esac

if [ "$SEED_ARG" = all ]; then
    SEEDS=(0 324 1278 765 8743)
else
    SEEDS=("$SEED_ARG")
fi

cd "$REPO"
for SEED in "${SEEDS[@]}"; do
    CMD=(python scripts/train_learntmask_textconditioning.py
        --save-frequency 400 --val-frequency 100000
        --logs "$RUN_ROOT"
        --batch-size 128 --epochs 200 --workers 8
        --dataset-type mad --model MGAP-ViT-256-18 --embed-dim 18 --mad-mode object_attribute_3
        --lr 1e-6
        --dataset-name MAD --load-dataset-name MAD
        --filter-algo-name LearntMask --filter-arch-name SAE
        --train-mask --train-mask-lr "$MASK_LR" --train-mask-epochs 25 --warmup-mask-epochs 5
        --n-layers "$N_LAYERS" --loss-name contra --train-mask-batch-size 128
        --use-pretrained-sae --expansion-factor 1
        --load-sae-batch-size 131072 --load-sae-epochs 200
        --load-k 12 --load-auxk 18 --load-sae-lr 1e-2
        --sae-pretrained-specific-checkpoint-path final
        --load-sae-logs "$LOAD_ROOT"
        ${ARCH_ARGS[@]+"${ARCH_ARGS[@]}"}
        --use-pretrained-clip --load-lr 5e-4 --load-batch-size 128 --load-workers 8
        --pretrained-specific-checkpoint-path 200
        --load-clip-logs "$LOAD_ROOT"
        --finetune-text-encoder --train-clip
        --train-data "$MAD_DATA" --val-data "$MAD_DATA" --mad-val "$MAD_DATA"
        --optimizer-name AdamW --clip-logit-scale-fixed 100.0 --use-mad-train-full-dataset
        --seed "$SEED"
        "$@")
    if [ -n "${DRY_RUN:-}" ]; then
        printf '%s\n' "${CMD[@]}"
        continue
    fi
    START="$(date '+%Y-%m-%d %H:%M:%S')"
    "${CMD[@]}"

    # Hide the per-epoch checkpoints of this run from --resume latest
    # (mgap/utils.py:get_latest_checkpoint), so that the evaluation loads the final one.
    find "$RUN_ROOT" -path "*/LearntMask/*/checkpoints/*_epoch_*.pt" -newermt "$START" -print0 |
        while IFS= read -r -d '' f; do
            mkdir -p "$(dirname "$f")/epochs"
            mv "$f" "$(dirname "$f")/epochs/"
        done
    echo "ARM=$ARM SEED=$SEED: mask written to $RUN_ROOT/$(mask_checkpoint_dir "$ARM" "$SEED")"
done
