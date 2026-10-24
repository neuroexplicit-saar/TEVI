#!/bin/bash
# Train CLIP (MGAP-ViT-256-18) on MAD.
#
# Usage: bash experiments/mad/01_train_clip.sh [extra flags]
#
# Reads MAD_DATA and RUN_ROOT from configs/paths.env, or from the environment. The model is
# written to RUN_ROOT/MGAP-ViT-256-18/MAD/<run name>/Contents. With DRY_RUN=1, the command is
# printed instead of run.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PATHS_ENV="${PATHS_ENV:-$REPO/configs/paths.env}"
if [ -f "$PATHS_ENV" ]; then
    source "$PATHS_ENV"
fi

: "${MAD_DATA:?set MAD_DATA to the MAD dataset folder}"
: "${RUN_ROOT:?set RUN_ROOT to the run tree}"

CMD=(python -m open_clip_train.main
    --save-frequency 40 --zeroshot-frequency 40 --val-frequency 40
    --train-data "$MAD_DATA" --val-data "$MAD_DATA" --mad-val "$MAD_DATA"
    --logs "$RUN_ROOT" --train-clip --load-dataset-name MAD
    --warmup-epochs 5 --batch-size 128 --lr 5e-4 --wd 0.1 --epochs 200 --workers 8
    --model MGAP-ViT-256-18 --dataset-type mad --mad-mode object_attribute_3 --embed-dim 18
    --image-mean 0.1033 --image-std 0.2913
    --filter-algo-name Identity --filter-arch-name Identity
    "$@")

cd "$REPO"
if [ -n "${DRY_RUN:-}" ]; then
    printf '%s\n' "${CMD[@]}"
    exit 0
fi
"${CMD[@]}"
