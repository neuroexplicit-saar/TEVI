#!/bin/bash
# Save the inputs of the SAE and mask training into the folder of the trained CLIP model:
# the image features with their attributes, the caption features with their multi-hot labels,
# the zero-shot templates, the text embeddings of the MAD concepts (the CG-SAE decoder) and
# the zero-shot embeddings of each attribute.
#
# Usage: bash experiments/mad/02_save_features.sh
#
# Reads MAD_DATA and RUN_ROOT from configs/paths.env, or from the environment. With DRY_RUN=1,
# the commands are printed instead of run.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PATHS_ENV="${PATHS_ENV:-$REPO/configs/paths.env}"
if [ -f "$PATHS_ENV" ]; then
    source "$PATHS_ENV"
fi
source "$REPO/experiments/mad/_names.sh"

: "${MAD_DATA:?set MAD_DATA to the MAD dataset folder}"
: "${RUN_ROOT:?set RUN_ROOT to the run tree}"

# The scripts read the model from, and write into, LOGS/NAME (NAME/checkpoints holds the model).
CLIP_ARGS=(--logs "$RUN_ROOT/$(dirname "$CLIP_RUN")" --name "$(basename "$CLIP_RUN")/Contents"
    --train-data "$MAD_DATA" --val-data "$MAD_DATA"
    --batch-size 128 --lr 5e-4 --wd 0.1 --epochs 200 --workers 8
    --model MGAP-ViT-256-18 --dataset-type mad --mad-mode object_attribute_3 --embed-dim 18
    --image-mean 0.1033 --image-std 0.2913)

CMDS=(
    "python scripts/save_features/save_openclip_features_mad.py --save-img-features"
    "python scripts/save_features/save_openclip_features_mad.py --save-text-features"
    "python scripts/save_features/save_openclip_features_mad.py --save-template-for-zeroshot"
    "python scripts/save_features/save_openclip_features_text_clipdissect_mad.py --specific-checkpoint-path 200"
    "python scripts/save_features/save_zeroshot_eval_embeddings_mad.py"
)

cd "$REPO"
for c in "${CMDS[@]}"; do
    read -r -a CMD <<< "$c"
    if [ -n "${DRY_RUN:-}" ]; then
        printf '%s\n' "${CMD[@]}" "${CLIP_ARGS[@]}"
        echo
        continue
    fi
    "${CMD[@]}" "${CLIP_ARGS[@]}"
done
