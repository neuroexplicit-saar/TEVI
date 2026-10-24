#!/bin/bash
# Train an SAE on the cached CLIP image features of MAD.
#   ARM=cgsae  CG-SAE, whose decoder is fixed to the text embeddings of the MAD concepts
#   ARM=tevi   SAE with a learnt decoder, used by TEVI
#
# Usage: bash experiments/mad/03_train_sae.sh ARM [extra flags]
#
# Reads RUN_ROOT and, optionally, LOAD_ROOT (default: RUN_ROOT) from configs/paths.env, or from
# the environment. With DRY_RUN=1, the command is printed instead of run.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PATHS_ENV="${PATHS_ENV:-$REPO/configs/paths.env}"
if [ -f "$PATHS_ENV" ]; then
    source "$PATHS_ENV"
fi

ARM="${1:?usage: $0 cgsae|tevi [extra flags]}"
shift

: "${RUN_ROOT:?set RUN_ROOT to the run tree}"
LOAD_ROOT="${LOAD_ROOT:-$RUN_ROOT}"

case "$ARM" in
    cgsae)
        SEED=49
        ARCH_ARGS=(--use-fixed-decoder-weights --decoder-weights-vocab mad)
        ;;
    tevi)
        SEED=0
        ARCH_ARGS=()
        ;;
    *)
        echo "ARM must be cgsae or tevi, got '$ARM'" >&2
        exit 1
        ;;
esac

CMD=(python scripts/train_sae_mad_or_cc3m.py
    --train-sae --train-sae-epochs 200 --train-sae-batch-size 131072
    --save-frequency 40 --val-frequency 1
    --logs "$RUN_ROOT" --load-clip-logs "$LOAD_ROOT"
    --load-batch-size 128 --epochs 200 --load-workers 8
    --dataset-type mad --dataset-name MAD --load-dataset-name MAD
    --auxk 18 --model MGAP-ViT-256-18 --embed-dim 18 --mad-mode object_attribute_3
    --load-lr 5e-4 --k 12 --expansion-factor 1 --train-sae-lr 1e-2
    --use-pretrained-clip --filter-arch-name SAE --pretrained-specific-checkpoint-path 200
    ${ARCH_ARGS[@]+"${ARCH_ARGS[@]}"}
    --seed "$SEED"
    "$@")

cd "$REPO"
if [ -n "${DRY_RUN:-}" ]; then
    printf '%s\n' "${CMD[@]}"
    exit 0
fi
"${CMD[@]}"
