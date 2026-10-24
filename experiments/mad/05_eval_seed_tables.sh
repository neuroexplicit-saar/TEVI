#!/bin/bash
# Evaluate the trained masks. With SEED=all, the results are averaged over the five seeds.
#
# Usage: bash experiments/mad/05_eval_seed_tables.sh ARM [SEED] [extra flags]
#   ARM is cgsae or tevi; SEED is 0, 324, 1278, 765, 8743, or all (default).
#
# Reads MAD_DATA, RUN_ROOT and, optionally, LOAD_ROOT (default: RUN_ROOT) and EVAL_CSV
# (default: RUN_ROOT/eval_mad_seed_table_ARM.csv) from configs/paths.env, or from the
# environment. With DRY_RUN=1, the commands are printed instead of run.

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
: "${RUN_ROOT:?set RUN_ROOT to the run tree with the trained masks}"
LOAD_ROOT="${LOAD_ROOT:-$RUN_ROOT}"
EVAL_CSV="${EVAL_CSV:-$RUN_ROOT/eval_mad_seed_table_${ARM}.csv}"

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
    CKPT_DIR="$RUN_ROOT/$(mask_checkpoint_dir "$ARM" "$SEED")"
    CKPT="$CKPT_DIR/$(mask_run "$ARM" "$SEED")-textconditioning.pt"

    # --precision amp is the default; it is part of the folder names (p_amp).
    CMD=(python open_clip/src/open_clip_train/custom_main.py
        --save-frequency 1 --val-frequency 100000
        --logs "$RUN_ROOT"
        --batch-size 128 --epochs 200 --workers 8
        --dataset-type mad --dataset-name MAD --load-dataset-name MAD
        --model MGAP-ViT-256-18 --embed-dim 18 --mad-mode object_attribute_3
        --lr 1e-6 --precision amp
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
        --eval-only --eval-before-train --resume latest --resume-combined-training
        --save-eval-csv --csv-path "$EVAL_CSV"
        "$@")
    if [ -n "${DRY_RUN:-}" ]; then
        printf '%s\n' "${CMD[@]}"
        continue
    fi

    # --resume latest loads the .pt file that sorts last in CKPT_DIR, so the final checkpoint
    # must be the only one there.
    N_PT=0
    if [ -d "$CKPT_DIR" ]; then
        N_PT=$(find "$CKPT_DIR" -maxdepth 1 -name '*.pt' | wc -l)
    fi
    if [ ! -f "$CKPT" ] || [ "$N_PT" -ne 1 ]; then
        echo "expected $CKPT to be the only .pt file in its folder (found $N_PT .pt files)" >&2
        exit 1
    fi

    ROWS_BEFORE=0
    if [ -f "$EVAL_CSV" ]; then
        ROWS_BEFORE=$(wc -l < "$EVAL_CSV")
    fi
    echo "evaluating ARM=$ARM SEED=$SEED from $CKPT"
    "${CMD[@]}"
    ROWS_AFTER=0
    if [ -f "$EVAL_CSV" ]; then
        ROWS_AFTER=$(wc -l < "$EVAL_CSV")
    fi
    if [ "$ROWS_AFTER" -le "$ROWS_BEFORE" ]; then
        echo "no result row was added to $EVAL_CSV" >&2
        exit 1
    fi
    echo "ARM=$ARM SEED=$SEED: result appended to $EVAL_CSV"
done

if [ -z "${DRY_RUN:-}" ] && [ "$SEED_ARG" = all ]; then
    python scripts/eval_scripts/make_mad_seed_table.py "$EVAL_CSV"
fi
