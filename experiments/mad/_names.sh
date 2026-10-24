# Folder names shared by the MAD run scripts (sourced, not run). They are the names that
# mgap/utils.py:common_init builds from the flags in these scripts.

CLIP_RUN=MGAP-ViT-256-18/MAD/model_MGAP-ViT-256-18-lr_0.0005-b_128-j_8-p_amp-mode_object_attribute_3
FT_RUN=Finetune/model_MGAP-ViT-256-18-lr_1e-06-b_128-j_8-p_amp-mode_object_attribute_3

# sae_config ARM: name of the SAE folder for ARM=cgsae|tevi
sae_config() {
    case "$1" in
        cgsae) echo lr_0.01-b_131072-e_200-k_12-auxk_18-ef_1-fdw_mad ;;
        tevi) echo lr_0.01-b_131072-e_200-k_12-auxk_18-ef_1 ;;
        *) echo "unknown ARM '$1'" >&2; return 1 ;;
    esac
}

# mask_run ARM SEED: name of the mask run folder. --clip-logit-scale-fixed 100.0 adds
# -logitscale100.0 (mgap/utils.py:get_mask_config_name), and seeds other than 0 add -seedN.
mask_run() {
    local name
    case "$1" in
        cgsae) name=masklr0.01-abltNone-e25-b128-warme5-losscontra-numlay6-FTtextenc ;;
        tevi) name=masklr0.0001-abltNone-e25-b128-warme5-losscontra-numlay3-FTtextenc ;;
        *) echo "unknown ARM '$1'" >&2; return 1 ;;
    esac
    name="${name}-logitscale100.0"
    if [ "$2" != 0 ]; then
        name="${name}-seed$2"
    fi
    echo "$name"
}

# mask_checkpoint_dir ARM SEED: folder of the mask checkpoint, relative to the run tree
mask_checkpoint_dir() {
    echo "$CLIP_RUN/$FT_RUN/SAE/$(sae_config "$1")/Frozen/LearntMask/$(mask_run "$1" "$2")/FromScratch/checkpoints"
}
