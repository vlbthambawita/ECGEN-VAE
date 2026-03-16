#!/usr/bin/env bash
# Train and test NVAE-ECG on PTB-XL with wandb logging.
#
# Uses the shared PTBXLDataset from data/dataset.py. Validation steps log
# reconstruction and generated sample panels to wandb.
#
# Usage:
#   ./run_nvae_ptbxl.sh [train|test|train_and_test]
#
# Default: train_and_test (train then run test on last checkpoint).
# Set RESUME=true to continue training from last.ckpt.
# All configuration via environment variables below.
#
# Requirements: run from ECGEN-VAE project root with dependencies installed
#   (torch, pytorch-lightning, wfdb, pandas, matplotlib, wandb, pillow).
# Set PTBXL_DIR to the PTB-XL root containing ptbxl_database.csv and records500/.

set -euo pipefail

# Project root (parent of nvae/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Optional: reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ============================================================================
# Configuration
# ============================================================================

# Data (REQUIRED: set PTBXL_DIR to your PTB-XL root containing ptbxl_database.csv)
PTBXL_DIR="${PTBXL_DIR:-/work/vajira/data/ptbxl/ptbxl}"

# Experiment
EXP_NAME="${EXP_NAME:-nvae_ptbxl}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/nvae/outputs/$EXP_NAME}"

# Data / dataloader
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PTBXL_MAX_SAMPLES="${PTBXL_MAX_SAMPLES:-}"   # empty = full dataset; set for debugging

# Architecture (matches Quick-start in nvae.py)
NUM_CHANNELS_ENC="${NUM_CHANNELS_ENC:-32}"
NUM_CHANNELS_DEC="${NUM_CHANNELS_DEC:-32}"
NUM_PREPROCESS_BLOCKS="${NUM_PREPROCESS_BLOCKS:-2}"
NUM_POSTPROCESS_BLOCKS="${NUM_POSTPROCESS_BLOCKS:-2}"
NUM_PREPROCESS_CELLS="${NUM_PREPROCESS_CELLS:-2}"
NUM_POSTPROCESS_CELLS="${NUM_POSTPROCESS_CELLS:-2}"
NUM_LATENT_SCALES="${NUM_LATENT_SCALES:-3}"
NUM_GROUPS_PER_SCALE="${NUM_GROUPS_PER_SCALE:-8}"
NUM_LATENT_PER_GROUP="${NUM_LATENT_PER_GROUP:-16}"
NUM_CELL_PER_COND_ENC="${NUM_CELL_PER_COND_ENC:-2}"
NUM_CELL_PER_COND_DEC="${NUM_CELL_PER_COND_DEC:-2}"

# Training
MAX_EPOCHS="${MAX_EPOCHS:-200}"
DEVICES="${DEVICES:-1}"
PRECISION="${PRECISION:-32}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0003}"
GRADIENT_CLIP_VAL="${GRADIENT_CLIP_VAL:-200.0}"

# Evaluation
NUM_IW_SAMPLES="${NUM_IW_SAMPLES:-200}"

# Weights & Biases
WANDB_OFF="${WANDB_OFF:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-nvae-ecg}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-$EXP_NAME}"
WANDB_TAGS="${WANDB_TAGS:-ptbxl}"

# Resume training from last checkpoint in OUTPUT_DIR
RESUME="${RESUME:-false}"

# Checkpoint for test-only mode (optional; otherwise uses OUTPUT_DIR/last.ckpt)
CKPT_PATH="${CKPT_PATH:-}"

# ============================================================================
# Helpers
# ============================================================================

print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
    echo ""
}

print_info() {
    echo "[INFO] $1"
}

print_error() {
    echo "[ERROR] $1" >&2
}

print_success() {
    echo "[SUCCESS] $1"
}

# ============================================================================
# Validation
# ============================================================================

validate_ptbxl_dir() {
    if [ ! -d "$PTBXL_DIR" ]; then
        print_error "PTB-XL directory does not exist: $PTBXL_DIR"
        print_error "Set PTBXL_DIR to the root containing ptbxl_database.csv"
        exit 1
    fi
    if [ ! -f "$PTBXL_DIR/ptbxl_database.csv" ]; then
        print_error "ptbxl_database.csv not found in $PTBXL_DIR"
        print_error "Download from https://physionet.org/content/ptb-xl/1.0.3/"
        exit 1
    fi
}

# ============================================================================
# Train
# ============================================================================

run_train() {
    print_header "NVAE-ECG Training (PTB-XL)"
    validate_ptbxl_dir

    print_info "PTBXL_DIR: $PTBXL_DIR"
    print_info "OUTPUT_DIR: $OUTPUT_DIR"
    print_info "BATCH_SIZE: $BATCH_SIZE  MAX_EPOCHS: $MAX_EPOCHS  DEVICES: $DEVICES"
    print_info "Wandb: project=$WANDB_PROJECT run=$WANDB_RUN_NAME"
    echo ""

    CMD="python nvae/nvae.py fit \
        --ptbxl_dir \"$PTBXL_DIR\" \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --num_channels_enc $NUM_CHANNELS_ENC \
        --num_channels_dec $NUM_CHANNELS_DEC \
        --num_preprocess_blocks $NUM_PREPROCESS_BLOCKS \
        --num_postprocess_blocks $NUM_POSTPROCESS_BLOCKS \
        --num_preprocess_cells $NUM_PREPROCESS_CELLS \
        --num_postprocess_cells $NUM_POSTPROCESS_CELLS \
        --num_latent_scales $NUM_LATENT_SCALES \
        --num_groups_per_scale $NUM_GROUPS_PER_SCALE \
        --num_latent_per_group $NUM_LATENT_PER_GROUP \
        --num_cell_per_cond_enc $NUM_CELL_PER_COND_ENC \
        --num_cell_per_cond_dec $NUM_CELL_PER_COND_DEC \
        --ada_groups --use_se --res_dist \
        --learning_rate $LEARNING_RATE \
        --warmup_epochs $WARMUP_EPOCHS \
        --weight_decay $WEIGHT_DECAY \
        --max_epochs $MAX_EPOCHS \
        --devices $DEVICES \
        --precision $PRECISION \
        --gradient_clip_val $GRADIENT_CLIP_VAL \
        --output_dir \"$OUTPUT_DIR\""

    if [ -n "$PTBXL_MAX_SAMPLES" ]; then
        CMD="$CMD --ptbxl_max_samples $PTBXL_MAX_SAMPLES"
    fi

    if [ "$WANDB_OFF" = "true" ]; then
        CMD="$CMD --wandb_off"
    else
        CMD="$CMD --wandb_project \"$WANDB_PROJECT\" --wandb_run_name \"$WANDB_RUN_NAME\""
        [ -n "$WANDB_ENTITY" ] && CMD="$CMD --wandb_entity \"$WANDB_ENTITY\""
        [ -n "$WANDB_TAGS" ] && CMD="$CMD --wandb_tags $WANDB_TAGS"
    fi

    if [ "$RESUME" = "true" ]; then
        LAST_CKPT="$OUTPUT_DIR/last.ckpt"
        if [ -f "$LAST_CKPT" ]; then
            CMD="$CMD --ckpt_path \"$LAST_CKPT\" --cont_training"
            print_info "Resuming from $LAST_CKPT"
        fi
    fi

    print_info "Executing: $CMD"
    echo ""
    eval $CMD
    print_success "Training completed."
}

# ============================================================================
# Test
# ============================================================================

run_test() {
    print_header "NVAE-ECG Test (IW-ELBO and metrics)"
    validate_ptbxl_dir

    if [ -n "$CKPT_PATH" ]; then
        USE_CKPT="$CKPT_PATH"
    else
        USE_CKPT="$OUTPUT_DIR/last.ckpt"
    fi

    if [ ! -f "$USE_CKPT" ]; then
        print_error "Checkpoint not found: $USE_CKPT"
        print_error "Run training first or set CKPT_PATH"
        exit 1
    fi

    print_info "Checkpoint: $USE_CKPT"
    print_info "PTBXL_DIR: $PTBXL_DIR"
    print_info "NUM_IW_SAMPLES: $NUM_IW_SAMPLES"
    echo ""

    CMD="python nvae/nvae.py test \
        --ptbxl_dir \"$PTBXL_DIR\" \
        --ckpt_path \"$USE_CKPT\" \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --num_channels_enc $NUM_CHANNELS_ENC \
        --num_channels_dec $NUM_CHANNELS_DEC \
        --num_preprocess_blocks $NUM_PREPROCESS_BLOCKS \
        --num_postprocess_blocks $NUM_POSTPROCESS_BLOCKS \
        --num_preprocess_cells $NUM_PREPROCESS_CELLS \
        --num_postprocess_cells $NUM_POSTPROCESS_CELLS \
        --num_latent_scales $NUM_LATENT_SCALES \
        --num_groups_per_scale $NUM_GROUPS_PER_SCALE \
        --num_latent_per_group $NUM_LATENT_PER_GROUP \
        --num_cell_per_cond_enc $NUM_CELL_PER_COND_ENC \
        --num_cell_per_cond_dec $NUM_CELL_PER_COND_DEC \
        --ada_groups --use_se --res_dist \
        --num_iw_samples $NUM_IW_SAMPLES \
        --output_dir \"$OUTPUT_DIR\""

    if [ -n "$PTBXL_MAX_SAMPLES" ]; then
        CMD="$CMD --ptbxl_max_samples $PTBXL_MAX_SAMPLES"
    fi

    if [ "$WANDB_OFF" = "true" ]; then
        CMD="$CMD --wandb_off"
    else
        CMD="$CMD --wandb_project \"$WANDB_PROJECT\" --wandb_run_name \"${WANDB_RUN_NAME}_test\""
        [ -n "$WANDB_ENTITY" ] && CMD="$CMD --wandb_entity \"$WANDB_ENTITY\""
    fi

    print_info "Executing: $CMD"
    echo ""
    eval $CMD
    print_success "Test completed."
}

# ============================================================================
# Main
# ============================================================================

MODE="${1:-train_and_test}"

case "$MODE" in
    train)
        run_train
        ;;
    test)
        run_test
        ;;
    train_and_test)
        run_train
        run_test
        ;;
    *)
        print_error "Unknown mode: $MODE"
        echo "Usage: $0 [train|test|train_and_test]"
        exit 1
        ;;
esac

print_header "Done."
