#!/usr/bin/env bash
# Unified training script for VQ-VAE-2 and Transformer Priors
#
# This script runs the complete training pipeline:
#   1. Train VQ-VAE-2 model
#   2. Extract codes from trained VQ-VAE-2
#   3. Train top prior transformer
#   4. Train bottom prior transformer
#
# Usage:
#   ./run_full_training.sh
#
# All configuration is done via environment variables (see below)

set -euo pipefail

# Enable expandable segments to reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================================
# Configuration - VQ-VAE-2
# ============================================================================

# Data path (REQUIRED - update this to your MIMIC-IV-ECG path)
DATA_DIR="${DATA_DIR:-/work/vajira/data/mimic_iv_original/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0}"

# Experiment settings
EXP_NAME="${EXP_NAME:-vqvae2_mimic}"
SEED="${SEED:-42}"
RUNS_ROOT="${RUNS_ROOT:-runs}"

# Data settings
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-null}"  # Set to null for full dataset
VAL_SPLIT="${VAL_SPLIT:-0.1}"
TEST_SPLIT="${TEST_SPLIT:-0.1}"

# Model settings
N_LEADS="${N_LEADS:-12}"
SIGNAL_LEN="${SIGNAL_LEN:-5000}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-128}"
RESIDUAL_CHANNELS="${RESIDUAL_CHANNELS:-64}"
N_RES_BLOCKS="${N_RES_BLOCKS:-4}"
N_EMBEDDINGS_TOP="${N_EMBEDDINGS_TOP:-512}"
N_EMBEDDINGS_BOT="${N_EMBEDDINGS_BOT:-512}"
EMBEDDING_DIM="${EMBEDDING_DIM:-64}"
COMMITMENT_COST="${COMMITMENT_COST:-0.25}"
EMA_DECAY="${EMA_DECAY:-0.99}"

# Training settings
LR="${LR:-0.0003}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-0}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-50}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
GRADIENT_CLIP="${GRADIENT_CLIP:-1.0}"
PATIENCE="${PATIENCE:-10}"
SAVE_TOP_K="${SAVE_TOP_K:-3}"

# Visualization settings
VIZ_EVERY_N_EPOCHS="${VIZ_EVERY_N_EPOCHS:-5}"
VIZ_NUM_SAMPLES="${VIZ_NUM_SAMPLES:-4}"

# Weights & Biases settings
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-ecg-vqvae2}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_TAGS="${WANDB_TAGS:-}"

# ============================================================================
# Configuration - Prior Training
# ============================================================================

# Codes directory
CODES_DIR="${CODES_DIR:-codes/vqvae2_mimic}"

# Extraction settings
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-32}"

# Training settings - Top Prior
TOP_BATCH_SIZE="${TOP_BATCH_SIZE:-16}"
TOP_MAX_EPOCHS="${TOP_MAX_EPOCHS:-100}"
TOP_LR="${TOP_LR:-0.0003}"
TOP_D_MODEL="${TOP_D_MODEL:-256}"
TOP_N_LAYERS="${TOP_N_LAYERS:-8}"
TOP_N_HEADS="${TOP_N_HEADS:-8}"

# Training settings - Bottom Prior
BOT_BATCH_SIZE="${BOT_BATCH_SIZE:-8}"
BOT_MAX_EPOCHS="${BOT_MAX_EPOCHS:-100}"
BOT_LR="${BOT_LR:-0.0003}"
BOT_D_MODEL="${BOT_D_MODEL:-512}"
BOT_N_LAYERS="${BOT_N_LAYERS:-12}"
BOT_N_HEADS="${BOT_N_HEADS:-8}"
# Resume from checkpoint (optional): set to path of last.ckpt to continue training
BOT_CKPT_PATH="${BOT_CKPT_PATH:-}"

# Weights & Biases settings for Prior
WANDB_PROJECT_PRIOR="${WANDB_PROJECT_PRIOR:-vqvae2-prior}"

# Resume VQ-VAE from checkpoint: set VQVAE_CKPT_PATH or RESUME_VQVAE=true to use last.ckpt
VQVAE_CKPT_PATH="${VQVAE_CKPT_PATH:-/work/vajira/DL2026/ECGEN-VAE/vqvae2/runs/vqvae2_mimic/seed_42/checkpoints/last.ckpt}"
RESUME_VQVAE="${RESUME_VQVAE:-true}"

# Resume/skip: set RESUME_STEP=4 to run only step 4 (useful with BOT_CKPT_PATH)
RESUME_STEP="${RESUME_STEP:-}"

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
    echo ""
}

print_stage() {
    echo ""
    echo "##########################################"
    echo "# STAGE $1"
    echo "##########################################"
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
# Validation Functions
# ============================================================================

validate_data_dir() {
    if [ ! -d "$DATA_DIR" ]; then
        print_error "Data directory does not exist: $DATA_DIR"
        print_error "Please set DATA_DIR environment variable or update the script"
        exit 1
    fi
    
    if [ ! -f "$DATA_DIR/machine_measurements.csv" ]; then
        print_error "machine_measurements.csv not found in $DATA_DIR"
        print_error "Please ensure you have the correct MIMIC-IV-ECG dataset path"
        exit 1
    fi
}

validate_vqvae_ckpt() {
    local CKPT_PATH="$1"
    if [ ! -f "$CKPT_PATH" ]; then
        print_error "VQ-VAE-2 checkpoint not found: $CKPT_PATH"
        exit 1
    fi
}

validate_codes_dir() {
    if [ ! -d "$CODES_DIR" ]; then
        print_error "Codes directory does not exist: $CODES_DIR"
        print_error "Code extraction may have failed"
        exit 1
    fi
    
    if [ ! -f "$CODES_DIR/codes_top.npy" ] || [ ! -f "$CODES_DIR/codes_bot.npy" ]; then
        print_error "Code files not found in $CODES_DIR"
        print_error "Code extraction may have failed"
        exit 1
    fi
}

# ============================================================================
# Stage 1: Train VQ-VAE-2
# ============================================================================

train_vqvae2() {
    print_stage "1/4: Training VQ-VAE-2"
    
    validate_data_dir
    
    # Resolve VQ-VAE checkpoint path for resume
    local VQVAE_RESUME_CKPT=""
    if [ "$RESUME_VQVAE" = "true" ] && [ -z "$VQVAE_CKPT_PATH" ]; then
        VQVAE_CKPT_PATH="$RUNS_ROOT/$EXP_NAME/seed_$SEED/checkpoints/last.ckpt"
    fi
    if [ -n "$VQVAE_CKPT_PATH" ] && [ -f "$VQVAE_CKPT_PATH" ]; then
        VQVAE_RESUME_CKPT="$VQVAE_CKPT_PATH"
        print_info "Resuming VQ-VAE from: $VQVAE_RESUME_CKPT"
    fi
    
    print_info "Configuration:"
    print_info "  Data directory: $DATA_DIR"
    print_info "  Experiment name: $EXP_NAME"
    print_info "  Batch size: $BATCH_SIZE"
    print_info "  Max epochs: $MAX_EPOCHS"
    print_info "  Learning rate: $LR"
    print_info "  Hidden channels: $HIDDEN_CHANNELS"
    print_info "  Embedding dim: $EMBEDDING_DIM"
    print_info "  Top codebook size: $N_EMBEDDINGS_TOP"
    print_info "  Bottom codebook size: $N_EMBEDDINGS_BOT"
    print_info "  Devices: $DEVICES"
    echo ""
    
    # Build command
    CMD="python vqvae2.py fit \
        --data-dir \"$DATA_DIR\" \
        --exp-name \"$EXP_NAME\" \
        --seed $SEED \
        --runs-root \"$RUNS_ROOT\" \
        --batch-size $BATCH_SIZE \
        --num-workers $NUM_WORKERS \
        --val-split $VAL_SPLIT \
        --test-split $TEST_SPLIT \
        --n-leads $N_LEADS \
        --signal-len $SIGNAL_LEN \
        --hidden-channels $HIDDEN_CHANNELS \
        --residual-channels $RESIDUAL_CHANNELS \
        --n-res-blocks $N_RES_BLOCKS \
        --n-embeddings-top $N_EMBEDDINGS_TOP \
        --n-embeddings-bot $N_EMBEDDINGS_BOT \
        --embedding-dim $EMBEDDING_DIM \
        --commitment-cost $COMMITMENT_COST \
        --ema-decay $EMA_DECAY \
        --lr $LR \
        --max-epochs $MAX_EPOCHS \
        --accelerator $ACCELERATOR \
        --devices $DEVICES \
        --log-every-n-steps $LOG_EVERY_N_STEPS \
        --check-val-every-n-epoch $CHECK_VAL_EVERY_N_EPOCH \
        --gradient-clip $GRADIENT_CLIP \
        --patience $PATIENCE \
        --save-top-k $SAVE_TOP_K \
        --viz-every-n-epochs $VIZ_EVERY_N_EPOCHS \
        --viz-num-samples $VIZ_NUM_SAMPLES"
    
    # Add max samples if not null
    if [ "$MAX_SAMPLES" != "null" ]; then
        CMD="$CMD --max-samples $MAX_SAMPLES"
    fi
    
    # Add checkpoint path for resume
    if [ -n "$VQVAE_RESUME_CKPT" ]; then
        CMD="$CMD --ckpt-path \"$VQVAE_RESUME_CKPT\""
    fi
    
    # Add W&B flags if enabled
    if [ "$WANDB_ENABLED" = "true" ]; then
        CMD="$CMD --wandb --wandb-project \"$WANDB_PROJECT\""
        
        if [ -n "$WANDB_ENTITY" ]; then
            CMD="$CMD --wandb-entity \"$WANDB_ENTITY\""
        fi
        
        if [ -n "$WANDB_RUN_NAME" ]; then
            CMD="$CMD --wandb-run-name \"$WANDB_RUN_NAME\""
        fi
        
        if [ -n "$WANDB_TAGS" ]; then
            CMD="$CMD --wandb-tags $WANDB_TAGS"
        fi
    fi
    
    print_info "Executing: $CMD"
    echo ""
    
    eval $CMD
    
    print_success "VQ-VAE-2 training completed!"
}

# ============================================================================
# Stage 2: Extract Codes
# ============================================================================

extract_codes() {
    local VQVAE_CKPT="$1"
    
    print_stage "2/4: Extracting Codes from VQ-VAE-2"
    
    validate_data_dir
    validate_vqvae_ckpt "$VQVAE_CKPT"
    
    print_info "Configuration:"
    print_info "  VQ-VAE-2 checkpoint: $VQVAE_CKPT"
    print_info "  Data directory: $DATA_DIR"
    print_info "  Output directory: $CODES_DIR"
    print_info "  Batch size: $EXTRACT_BATCH_SIZE"
    if [ -n "$MAX_SAMPLES" ] && [ "$MAX_SAMPLES" != "null" ]; then
        print_info "  Max samples: $MAX_SAMPLES"
    else
        print_info "  Max samples: (all)"
    fi
    echo ""
    
    # Build command
    CMD="python transformer_prior.py extract \
        --vqvae-ckpt \"$VQVAE_CKPT\" \
        --data-dir \"$DATA_DIR\" \
        --out-dir \"$CODES_DIR\" \
        --batch-size $EXTRACT_BATCH_SIZE"
    
    # Add max samples if set
    if [ -n "$MAX_SAMPLES" ] && [ "$MAX_SAMPLES" != "null" ]; then
        CMD="$CMD --max-samples $MAX_SAMPLES"
    fi
    
    print_info "Executing: $CMD"
    echo ""
    
    eval $CMD
    
    print_success "Code extraction completed!"
}

# ============================================================================
# Stage 3: Train Top Prior
# ============================================================================

train_top_prior() {
    print_stage "3/4: Training Top Prior"
    
    validate_codes_dir
    
    print_info "Configuration:"
    print_info "  Codes directory: $CODES_DIR"
    print_info "  Batch size: $TOP_BATCH_SIZE"
    print_info "  Max epochs: $TOP_MAX_EPOCHS"
    print_info "  Learning rate: $TOP_LR"
    print_info "  Model dimension: $TOP_D_MODEL"
    print_info "  Number of layers: $TOP_N_LAYERS"
    print_info "  Number of heads: $TOP_N_HEADS"
    print_info "  GPU: $DEVICES"
    print_info "  W&B enabled: $WANDB_ENABLED"
    echo ""
    
    CMD="python transformer_prior.py fit_top \
        --codes-dir \"$CODES_DIR\" \
        --batch-size $TOP_BATCH_SIZE \
        --max-epochs $TOP_MAX_EPOCHS \
        --lr $TOP_LR \
        --d-model $TOP_D_MODEL \
        --n-layers $TOP_N_LAYERS \
        --n-heads $TOP_N_HEADS"
    
    if [ -n "$DEVICES" ]; then
        CMD="$CMD --gpus $DEVICES"
    fi
    
    # Add W&B flags if enabled
    if [ "$WANDB_ENABLED" = "true" ]; then
        CMD="$CMD --wandb --wandb-project \"$WANDB_PROJECT_PRIOR\""
        
        if [ -n "$WANDB_ENTITY" ]; then
            CMD="$CMD --wandb-entity \"$WANDB_ENTITY\""
        fi
        
        if [ -n "$WANDB_RUN_NAME" ]; then
            CMD="$CMD --wandb-run-name \"${WANDB_RUN_NAME}_top\""
        fi
    fi
    
    print_info "Executing: $CMD"
    echo ""
    
    eval $CMD
    
    print_success "Top prior training completed!"
}

# ============================================================================
# Stage 4: Train Bottom Prior
# ============================================================================

train_bottom_prior() {
    print_stage "4/4: Training Bottom Prior"
    
    validate_codes_dir
    
    print_info "Configuration:"
    print_info "  Codes directory: $CODES_DIR"
    print_info "  Batch size: $BOT_BATCH_SIZE"
    print_info "  Max epochs: $BOT_MAX_EPOCHS"
    print_info "  Learning rate: $BOT_LR"
    print_info "  Model dimension: $BOT_D_MODEL"
    print_info "  Number of layers: $BOT_N_LAYERS"
    print_info "  Number of heads: $BOT_N_HEADS"
    print_info "  GPU: $DEVICES"
    print_info "  W&B enabled: $WANDB_ENABLED"
    echo ""
    
    CMD="python transformer_prior.py fit_bot \
        --codes-dir \"$CODES_DIR\" \
        --batch-size $BOT_BATCH_SIZE \
        --max-epochs $BOT_MAX_EPOCHS \
        --lr $BOT_LR \
        --d-model $BOT_D_MODEL \
        --n-layers $BOT_N_LAYERS \
        --n-heads $BOT_N_HEADS"
    
    if [ -n "$BOT_CKPT_PATH" ]; then
        CMD="$CMD --ckpt-path \"$BOT_CKPT_PATH\""
        print_info "Resuming from checkpoint: $BOT_CKPT_PATH"
    fi
    
    if [ -n "$DEVICES" ]; then
        CMD="$CMD --gpus $DEVICES"
    fi
    
    # Add W&B flags if enabled
    if [ "$WANDB_ENABLED" = "true" ]; then
        CMD="$CMD --wandb --wandb-project \"$WANDB_PROJECT_PRIOR\""
        
        if [ -n "$WANDB_ENTITY" ]; then
            CMD="$CMD --wandb-entity \"$WANDB_ENTITY\""
        fi
        
        if [ -n "$WANDB_RUN_NAME" ]; then
            CMD="$CMD --wandb-run-name \"${WANDB_RUN_NAME}_bot\""
        fi
    fi
    
    print_info "Executing: $CMD"
    echo ""
    
    eval $CMD
    
    print_success "Bottom prior training completed!"
}

# ============================================================================
# Main Pipeline
# ============================================================================

print_header "VQ-VAE-2 Full Training Pipeline"

# Check if required Python scripts exist
if [ ! -f "vqvae2.py" ]; then
    print_error "vqvae2.py not found in current directory"
    print_error "Please run this script from the vqvae2 directory"
    exit 1
fi

if [ ! -f "transformer_prior.py" ]; then
    print_error "transformer_prior.py not found in current directory"
    print_error "Please run this script from the vqvae2 directory"
    exit 1
fi

print_info "Starting full training pipeline with 4 stages:"
print_info "  1. Train VQ-VAE-2"
print_info "  2. Extract codes"
print_info "  3. Train top prior"
print_info "  4. Train bottom prior"
if [ -n "$RESUME_STEP" ]; then
    print_info "RESUME_STEP=$RESUME_STEP: skipping stages 1–$((RESUME_STEP - 1))"
fi
echo ""

if [ -z "$RESUME_STEP" ] || [ "$RESUME_STEP" -le 1 ]; then
    # Stage 1: Train VQ-VAE-2
    train_vqvae2
fi

# Determine VQ-VAE-2 checkpoint path
VQVAE_CKPT="$RUNS_ROOT/$EXP_NAME/seed_$SEED/checkpoints/last.ckpt"
if [ -z "$RESUME_STEP" ] || [ "$RESUME_STEP" -le 2 ]; then
    # Validate checkpoint exists
    if [ ! -f "$VQVAE_CKPT" ]; then
        print_error "Expected checkpoint not found: $VQVAE_CKPT"
        print_error "Training may have failed or checkpoint saved to different location"
        exit 1
    fi
    # Stage 2: Extract codes
    extract_codes "$VQVAE_CKPT"
fi

if [ -z "$RESUME_STEP" ] || [ "$RESUME_STEP" -le 3 ]; then
    # Stage 3: Train top prior
    train_top_prior
fi

# Stage 4: Train bottom prior
train_bottom_prior

# ============================================================================
# Summary
# ============================================================================

print_header "Training Pipeline Complete!"

print_info "Summary:"
print_info "  ✓ VQ-VAE-2 trained and saved to: $VQVAE_CKPT"
print_info "  ✓ Codes extracted to: $CODES_DIR"
print_info "  ✓ Top prior trained"
print_info "  ✓ Bottom prior trained"
echo ""
print_info "Next steps:"
print_info "  - To generate samples: ./run_train_prior.sh sample"
print_info "  - To plot samples: ./run_train_prior.sh plot"
echo ""

print_header "All Done!"
