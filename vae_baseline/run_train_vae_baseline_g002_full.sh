#!/usr/bin/env bash
# Shell script to train VAE baseline model
# Supports fresh training and resuming from checkpoints
#
# Usage:
#   ./run_train_vae_baseline.sh [train|resume]
#
# Examples:
#   ./run_train_vae_baseline.sh              # Auto-detect (resume if checkpoint exists)
#   ./run_train_vae_baseline.sh train        # Start fresh training
#   ./run_train_vae_baseline.sh resume       # Resume from last checkpoint
#   RESUME_CHECKPOINT=path/to/ckpt ./run_train_vae_baseline.sh resume  # Resume from specific checkpoint

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

# Data path (REQUIRED - update this to your MIMIC-IV-ECG path)
DATA_DIR="${DATA_DIR:-/work/vajira/data/mimic_iv_original/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0}"

# Experiment settings
EXP_NAME="${EXP_NAME:-vae_baseline_mimic}"
SEED="${SEED:-42}"
RUNS_ROOT="${RUNS_ROOT:-runs}"

# Resume checkpoint (can be set via env var)
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

# Data settings
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-null}"  # Set to null for full dataset
VAL_SPLIT="${VAL_SPLIT:-0.1}"
TEST_SPLIT="${TEST_SPLIT:-0.1}"

# Model settings
IN_CHANNELS="${IN_CHANNELS:-12}"
BASE_CHANNELS="${BASE_CHANNELS:-64}"
LATENT_CHANNELS="${LATENT_CHANNELS:-8}"
NUM_RES_BLOCKS="${NUM_RES_BLOCKS:-2}"
KL_WEIGHT="${KL_WEIGHT:-0.0001}"
SEQ_LENGTH="${SEQ_LENGTH:-5000}"

# Training settings
LR="${LR:-0.0001}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-0}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-1000}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
GRADIENT_CLIP="${GRADIENT_CLIP:-1.0}"
PATIENCE="${PATIENCE:-10}"
SAVE_TOP_K="${SAVE_TOP_K:-3}"

# Visualization settings
VIZ_EVERY_N_EPOCHS="${VIZ_EVERY_N_EPOCHS:-5}"
VIZ_NUM_SAMPLES="${VIZ_NUM_SAMPLES:-4}"

# Weights & Biases settings
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-ecg-vae-baseline}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_TAGS="${WANDB_TAGS:-}"

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

print_info() {
    echo "[INFO] $1"
}

print_error() {
    echo "[ERROR] $1" >&2
}

# ============================================================================
# Validation
# ============================================================================

# Check if data directory exists
if [ ! -d "${DATA_DIR}" ]; then
    print_error "Data directory not found: ${DATA_DIR}"
    print_error "Please set DATA_DIR environment variable or update the script"
    exit 1
fi

# Check if training script exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_vae_baseline.py"

if [ ! -f "${TRAIN_SCRIPT}" ]; then
    print_error "Training script not found: ${TRAIN_SCRIPT}"
    exit 1
fi

# ============================================================================
# Mode Selection and Argument Parsing
# ============================================================================

MODE="${1:-auto}"

if [[ ! "${MODE}" =~ ^(train|resume|auto)$ ]]; then
    print_error "Invalid mode: ${MODE}"
    echo ""
    echo "Usage: $0 [train|resume]"
    echo ""
    echo "Arguments:"
    echo "  train  - Start fresh training"
    echo "  resume - Resume from checkpoint"
    echo "  (none) - Auto-detect (resume if checkpoint exists, else train)"
    echo ""
    echo "Examples:"
    echo "  $0                    # Auto-detect mode"
    echo "  $0 train              # Start fresh training"
    echo "  $0 resume             # Resume from last checkpoint"
    echo ""
    echo "Environment Variables:"
    echo "  RESUME_CHECKPOINT - Path to checkpoint for resume mode"
    echo "  DATA_DIR          - Path to MIMIC-IV-ECG dataset"
    echo "  MAX_SAMPLES       - Limit dataset size (for testing)"
    echo "  WANDB_ENABLED     - Enable Weights & Biases logging (default: true)"
    echo ""
    exit 1
fi

# Determine checkpoint directory
CHECKPOINT_DIR="${RUNS_ROOT}/${EXP_NAME}/seed_${SEED}/checkpoints"
LAST_CHECKPOINT="${CHECKPOINT_DIR}/last.ckpt"

# Handle mode logic
if [ "${MODE}" = "auto" ]; then
    if [ -f "${LAST_CHECKPOINT}" ]; then
        print_info "Found existing checkpoint: ${LAST_CHECKPOINT}"
        print_info "Auto-detecting mode: resume"
        MODE="resume"
        RESUME_CHECKPOINT="${LAST_CHECKPOINT}"
    else
        print_info "No checkpoint found, starting fresh training"
        MODE="train"
    fi
elif [ "${MODE}" = "resume" ]; then
    # If resume mode but no checkpoint specified, try to find last.ckpt
    if [ -z "${RESUME_CHECKPOINT}" ]; then
        if [ -f "${LAST_CHECKPOINT}" ]; then
            RESUME_CHECKPOINT="${LAST_CHECKPOINT}"
            print_info "Using checkpoint: ${RESUME_CHECKPOINT}"
        else
            print_error "Resume mode requested but no checkpoint found"
            echo ""
            echo "Checkpoint not found at: ${LAST_CHECKPOINT}"
            echo ""
            echo "Please either:"
            echo "  1. Specify checkpoint explicitly:"
            echo "     RESUME_CHECKPOINT=path/to/checkpoint.ckpt $0 resume"
            echo ""
            echo "  2. Train the model first:"
            echo "     $0 train"
            echo ""
            exit 1
        fi
    elif [ ! -f "${RESUME_CHECKPOINT}" ]; then
        print_error "Specified checkpoint not found: ${RESUME_CHECKPOINT}"
        exit 1
    fi
fi

# ============================================================================
# Training
# ============================================================================

print_header "Training VAE Baseline Model"

print_info "Mode: ${MODE}"
print_info "Experiment: ${EXP_NAME}"
print_info "Data directory: ${DATA_DIR}"
print_info "Batch size: ${BATCH_SIZE}"
print_info "Max samples: ${MAX_SAMPLES}"
print_info "Max epochs: ${MAX_EPOCHS}"
print_info "Learning rate: ${LR}"
print_info "KL weight: ${KL_WEIGHT}"
print_info "Wandb enabled: ${WANDB_ENABLED}"

if [ "${MODE}" = "resume" ]; then
    print_info "Resuming from: ${RESUME_CHECKPOINT}"
fi

# Build command
CMD="python ${TRAIN_SCRIPT} \
    --exp-name ${EXP_NAME} \
    --seed ${SEED} \
    --runs-root ${RUNS_ROOT} \
    --data-dir ${DATA_DIR} \
    --batch-size ${BATCH_SIZE} \
    --num-workers ${NUM_WORKERS} \
    --val-split ${VAL_SPLIT} \
    --test-split ${TEST_SPLIT} \
    --in-channels ${IN_CHANNELS} \
    --base-channels ${BASE_CHANNELS} \
    --latent-channels ${LATENT_CHANNELS} \
    --num-res-blocks ${NUM_RES_BLOCKS} \
    --kl-weight ${KL_WEIGHT} \
    --seq-length ${SEQ_LENGTH} \
    --lr ${LR} \
    --max-epochs ${MAX_EPOCHS} \
    --accelerator ${ACCELERATOR} \
    --devices ${DEVICES} \
    --log-every-n-steps ${LOG_EVERY_N_STEPS} \
    --check-val-every-n-epoch ${CHECK_VAL_EVERY_N_EPOCH} \
    --gradient-clip ${GRADIENT_CLIP} \
    --patience ${PATIENCE} \
    --save-top-k ${SAVE_TOP_K} \
    --viz-every-n-epochs ${VIZ_EVERY_N_EPOCHS} \
    --viz-num-samples ${VIZ_NUM_SAMPLES}"

# Add wandb if enabled
if [ "${WANDB_ENABLED}" = "true" ]; then
    CMD="${CMD} --wandb"
    CMD="${CMD} --wandb-project ${WANDB_PROJECT}"
    
    if [ -n "${WANDB_ENTITY}" ]; then
        CMD="${CMD} --wandb-entity ${WANDB_ENTITY}"
    fi
    
    if [ -n "${WANDB_RUN_NAME}" ]; then
        CMD="${CMD} --wandb-run-name ${WANDB_RUN_NAME}"
    fi
    
    if [ -n "${WANDB_TAGS}" ]; then
        CMD="${CMD} --wandb-tags ${WANDB_TAGS}"
    fi
fi

# Add max-samples if specified
if [ -n "${MAX_SAMPLES}" ] && [ "${MAX_SAMPLES}" != "null" ]; then
    CMD="${CMD} --max-samples ${MAX_SAMPLES}"
fi

# Add resume checkpoint if in resume mode
if [ "${MODE}" = "resume" ] && [ -n "${RESUME_CHECKPOINT}" ]; then
    CMD="${CMD} --resume ${RESUME_CHECKPOINT}"
fi

print_info "Running command:"
echo "${CMD}"
echo ""

# Run training
eval "${CMD}"

# Check if training succeeded
if [ $? -ne 0 ]; then
    print_error "Training failed"
    exit 1
fi

# ============================================================================
# Summary
# ============================================================================

print_header "Training Complete!"

echo "Results:"
echo "  - Checkpoints: ${CHECKPOINT_DIR}/"
echo "  - Samples: ${RUNS_ROOT}/${EXP_NAME}/seed_${SEED}/samples/"
echo "  - TensorBoard logs: ${RUNS_ROOT}/${EXP_NAME}/seed_${SEED}/tb/"
echo ""

echo "To view training progress:"
echo "  TensorBoard: tensorboard --logdir=${RUNS_ROOT}"
if [ "${WANDB_ENABLED}" = "true" ]; then
    echo "  Weights & Biases: https://wandb.ai/${WANDB_ENTITY:-your-username}/${WANDB_PROJECT}"
fi
echo ""

echo "To resume training:"
echo "  $0 resume"
echo ""

echo "To start fresh training:"
echo "  $0 train"
echo ""

# Find best checkpoint
BEST_CHECKPOINT=$(find "${CHECKPOINT_DIR}" -name "epoch*.ckpt" -type f 2>/dev/null | sort | tail -n 1)

if [ -n "${BEST_CHECKPOINT}" ]; then
    echo "Best checkpoint: ${BEST_CHECKPOINT}"
elif [ -f "${LAST_CHECKPOINT}" ]; then
    echo "Last checkpoint: ${LAST_CHECKPOINT}"
fi

echo ""

