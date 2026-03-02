#!/usr/bin/env bash
# Shell script to train VQ-VAE model in two stages
# Stage 1: Train VQ-VAE (encoder + vector quantizer + decoder)
# Stage 2: Train PixelCNN Prior on discrete codes
#
# Usage:
#   ./run_train_vqvae.sh [1|2|both] [vqvae_checkpoint_path]
#
# Examples:
#   ./run_train_vqvae.sh both                    # Train both stages
#   ./run_train_vqvae.sh 1                       # Train Stage 1 only
#   ./run_train_vqvae.sh 2 path/to/vqvae.ckpt    # Train Stage 2 only with checkpoint
#   VQVAE_CHECKPOINT=path/to/vqvae.ckpt ./run_train_vqvae.sh 2  # Alternative for Stage 2

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

# Data path (REQUIRED - update this to your MIMIC-IV-ECG path)
DATA_DIR="${DATA_DIR:-/work/vajira/DATA/SEARCH/MIMIC_IV_ECG_raw_v1/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0}"

# Experiment settings
EXP_NAME_STAGE1="${EXP_NAME_STAGE1:-vqvae_mimic_standalone}"
EXP_NAME_STAGE2="${EXP_NAME_STAGE2:-prior_mimic_standalone}"
SEED="${SEED:-42}"
RUNS_ROOT="${RUNS_ROOT:-runs}"

# VQ-VAE checkpoint for Stage 2 (can be set via env var or command line)
VQVAE_CHECKPOINT="${VQVAE_CHECKPOINT:-}"

# Data settings
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"  # Set to null for full dataset
VAL_SPLIT="${VAL_SPLIT:-0.1}"
TEST_SPLIT="${TEST_SPLIT:-0.1}"

# Model settings (Stage 1)
IN_CHANNELS="${IN_CHANNELS:-12}"
BASE_CHANNELS="${BASE_CHANNELS:-64}"
LATENT_CHANNELS="${LATENT_CHANNELS:-64}"
NUM_RES_BLOCKS="${NUM_RES_BLOCKS:-2}"
NUM_EMBEDDINGS="${NUM_EMBEDDINGS:-512}"
COMMITMENT_COST="${COMMITMENT_COST:-0.25}"
SEQ_LENGTH="${SEQ_LENGTH:-5000}"

# Model settings (Stage 2)
HIDDEN_DIM="${HIDDEN_DIM:-128}"
NUM_LAYERS="${NUM_LAYERS:-3}"

# Training settings
LR_STAGE1="${LR_STAGE1:-0.0001}"
LR_STAGE2="${LR_STAGE2:-0.001}"
MAX_EPOCHS_STAGE1="${MAX_EPOCHS_STAGE1:-10}"
MAX_EPOCHS_STAGE2="${MAX_EPOCHS_STAGE2:-10}"
ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-0}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-50}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
GRADIENT_CLIP="${GRADIENT_CLIP:-1.0}"
PATIENCE="${PATIENCE:-10}"
SAVE_TOP_K="${SAVE_TOP_K:-3}"

# Visualization settings (Stage 1)
VIZ_EVERY_N_EPOCHS="${VIZ_EVERY_N_EPOCHS:-5}"
VIZ_NUM_SAMPLES="${VIZ_NUM_SAMPLES:-4}"

# Weights & Biases settings
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-ecg-vqvae}"
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
TRAIN_SCRIPT="${SCRIPT_DIR}/train_vqvae_standalone.py"

if [ ! -f "${TRAIN_SCRIPT}" ]; then
    print_error "Training script not found: ${TRAIN_SCRIPT}"
    exit 1
fi

# ============================================================================
# Stage Selection and Argument Parsing
# ============================================================================

STAGE="${1:-both}"

# If second argument provided, use it as VQVAE_CHECKPOINT
if [ -n "${2:-}" ]; then
    VQVAE_CHECKPOINT="${2}"
fi

if [[ ! "${STAGE}" =~ ^(1|2|both)$ ]]; then
    print_error "Invalid stage: ${STAGE}"
    echo ""
    echo "Usage: $0 [1|2|both] [vqvae_checkpoint_path]"
    echo ""
    echo "Arguments:"
    echo "  1    - Train VQ-VAE only (Stage 1)"
    echo "  2    - Train Prior only (Stage 2) - requires VQ-VAE checkpoint"
    echo "  both - Train both stages sequentially (default)"
    echo ""
    echo "  vqvae_checkpoint_path - Path to VQ-VAE checkpoint (required for Stage 2)"
    echo ""
    echo "Examples:"
    echo "  $0 both                                    # Train both stages"
    echo "  $0 1                                       # Train Stage 1 only"
    echo "  $0 2 runs/vqvae/seed_42/checkpoints/best.ckpt  # Train Stage 2 with checkpoint"
    echo ""
    echo "Environment Variables:"
    echo "  VQVAE_CHECKPOINT - Alternative way to specify checkpoint for Stage 2"
    echo "  DATA_DIR         - Path to MIMIC-IV-ECG dataset"
    echo "  MAX_SAMPLES      - Limit dataset size (for testing)"
    echo "  WANDB_ENABLED    - Enable Weights & Biases logging (default: true)"
    echo ""
    exit 1
fi

# ============================================================================
# Pre-flight Checks
# ============================================================================

# If Stage 2 only and no checkpoint provided, try to help user find one
if [[ "${STAGE}" == "2" && -z "${VQVAE_CHECKPOINT:-}" ]]; then
    print_info "Looking for available VQ-VAE checkpoints..."
    
    # Search for checkpoints in the default location
    if [ -d "${RUNS_ROOT}" ]; then
        FOUND_CHECKPOINTS=$(find "${RUNS_ROOT}" -path "*/vqvae*/seed_*/checkpoints/*.ckpt" -type f 2>/dev/null | head -5)
        
        if [ -n "${FOUND_CHECKPOINTS}" ]; then
            echo ""
            echo "Found the following VQ-VAE checkpoints:"
            echo "${FOUND_CHECKPOINTS}" | nl
            echo ""
            echo "You can use one of these checkpoints by running:"
            FIRST_CHECKPOINT=$(echo "${FOUND_CHECKPOINTS}" | head -1)
            echo "  $0 2 ${FIRST_CHECKPOINT}"
            echo ""
        fi
    fi
fi

# ============================================================================
# Stage 1: Train VQ-VAE
# ============================================================================

if [[ "${STAGE}" == "1" || "${STAGE}" == "both" ]]; then
    print_header "STAGE 1: Training VQ-VAE"
    
    print_info "Experiment: ${EXP_NAME_STAGE1}"
    print_info "Data directory: ${DATA_DIR}"
    print_info "Batch size: ${BATCH_SIZE}"
    print_info "Max samples: ${MAX_SAMPLES}"
    print_info "Max epochs: ${MAX_EPOCHS_STAGE1}"
    print_info "Learning rate: ${LR_STAGE1}"
    print_info "Codebook size: ${NUM_EMBEDDINGS}"
    print_info "Wandb enabled: ${WANDB_ENABLED}"
    
    # Build command
    CMD="python ${TRAIN_SCRIPT} \
        --stage 1 \
        --exp-name ${EXP_NAME_STAGE1} \
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
        --num-embeddings ${NUM_EMBEDDINGS} \
        --commitment-cost ${COMMITMENT_COST} \
        --seq-length ${SEQ_LENGTH} \
        --lr ${LR_STAGE1} \
        --max-epochs ${MAX_EPOCHS_STAGE1} \
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
    
    print_info "Running command:"
    echo "${CMD}"
    echo ""
    
    # Run training
    eval "${CMD}"
    
    # Check if training succeeded
    if [ $? -ne 0 ]; then
        print_error "Stage 1 training failed"
        exit 1
    fi
    
    # Find best checkpoint
    CHECKPOINT_DIR="${RUNS_ROOT}/${EXP_NAME_STAGE1}/seed_${SEED}/checkpoints"
    BEST_CHECKPOINT=$(find "${CHECKPOINT_DIR}" -name "epoch*.ckpt" -type f | sort | tail -n 1)
    
    if [ -z "${BEST_CHECKPOINT}" ]; then
        BEST_CHECKPOINT="${CHECKPOINT_DIR}/last.ckpt"
    fi
    
    print_header "Stage 1 Complete"
    print_info "Best checkpoint: ${BEST_CHECKPOINT}"
    
    # Export for Stage 2
    export VQVAE_CHECKPOINT="${BEST_CHECKPOINT}"
fi

# ============================================================================
# Stage 2: Train Prior
# ============================================================================

if [[ "${STAGE}" == "2" || "${STAGE}" == "both" ]]; then
    print_header "STAGE 2: Training PixelCNN Prior"
    
    # Check if VQ-VAE checkpoint is provided
    if [ -z "${VQVAE_CHECKPOINT:-}" ]; then
        print_error "VQ-VAE checkpoint not specified for Stage 2"
        echo ""
        echo "Please provide the checkpoint in one of these ways:"
        echo "  1. Command line argument:"
        echo "     $0 2 runs/vqvae_mimic_standalone/seed_42/checkpoints/best.ckpt"
        echo ""
        echo "  2. Environment variable:"
        echo "     VQVAE_CHECKPOINT=runs/vqvae_mimic_standalone/seed_42/checkpoints/best.ckpt $0 2"
        echo ""
        echo "  3. Run Stage 1 first (it will automatically pass checkpoint to Stage 2):"
        echo "     $0 both"
        echo ""
        exit 1
    fi
    
    if [ ! -f "${VQVAE_CHECKPOINT}" ]; then
        print_error "VQ-VAE checkpoint not found: ${VQVAE_CHECKPOINT}"
        echo ""
        echo "Please check that the file exists and the path is correct."
        echo "If you haven't trained Stage 1 yet, run:"
        echo "  $0 1"
        echo ""
        exit 1
    fi
    
    print_info "Experiment: ${EXP_NAME_STAGE2}"
    print_info "VQ-VAE checkpoint: ${VQVAE_CHECKPOINT}"
    print_info "Data directory: ${DATA_DIR}"
    print_info "Batch size: ${BATCH_SIZE}"
    print_info "Max samples: ${MAX_SAMPLES}"
    print_info "Max epochs: ${MAX_EPOCHS_STAGE2}"
    print_info "Learning rate: ${LR_STAGE2}"
    print_info "Wandb enabled: ${WANDB_ENABLED}"
    
    # Build command
    CMD="python ${TRAIN_SCRIPT} \
        --stage 2 \
        --exp-name ${EXP_NAME_STAGE2} \
        --seed ${SEED} \
        --runs-root ${RUNS_ROOT} \
        --data-dir ${DATA_DIR} \
        --batch-size ${BATCH_SIZE} \
        --num-workers ${NUM_WORKERS} \
        --val-split ${VAL_SPLIT} \
        --test-split ${TEST_SPLIT} \
        --num-embeddings ${NUM_EMBEDDINGS} \
        --hidden-dim ${HIDDEN_DIM} \
        --num-layers ${NUM_LAYERS} \
        --seq-length ${SEQ_LENGTH} \
        --lr ${LR_STAGE2} \
        --max-epochs ${MAX_EPOCHS_STAGE2} \
        --accelerator ${ACCELERATOR} \
        --devices ${DEVICES} \
        --log-every-n-steps ${LOG_EVERY_N_STEPS} \
        --check-val-every-n-epoch ${CHECK_VAL_EVERY_N_EPOCH} \
        --gradient-clip ${GRADIENT_CLIP} \
        --patience ${PATIENCE} \
        --save-top-k ${SAVE_TOP_K} \
        --vqvae-checkpoint ${VQVAE_CHECKPOINT}"
    
    # Add wandb if enabled
    if [ "${WANDB_ENABLED}" = "true" ]; then
        CMD="${CMD} --wandb"
        CMD="${CMD} --wandb-project ${WANDB_PROJECT}"
        
        if [ -n "${WANDB_ENTITY}" ]; then
            CMD="${CMD} --wandb-entity ${WANDB_ENTITY}"
        fi
        
        if [ -n "${WANDB_RUN_NAME}" ]; then
            CMD="${CMD} --wandb-run-name ${WANDB_RUN_NAME}_stage2"
        fi
        
        if [ -n "${WANDB_TAGS}" ]; then
            CMD="${CMD} --wandb-tags ${WANDB_TAGS}"
        fi
    fi
    
    # Add max-samples if specified
    if [ -n "${MAX_SAMPLES}" ] && [ "${MAX_SAMPLES}" != "null" ]; then
        CMD="${CMD} --max-samples ${MAX_SAMPLES}"
    fi
    
    print_info "Running command:"
    echo "${CMD}"
    echo ""
    
    # Run training
    eval "${CMD}"
    
    # Check if training succeeded
    if [ $? -ne 0 ]; then
        print_error "Stage 2 training failed"
        exit 1
    fi
    
    # Find best checkpoint
    CHECKPOINT_DIR="${RUNS_ROOT}/${EXP_NAME_STAGE2}/seed_${SEED}/checkpoints"
    BEST_CHECKPOINT=$(find "${CHECKPOINT_DIR}" -name "epoch*.ckpt" -type f | sort | tail -n 1)
    
    if [ -z "${BEST_CHECKPOINT}" ]; then
        BEST_CHECKPOINT="${CHECKPOINT_DIR}/last.ckpt"
    fi
    
    print_header "Stage 2 Complete"
    print_info "Best checkpoint: ${BEST_CHECKPOINT}"
fi

# ============================================================================
# Summary
# ============================================================================

print_header "Training Complete!"

if [[ "${STAGE}" == "1" || "${STAGE}" == "both" ]]; then
    echo "Stage 1 (VQ-VAE) results:"
    echo "  - Checkpoints: ${RUNS_ROOT}/${EXP_NAME_STAGE1}/seed_${SEED}/checkpoints/"
    echo "  - Samples: ${RUNS_ROOT}/${EXP_NAME_STAGE1}/seed_${SEED}/samples/"
    echo "  - TensorBoard logs: ${RUNS_ROOT}/${EXP_NAME_STAGE1}/seed_${SEED}/tb/"
    echo ""
fi

if [[ "${STAGE}" == "2" || "${STAGE}" == "both" ]]; then
    echo "Stage 2 (Prior) results:"
    echo "  - Checkpoints: ${RUNS_ROOT}/${EXP_NAME_STAGE2}/seed_${SEED}/checkpoints/"
    echo "  - TensorBoard logs: ${RUNS_ROOT}/${EXP_NAME_STAGE2}/seed_${SEED}/tb/"
    echo ""
fi

echo "To view training progress:"
echo "  TensorBoard: tensorboard --logdir=${RUNS_ROOT}"
if [ "${WANDB_ENABLED}" = "true" ]; then
    echo "  Weights & Biases: https://wandb.ai/${WANDB_ENTITY:-your-username}/${WANDB_PROJECT}"
fi
echo ""

if [[ "${STAGE}" == "both" ]]; then
    echo "Both stages completed successfully!"
    echo "You can now use the trained models for ECG generation."
fi

echo ""
echo "Quick Reference:"
echo "  Train Stage 1 only:  $0 1"
echo "  Train Stage 2 only:  $0 2 <checkpoint_path>"
echo "  Train both stages:   $0 both"
echo ""
