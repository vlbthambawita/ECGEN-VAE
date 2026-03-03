#!/usr/bin/env bash
# Shell script to train Conditional VQ-VAE-2 model for ECG generation
#
# Usage:
#   ./run_train_vqvae2_cond.sh [fit|test|sample]
#
# Examples:
#   ./run_train_vqvae2_cond.sh fit                           # Train Conditional VQ-VAE-2
#   ./run_train_vqvae2_cond.sh test path/to/checkpoint.ckpt  # Test model
#   ./run_train_vqvae2_cond.sh sample path/to/checkpoint.ckpt # Generate samples (requires 9 feature env vars)

set -euo pipefail

# Resolve script directory and set PYTHONPATH for vqvae2 import
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}/../vqvae2:${PYTHONPATH:-}"

# ============================================================================
# Configuration
# ============================================================================

# Data path (REQUIRED - update this to your MIMIC-IV-ECG path)
DATA_DIR="${DATA_DIR:-/work/vajira/DATA/SEARCH/MIMIC_IV_ECG_raw_v1/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0}"

# Experiment settings
EXP_NAME="${EXP_NAME:-cond_vqvae2_mimic}"
SEED="${SEED:-42}"
RUNS_ROOT="${RUNS_ROOT:-runs}"

# Data settings
BATCH_SIZE="${BATCH_SIZE:-32}"
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
COND_DIM="${COND_DIM:-128}"

# Training settings
LR="${LR:-0.0003}"
B1="${B1:-0.9}"
B2="${B2:-0.999}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-0}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-50}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
GRADIENT_CLIP="${GRADIENT_CLIP:-1.0}"
PATIENCE="${PATIENCE:-15}"
SAVE_TOP_K="${SAVE_TOP_K:-3}"

# Visualization settings
VIZ_EVERY_N_EPOCHS="${VIZ_EVERY_N_EPOCHS:-5}"
VIZ_NUM_SAMPLES="${VIZ_NUM_SAMPLES:-4}"

# Weights & Biases settings
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-ecg-cond-vqvae2}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_TAGS="${WANDB_TAGS:-}"

# Sampling settings
N_SAMPLES="${N_SAMPLES:-8}"
TEMPERATURE="${TEMPERATURE:-1.0}"
OUTPUT_FILE="${OUTPUT_FILE:-cond_samples.npy}"
PLOT="${PLOT:-false}"

# Clinical feature inputs (normalized) for sampling - default 0.0
RR_INTERVAL="${RR_INTERVAL:-0.0}"
P_ONSET="${P_ONSET:-0.0}"
P_END="${P_END:-0.0}"
QRS_ONSET="${QRS_ONSET:-0.0}"
QRS_END="${QRS_END:-0.0}"
T_END="${T_END:-0.0}"
P_AXIS="${P_AXIS:-0.0}"
QRS_AXIS="${QRS_AXIS:-0.0}"
T_AXIS="${T_AXIS:-0.0}"

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

# ============================================================================
# Training Commands
# ============================================================================

train_cond_vqvae2() {
    print_header "Training Conditional VQ-VAE-2"

    validate_data_dir

    print_info "Configuration:"
    print_info "  Data directory: $DATA_DIR"
    print_info "  Experiment name: $EXP_NAME"
    print_info "  Batch size: $BATCH_SIZE"
    print_info "  Max epochs: $MAX_EPOCHS"
    print_info "  Learning rate: $LR"
    print_info "  Condition dim: $COND_DIM"
    print_info "  Hidden channels: $HIDDEN_CHANNELS"
    print_info "  Embedding dim: $EMBEDDING_DIM"
    print_info "  Top codebook size: $N_EMBEDDINGS_TOP"
    print_info "  Bottom codebook size: $N_EMBEDDINGS_BOT"
    print_info "  Devices: $DEVICES"
    echo ""

    # Build command
    CMD="python vqvae2_conditional.py fit \
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
        --cond-dim $COND_DIM \
        --lr $LR \
        --b1 $B1 \
        --b2 $B2 \
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
}

test_cond_vqvae2() {
    local CKPT_PATH="$1"

    print_header "Testing Conditional VQ-VAE-2"

    if [ -z "$CKPT_PATH" ]; then
        print_error "Checkpoint path is required for testing"
        print_error "Usage: $0 test <checkpoint_path>"
        exit 1
    fi

    if [ ! -f "$CKPT_PATH" ]; then
        print_error "Checkpoint not found: $CKPT_PATH"
        exit 1
    fi

    validate_data_dir

    print_info "Configuration:"
    print_info "  Checkpoint: $CKPT_PATH"
    print_info "  Data directory: $DATA_DIR"
    print_info "  Devices: $DEVICES"
    echo ""

    CMD="python vqvae2_conditional.py test \
        --data-dir \"$DATA_DIR\" \
        --ckpt-path \"$CKPT_PATH\" \
        --batch-size $BATCH_SIZE \
        --num-workers $NUM_WORKERS \
        --seed $SEED \
        --accelerator $ACCELERATOR \
        --devices $DEVICES"

    print_info "Executing: $CMD"
    echo ""

    eval $CMD
}

sample_cond_vqvae2() {
    local CKPT_PATH="$1"

    print_header "Sampling from Conditional VQ-VAE-2"

    if [ -z "$CKPT_PATH" ]; then
        print_error "Checkpoint path is required for sampling"
        print_error "Usage: $0 sample <checkpoint_path>"
        exit 1
    fi

    if [ ! -f "$CKPT_PATH" ]; then
        print_error "Checkpoint not found: $CKPT_PATH"
        exit 1
    fi

    print_info "Configuration:"
    print_info "  Checkpoint: $CKPT_PATH"
    print_info "  Number of samples: $N_SAMPLES"
    print_info "  Temperature: $TEMPERATURE"
    print_info "  Output file: $OUTPUT_FILE"
    print_info "  Condition: RR=$RR_INTERVAL P_onset=$P_ONSET P_end=$P_END QRS_onset=$QRS_ONSET QRS_end=$QRS_END T_end=$T_END P_axis=$P_AXIS QRS_axis=$QRS_AXIS T_axis=$T_AXIS"
    echo ""

    CMD="python vqvae2_conditional.py sample \
        --ckpt-path \"$CKPT_PATH\" \
        --n-samples $N_SAMPLES \
        --temperature $TEMPERATURE \
        --out \"$OUTPUT_FILE\" \
        --rr-interval $RR_INTERVAL \
        --p-onset $P_ONSET \
        --p-end $P_END \
        --qrs-onset $QRS_ONSET \
        --qrs-end $QRS_END \
        --t-end $T_END \
        --p-axis $P_AXIS \
        --qrs-axis $QRS_AXIS \
        --t-axis $T_AXIS"

    if [ "$PLOT" = "true" ]; then
        CMD="$CMD --plot"
    fi

    print_info "Executing: $CMD"
    echo ""

    eval $CMD
}

# ============================================================================
# Main
# ============================================================================

print_header "Conditional VQ-VAE-2 Training Script"

# Check if vqvae2_conditional.py exists
if [ ! -f "vqvae2_conditional.py" ]; then
    print_error "vqvae2_conditional.py not found in current directory"
    print_error "Please run this script from the vqvae2_conditional directory"
    exit 1
fi

# Parse command
COMMAND="${1:-}"

case "$COMMAND" in
    fit|train)
        train_cond_vqvae2
        ;;
    test|eval)
        CKPT_PATH="${2:-}"
        test_cond_vqvae2 "$CKPT_PATH"
        ;;
    sample|generate)
        CKPT_PATH="${2:-}"
        sample_cond_vqvae2 "$CKPT_PATH"
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        echo ""
        echo "Usage: $0 [fit|test|sample] [checkpoint_path]"
        echo ""
        echo "Commands:"
        echo "  fit                           Train Conditional VQ-VAE-2 model"
        echo "  test <checkpoint_path>        Test a trained model"
        echo "  sample <checkpoint_path>      Generate samples from a trained model (uses RR_INTERVAL, P_ONSET, etc.)"
        echo ""
        echo "Environment Variables:"
        echo "  DATA_DIR                      Path to MIMIC-IV-ECG dataset"
        echo "  EXP_NAME                      Experiment name (default: cond_vqvae2_mimic)"
        echo "  COND_DIM                      Condition embedding dimension (default: 128)"
        echo "  BATCH_SIZE                    Batch size (default: 32)"
        echo "  MAX_EPOCHS                    Maximum epochs (default: 200)"
        echo "  LR                            Learning rate (default: 0.0003)"
        echo "  WANDB_ENABLED                 Enable W&B logging (default: true)"
        echo "  DEVICES                       GPU device IDs (default: 0)"
        echo "  RR_INTERVAL, P_ONSET, ...     Normalized clinical features for sampling (default: 0.0)"
        echo ""
        echo "Examples:"
        echo "  $0 fit"
        echo "  BATCH_SIZE=64 MAX_EPOCHS=100 $0 fit"
        echo "  $0 test runs/cond_vqvae2_mimic/seed_42/checkpoints/last.ckpt"
        echo "  RR_INTERVAL=0.34 P_ONSET=-0.56 N_SAMPLES=8 $0 sample runs/cond_vqvae2_mimic/seed_42/checkpoints/last.ckpt"
        exit 1
        ;;
esac

print_header "Done!"
