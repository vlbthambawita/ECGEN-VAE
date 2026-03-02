#!/usr/bin/env bash
# Shell script to train Transformer Priors for VQ-VAE-2 ECG generation
#
# Usage:
#   ./run_train_prior.sh [extract|fit_top|fit_bot|sample]
#
# Examples:
#   ./run_train_prior.sh extract                    # Extract codes from VQ-VAE-2
#   ./run_train_prior.sh fit_top                    # Train top prior
#   ./run_train_prior.sh fit_bot                    # Train bottom prior
#   ./run_train_prior.sh sample                     # Generate ECG samples

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

# Data path (REQUIRED - update this to your MIMIC-IV-ECG path)
DATA_DIR="${DATA_DIR:-/work/vajira/DATA/SEARCH/MIMIC_IV_ECG_raw_v1/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0}"

# VQ-VAE-2 checkpoint (REQUIRED for extract and sample)
VQVAE_CKPT="${VQVAE_CKPT:-runs/vqvae2_mimic/seed_42/checkpoints/last.ckpt}"

# Codes directory
CODES_DIR="${CODES_DIR:-codes/vqvae2_mimic}"

# Experiment settings
EXP_NAME="${EXP_NAME:-prior_vqvae2}"
SEED="${SEED:-42}"

# Extraction settings
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-32}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"  # Empty for full dataset

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

# GPU settings
GPUS="${GPUS:-0}"

# Weights & Biases settings
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-vqvae2-prior}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"

# Prior checkpoints (for sampling)
TOP_PRIOR_CKPT="${TOP_PRIOR_CKPT:-logs/top_prior/version_0/checkpoints/last.ckpt}"
BOT_PRIOR_CKPT="${BOT_PRIOR_CKPT:-logs/bot_prior/version_0/checkpoints/last.ckpt}"

# Sampling settings
N_SAMPLES="${N_SAMPLES:-16}"
TOP_TEMP="${TOP_TEMP:-1.0}"
BOT_TEMP="${BOT_TEMP:-1.0}"
TOP_P="${TOP_P:-0.95}"
OUTPUT_FILE="${OUTPUT_FILE:-generated_ecgs.npy}"

# Plotting settings
PLOT_DIR="${PLOT_DIR:-plots}"
PLOT_N_SAMPLES="${PLOT_N_SAMPLES:-}"
PLOT_PREFIX="${PLOT_PREFIX:-ecg_}"
PLOT_STYLE="${PLOT_STYLE:-}"
PLOT_COLUMNS="${PLOT_COLUMNS:-2}"

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

validate_vqvae_ckpt() {
    if [ ! -f "$VQVAE_CKPT" ]; then
        print_error "VQ-VAE-2 checkpoint not found: $VQVAE_CKPT"
        print_error "Please train VQ-VAE-2 first or set VQVAE_CKPT to correct path"
        exit 1
    fi
}

validate_codes_dir() {
    if [ ! -d "$CODES_DIR" ]; then
        print_error "Codes directory does not exist: $CODES_DIR"
        print_error "Please run 'extract' command first"
        exit 1
    fi
    
    if [ ! -f "$CODES_DIR/codes_top.npy" ] || [ ! -f "$CODES_DIR/codes_bot.npy" ]; then
        print_error "Code files not found in $CODES_DIR"
        print_error "Please run 'extract' command first"
        exit 1
    fi
}

validate_prior_ckpts() {
    if [ ! -f "$TOP_PRIOR_CKPT" ]; then
        print_error "Top prior checkpoint not found: $TOP_PRIOR_CKPT"
        print_error "Please train top prior first or set TOP_PRIOR_CKPT"
        exit 1
    fi
    
    if [ ! -f "$BOT_PRIOR_CKPT" ]; then
        print_error "Bottom prior checkpoint not found: $BOT_PRIOR_CKPT"
        print_error "Please train bottom prior first or set BOT_PRIOR_CKPT"
        exit 1
    fi
}

# ============================================================================
# Commands
# ============================================================================

extract_codes() {
    print_header "Extracting Codes from VQ-VAE-2"
    
    validate_data_dir
    validate_vqvae_ckpt
    
    print_info "Configuration:"
    print_info "  VQ-VAE-2 checkpoint: $VQVAE_CKPT"
    print_info "  Data directory: $DATA_DIR"
    print_info "  Output directory: $CODES_DIR"
    print_info "  Batch size: $EXTRACT_BATCH_SIZE"
    if [ -n "$MAX_SAMPLES" ]; then
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
    if [ -n "$MAX_SAMPLES" ]; then
        CMD="$CMD --max-samples $MAX_SAMPLES"
    fi
    
    print_info "Executing: $CMD"
    echo ""
    
    eval $CMD
}

train_top_prior() {
    print_header "Training Top Prior"
    
    validate_codes_dir
    
    print_info "Configuration:"
    print_info "  Codes directory: $CODES_DIR"
    print_info "  Batch size: $TOP_BATCH_SIZE"
    print_info "  Max epochs: $TOP_MAX_EPOCHS"
    print_info "  Learning rate: $TOP_LR"
    print_info "  Model dimension: $TOP_D_MODEL"
    print_info "  Number of layers: $TOP_N_LAYERS"
    print_info "  Number of heads: $TOP_N_HEADS"
    print_info "  GPU: $GPUS"
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
    
    if [ -n "$GPUS" ]; then
        CMD="$CMD --gpus $GPUS"
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
    fi
    
    print_info "Executing: $CMD"
    echo ""
    
    eval $CMD
}

train_bottom_prior() {
    print_header "Training Bottom Prior"
    
    validate_codes_dir
    
    print_info "Configuration:"
    print_info "  Codes directory: $CODES_DIR"
    print_info "  Batch size: $BOT_BATCH_SIZE"
    print_info "  Max epochs: $BOT_MAX_EPOCHS"
    print_info "  Learning rate: $BOT_LR"
    print_info "  Model dimension: $BOT_D_MODEL"
    print_info "  Number of layers: $BOT_N_LAYERS"
    print_info "  Number of heads: $BOT_N_HEADS"
    print_info "  GPU: $GPUS"
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
    
    if [ -n "$GPUS" ]; then
        CMD="$CMD --gpus $GPUS"
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
    fi
    
    print_info "Executing: $CMD"
    echo ""
    
    eval $CMD
}

sample_ecgs() {
    print_header "Generating ECG Samples"
    
    validate_vqvae_ckpt
    validate_prior_ckpts
    
    print_info "Configuration:"
    print_info "  VQ-VAE-2 checkpoint: $VQVAE_CKPT"
    print_info "  Top prior checkpoint: $TOP_PRIOR_CKPT"
    print_info "  Bottom prior checkpoint: $BOT_PRIOR_CKPT"
    print_info "  Number of samples: $N_SAMPLES"
    print_info "  Top temperature: $TOP_TEMP"
    print_info "  Bottom temperature: $BOT_TEMP"
    print_info "  Top-p (nucleus): $TOP_P"
    print_info "  Output file: $OUTPUT_FILE"
    echo ""
    
    CMD="python transformer_prior.py sample \
        --vqvae-ckpt \"$VQVAE_CKPT\" \
        --top-prior-ckpt \"$TOP_PRIOR_CKPT\" \
        --bot-prior-ckpt \"$BOT_PRIOR_CKPT\" \
        --n-samples $N_SAMPLES \
        --top-temp $TOP_TEMP \
        --bot-temp $BOT_TEMP \
        --top-p $TOP_P \
        --out \"$OUTPUT_FILE\""
    
    print_info "Executing: $CMD"
    echo ""
    
    eval $CMD
}

plot_ecgs() {
    print_header "Plotting Generated ECGs"
    
    if [ ! -f "$OUTPUT_FILE" ]; then
        print_error "ECG file not found: $OUTPUT_FILE"
        print_error "Please run 'sample' command first or set OUTPUT_FILE to correct path"
        exit 1
    fi
    
    print_info "Configuration:"
    print_info "  Input file: $OUTPUT_FILE"
    print_info "  Output directory: $PLOT_DIR"
    print_info "  Filename prefix: $PLOT_PREFIX"
    if [ -n "$PLOT_N_SAMPLES" ]; then
        print_info "  Number of plots: $PLOT_N_SAMPLES"
    else
        print_info "  Number of plots: (all)"
    fi
    if [ -n "$PLOT_STYLE" ]; then
        print_info "  Plot style: $PLOT_STYLE"
    fi
    print_info "  Columns: $PLOT_COLUMNS"
    echo ""
    
    CMD="python plot_ecgs.py --input \"$OUTPUT_FILE\" --output-dir \"$PLOT_DIR\" --prefix \"$PLOT_PREFIX\" --columns $PLOT_COLUMNS"
    
    if [ -n "$PLOT_N_SAMPLES" ]; then
        CMD="$CMD --n-samples $PLOT_N_SAMPLES"
    fi
    
    if [ -n "$PLOT_STYLE" ]; then
        CMD="$CMD --style \"$PLOT_STYLE\""
    fi
    
    print_info "Executing: $CMD"
    echo ""
    
    eval $CMD
}

# ============================================================================
# Main
# ============================================================================

print_header "VQ-VAE-2 Transformer Prior Training Script"

# Check if transformer_prior.py exists
if [ ! -f "transformer_prior.py" ]; then
    print_error "transformer_prior.py not found in current directory"
    print_error "Please run this script from the vqvae2 directory"
    exit 1
fi

# Parse command
COMMAND="${1:-}"

case "$COMMAND" in
    extract)
        extract_codes
        ;;
    fit_top|train_top)
        train_top_prior
        ;;
    fit_bot|train_bot)
        train_bottom_prior
        ;;
    sample|generate)
        sample_ecgs
        ;;
    plot)
        plot_ecgs
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        echo ""
        echo "Usage: $0 [extract|fit_top|fit_bot|sample|plot]"
        echo ""
        echo "Commands:"
        echo "  extract                       Extract codes from trained VQ-VAE-2"
        echo "  fit_top                       Train top prior transformer"
        echo "  fit_bot                       Train bottom prior transformer"
        echo "  sample                        Generate ECG samples using trained priors"
        echo "  plot                          Plot generated ECG samples"
        echo ""
        echo "Environment Variables:"
        echo "  DATA_DIR                      Path to MIMIC-IV-ECG dataset"
        echo "  VQVAE_CKPT                    Path to VQ-VAE-2 checkpoint"
        echo "  CODES_DIR                     Directory for extracted codes"
        echo "  TOP_BATCH_SIZE                Top prior batch size (default: 16)"
        echo "  TOP_MAX_EPOCHS                Top prior max epochs (default: 100)"
        echo "  TOP_LR                        Top prior learning rate (default: 0.0003)"
        echo "  BOT_BATCH_SIZE                Bottom prior batch size (default: 16)"
        echo "  BOT_MAX_EPOCHS                Bottom prior max epochs (default: 100)"
        echo "  BOT_LR                        Bottom prior learning rate (default: 0.0003)"
        echo "  GPUS                          GPU device ID (default: 0)"
        echo "  WANDB_ENABLED                 Enable W&B logging (default: true)"
        echo "  WANDB_PROJECT                 W&B project name (default: vqvae2-prior)"
        echo "  WANDB_ENTITY                  W&B entity (username/team)"
        echo "  WANDB_RUN_NAME                W&B run name"
        echo "  N_SAMPLES                     Number of samples to generate (default: 16)"
        echo "  OUTPUT_FILE                   Output file for generated ECGs (default: generated_ecgs.npy)"
        echo "  PLOT_DIR                      Output directory for plots (default: plots)"
        echo "  PLOT_N_SAMPLES                Number of samples to plot (default: all)"
        echo "  PLOT_PREFIX                   Filename prefix for plots (default: ecg_)"
        echo "  PLOT_STYLE                    Plot style: standard or 'bw' (default: standard)"
        echo "  PLOT_COLUMNS                  Number of columns in plot layout (default: 2)"
        echo ""
        echo "Examples:"
        echo "  $0 extract"
        echo "  $0 fit_top"
        echo "  $0 fit_bot"
        echo "  WANDB_ENABLED=false $0 fit_top"
        echo "  N_SAMPLES=32 TOP_TEMP=0.8 $0 sample"
        echo "  $0 plot"
        echo "  PLOT_N_SAMPLES=8 PLOT_STYLE=bw $0 plot"
        exit 1
        ;;
esac

print_header "Done!"
