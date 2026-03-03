#!/usr/bin/env bash
# Shell script to train Conditional Transformer Priors for VQ-VAE-2 ECG generation
#
# Usage:
#   ./run_train_prior_cond.sh [extract|fit_top|fit_bot|sample]
#
# Examples:
#   ./run_train_prior_cond.sh extract
#   ./run_train_prior_cond.sh fit_top
#   ./run_train_prior_cond.sh fit_bot
#   ./run_train_prior_cond.sh sample

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}/../vqvae2:${PYTHONPATH:-}"

# Configuration
DATA_DIR="${DATA_DIR:-/work/vajira/DATA/SEARCH/MIMIC_IV_ECG_raw_v1/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0}"
VQVAE_CKPT="${VQVAE_CKPT:-runs/cond_vqvae2_mimic/seed_42/checkpoints/last.ckpt}"
CODES_DIR="${CODES_DIR:-codes/cond_vqvae2_mimic}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-32}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"

TOP_BATCH_SIZE="${TOP_BATCH_SIZE:-16}"
TOP_MAX_EPOCHS="${TOP_MAX_EPOCHS:-100}"
TOP_LR="${TOP_LR:-0.0003}"
TOP_D_MODEL="${TOP_D_MODEL:-256}"
TOP_N_LAYERS="${TOP_N_LAYERS:-8}"
TOP_N_HEADS="${TOP_N_HEADS:-8}"
TOP_COND_DIM="${TOP_COND_DIM:-128}"

BOT_BATCH_SIZE="${BOT_BATCH_SIZE:-8}"
BOT_MAX_EPOCHS="${BOT_MAX_EPOCHS:-100}"
BOT_LR="${BOT_LR:-0.0003}"
BOT_D_MODEL="${BOT_D_MODEL:-512}"
BOT_N_LAYERS="${BOT_N_LAYERS:-12}"
BOT_N_HEADS="${BOT_N_HEADS:-8}"
BOT_COND_DIM="${BOT_COND_DIM:-128}"

GPUS="${GPUS:-0}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-cond-vqvae2-prior}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"

TOP_PRIOR_CKPT="${TOP_PRIOR_CKPT:-logs/cond_top_prior/version_0/checkpoints/last.ckpt}"
BOT_PRIOR_CKPT="${BOT_PRIOR_CKPT:-logs/cond_bot_prior/version_0/checkpoints/last.ckpt}"
N_SAMPLES="${N_SAMPLES:-8}"
TOP_TEMP="${TOP_TEMP:-1.0}"
BOT_TEMP="${BOT_TEMP:-1.0}"
TOP_P="${TOP_P:-0.95}"
OUTPUT_FILE="${OUTPUT_FILE:-cond_generated_ecgs.npy}"
PLOT="${PLOT:-true}"
COND_DIM="${COND_DIM:-128}"

RR_INTERVAL="${RR_INTERVAL:-0.0}"
P_ONSET="${P_ONSET:-0.0}"
P_END="${P_END:-0.0}"
QRS_ONSET="${QRS_ONSET:-0.0}"
QRS_END="${QRS_END:-0.0}"
T_END="${T_END:-0.0}"
P_AXIS="${P_AXIS:-0.0}"
QRS_AXIS="${QRS_AXIS:-0.0}"
T_AXIS="${T_AXIS:-0.0}"

print_header() { echo ""; echo "=========================================="; echo "$1"; echo "=========================================="; echo ""; }
print_info() { echo "[INFO] $1"; }
print_error() { echo "[ERROR] $1" >&2; }

validate_data_dir() {
    [ -d "$DATA_DIR" ] || { print_error "Data directory does not exist: $DATA_DIR"; exit 1; }
    [ -f "$DATA_DIR/machine_measurements.csv" ] || { print_error "machine_measurements.csv not found"; exit 1; }
}
validate_vqvae_ckpt() {
    [ -f "$VQVAE_CKPT" ] || { print_error "Cond-VQVAE2 checkpoint not found: $VQVAE_CKPT"; exit 1; }
}
validate_codes_dir() {
    [ -d "$CODES_DIR" ] || { print_error "Codes directory does not exist: $CODES_DIR"; exit 1; }
    [ -f "$CODES_DIR/codes_top.npy" ] && [ -f "$CODES_DIR/codes_bot.npy" ] && [ -f "$CODES_DIR/features.npy" ] || \
        { print_error "Need codes_top.npy, codes_bot.npy, features.npy"; exit 1; }
}
validate_prior_ckpts() {
    [ -f "$TOP_PRIOR_CKPT" ] || { print_error "Top prior checkpoint not found"; exit 1; }
    [ -f "$BOT_PRIOR_CKPT" ] || { print_error "Bottom prior checkpoint not found"; exit 1; }
}

extract_codes() {
    print_header "Extracting Codes + Features from Cond-VQVAE2"
    validate_data_dir; validate_vqvae_ckpt
    CMD="python cond_transformer_prior.py extract --vqvae-ckpt \"$VQVAE_CKPT\" --data-dir \"$DATA_DIR\" --out-dir \"$CODES_DIR\" --batch-size $EXTRACT_BATCH_SIZE"
    [ -n "$MAX_SAMPLES" ] && CMD="$CMD --max-samples $MAX_SAMPLES"
    print_info "Executing: $CMD"; echo ""; eval $CMD
}

train_top_prior() {
    print_header "Training Conditional Top Prior"
    validate_codes_dir
    CMD="python cond_transformer_prior.py fit_top --codes-dir \"$CODES_DIR\" --batch-size $TOP_BATCH_SIZE --max-epochs $TOP_MAX_EPOCHS --lr $TOP_LR --d-model $TOP_D_MODEL --n-layers $TOP_N_LAYERS --n-heads $TOP_N_HEADS --cond-dim $TOP_COND_DIM"
    [ -n "$GPUS" ] && CMD="$CMD --gpus $GPUS"
    [ "$WANDB_ENABLED" = "true" ] && CMD="$CMD --wandb --wandb-project \"$WANDB_PROJECT\""
    [ -n "$WANDB_ENTITY" ] && CMD="$CMD --wandb-entity \"$WANDB_ENTITY\""
    [ -n "$WANDB_RUN_NAME" ] && CMD="$CMD --wandb-run-name \"$WANDB_RUN_NAME\""
    print_info "Executing: $CMD"; echo ""; eval $CMD
}

train_bottom_prior() {
    print_header "Training Conditional Bottom Prior"
    validate_codes_dir
    CMD="python cond_transformer_prior.py fit_bot --codes-dir \"$CODES_DIR\" --batch-size $BOT_BATCH_SIZE --max-epochs $BOT_MAX_EPOCHS --lr $BOT_LR --d-model $BOT_D_MODEL --n-layers $BOT_N_LAYERS --n-heads $BOT_N_HEADS --cond-dim $BOT_COND_DIM"
    [ -n "$GPUS" ] && CMD="$CMD --gpus $GPUS"
    [ "$WANDB_ENABLED" = "true" ] && CMD="$CMD --wandb --wandb-project \"$WANDB_PROJECT\""
    [ -n "$WANDB_ENTITY" ] && CMD="$CMD --wandb-entity \"$WANDB_ENTITY\""
    [ -n "$WANDB_RUN_NAME" ] && CMD="$CMD --wandb-run-name \"$WANDB_RUN_NAME\""
    print_info "Executing: $CMD"; echo ""; eval $CMD
}

sample_ecgs() {
    print_header "Generating Conditional ECG Samples"
    validate_vqvae_ckpt; validate_prior_ckpts
    CMD="python cond_transformer_prior.py sample --vqvae-ckpt \"$VQVAE_CKPT\" --top-prior-ckpt \"$TOP_PRIOR_CKPT\" --bot-prior-ckpt \"$BOT_PRIOR_CKPT\" --n-samples $N_SAMPLES --top-temp $TOP_TEMP --bot-temp $BOT_TEMP --top-p $TOP_P --cond-dim $COND_DIM --out \"$OUTPUT_FILE\" --rr-interval $RR_INTERVAL --p-onset $P_ONSET --p-end $P_END --qrs-onset $QRS_ONSET --qrs-end $QRS_END --t-end $T_END --p-axis $P_AXIS --qrs-axis $QRS_AXIS --t-axis $T_AXIS"
    [ "$PLOT" = "true" ] && CMD="$CMD --plot"
    print_info "Executing: $CMD"; echo ""; eval $CMD
}

print_header "Conditional VQ-VAE-2 Transformer Prior Training Script"
[ -f "cond_transformer_prior.py" ] || { print_error "cond_transformer_prior.py not found"; exit 1; }

COMMAND="${1:-}"
case "$COMMAND" in
    extract) extract_codes ;;
    fit_top|train_top) train_top_prior ;;
    fit_bot|train_bot) train_bottom_prior ;;
    sample|generate) sample_ecgs ;;
    *)
        print_error "Unknown command: $COMMAND"
        echo ""
        echo "Usage: $0 [extract|fit_top|fit_bot|sample]"
        echo ""
        echo "Commands:"
        echo "  extract                       Extract codes + features from Cond-VQVAE2"
        echo "  fit_top                       Train conditional top prior transformer"
        echo "  fit_bot                       Train conditional bottom prior transformer"
        echo "  sample                        Generate ECG samples (uses RR_INTERVAL, P_ONSET, etc.)"
        echo ""
        echo "Environment Variables:"
        echo "  DATA_DIR, VQVAE_CKPT, CODES_DIR"
        echo "  TOP_BATCH_SIZE, TOP_MAX_EPOCHS, TOP_LR, TOP_COND_DIM"
        echo "  BOT_BATCH_SIZE, BOT_MAX_EPOCHS, BOT_LR, BOT_COND_DIM"
        echo "  GPUS, WANDB_ENABLED, WANDB_PROJECT"
        echo "  N_SAMPLES, OUTPUT_FILE, PLOT, COND_DIM (must match prior training; default 128)"
        echo "  RR_INTERVAL, P_ONSET, P_END, QRS_ONSET, QRS_END, T_END, P_AXIS, QRS_AXIS, T_AXIS"
        echo ""
        echo "Examples:"
        echo "  $0 extract"
        echo "  $0 fit_top"
        echo "  N_SAMPLES=16 RR_INTERVAL=0.34 PLOT=true $0 sample"
        exit 1
        ;;
esac
print_header "Done!"
