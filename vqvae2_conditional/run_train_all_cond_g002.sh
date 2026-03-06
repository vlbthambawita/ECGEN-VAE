#!/usr/bin/env bash
# Combined script to train Conditional VQ-VAE-2 and Transformer Priors in one run
#
# Pipeline: Train VQ-VAE-2 -> Extract Codes -> Train Top Prior -> Train Bottom Prior
# No sampling at the end.
#
# Usage:
#   ./run_train_all_cond_g002.sh           # Run full pipeline (VQ-VAE-2 + priors)
#   ./run_train_all_cond_g002.sh extract   # Skip VQ-VAE-2, start from extract step
#   ./run_train_all_cond_g002.sh priors    # Alias for starting from extract step
#
# Override via environment:
#   EXP_NAME, SEED, RUNS_ROOT, DATA_DIR, MAX_SAMPLES
#   GPUS (sets both DEVICES and GPUS for sub-scripts)
#   MAX_EPOCHS, TOP_MAX_EPOCHS, BOT_MAX_EPOCHS, etc.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}/../vqvae2:${PYTHONPATH:-}"

# -----------------------------------------------------------------------------
# Shared configuration
# -----------------------------------------------------------------------------
DATA_DIR="${DATA_DIR:-/work/vajira/data/mimic_iv_original/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0}"
EXP_NAME="${EXP_NAME:-cond_vqvae2_mimic}"
SEED="${SEED:-42}"
RUNS_ROOT="${RUNS_ROOT:-runs}"
# Leave MAX_SAMPLES empty by default so it is only passed when explicitly set.
MAX_SAMPLES="${MAX_SAMPLES:-}"

# GPU: prior script uses GPUS, vqvae2 uses DEVICES
GPUS="${GPUS:-0}"
export DEVICES="${DEVICES:-$GPUS}"
export GPUS="$GPUS"

# Export for sub-scripts
export DATA_DIR EXP_NAME SEED RUNS_ROOT MAX_SAMPLES

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
    echo ""
}
print_info() { echo "[INFO] $1"; }
print_error() { echo "[ERROR] $1" >&2; }

# -----------------------------------------------------------------------------
# Pre-flight
# -----------------------------------------------------------------------------
[ -f "vqvae2_conditional.py" ] || { print_error "vqvae2_conditional.py not found"; exit 1; }
[ -f "cond_transformer_prior.py" ] || { print_error "cond_transformer_prior.py not found"; exit 1; }

# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
MODE="${1:-all}"  # all | extract | priors

print_header "Combined Training: VQ-VAE-2 + Transformer Priors (mode: $MODE)"

# Step 1: Train Conditional VQ-VAE-2 (optional based on MODE)
if [ "$MODE" = "all" ]; then
    print_header "Step 1/4: Training Conditional VQ-VAE-2"
    ./run_train_vqvae2_cond.sh fit
else
    print_info "Skipping Step 1/4 (VQ-VAE-2 training) because MODE='$MODE'"
fi

# Step 2-4: Prior pipeline (extract -> fit_top -> fit_bot)
export VQVAE_CKPT="${RUNS_ROOT}/${EXP_NAME}/seed_${SEED}/checkpoints/last.ckpt"
export CODES_DIR="codes/${EXP_NAME}"

if [ ! -f "$VQVAE_CKPT" ]; then
    print_error "VQ-VAE checkpoint not found: $VQVAE_CKPT"
    exit 1
fi

print_header "Step 2/4: Extracting Codes"
./run_train_prior_cond.sh extract

print_header "Step 3/4: Training Conditional Top Prior"
./run_train_prior_cond.sh fit_top

print_header "Step 4/4: Training Conditional Bottom Prior"
./run_train_prior_cond.sh fit_bot

print_header "Done! Both models trained."
