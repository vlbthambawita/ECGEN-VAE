#!/usr/bin/env bash
# Quick script to generate ECG samples from trained models

set -euo pipefail

# Default settings
PRIOR_CHECKPOINT="${PRIOR_CHECKPOINT:-/work/vajira/DL2026/ECGEN-VAE/runs/prior_mimic_standalone/seed_42/checkpoints/last.ckpt}"
N_SAMPLES="${N_SAMPLES:-16}"
TEMPERATURE="${TEMPERATURE:-1.0}"
OUTPUT_DIR="${OUTPUT_DIR:-generated_samples}"
DEVICE="${DEVICE:-cuda}"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Helper functions
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

# Parse command line arguments
if [ $# -ge 1 ]; then
    PRIOR_CHECKPOINT="$1"
fi

if [ $# -ge 2 ]; then
    N_SAMPLES="$2"
fi

if [ $# -ge 3 ]; then
    TEMPERATURE="$3"
fi

# Check if checkpoint is provided
if [ -z "${PRIOR_CHECKPOINT}" ]; then
    print_error "Prior checkpoint not specified"
    echo ""
    echo "Usage: $0 <prior_checkpoint> [n_samples] [temperature]"
    echo ""
    echo "Examples:"
    echo "  $0 runs/prior_mimic_standalone/seed_42/checkpoints/best.ckpt"
    echo "  $0 runs/prior_mimic_standalone/seed_42/checkpoints/best.ckpt 32"
    echo "  $0 runs/prior_mimic_standalone/seed_42/checkpoints/best.ckpt 32 0.8"
    echo ""
    echo "Environment variables:"
    echo "  PRIOR_CHECKPOINT - Path to Prior checkpoint"
    echo "  N_SAMPLES        - Number of samples (default: 16)"
    echo "  TEMPERATURE      - Sampling temperature (default: 1.0)"
    echo "  OUTPUT_DIR       - Output directory (default: generated_samples)"
    echo "  DEVICE           - Device to use (default: cuda)"
    echo ""
    
    # Try to find available checkpoints
    print_info "Looking for available Prior checkpoints..."
    FOUND_CHECKPOINTS=$(find runs -path "*/prior*/seed_*/checkpoints/*.ckpt" -type f 2>/dev/null | head -5)
    
    if [ -n "${FOUND_CHECKPOINTS}" ]; then
        echo ""
        echo "Found the following Prior checkpoints:"
        echo "${FOUND_CHECKPOINTS}" | nl
        echo ""
        echo "You can use one of these by running:"
        FIRST_CHECKPOINT=$(echo "${FOUND_CHECKPOINTS}" | head -1)
        echo "  $0 ${FIRST_CHECKPOINT}"
        echo ""
    fi
    
    exit 1
fi

# Check if checkpoint exists
if [ ! -f "${PRIOR_CHECKPOINT}" ]; then
    print_error "Prior checkpoint not found: ${PRIOR_CHECKPOINT}"
    exit 1
fi

# Print configuration
print_header "Generating ECG Samples"
print_info "Prior checkpoint: ${PRIOR_CHECKPOINT}"
print_info "Number of samples: ${N_SAMPLES}"
print_info "Temperature: ${TEMPERATURE}"
print_info "Output directory: ${OUTPUT_DIR}"
print_info "Device: ${DEVICE}"

# Run generation
python "${SCRIPT_DIR}/generate_samples.py" \
    --prior-checkpoint "${PRIOR_CHECKPOINT}" \
    --n-samples "${N_SAMPLES}" \
    --temperature "${TEMPERATURE}" \
    --output-dir "${OUTPUT_DIR}" \
    --device "${DEVICE}"

# Check if generation succeeded
if [ $? -eq 0 ]; then
    print_header "Generation Complete!"
    echo "Generated samples saved to: ${OUTPUT_DIR}"
    echo ""
    echo "Files created:"
    echo "  - samples_grid_n${N_SAMPLES}_t${TEMPERATURE}.png  (grid visualization)"
    echo "  - samples_n${N_SAMPLES}_t${TEMPERATURE}.npy       (numpy array)"
    echo "  - individual/sample_*.png                          (individual plots)"
    echo ""
    echo "To generate with different settings:"
    echo "  N_SAMPLES=32 TEMPERATURE=0.8 $0 ${PRIOR_CHECKPOINT}"
else
    print_error "Generation failed"
    exit 1
fi
