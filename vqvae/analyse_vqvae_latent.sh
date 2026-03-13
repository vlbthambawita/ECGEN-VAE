#!/usr/bin/env bash
# ===========================================================================
# run_analyse_vqvae_latent.sh
# ===========================================================================
# Runs the VQ-VAE latent space analysis script using the same configuration
# that was used to train the model (from run_train_vqvae_g002_full_v2.sh).
#
# Usage:
#   ./run_analyse_vqvae_latent.sh [vqvae_checkpoint_path]
#
# Examples:
#   # Auto-find the best checkpoint from the default experiment
#   ./run_analyse_vqvae_latent.sh
#
#   # Explicitly provide a checkpoint
#   ./run_analyse_vqvae_latent.sh runs/vqvae_mimic_standalone_v2/seed_42/checkpoints/best.ckpt
#
#   # Analyse more samples
#   MAX_SAMPLES=5000 ./run_analyse_vqvae_latent.sh
#
#   # Run on CPU (no GPU)
#   DEVICE=cpu ./run_analyse_vqvae_latent.sh
# ===========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Data & experiment paths — keep in sync with training script
# ---------------------------------------------------------------------------
DATA_DIR="${DATA_DIR:-/work/vajira/data/mimic_iv_original/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0}"
EXP_NAME_STAGE1="${EXP_NAME_STAGE1:-vqvae_mimic_standalone_v2}"
SEED="${SEED:-42}"
RUNS_ROOT="${RUNS_ROOT:-runs}"

# ---------------------------------------------------------------------------
# Checkpoint — can be overridden via env var or first CLI argument
# ---------------------------------------------------------------------------
VQVAE_CHECKPOINT="${VQVAE_CHECKPOINT:-}"

if [ -n "${1:-}" ]; then
    VQVAE_CHECKPOINT="${1}"
fi

# Auto-locate best.ckpt if not provided
if [ -z "${VQVAE_CHECKPOINT}" ]; then
    CANDIDATE="${RUNS_ROOT}/${EXP_NAME_STAGE1}/seed_${SEED}/checkpoints/best.ckpt"
    if [ -f "${CANDIDATE}" ]; then
        VQVAE_CHECKPOINT="${CANDIDATE}"
        echo "[INFO] Auto-found checkpoint: ${VQVAE_CHECKPOINT}"
    else
        # Fall back to latest epoch checkpoint
        VQVAE_CHECKPOINT=$(find "${RUNS_ROOT}" \
            -path "*/${EXP_NAME_STAGE1}/seed_${SEED}/checkpoints/epoch*.ckpt" \
            -type f 2>/dev/null | sort | tail -n 1 || true)
        if [ -z "${VQVAE_CHECKPOINT}" ]; then
            echo "[ERROR] No VQ-VAE checkpoint found."
            echo "        Please provide one as an argument:"
            echo "        $0 path/to/best.ckpt"
            exit 1
        fi
        echo "[INFO] Auto-found checkpoint: ${VQVAE_CHECKPOINT}"
    fi
fi

if [ ! -f "${VQVAE_CHECKPOINT}" ]; then
    echo "[ERROR] Checkpoint not found: ${VQVAE_CHECKPOINT}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Analysis settings — tweak these freely
# ---------------------------------------------------------------------------

# How many test samples to analyse. Increase for more robust statistics.
# The test split is 10% of the dataset; set to empty string for all test samples.
MAX_SAMPLES="${MAX_SAMPLES:-2000}"

# Output directory for figures and stats.json
OUTPUT_DIR="${OUTPUT_DIR:-latent_analysis}"

# Data loading
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# Must match training
SEQ_LENGTH="${SEQ_LENGTH:-5000}"
VAL_SPLIT="${VAL_SPLIT:-0.1}"
TEST_SPLIT="${TEST_SPLIT:-0.1}"

# Reconstruction plot: how many samples & leads to display
N_RECON_SAMPLES="${N_RECON_SAMPLES:-8}"
N_RECON_LEADS="${N_RECON_LEADS:-6}"

# Code heatmap: rows (samples)
N_HEATMAP_SAMPLES="${N_HEATMAP_SAMPLES:-60}"

# Device: 'cuda' or 'cpu'
DEVICE="${DEVICE:-cuda}"

# ---------------------------------------------------------------------------
# Locate the analysis script (same directory as this shell script)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_SCRIPT="${SCRIPT_DIR}/analyse_vqvae_latent.py"

if [ ! -f "${ANALYSIS_SCRIPT}" ]; then
    echo "[ERROR] Analysis script not found: ${ANALYSIS_SCRIPT}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Check / install optional dependency
# ---------------------------------------------------------------------------
if ! python -c "import umap" 2>/dev/null; then
    echo "[INFO] umap-learn not found — UMAP plots will be skipped."
    echo "       To enable them: pip install umap-learn"
fi

# ---------------------------------------------------------------------------
# Print configuration
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "  VQ-VAE Latent Space Analysis"
echo "=========================================="
echo "  Checkpoint  : ${VQVAE_CHECKPOINT}"
echo "  Data dir    : ${DATA_DIR}"
echo "  Max samples : ${MAX_SAMPLES}"
echo "  Output dir  : ${OUTPUT_DIR}"
echo "  Device      : ${DEVICE}"
echo "  Batch size  : ${BATCH_SIZE}"
echo "  Recon grid  : ${N_RECON_SAMPLES} samples × ${N_RECON_LEADS} leads"
echo "  Heatmap rows: ${N_HEATMAP_SAMPLES}"
echo "=========================================="
echo ""

# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------
CMD="python ${ANALYSIS_SCRIPT} \
    --checkpoint ${VQVAE_CHECKPOINT} \
    --data-dir ${DATA_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --batch-size ${BATCH_SIZE} \
    --num-workers ${NUM_WORKERS} \
    --seq-length ${SEQ_LENGTH} \
    --val-split ${VAL_SPLIT} \
    --test-split ${TEST_SPLIT} \
    --seed ${SEED} \
    --n-recon-samples ${N_RECON_SAMPLES} \
    --n-recon-leads ${N_RECON_LEADS} \
    --n-heatmap-samples ${N_HEATMAP_SAMPLES} \
    --device ${DEVICE}"

# Only pass --max-samples if non-empty
if [ -n "${MAX_SAMPLES}" ]; then
    CMD="${CMD} --max-samples ${MAX_SAMPLES}"
fi

echo "[INFO] Running:"
echo "  ${CMD}"
echo ""

eval "${CMD}"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "  Analysis complete!"
echo "  Figures: ${OUTPUT_DIR}/"
echo ""
echo "  01_codebook_usage.png       — histogram + CDF of code frequencies"
echo "  03_codebook_pca.png         — PCA of codebook embedding vectors"
echo "  04_codebook_umap.png        — UMAP of codebook embedding vectors"
echo "  05_code_heatmap.png         — discrete code sequences per sample"
echo "  06_positional_code_freq.png — top codes at each latent position"
echo "  07_sample_latent_pca.png    — PCA of per-sample mean latent"
echo "  08_sample_latent_umap.png   — UMAP of per-sample mean latent"
echo "  09_reconstructions.png      — real vs reconstructed ECG overlays"
echo "  10_loss_scatter.png         — recon & VQ loss distributions"
echo "  stats.json                  — numeric summary"
echo "=========================================="
echo ""
echo "Tip — to analyse more/fewer samples:"
echo "  MAX_SAMPLES=5000 $0 ${VQVAE_CHECKPOINT}"
echo "  MAX_SAMPLES=500  $0 ${VQVAE_CHECKPOINT}"
echo ""