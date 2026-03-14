#!/usr/bin/env bash
# Finetune VQ-VAE on PTB-XL Hypertrophy (HYP) subset

set -euo pipefail

# Paths (customize these)
DATA_DIR="${DATA_DIR:-/work/vajira/data/ptbxl/ptbxl}"
CHECKPOINT="${CHECKPOINT:-/work/vajira/DL2026/ECGEN-VAE/runs/vqvae_ptbxl_hyp_finetune_20260314_165552/seed_42/checkpoints/last.ckpt}"
EXP_BASE="${EXP_BASE:-vqvae_ptbxl_hyp_finetune}"
WANDB_PROJECT="${WANDB_PROJECT:-vqvae_g002}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"   # set to "true" if wandb works

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME="${EXP_BASE}_${TIMESTAMP}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Wandb: set dir so init works in headless; use WANDB_MODE=offline if no network
export WANDB_DIR="${WANDB_DIR:-${PROJECT_ROOT}/wandb_logs}"
mkdir -p "${WANDB_DIR}"
[[ -n "${WANDB_MODE:-}" ]] && export WANDB_MODE

# Resolve checkpoint to absolute path (relative paths are from PROJECT_ROOT)
if [[ "${CHECKPOINT}" != /* ]]; then
  CHECKPOINT="${PROJECT_ROOT}/${CHECKPOINT}"
fi

echo "Finetuning VQ-VAE on PTB-XL HYP"
echo "  Data:      ${DATA_DIR}"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Exp name:  ${EXP_NAME}"
echo "  W&B:       ${WANDB_ENABLED} (set WANDB_ENABLED=true to enable)"
echo ""

WANDB_ARGS=()
[[ "${WANDB_ENABLED}" = "true" ]] && WANDB_ARGS=(--wandb --wandb-project "${WANDB_PROJECT}")

python vqvae/train_vqvae_standalone.py --stage 1 \
  --dataset ptbxl \
  --data-dir "${DATA_DIR}" \
  --ptbxl-scp-class HYP \
  --load-checkpoint "${CHECKPOINT}" \
  --lr 1e-5 \
  --exp-name "${EXP_NAME}" \
  "${WANDB_ARGS[@]}" \
  --batch-size 32 \
  --max-epochs 200 \
  "$@"
