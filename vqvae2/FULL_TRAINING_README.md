# Full Training Pipeline

This document describes the unified training script `run_full_training.sh` that executes the complete VQ-VAE-2 and Prior training pipeline.

## Overview

The script runs 4 sequential stages:

1. **Train VQ-VAE-2**: Trains the VQ-VAE-2 model on ECG data
2. **Extract Codes**: Extracts discrete codes from the trained VQ-VAE-2
3. **Train Top Prior**: Trains the top-level transformer prior
4. **Train Bottom Prior**: Trains the bottom-level transformer prior

## Usage

### Basic Usage

```bash
cd /work/vajira/DL2026/ECGEN-VAE/vqvae2
./run_full_training.sh
```

This will run the entire pipeline with default settings.

### Custom Configuration

All configuration is done via environment variables. You can override any setting:

```bash
# Example: Train with custom batch size and epochs
BATCH_SIZE=64 MAX_EPOCHS=100 ./run_full_training.sh

# Example: Disable W&B logging
WANDB_ENABLED=false ./run_full_training.sh

# Example: Use different GPU
DEVICES=1 ./run_full_training.sh

# Example: Train on full dataset (no sample limit)
MAX_SAMPLES=null ./run_full_training.sh
```

## Key Configuration Variables

### Data Settings
- `DATA_DIR`: Path to MIMIC-IV-ECG dataset (default: `/work/vajira/DATA/SEARCH/MIMIC_IV_ECG_raw_v1/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0`)
- `MAX_SAMPLES`: Maximum samples to use (default: `1000`, set to `null` for full dataset)
- `BATCH_SIZE`: Batch size for VQ-VAE-2 training (default: `32`)

### VQ-VAE-2 Training
- `MAX_EPOCHS`: Maximum training epochs (default: `200`)
- `LR`: Learning rate (default: `0.0003`)
- `HIDDEN_CHANNELS`: Hidden channels (default: `128`)
- `N_EMBEDDINGS_TOP`: Top codebook size (default: `512`)
- `N_EMBEDDINGS_BOT`: Bottom codebook size (default: `512`)
- `EMBEDDING_DIM`: Embedding dimension (default: `64`)

### Prior Training
- `TOP_BATCH_SIZE`: Top prior batch size (default: `16`)
- `TOP_MAX_EPOCHS`: Top prior max epochs (default: `100`)
- `TOP_LR`: Top prior learning rate (default: `0.0003`)
- `BOT_BATCH_SIZE`: Bottom prior batch size (default: `8`)
- `BOT_MAX_EPOCHS`: Bottom prior max epochs (default: `100`)
- `BOT_LR`: Bottom prior learning rate (default: `0.0003`)

### GPU Settings
- `DEVICES`: GPU device ID (default: `0`)
- `ACCELERATOR`: Accelerator type (default: `gpu`)

### Weights & Biases
- `WANDB_ENABLED`: Enable W&B logging (default: `true`)
- `WANDB_PROJECT`: W&B project for VQ-VAE-2 (default: `ecg-vqvae2`)
- `WANDB_PROJECT_PRIOR`: W&B project for priors (default: `vqvae2-prior`)
- `WANDB_ENTITY`: W&B entity (username/team)
- `WANDB_RUN_NAME`: W&B run name

### Output Paths
- `RUNS_ROOT`: Root directory for VQ-VAE-2 runs (default: `runs`)
- `CODES_DIR`: Directory for extracted codes (default: `codes/vqvae2_mimic`)
- `EXP_NAME`: Experiment name (default: `vqvae2_mimic`)
- `SEED`: Random seed (default: `42`)

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Full Training Pipeline                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Train VQ-VAE-2                                      │
│ - Trains encoder/decoder with vector quantization           │
│ - Saves checkpoint to: runs/vqvae2_mimic/seed_42/           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Extract Codes                                       │
│ - Encodes all ECGs to discrete codes                        │
│ - Saves to: codes/vqvae2_mimic/                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Train Top Prior                                     │
│ - Trains transformer on top-level codes                     │
│ - Saves to: logs/top_prior/                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 4: Train Bottom Prior                                  │
│ - Trains transformer on bottom-level codes                  │
│ - Saves to: logs/bot_prior/                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                          Complete!
```

## Error Handling

The script uses `set -euo pipefail` to stop immediately if any stage fails. Each stage validates its prerequisites before running:

- **Stage 1**: Validates data directory exists
- **Stage 2**: Validates VQ-VAE-2 checkpoint exists
- **Stage 3**: Validates extracted codes exist
- **Stage 4**: Validates extracted codes exist

## After Training

Once the pipeline completes, you can:

1. **Generate samples**:
   ```bash
   ./run_train_prior.sh sample
   ```

2. **Plot generated ECGs**:
   ```bash
   ./run_train_prior.sh plot
   ```

## Output Structure

After successful completion:

```
vqvae2/
├── runs/vqvae2_mimic/seed_42/
│   └── checkpoints/
│       └── last.ckpt              # VQ-VAE-2 checkpoint
├── codes/vqvae2_mimic/
│   ├── codes_top.npy              # Top-level codes
│   └── codes_bot.npy              # Bottom-level codes
├── logs/
│   ├── top_prior/version_0/
│   │   └── checkpoints/
│   │       └── last.ckpt          # Top prior checkpoint
│   └── bot_prior/version_0/
│       └── checkpoints/
│           └── last.ckpt          # Bottom prior checkpoint
```

## Comparison with Individual Scripts

### Before (Manual Process)
```bash
# Step 1: Train VQ-VAE-2
./run_train_vqvae2.sh fit

# Step 2: Extract codes
./run_train_prior.sh extract

# Step 3: Train top prior
./run_train_prior.sh fit_top

# Step 4: Train bottom prior
./run_train_prior.sh fit_bot
```

### After (Automated)
```bash
# All steps in one command
./run_full_training.sh
```

## Notes

- The script automatically determines checkpoint paths between stages
- All environment variables from both original scripts are supported
- Sample generation and plotting are NOT included (use `run_train_prior.sh` for these)
- Progress is clearly indicated with stage headers
- Each stage prints its configuration before running
