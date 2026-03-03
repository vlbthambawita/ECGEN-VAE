# VAE Baseline Training Script

This directory contains a standalone training script for a basic Variational Autoencoder (VAE) model for ECG signal generation.

## Overview

The `train_vae_baseline.py` script implements a complete VAE training pipeline in a single file, including:
- VAE model architecture (Encoder, Decoder with reparameterization trick)
- MIMIC-IV-ECG dataset loading
- PyTorch Lightning training loop
- Visualization callbacks
- TensorBoard and Weights & Biases logging support

## Model Architecture

- **Encoder**: 1D CNN with residual blocks → mean and logvar
- **Decoder**: 1D transposed CNN with residual blocks
- **Loss**: MSE reconstruction + KL divergence
- **Latent Space**: Continuous (not quantized)

### Default Configuration

- Input: 12 ECG leads × 5000 timesteps (10s @ 500Hz)
- Architecture: 64 base channels → [1,2,4,4] multipliers → 8 latent channels
- Training: batch_size=32, lr=1e-4, max_epochs=100
- Loss: KL weight=0.0001

## Usage

### Basic Training

```bash
python train_vae_baseline.py \
  --data-dir /path/to/mimic-iv-ecg \
  --exp-name vae_baseline_exp
```

### With Custom Parameters

```bash
python train_vae_baseline.py \
  --data-dir /path/to/mimic-iv-ecg \
  --exp-name vae_custom \
  --batch-size 64 \
  --lr 2e-4 \
  --max-epochs 200 \
  --kl-weight 0.0005 \
  --latent-channels 16
```

### With Weights & Biases

```bash
python train_vae_baseline.py \
  --data-dir /path/to/mimic-iv-ecg \
  --exp-name vae_wandb \
  --wandb \
  --wandb-project ecg-vae \
  --wandb-entity your-username
```

### Resume Training

```bash
python train_vae_baseline.py \
  --data-dir /path/to/mimic-iv-ecg \
  --exp-name vae_baseline_exp \
  --resume runs/vae_baseline_exp/seed_42/checkpoints/last.ckpt
```

### Quick Test (Small Dataset)

```bash
python train_vae_baseline.py \
  --data-dir /path/to/mimic-iv-ecg \
  --exp-name vae_test \
  --max-samples 1000 \
  --max-epochs 10 \
  --skip-missing-check
```

## Command-Line Arguments

### Experiment
- `--exp-name`: Experiment name (default: vae_baseline)
- `--seed`: Random seed (default: 42)
- `--runs-root`: Root directory for runs (default: runs)

### Data
- `--data-dir`: Path to MIMIC-IV-ECG dataset (required)
- `--batch-size`: Batch size (default: 32)
- `--num-workers`: Number of data loading workers (default: 4)
- `--max-samples`: Max samples for debugging (default: None)
- `--skip-missing-check`: Skip missing file check
- `--val-split`: Validation split ratio (default: 0.1)
- `--test-split`: Test split ratio (default: 0.1)

### Model
- `--in-channels`: Number of ECG leads (default: 12)
- `--base-channels`: Base number of channels (default: 64)
- `--latent-channels`: Latent channels (default: 8)
- `--num-res-blocks`: Number of residual blocks (default: 2)
- `--kl-weight`: KL divergence weight (default: 0.0001)
- `--seq-length`: ECG sequence length (default: 5000)

### Training
- `--lr`: Learning rate (default: 1e-4)
- `--max-epochs`: Maximum number of epochs (default: 100)
- `--accelerator`: Accelerator type (default: gpu)
- `--devices`: Device IDs (default: [0])
- `--gradient-clip`: Gradient clipping value (default: 1.0)
- `--patience`: Early stopping patience (default: 10)

### Visualization
- `--viz-every-n-epochs`: Generate visualizations every N epochs (default: 5)
- `--viz-num-samples`: Number of samples to visualize (default: 4)

### Resume
- `--resume`: Path to checkpoint to resume training from (default: None)

### Logging
- `--wandb`: Enable Weights & Biases logging
- `--wandb-project`: W&B project name (default: ecg-vae)
- `--wandb-entity`: W&B entity (username/team)
- `--wandb-run-name`: W&B run name (auto-generated if not set)
- `--wandb-tags`: W&B tags

## Output Structure

```
runs/
└── {exp_name}/
    └── seed_{seed}/
        ├── checkpoints/
        │   ├── epoch{epoch:03d}-step{step:06d}.ckpt
        │   └── last.ckpt
        ├── samples/
        │   └── epoch_{epoch:04d}.png
        └── tb/
            └── tensorboard_logs/
```

## Key Differences from VQ-VAE

| Feature | VQ-VAE | Basic VAE |
|---------|---------|-----------|
| Latent Space | Discrete codes | Continuous distribution |
| Quantization | VectorQuantizer | Reparameterization trick |
| Loss | MSE + VQ loss | MSE + KL divergence |
| Training | 2-stage (VAE + Prior) | Single-stage |
| Sampling | Requires prior model | Direct sampling from latent |

## Requirements

- Python 3.8+
- PyTorch 1.12+
- PyTorch Lightning 2.0+
- wfdb
- pandas
- numpy
- matplotlib
- scikit-learn

Optional:
- wandb (for Weights & Biases logging)

## Notes

- The script uses subject-level splits to avoid data leakage
- ECG signals are normalized per-sample (zero mean, unit variance)
- Features (9 MIMIC measurements) are normalized using training set statistics
- Checkpoints are saved based on validation loss
- Visualizations show real vs reconstructed ECGs for the first validation batch

## Comparison with Other Implementations

This baseline script is designed to be:
1. **Self-contained**: All components in a single file
2. **Simple**: Standard VAE without vector quantization
3. **Comparable**: Same architecture as deepfakeECGLDM VAE
4. **Flexible**: Easy to modify hyperparameters via command-line

For production use or more complex experiments, consider using the modular implementations in the parent directories.
