# Training Stage 2 Only (PixelCNN Prior)

This guide explains how to train only Stage 2 (PixelCNN Prior) using a pre-trained VQ-VAE checkpoint.

## Why Train Stage 2 Only?

You might want to train only Stage 2 if:

1. **You already have a trained VQ-VAE** from a previous run
2. **You want to experiment with different Prior architectures** without retraining the VQ-VAE
3. **You're doing hyperparameter tuning** for the Prior model
4. **You're resuming training** after an interruption

## Quick Start

### Method 1: Command Line Argument (Recommended)

```bash
./run_train_vqvae.sh 2 runs/vqvae_mimic_standalone/seed_42/checkpoints/epoch003-step000128.ckpt
```

### Method 2: Environment Variable

```bash
export VQVAE_CHECKPOINT=runs/vqvae_mimic_standalone/seed_42/checkpoints/epoch003-step000128.ckpt
./run_train_vqvae.sh 2
```

### Method 3: Inline Environment Variable

```bash
VQVAE_CHECKPOINT=runs/vqvae_mimic_standalone/seed_42/checkpoints/epoch003-step000128.ckpt ./run_train_vqvae.sh 2
```

## Finding Available Checkpoints

If you don't know which checkpoint to use, simply run:

```bash
./run_train_vqvae.sh 2
```

The script will automatically search for available VQ-VAE checkpoints and display them.
