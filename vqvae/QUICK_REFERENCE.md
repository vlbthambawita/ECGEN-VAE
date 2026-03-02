# Quick Reference Card

## Training

### Train Both Stages
```bash
./run_train_vqvae.sh both
```

### Train Stage 1 Only (VQ-VAE)
```bash
./run_train_vqvae.sh 1
```

### Train Stage 2 Only (Prior)
```bash
./run_train_vqvae.sh 2 runs/vqvae_mimic_standalone/seed_42/checkpoints/best.ckpt
```

### Quick Test (Small Dataset)
```bash
MAX_SAMPLES=100 MAX_EPOCHS_STAGE1=5 ./run_train_vqvae.sh 1
```

## Generation

### Generate Samples (Simple)
```bash
./generate.sh runs/prior_mimic_standalone/seed_42/checkpoints/best.ckpt
```

### Generate with Custom Settings
```bash
./generate.sh <checkpoint> <n_samples> <temperature>
# Example: 32 samples with temperature 0.8
./generate.sh runs/prior_mimic_standalone/seed_42/checkpoints/best.ckpt 32 0.8
```

### Generate with Multiple Temperatures
```bash
python generate_samples.py \
    --prior-checkpoint runs/prior_mimic_standalone/seed_42/checkpoints/best.ckpt \
    --n-samples 16 \
    --temperatures 0.5 0.8 1.0 1.2 1.5
```

## Finding Checkpoints

### Find VQ-VAE Checkpoints
```bash
find runs -path "*/vqvae*/seed_*/checkpoints/*.ckpt" -type f
```

### Find Prior Checkpoints
```bash
find runs -path "*/prior*/seed_*/checkpoints/*.ckpt" -type f
```

### Auto-Find (Just Run Without Args)
```bash
./run_train_vqvae.sh 2  # Shows available VQ-VAE checkpoints
./generate.sh           # Shows available Prior checkpoints
```

## Monitoring

### TensorBoard
```bash
tensorboard --logdir=runs
```

### Check Training Progress
```bash
# List all experiments
ls -lh runs/

# Check latest checkpoint
ls -lht runs/*/seed_*/checkpoints/*.ckpt | head -5

# View sample reconstructions
ls -lht runs/*/seed_*/samples/*.png | head -5
```

## Common Environment Variables

### Training
```bash
DATA_DIR=/path/to/mimic-iv-ecg
MAX_SAMPLES=1000
MAX_EPOCHS_STAGE1=100
MAX_EPOCHS_STAGE2=100
BATCH_SIZE=32
LR_STAGE1=0.0001
LR_STAGE2=0.001
WANDB_ENABLED=true
```

### Generation
```bash
PRIOR_CHECKPOINT=runs/prior_mimic_standalone/seed_42/checkpoints/best.ckpt
N_SAMPLES=16
TEMPERATURE=1.0
OUTPUT_DIR=generated_samples
DEVICE=cuda
```

## Temperature Guide

| Temperature | Effect | Use Case |
|-------------|--------|----------|
| 0.5 | Conservative | Most realistic, less diverse |
| 0.8 | Slightly creative | Good balance |
| 1.0 | Balanced | Default, recommended |
| 1.2 | More diverse | More variation |
| 1.5 | Very diverse | High creativity, may have artifacts |

## File Locations

### After Training
```
runs/
├── vqvae_mimic_standalone/seed_42/
│   ├── checkpoints/best.ckpt          ← Use for Stage 2
│   ├── samples/epoch_*.png            ← Check reconstruction quality
│   └── tb/                            ← TensorBoard logs
└── prior_mimic_standalone/seed_42/
    ├── checkpoints/best.ckpt          ← Use for generation
    └── tb/                            ← TensorBoard logs
```

### After Generation
```
generated_samples/
├── samples_grid_n16_t1.00.png         ← Quick overview
├── samples_n16_t1.00.npy              ← Raw data for analysis
└── individual/
    └── sample_*.png                   ← Detailed views
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Checkpoint not found | Run `./run_train_vqvae.sh 2` or `./generate.sh` to see available checkpoints |
| Out of memory | Reduce `BATCH_SIZE` or `N_SAMPLES` |
| Training too slow | Increase `NUM_WORKERS` or reduce `MAX_SAMPLES` |
| Poor quality samples | Try different temperature (0.8-1.2) or use best checkpoint |
| VQ-VAE not loaded | Make sure you're using a Prior checkpoint, not VQ-VAE checkpoint |

## Complete Workflow Example

```bash
# 1. Quick test with small dataset
MAX_SAMPLES=100 MAX_EPOCHS_STAGE1=5 ./run_train_vqvae.sh 1

# 2. Train Prior with same small dataset
MAX_SAMPLES=100 MAX_EPOCHS_STAGE2=5 \
./run_train_vqvae.sh 2 runs/vqvae_mimic_standalone/seed_42/checkpoints/best.ckpt

# 3. Generate samples to check quality
./generate.sh runs/prior_mimic_standalone/seed_42/checkpoints/best.ckpt 8 1.0

# 4. If quality is good, train on full dataset
MAX_EPOCHS_STAGE1=100 MAX_EPOCHS_STAGE2=100 ./run_train_vqvae.sh both

# 5. Generate final samples
./generate.sh runs/prior_mimic_standalone/seed_42/checkpoints/best.ckpt 100 1.0
```

## Documentation

- `README.md` - Complete documentation with architecture details
- `USAGE_EXAMPLES.md` - Training examples and configurations
- `STAGE2_ONLY.md` - Guide for training Stage 2 only
- `GENERATION_GUIDE.md` - Comprehensive generation guide
- `QUICK_REFERENCE.md` - This file

## Key Commands Summary

```bash
# Training
./run_train_vqvae.sh [1|2|both] [vqvae_checkpoint]

# Generation
./generate.sh <prior_checkpoint> [n_samples] [temperature]
python generate_samples.py --prior-checkpoint <path> [options]

# Monitoring
tensorboard --logdir=runs

# Finding checkpoints
find runs -name "*.ckpt"
```

## Getting Help

```bash
# Training script help
./run_train_vqvae.sh

# Generation script help
python generate_samples.py --help
./generate.sh
```
