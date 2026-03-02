# Quick Start Guide

## TL;DR

The checkpoint error you encountered is because you're trying to load a **deepfakeECGLDM** checkpoint into the **VAE baseline** script. They have incompatible model structures.

## Quick Solutions

### Solution 1: Train from Scratch (Easiest)
```bash
cd /work/vajira/DL2026/ECGEN-VAE/vae_baseline
./run_train_vae_baseline.sh train
```

### Solution 2: Convert Checkpoint (If you need pre-trained weights)
```bash
cd /work/vajira/DL2026/ECGEN-VAE/vae_baseline

# Convert using shell script (easiest)
./convert_deepfake_checkpoint.sh

# Or specify custom paths
./convert_deepfake_checkpoint.sh /path/to/input.ckpt output.ckpt

# Or use Python script directly
python convert_checkpoint.py \
  --input /work/vajira/DL2026/deepfakeECGLDM/ecg_diffusion/outputs/mimic_v3_vae_continue/vae/checkpoints/last.ckpt \
  --output converted_checkpoint.ckpt

# Use converted checkpoint
RESUME_CHECKPOINT=converted_checkpoint.ckpt ./run_train_vae_baseline.sh resume
```

### Solution 3: Use Original Script (If you want to continue with deepfakeECGLDM)
```bash
cd /work/vajira/DL2026/deepfakeECGLDM/ecg_diffusion
./scripts/train_vae.sh
```

## Why the Error Occurred

The error message shows:
```
Missing key(s): "vae.encoder.conv_in.weight", ...
Unexpected key(s): "model.encoder.conv_in.weight", ...
```

This means:
- **deepfakeECGLDM** stores VAE as `self.model` → checkpoint keys: `model.encoder.*`
- **VAE baseline** stores VAE as `self.vae` → expects keys: `vae.encoder.*`

## What to Do Now

### If you want to use the VAE baseline script:
```bash
cd /work/vajira/DL2026/ECGEN-VAE/vae_baseline
./run_train_vae_baseline.sh train
```

### If you want to continue with deepfakeECGLDM:
```bash
cd /work/vajira/DL2026/deepfakeECGLDM/ecg_diffusion
./scripts/train_vae.sh
```

## Files Created

1. **`convert_checkpoint.py`** - Converts deepfakeECGLDM checkpoints to VAE baseline format
2. **`CHECKPOINT_COMPATIBILITY.md`** - Detailed compatibility guide
3. **`QUICK_START.md`** - This file

## Training from Scratch (Recommended)

```bash
cd /work/vajira/DL2026/ECGEN-VAE/vae_baseline

# Basic training
./run_train_vae_baseline.sh train

# Quick test (small dataset)
MAX_SAMPLES=1000 MAX_EPOCHS=10 ./run_train_vae_baseline.sh train

# Production training
DATA_DIR=/path/to/mimic-iv-ecg \
BATCH_SIZE=64 \
MAX_EPOCHS=200 \
WANDB_ENABLED=true \
./run_train_vae_baseline.sh train
```

## Key Differences

| Feature | deepfakeECGLDM | VAE Baseline |
|---------|----------------|--------------|
| Structure | Modular (multiple files) | Standalone (single file) |
| Model attribute | `self.model` | `self.vae` |
| Checkpoint keys | `model.*` | `vae.*` |
| Resume support | Via `--resume` | Via `--resume` + shell script modes |

## Next Steps

1. **Choose your approach** (train from scratch, convert, or use original)
2. **Run the training** with appropriate command
3. **Monitor progress** with TensorBoard or W&B
4. **Check outputs** in `runs/` directory

## Need Help?

- **Checkpoint conversion**: See `CHECKPOINT_COMPATIBILITY.md`
- **Training guide**: See `TRAINING_GUIDE.md`
- **Resume functionality**: See `RESUME_FUNCTIONALITY.md`
- **General usage**: See `README.md`
