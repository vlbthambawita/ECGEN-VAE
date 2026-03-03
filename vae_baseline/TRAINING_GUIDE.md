# VAE Baseline Training Guide

This guide explains how to use the training script for the VAE baseline model.

## Quick Start

### Basic Training
```bash
cd /work/vajira/DL2026/ECGEN-VAE/vae_baseline
./run_train_vae_baseline.sh
```

This will automatically:
- Start fresh training if no checkpoint exists
- Resume training if a checkpoint is found

## Usage Modes

### 1. Auto Mode (Default)
```bash
./run_train_vae_baseline.sh
```
Automatically detects whether to train or resume based on checkpoint existence.

### 2. Fresh Training
```bash
./run_train_vae_baseline.sh train
```
Always starts fresh training, ignoring any existing checkpoints.

### 3. Resume Training
```bash
./run_train_vae_baseline.sh resume
```
Resumes from the last checkpoint. Fails if no checkpoint exists.

## Configuration

All settings can be customized via environment variables:

### Data Configuration
```bash
DATA_DIR=/path/to/mimic-iv-ecg \
BATCH_SIZE=64 \
NUM_WORKERS=8 \
./run_train_vae_baseline.sh
```

### Model Configuration
```bash
LATENT_CHANNELS=16 \
KL_WEIGHT=0.0005 \
BASE_CHANNELS=128 \
./run_train_vae_baseline.sh
```

### Training Configuration
```bash
MAX_EPOCHS=200 \
LR=0.0002 \
PATIENCE=20 \
./run_train_vae_baseline.sh
```

### Quick Test (Small Dataset)
```bash
MAX_SAMPLES=1000 \
MAX_EPOCHS=10 \
./run_train_vae_baseline.sh train
```

## Environment Variables Reference

### Data Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `/work/vajira/data/mimic_iv_original/...` | Path to MIMIC-IV-ECG dataset |
| `BATCH_SIZE` | `32` | Training batch size |
| `NUM_WORKERS` | `4` | Number of data loading workers |
| `MAX_SAMPLES` | `null` | Limit dataset size (null = full dataset) |
| `VAL_SPLIT` | `0.1` | Validation split ratio |
| `TEST_SPLIT` | `0.1` | Test split ratio |

### Experiment Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `EXP_NAME` | `vae_baseline_mimic` | Experiment name |
| `SEED` | `42` | Random seed |
| `RUNS_ROOT` | `runs` | Root directory for outputs |

### Model Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `IN_CHANNELS` | `12` | Number of ECG leads |
| `BASE_CHANNELS` | `64` | Base number of channels |
| `LATENT_CHANNELS` | `8` | Latent space dimensions |
| `NUM_RES_BLOCKS` | `2` | Number of residual blocks |
| `KL_WEIGHT` | `0.0001` | KL divergence weight |
| `SEQ_LENGTH` | `5000` | ECG sequence length |

### Training Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `LR` | `0.0001` | Learning rate |
| `MAX_EPOCHS` | `100` | Maximum training epochs |
| `ACCELERATOR` | `gpu` | Device type (gpu/cpu) |
| `DEVICES` | `0` | GPU device ID |
| `GRADIENT_CLIP` | `1.0` | Gradient clipping value |
| `PATIENCE` | `10` | Early stopping patience |
| `SAVE_TOP_K` | `3` | Number of best checkpoints to keep |

### Visualization Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `VIZ_EVERY_N_EPOCHS` | `5` | Visualization frequency |
| `VIZ_NUM_SAMPLES` | `4` | Number of samples to visualize |

### Logging Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `WANDB_ENABLED` | `true` | Enable Weights & Biases |
| `WANDB_PROJECT` | `ecg-vae` | W&B project name |
| `WANDB_ENTITY` | `` | W&B entity (username/team) |
| `WANDB_RUN_NAME` | `` | W&B run name (auto-generated) |
| `WANDB_TAGS` | `` | W&B tags (space-separated) |

## Resume from Specific Checkpoint

```bash
RESUME_CHECKPOINT=runs/vae_exp/seed_42/checkpoints/epoch050.ckpt \
./run_train_vae_baseline.sh resume
```

## Output Structure

After training, outputs are organized as:
```
runs/
└── vae_baseline_mimic/
    └── seed_42/
        ├── checkpoints/
        │   ├── epoch001-step000100.ckpt
        │   ├── epoch002-step000200.ckpt
        │   ├── ...
        │   └── last.ckpt
        ├── samples/
        │   ├── epoch_0005.png
        │   ├── epoch_0010.png
        │   └── ...
        └── tb/
            └── (TensorBoard logs)
```

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir=runs
```
Then open http://localhost:6006

### Weights & Biases
If W&B is enabled, view at:
```
https://wandb.ai/your-username/ecg-vae
```

## Common Use Cases

### 1. Production Training
```bash
DATA_DIR=/path/to/mimic-iv-ecg \
EXP_NAME=vae_production \
MAX_EPOCHS=200 \
WANDB_ENABLED=true \
./run_train_vae_baseline.sh train
```

### 2. Hyperparameter Search
```bash
# Experiment 1: Higher KL weight
KL_WEIGHT=0.001 EXP_NAME=vae_kl001 ./run_train_vae_baseline.sh train

# Experiment 2: Larger latent space
LATENT_CHANNELS=16 EXP_NAME=vae_lat16 ./run_train_vae_baseline.sh train

# Experiment 3: Different learning rate
LR=0.0002 EXP_NAME=vae_lr0002 ./run_train_vae_baseline.sh train
```

### 3. Quick Debugging
```bash
MAX_SAMPLES=100 \
MAX_EPOCHS=2 \
WANDB_ENABLED=false \
./run_train_vae_baseline.sh train
```

### 4. Multi-GPU Training
```bash
DEVICES="0 1" \
BATCH_SIZE=64 \
./run_train_vae_baseline.sh train
```

### 5. Resume After Interruption
```bash
# Automatically resumes from last checkpoint
./run_train_vae_baseline.sh resume
```

## Troubleshooting

### Issue: "Data directory not found"
**Solution**: Set the correct DATA_DIR:
```bash
DATA_DIR=/correct/path/to/mimic-iv-ecg ./run_train_vae_baseline.sh
```

### Issue: "Training script not found"
**Solution**: Make sure you're running from the vae_baseline directory:
```bash
cd /work/vajira/DL2026/ECGEN-VAE/vae_baseline
./run_train_vae_baseline.sh
```

### Issue: "Resume mode requested but no checkpoint found"
**Solution**: Train the model first:
```bash
./run_train_vae_baseline.sh train
```

### Issue: Out of memory
**Solution**: Reduce batch size:
```bash
BATCH_SIZE=16 ./run_train_vae_baseline.sh
```

### Issue: Training too slow
**Solution**: Increase number of workers:
```bash
NUM_WORKERS=8 ./run_train_vae_baseline.sh
```

## Comparison with VQ-VAE Script

| Feature | VQ-VAE Script | VAE Baseline Script |
|---------|---------------|---------------------|
| Training stages | 2 (VAE + Prior) | 1 (VAE only) |
| Mode argument | `--stage 1\|2\|both` | `train\|resume` |
| Checkpoint usage | Pass between stages | Resume same model |
| Complexity | Higher | Simpler |
| Use case | Discrete latent generation | Continuous latent generation |

## Tips

1. **Start with default settings** for initial experiments
2. **Use MAX_SAMPLES** for quick testing before full training
3. **Enable W&B** for better experiment tracking
4. **Monitor validation loss** to detect overfitting
5. **Save top-k checkpoints** to compare different epochs
6. **Use resume mode** to continue interrupted training
7. **Adjust KL_WEIGHT** if reconstruction quality is poor

## Next Steps

After training completes:
1. Check reconstruction quality in `samples/` directory
2. Analyze training curves in TensorBoard
3. Load best checkpoint for inference
4. Compare with VQ-VAE results
5. Tune hyperparameters based on results

## Example Training Session

```bash
# 1. Start training
./run_train_vae_baseline.sh train

# 2. Monitor in another terminal
tensorboard --logdir=runs

# 3. If interrupted, resume
./run_train_vae_baseline.sh resume

# 4. After completion, check results
ls runs/vae_baseline_mimic/seed_42/checkpoints/
ls runs/vae_baseline_mimic/seed_42/samples/
```
