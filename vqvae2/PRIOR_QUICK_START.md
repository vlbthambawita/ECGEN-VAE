# VQ-VAE-2 Prior Training - Quick Start Guide

## Prerequisites

✓ VQ-VAE-2 trained and checkpoint available at `runs/vqvae2_mimic/seed_42/checkpoints/last.ckpt`

## Complete Pipeline (4 Steps)

### Step 1: Extract Codes

```bash
./run_train_prior.sh extract
```

**What it does**: Extracts discrete codes from ECG signals using trained VQ-VAE-2

**Output**: 
- `codes/vqvae2_mimic/codes_top.npy` (N, 78)
- `codes/vqvae2_mimic/codes_bot.npy` (N, 625)

**Time**: ~1 min per 1000 samples

---

### Step 2: Train Top Prior

```bash
./run_train_prior.sh fit_top
```

**What it does**: Trains unconditional transformer on top codes

**Output**: `logs/top_prior/version_0/checkpoints/last.ckpt`

**Time**: ~10 sec/epoch (default: 100 epochs)

**Memory**: ~4 GB GPU

---

### Step 3: Train Bottom Prior

```bash
./run_train_prior.sh fit_bot
```

**What it does**: Trains conditional transformer on bottom codes

**Output**: `logs/bot_prior/version_0/checkpoints/last.ckpt`

**Time**: ~30 sec/epoch (default: 100 epochs)

**Memory**: ~8 GB GPU

---

### Step 4: Generate Samples

```bash
./run_train_prior.sh sample
```

**What it does**: Generates new ECG signals using trained priors

**Output**: `generated_ecgs.npy` (16, 12, 5000)

**Time**: ~15 seconds for 16 samples

---

## Common Customizations

### Extract Subset (for Testing)

```bash
MAX_SAMPLES=100 ./run_train_prior.sh extract
```

### Train with Custom Settings

```bash
# Top prior with larger model
TOP_D_MODEL=512 TOP_N_LAYERS=12 ./run_train_prior.sh fit_top

# Bottom prior with more epochs
BOT_MAX_EPOCHS=200 ./run_train_prior.sh fit_bot
```

### Generate More Samples

```bash
N_SAMPLES=100 ./run_train_prior.sh sample
```

### Control Sample Diversity

```bash
# More conservative (higher quality)
TOP_TEMP=0.8 BOT_TEMP=0.8 ./run_train_prior.sh sample

# More diverse
TOP_TEMP=1.2 BOT_TEMP=1.2 ./run_train_prior.sh sample
```

### Disable Wandb (Use CSV Only)

```bash
WANDB_ENABLED=false ./run_train_prior.sh fit_top
WANDB_ENABLED=false ./run_train_prior.sh fit_bot
```

### Custom Wandb Settings

```bash
WANDB_PROJECT=my-ecg-project WANDB_RUN_NAME=experiment1 ./run_train_prior.sh fit_top
```

---

## Debug Mode (Fast Testing)

```bash
# Extract small subset
MAX_SAMPLES=100 ./run_train_prior.sh extract

# Train for 2 epochs only
TOP_MAX_EPOCHS=2 ./run_train_prior.sh fit_top
BOT_MAX_EPOCHS=2 ./run_train_prior.sh fit_bot

# Generate 4 samples
N_SAMPLES=4 ./run_train_prior.sh sample
```

**Total time**: ~5 minutes

---

## Verify Output

```bash
# Check extracted codes
python3 -c "import numpy as np; print(np.load('codes/vqvae2_mimic/codes_top.npy').shape)"
# Expected: (N, 78)

# Check generated ECGs
python3 -c "import numpy as np; print(np.load('generated_ecgs.npy').shape)"
# Expected: (16, 12, 5000)
```

---

## Troubleshooting

### "VQ-VAE-2 checkpoint not found"

Train VQ-VAE-2 first:
```bash
cd /work/vajira/DL2026/ECGEN-VAE/vqvae2
./run_train_vqvae2.sh fit
```

### "CUDA out of memory"

Reduce batch sizes:
```bash
TOP_BATCH_SIZE=8 BOT_BATCH_SIZE=4 ./run_train_prior.sh fit_top
```

### "Codes directory not found"

Run extract first:
```bash
./run_train_prior.sh extract
```

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_SAMPLES` | (all) | Limit samples for extraction |
| `TOP_MAX_EPOCHS` | 100 | Top prior training epochs |
| `BOT_MAX_EPOCHS` | 100 | Bottom prior training epochs |
| `TOP_BATCH_SIZE` | 16 | Top prior batch size |
| `BOT_BATCH_SIZE` | 16 | Bottom prior batch size |
| `WANDB_ENABLED` | true | Enable Weights & Biases logging |
| `WANDB_PROJECT` | vqvae2-prior | W&B project name |
| `WANDB_ENTITY` | - | W&B entity (username/team) |
| `WANDB_RUN_NAME` | - | W&B run name |
| `N_SAMPLES` | 16 | Number of samples to generate |
| `TOP_TEMP` | 1.0 | Top prior temperature |
| `BOT_TEMP` | 1.0 | Bottom prior temperature |
| `GPUS` | 0 | GPU device ID |

---

## Full Example

```bash
# Navigate to vqvae2 directory
cd /work/vajira/DL2026/ECGEN-VAE/vqvae2

# Run complete pipeline
./run_train_prior.sh extract
./run_train_prior.sh fit_top
./run_train_prior.sh fit_bot
./run_train_prior.sh sample

# Verify output
python3 -c "import numpy as np; ecgs = np.load('generated_ecgs.npy'); print(f'Generated {ecgs.shape[0]} ECGs with shape {ecgs.shape}')"
```

**Expected output**: `Generated 16 ECGs with shape (16, 12, 5000)`

---

## Next Steps

After successful generation:

1. **Visualize samples**: Plot generated ECGs
2. **Evaluate quality**: Compare with real ECGs
3. **Scale up**: Generate larger datasets
4. **Fine-tune**: Adjust sampling parameters

For detailed documentation, see `PRIOR_README.md`
