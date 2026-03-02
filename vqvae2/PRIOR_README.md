# Transformer Prior for VQ-VAE-2 ECG Generation

Stage 2 of the VQ-VAE-2 pipeline: Training autoregressive priors to generate ECG signals.

## Overview

After training VQ-VAE-2 (Stage 1), we train two Transformer-based priors:

1. **Top Prior**: Learns the distribution over top-level codes (global structure)
2. **Bottom Prior**: Learns the distribution over bottom-level codes conditioned on top codes (local details)

## Quick Start

### Using the Shell Script (Recommended)

```bash
# Step 1: Extract codes from trained VQ-VAE-2
./run_train_prior.sh extract

# Step 2: Train top prior
./run_train_prior.sh fit_top

# Step 3: Train bottom prior
./run_train_prior.sh fit_bot

# Step 4: Generate ECG samples
./run_train_prior.sh sample
```

### Using Python Directly

```bash
# Step 1: Extract codes
python transformer_prior.py extract \
    --vqvae-ckpt runs/vqvae2_mimic/seed_42/checkpoints/last.ckpt \
    --data-dir /path/to/mimic \
    --out-dir codes/vqvae2_mimic \
    --batch-size 32

# Step 2: Train top prior
python transformer_prior.py fit_top \
    --codes-dir codes/vqvae2_mimic \
    --max-epochs 100 \
    --batch-size 16 \
    --gpus 1

# Step 3: Train bottom prior
python transformer_prior.py fit_bot \
    --codes-dir codes/vqvae2_mimic \
    --max-epochs 100 \
    --batch-size 16 \
    --gpus 1

# Step 4: Generate samples
python transformer_prior.py sample \
    --vqvae-ckpt runs/vqvae2_mimic/seed_42/checkpoints/last.ckpt \
    --top-prior-ckpt logs/top_prior/version_0/checkpoints/last.ckpt \
    --bot-prior-ckpt logs/bot_prior/version_0/checkpoints/last.ckpt \
    --n-samples 16 \
    --out generated_ecgs.npy
```

## Architecture

### Top Prior (Unconditional)

- **Input**: Top codes (sequence length: 78)
- **Architecture**: GPT-style causal transformer
- **Output**: Distribution over top codebook (512 embeddings)
- **Parameters**: ~6.6M
- **Default config**:
  - d_model: 256
  - n_layers: 8
  - n_heads: 8
  - d_ff: 1024

### Bottom Prior (Conditional)

- **Input**: Bottom codes (sequence length: 625)
- **Condition**: Top codes (via cross-attention)
- **Architecture**: Conditioned causal transformer
- **Output**: Distribution over bottom codebook (512 embeddings)
- **Parameters**: ~51.2M
- **Default config**:
  - d_model: 512
  - n_layers: 12
  - n_heads: 8
  - d_ff: 2048

## Configuration

### Environment Variables (Shell Script)

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `/work/vajira/DATA/...` | Path to MIMIC-IV-ECG dataset |
| `VQVAE_CKPT` | `runs/vqvae2_mimic/.../last.ckpt` | VQ-VAE-2 checkpoint |
| `CODES_DIR` | `codes/vqvae2_mimic` | Directory for extracted codes |
| `TOP_BATCH_SIZE` | `16` | Top prior batch size |
| `TOP_MAX_EPOCHS` | `100` | Top prior training epochs |
| `TOP_LR` | `0.0003` | Top prior learning rate |
| `BOT_BATCH_SIZE` | `16` | Bottom prior batch size |
| `BOT_MAX_EPOCHS` | `100` | Bottom prior training epochs |
| `BOT_LR` | `0.0003` | Bottom prior learning rate |
| `GPUS` | `0` | GPU device ID |
| `WANDB_ENABLED` | `true` | Enable Weights & Biases logging |
| `WANDB_PROJECT` | `vqvae2-prior` | W&B project name |
| `WANDB_ENTITY` | `` | W&B entity (username/team) |
| `WANDB_RUN_NAME` | `` | W&B run name |
| `N_SAMPLES` | `16` | Number of samples to generate |
| `TOP_TEMP` | `1.0` | Top prior sampling temperature |
| `BOT_TEMP` | `1.0` | Bottom prior sampling temperature |
| `TOP_P` | `0.95` | Nucleus sampling threshold |

## Pipeline Details

### Step 1: Code Extraction

Extracts discrete codes from ECG signals using trained VQ-VAE-2:

```bash
python transformer_prior.py extract \
    --vqvae-ckpt runs/vqvae2_mimic/seed_42/checkpoints/last.ckpt \
    --data-dir /path/to/mimic \
    --out-dir codes/vqvae2_mimic \
    --batch-size 32 \
    --max-samples 1000  # Optional: for debugging
```

**Output**:
- `codes/vqvae2_mimic/codes_top.npy` - shape (N, 78)
- `codes/vqvae2_mimic/codes_bot.npy` - shape (N, 625)

### Step 2: Train Top Prior

Trains unconditional transformer on top codes:

```bash
python transformer_prior.py fit_top \
    --codes-dir codes/vqvae2_mimic \
    --max-epochs 100 \
    --batch-size 16 \
    --lr 0.0003 \
    --d-model 256 \
    --n-layers 8 \
    --n-heads 8 \
    --gpus 1 \
    --wandb \
    --wandb-project vqvae2-prior \
    --wandb-run-name top_prior_run1
```

**Checkpoints saved to**: `logs/top_prior/version_0/checkpoints/`

**Metrics logged**:
- `train/loss` - Cross-entropy loss
- `train/acc` - Token prediction accuracy
- `val/loss` - Validation loss
- `val/acc` - Validation accuracy

**Logging**: Metrics are logged to both CSV files and Weights & Biases (if enabled)

### Step 3: Train Bottom Prior

Trains conditional transformer on bottom codes:

```bash
python transformer_prior.py fit_bot \
    --codes-dir codes/vqvae2_mimic \
    --max-epochs 100 \
    --batch-size 16 \
    --lr 0.0003 \
    --d-model 512 \
    --n-layers 12 \
    --n-heads 8 \
    --gpus 1 \
    --wandb \
    --wandb-project vqvae2-prior \
    --wandb-run-name bot_prior_run1
```

**Checkpoints saved to**: `logs/bot_prior/version_0/checkpoints/`

**Note**: Bottom prior is larger and takes longer to train due to cross-attention mechanism.

**Logging**: Metrics are logged to both CSV files and Weights & Biases (if enabled)

### Step 4: Generate Samples

Generates new ECG signals using trained priors:

```bash
python transformer_prior.py sample \
    --vqvae-ckpt runs/vqvae2_mimic/seed_42/checkpoints/last.ckpt \
    --top-prior-ckpt logs/top_prior/version_0/checkpoints/last.ckpt \
    --bot-prior-ckpt logs/bot_prior/version_0/checkpoints/last.ckpt \
    --n-samples 32 \
    --top-temp 1.0 \
    --bot-temp 1.0 \
    --top-p 0.95 \
    --out generated_ecgs.npy
```

**Output**: `generated_ecgs.npy` - shape (32, 12, 5000)

**Sampling parameters**:
- `--top-temp`: Temperature for top prior (lower = more conservative)
- `--bot-temp`: Temperature for bottom prior
- `--top-p`: Nucleus sampling threshold (0.95 = top 95% probability mass)

## Training Tips

### Memory Optimization

If you encounter OOM errors:

```bash
# Reduce batch size
TOP_BATCH_SIZE=8 BOT_BATCH_SIZE=4 ./run_train_prior.sh fit_top

# Or reduce model size
python transformer_prior.py fit_bot \
    --d-model 256 \
    --n-layers 8 \
    --batch-size 8
```

### Faster Training (Debug Mode)

For quick testing:

```bash
# Extract small subset
python transformer_prior.py extract \
    --vqvae-ckpt runs/vqvae2_mimic/seed_42/checkpoints/last.ckpt \
    --data-dir /path/to/mimic \
    --out-dir codes/test \
    --max-samples 100

# Train for few epochs
python transformer_prior.py fit_top \
    --codes-dir codes/test \
    --max-epochs 2 \
    --batch-size 16
```

### Quality vs Diversity Trade-off

```bash
# More conservative (higher quality, less diversity)
python transformer_prior.py sample \
    --top-temp 0.8 \
    --bot-temp 0.8 \
    --top-p 0.9 \
    --n-samples 16 \
    --out conservative_samples.npy

# More diverse (lower quality, more diversity)
python transformer_prior.py sample \
    --top-temp 1.2 \
    --bot-temp 1.2 \
    --top-p 0.98 \
    --n-samples 16 \
    --out diverse_samples.npy
```

## Output Structure

```
vqvae2/
├── codes/
│   └── vqvae2_mimic/
│       ├── codes_top.npy      # Extracted top codes (N, 78)
│       └── codes_bot.npy      # Extracted bottom codes (N, 625)
├── logs/
│   ├── top_prior/
│   │   └── version_0/
│   │       ├── checkpoints/
│   │       │   ├── last.ckpt
│   │       │   └── top_prior-epoch=XXX-val/loss=X.XXXX.ckpt
│   │       └── metrics.csv
│   └── bot_prior/
│       └── version_0/
│           ├── checkpoints/
│           │   ├── last.ckpt
│           │   └── bot_prior-epoch=XXX-val/loss=X.XXXX.ckpt
│           └── metrics.csv
└── generated_ecgs.npy         # Generated samples (N, 12, 5000)
```

## Monitoring Training

### Weights & Biases

If W&B is enabled, metrics are automatically synced to your W&B project:

```bash
# Train with W&B enabled (default)
./run_train_prior.sh fit_top

# Train without W&B
WANDB_ENABLED=false ./run_train_prior.sh fit_top

# Custom W&B project
WANDB_PROJECT=my-ecg-project ./run_train_prior.sh fit_top
```

View your runs at: https://wandb.ai/your-entity/your-project

### CSV Logs

Training metrics are also saved to CSV files:

```bash
# View top prior training progress
cat logs/top_prior/version_0/metrics.csv

# View bottom prior training progress
cat logs/bot_prior/version_0/metrics.csv
```

### Key Metrics

- **Loss**: Should decrease over time
- **Accuracy**: Should increase (typically reaches 10-30% for discrete codes)
- **Learning rate**: Follows cosine schedule with warmup

## Troubleshooting

### Issue: "VQ-VAE-2 checkpoint not found"

**Solution**: Train VQ-VAE-2 first or set correct path:

```bash
VQVAE_CKPT=path/to/your/checkpoint.ckpt ./run_train_prior.sh extract
```

### Issue: "Codes directory not found"

**Solution**: Run extract command first:

```bash
./run_train_prior.sh extract
```

### Issue: "CUDA out of memory"

**Solution**: Reduce batch size or model size:

```bash
TOP_BATCH_SIZE=8 BOT_BATCH_SIZE=4 ./run_train_prior.sh fit_top
```

### Issue: Low accuracy / High loss

**Possible causes**:
- Not enough training epochs (try 200-300)
- Learning rate too high/low (try 1e-4 to 5e-4)
- Model too small (increase d_model or n_layers)

## Performance Benchmarks

Tested on NVIDIA RTX 3090:

| Operation | Batch Size | Time | Memory |
|-----------|------------|------|--------|
| Code extraction | 32 | ~1 min/1000 samples | ~2 GB |
| Top prior training | 16 | ~10 sec/epoch | ~4 GB |
| Bottom prior training | 16 | ~30 sec/epoch | ~8 GB |
| Sampling (16 samples) | - | ~15 sec | ~3 GB |

## References

- Razavi, A., et al. (2019). "Generating Diverse High-Fidelity Images with VQ-VAE-2." NeurIPS 2019.
- Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS 2017.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{razavi2019vqvae2,
  title={Generating Diverse High-Fidelity Images with VQ-VAE-2},
  author={Razavi, Ali and van den Oord, Aaron and Vinyals, Oriol},
  booktitle={NeurIPS},
  year={2019}
}
```

## Next Steps

After training the priors, you can:

1. **Generate large datasets**: Use trained priors to generate thousands of synthetic ECGs
2. **Evaluate quality**: Compare generated ECGs with real ones using clinical metrics
3. **Fine-tune**: Adjust sampling parameters for specific use cases
4. **Conditional generation**: Extend the model to condition on clinical features

## Complete Example

```bash
# Full pipeline from scratch
cd /work/vajira/DL2026/ECGEN-VAE/vqvae2

# 1. Extract codes (assuming VQ-VAE-2 is trained)
./run_train_prior.sh extract

# 2. Train top prior
TOP_MAX_EPOCHS=100 ./run_train_prior.sh fit_top

# 3. Train bottom prior
BOT_MAX_EPOCHS=100 ./run_train_prior.sh fit_bot

# 4. Generate samples
N_SAMPLES=32 ./run_train_prior.sh sample

# 5. Verify output
python3 -c "import numpy as np; print(np.load('generated_ecgs.npy').shape)"
# Expected output: (32, 12, 5000)
```
