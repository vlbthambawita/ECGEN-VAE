# VQ-VAE-2 for 12-Lead ECG Generation

Implementation of VQ-VAE-2 (Razavi et al., NeurIPS 2019) adapted for 12-lead ECG signals.

## Architecture

VQ-VAE-2 uses a two-level hierarchical vector quantization:

- **Bottom Level**: Encodes local morphology (QRS complexes, P/T waves)
  - Input: 12 × 5000 → Encoded: 64 × 625 (stride 8)
  
- **Top Level**: Encodes global structure (rhythm, segments)
  - Input: 64 × 625 → Encoded: 64 × 79 (stride ~8)

### Key Features

- **EMA Codebook Updates**: Exponential moving average for stable learning
- **Hierarchical Conditioning**: Top features condition bottom quantization
- **L1 Reconstruction Loss**: More robust for ECG signals than MSE
- **MIMIC-IV-ECG Dataset**: Full integration with existing dataset loader

## Installation

```bash
# Required packages
pip install torch pytorch-lightning numpy pandas matplotlib wfdb scikit-learn

# Optional: Weights & Biases logging
pip install wandb
```

## Quick Start

### 1. Using the Shell Script (Recommended)

```bash
# Train VQ-VAE-2
./run_train_vqvae2.sh fit

# With custom settings
BATCH_SIZE=64 MAX_EPOCHS=100 ./run_train_vqvae2.sh fit

# Test a trained model
./run_train_vqvae2.sh test runs/vqvae2_mimic/seed_42/checkpoints/last.ckpt

# Generate samples
N_SAMPLES=32 ./run_train_vqvae2.sh sample runs/vqvae2_mimic/seed_42/checkpoints/last.ckpt
```

### 2. Using Python Directly

```bash
# Train
python vqvae2.py fit --data-dir /path/to/mimic --devices 0

# Test
python vqvae2.py test --data-dir /path/to/mimic --ckpt-path last.ckpt

# Sample
python vqvae2.py sample --ckpt-path last.ckpt --n-samples 16 --out samples.npy
```

## Configuration

### Environment Variables (Shell Script)

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `/work/vajira/data/mimic_iv_original/...` | Path to MIMIC-IV-ECG dataset |
| `EXP_NAME` | `vqvae2_mimic` | Experiment name |
| `BATCH_SIZE` | `32` | Batch size |
| `MAX_EPOCHS` | `200` | Maximum training epochs |
| `LR` | `0.0003` | Learning rate |
| `HIDDEN_CHANNELS` | `128` | Hidden channels in encoder/decoder |
| `N_EMBEDDINGS_TOP` | `512` | Top codebook size |
| `N_EMBEDDINGS_BOT` | `512` | Bottom codebook size |
| `EMBEDDING_DIM` | `64` | Codebook vector dimension |
| `EMA_DECAY` | `0.99` | EMA decay for codebook (0 to disable) |
| `WANDB_ENABLED` | `true` | Enable Weights & Biases logging |
| `DEVICES` | `0` | GPU device IDs |

### Model Hyperparameters

```python
VQVAE2Config(
    n_leads=12,              # Number of ECG leads
    signal_len=5000,         # Signal length (10s × 500Hz)
    hidden_channels=128,     # Encoder/decoder hidden channels
    residual_channels=64,    # Residual block channels
    n_res_blocks=4,          # Number of residual blocks
    n_embeddings_top=512,    # Top codebook size
    n_embeddings_bot=512,    # Bottom codebook size
    embedding_dim=64,        # Codebook vector dimension
    commitment_cost=0.25,    # β in VQ loss
    ema_decay=0.99,          # EMA decay (0 to disable)
    enc_bot_strides=(2,2,2), # Bottom encoder strides (×8)
    enc_top_strides=(2,2,2), # Top encoder strides (×8)
    lr=3e-4,                 # Learning rate
)
```

## Usage Examples

### Basic Training

```bash
# Minimal training command
python vqvae2.py fit --data-dir /path/to/mimic --devices 0
```

### Training with Custom Settings

```bash
python vqvae2.py fit \
    --data-dir /path/to/mimic \
    --exp-name vqvae2_exp1 \
    --batch-size 64 \
    --max-epochs 100 \
    --lr 0.0005 \
    --hidden-channels 256 \
    --n-embeddings-top 1024 \
    --n-embeddings-bot 1024 \
    --embedding-dim 128 \
    --devices 0 1 \
    --wandb \
    --wandb-project my-ecg-project
```

### Debug Mode (Small Dataset)

```bash
python vqvae2.py fit \
    --data-dir /path/to/mimic \
    --max-samples 100 \
    --max-epochs 2 \
    --devices 0
```

### Testing

```bash
python vqvae2.py test \
    --data-dir /path/to/mimic \
    --ckpt-path runs/vqvae2_mimic/seed_42/checkpoints/last.ckpt \
    --devices 0
```

### Sampling

```bash
# Generate 32 samples with temperature 1.0
python vqvae2.py sample \
    --ckpt-path runs/vqvae2_mimic/seed_42/checkpoints/last.ckpt \
    --n-samples 32 \
    --temperature 1.0 \
    --out samples_t1.0.npy

# Generate with different temperature
python vqvae2.py sample \
    --ckpt-path runs/vqvae2_mimic/seed_42/checkpoints/last.ckpt \
    --n-samples 32 \
    --temperature 0.8 \
    --out samples_t0.8.npy
```

## Output Structure

```
runs/
└── vqvae2_mimic/
    └── seed_42/
        ├── checkpoints/
        │   ├── epoch000-step000128.ckpt
        │   ├── epoch001-step000256.ckpt
        │   └── last.ckpt
        ├── samples/
        │   ├── epoch_0000.png
        │   ├── epoch_0005.png
        │   └── ...
        └── tb/
            └── version_0/
                └── events.out.tfevents...
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir runs/vqvae2_mimic/seed_42/tb
```

### Weights & Biases

If W&B is enabled, logs are automatically synced to your W&B project.

## Key Metrics

During training, the following metrics are logged:

- `train/total_loss`: Total loss (reconstruction + VQ)
- `train/recon_loss`: L1 reconstruction loss
- `train/vq_loss`: Combined VQ loss (top + bottom)
- `train/unique_codes_bot`: Unique codes used in bottom codebook
- `train/unique_codes_top`: Unique codes used in top codebook
- `train/codebook_usage_bot`: Fraction of bottom codebook used
- `train/codebook_usage_top`: Fraction of top codebook used
- `val/*`: Same metrics for validation set

## Architecture Details

### Temporal Dimensions

```
Input:          B × 12 × 5000
↓ enc_bot (stride 8)
Bottom latent:  B × 64 × 625
↓ enc_top (stride 8)
Top latent:     B × 64 × 79
↓ vq_top
Top quantized:  B × 64 × 79
↓ dec_top (upsample)
Top upsampled:  B × 64 × 625
↓ vq_bot (conditioned on top)
Bottom quantized: B × 64 × 625
↓ dec_bot (conditioned on top)
Reconstructed:  B × 12 × 5000
```

### Loss Function

```python
recon_loss = L1_loss(x_recon, x)
vq_loss_bot = VQ_loss(z_bot, z_q_bot)
vq_loss_top = VQ_loss(z_top, z_q_top)
total_loss = recon_loss + vq_loss_bot + vq_loss_top
```

### EMA Codebook Update

```python
# Exponential moving average update (more stable than gradient-based)
ema_cluster_size = decay * ema_cluster_size + (1 - decay) * cluster_size
ema_dw = decay * ema_dw + (1 - decay) * dw
codebook = ema_dw / (ema_cluster_size + epsilon)
```

## Comparison with VQ-VAE

| Feature | VQ-VAE | VQ-VAE-2 |
|---------|--------|----------|
| Levels | 1 | 2 (hierarchical) |
| Codebooks | 1 | 2 (top + bottom) |
| Latent resolution | 625 | 79 (top) + 625 (bottom) |
| Conditioning | None | Top conditions bottom |
| Reconstruction quality | Good | Better (captures both local and global) |

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
BATCH_SIZE=16 ./run_train_vqvae2.sh fit

# Or reduce model size
python vqvae2.py fit --hidden-channels 64 --embedding-dim 32
```

### Low Codebook Usage

- Increase `ema_decay` (e.g., 0.999)
- Reduce `commitment_cost` (e.g., 0.1)
- Increase codebook size

### Poor Reconstruction

- Increase `hidden_channels` (e.g., 256)
- Increase `n_res_blocks` (e.g., 6)
- Train for more epochs
- Reduce learning rate

## References

- Razavi, A., et al. (2019). "Generating Diverse High-Fidelity Images with VQ-VAE-2." NeurIPS 2019.
- van den Oord, A., et al. (2017). "Neural Discrete Representation Learning." NeurIPS 2017.

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

## License

This implementation follows the structure of the existing VQ-VAE codebase in this repository.
