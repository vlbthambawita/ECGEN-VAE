# VQ-VAE-2 Quick Start Guide

## 1. Installation Check

```bash
cd /work/vajira/DL2026/ECGEN-VAE/vqvae2

# Verify Python script
python vqvae2.py --help

# Verify shell script
./run_train_vqvae2.sh
```

## 2. Quick Training (Debug Mode)

Test with a small dataset to ensure everything works:

```bash
python vqvae2.py fit \
    --data-dir /path/to/mimic \
    --max-samples 100 \
    --max-epochs 2 \
    --batch-size 16 \
    --devices 0
```

Expected output:
- Dataset loads successfully
- Training starts without errors
- Checkpoints saved to `runs/vqvae2_mimic/seed_42/checkpoints/`
- Visualizations saved to `runs/vqvae2_mimic/seed_42/samples/`

## 3. Full Training

Once debug mode works, start full training:

```bash
# Using shell script (recommended)
./run_train_vqvae2.sh fit

# Or using Python directly
python vqvae2.py fit \
    --data-dir /path/to/mimic \
    --batch-size 32 \
    --max-epochs 200 \
    --devices 0 \
    --wandb
```

## 4. Monitor Training

### TensorBoard
```bash
tensorboard --logdir runs/vqvae2_mimic/seed_42/tb
```

### Key Metrics to Watch
- `val/total_loss` should decrease steadily
- `val/codebook_usage_bot` should be > 0.7
- `val/codebook_usage_top` should be > 0.7
- Check visualizations in `runs/vqvae2_mimic/seed_42/samples/`

## 5. Test Trained Model

```bash
./run_train_vqvae2.sh test runs/vqvae2_mimic/seed_42/checkpoints/last.ckpt
```

## 6. Generate Samples

```bash
# Generate 16 samples
./run_train_vqvae2.sh sample runs/vqvae2_mimic/seed_42/checkpoints/last.ckpt

# Generate 32 samples with custom temperature
N_SAMPLES=32 TEMPERATURE=0.8 ./run_train_vqvae2.sh sample runs/vqvae2_mimic/seed_42/checkpoints/last.ckpt
```

## Common Issues

### Issue: Out of Memory
**Solution**: Reduce batch size
```bash
BATCH_SIZE=16 ./run_train_vqvae2.sh fit
```

### Issue: Dataset not found
**Solution**: Set correct DATA_DIR
```bash
DATA_DIR=/correct/path/to/mimic ./run_train_vqvae2.sh fit
```

### Issue: Low codebook usage (< 0.5)
**Solution**: Adjust hyperparameters
```bash
python vqvae2.py fit \
    --data-dir /path/to/mimic \
    --ema-decay 0.999 \
    --commitment-cost 0.1 \
    --devices 0
```

## Expected Training Time

- **Debug mode** (100 samples, 2 epochs): ~2 minutes
- **Full training** (full dataset, 200 epochs): ~24-48 hours on single GPU

## File Locations

After training, you'll find:

```
runs/vqvae2_mimic/seed_42/
├── checkpoints/
│   ├── epoch000-step000128.ckpt
│   ├── epoch001-step000256.ckpt
│   └── last.ckpt                    ← Use this for inference
├── samples/
│   ├── epoch_0000.png               ← Reconstruction visualizations
│   ├── epoch_0005.png
│   └── ...
└── tb/
    └── version_0/
        └── events.out.tfevents...   ← TensorBoard logs
```

## Next Steps

1. **Train VQ-VAE-2**: Complete Stage 1 (this script)
2. **Train Prior**: Implement PixelCNN prior for hierarchical generation
3. **Generate ECGs**: Use trained prior to generate realistic ECGs

## Help

For detailed documentation, see:
- `README.md` - Full documentation
- `COMPARISON.md` - VQ-VAE vs VQ-VAE-2 comparison
- `vqvae2.py --help` - CLI reference

For issues, check:
- TensorBoard logs
- Console output
- Checkpoint files exist
