# VQ-VAE Training Script Usage Examples

This document provides practical examples for using the `run_train_vqvae.sh` script.

## Basic Usage

### Train Both Stages (Default)

Train VQ-VAE (Stage 1) followed by PixelCNN Prior (Stage 2):

```bash
./run_train_vqvae.sh both
```

or simply:

```bash
./run_train_vqvae.sh
```

### Train Stage 1 Only (VQ-VAE)

```bash
./run_train_vqvae.sh 1
```

### Train Stage 2 Only (PixelCNN Prior)

**Option 1: Command line argument**
```bash
./run_train_vqvae.sh 2 runs/vqvae_mimic_standalone/seed_42/checkpoints/epoch003-step000128.ckpt
```

**Option 2: Environment variable**
```bash
VQVAE_CHECKPOINT=runs/vqvae_mimic_standalone/seed_42/checkpoints/epoch003-step000128.ckpt ./run_train_vqvae.sh 2
```

**Option 3: Export and run**
```bash
export VQVAE_CHECKPOINT=runs/vqvae_mimic_standalone/seed_42/checkpoints/epoch003-step000128.ckpt
./run_train_vqvae.sh 2
```

## Advanced Configuration

### Custom Data Directory

```bash
DATA_DIR=/path/to/your/mimic-iv-ecg ./run_train_vqvae.sh both
```

### Limit Dataset Size (for Testing)

```bash
MAX_SAMPLES=100 ./run_train_vqvae.sh 1
```

### Custom Experiment Names

```bash
EXP_NAME_STAGE1=my_vqvae_experiment \
EXP_NAME_STAGE2=my_prior_experiment \
./run_train_vqvae.sh both
```

### Adjust Training Epochs

```bash
MAX_EPOCHS_STAGE1=50 \
MAX_EPOCHS_STAGE2=30 \
./run_train_vqvae.sh both
```

### Custom Learning Rates

```bash
LR_STAGE1=0.0002 \
LR_STAGE2=0.0005 \
./run_train_vqvae.sh both
```

### Disable Weights & Biases

```bash
WANDB_ENABLED=false ./run_train_vqvae.sh both
```

### Custom Batch Size

```bash
BATCH_SIZE=64 ./run_train_vqvae.sh both
```

### Use CPU Instead of GPU

```bash
ACCELERATOR=cpu ./run_train_vqvae.sh 1
```

## Common Workflows

### Workflow 1: Train and Experiment

1. **Quick test with small dataset:**
   ```bash
   MAX_SAMPLES=100 MAX_EPOCHS_STAGE1=5 ./run_train_vqvae.sh 1
   ```

2. **If results look good, train Stage 2:**
   ```bash
   MAX_SAMPLES=100 MAX_EPOCHS_STAGE2=5 ./run_train_vqvae.sh 2 runs/vqvae_mimic_standalone/seed_42/checkpoints/last.ckpt
   ```

3. **Full training:**
   ```bash
   MAX_EPOCHS_STAGE1=100 MAX_EPOCHS_STAGE2=100 ./run_train_vqvae.sh both
   ```

### Workflow 2: Resume Training

If you already have a trained VQ-VAE and want to try different Prior configurations:

```bash
# Train Prior with 3 layers
NUM_LAYERS=3 HIDDEN_DIM=128 \
EXP_NAME_STAGE2=prior_3layers \
./run_train_vqvae.sh 2 runs/vqvae_mimic_standalone/seed_42/checkpoints/best.ckpt

# Train Prior with 5 layers
NUM_LAYERS=5 HIDDEN_DIM=256 \
EXP_NAME_STAGE2=prior_5layers \
./run_train_vqvae.sh 2 runs/vqvae_mimic_standalone/seed_42/checkpoints/best.ckpt
```

### Workflow 3: Different Seeds

Train multiple models with different random seeds:

```bash
# Seed 42
SEED=42 EXP_NAME_STAGE1=vqvae_seed42 ./run_train_vqvae.sh 1

# Seed 123
SEED=123 EXP_NAME_STAGE1=vqvae_seed123 ./run_train_vqvae.sh 1

# Seed 456
SEED=456 EXP_NAME_STAGE1=vqvae_seed456 ./run_train_vqvae.sh 1
```

### Workflow 4: Hyperparameter Search for Stage 2

```bash
# Get the VQ-VAE checkpoint
VQVAE_CKPT=runs/vqvae_mimic_standalone/seed_42/checkpoints/best.ckpt

# Try different hidden dimensions
for HIDDEN in 64 128 256; do
    HIDDEN_DIM=$HIDDEN \
    EXP_NAME_STAGE2=prior_hidden${HIDDEN} \
    ./run_train_vqvae.sh 2 $VQVAE_CKPT
done

# Try different number of layers
for LAYERS in 2 3 4 5; do
    NUM_LAYERS=$LAYERS \
    EXP_NAME_STAGE2=prior_layers${LAYERS} \
    ./run_train_vqvae.sh 2 $VQVAE_CKPT
done
```

## Environment Variables Reference

### Required
- `DATA_DIR` - Path to MIMIC-IV-ECG dataset (default: `/work/vajira/DATA/SEARCH/MIMIC_IV_ECG_raw_v1/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0`)

### Experiment Settings
- `EXP_NAME_STAGE1` - Stage 1 experiment name (default: `vqvae_mimic_standalone`)
- `EXP_NAME_STAGE2` - Stage 2 experiment name (default: `prior_mimic_standalone`)
- `SEED` - Random seed (default: `42`)
- `RUNS_ROOT` - Root directory for outputs (default: `runs`)
- `VQVAE_CHECKPOINT` - Path to VQ-VAE checkpoint for Stage 2

### Data Settings
- `BATCH_SIZE` - Batch size (default: `32`)
- `NUM_WORKERS` - Data loading workers (default: `4`)
- `MAX_SAMPLES` - Limit dataset size (default: full dataset)
- `VAL_SPLIT` - Validation split ratio (default: `0.1`)
- `TEST_SPLIT` - Test split ratio (default: `0.1`)

### Model Settings (Stage 1)
- `IN_CHANNELS` - Number of ECG leads (default: `12`)
- `BASE_CHANNELS` - Base channels (default: `64`)
- `LATENT_CHANNELS` - Latent channels (default: `64`)
- `NUM_RES_BLOCKS` - Residual blocks per stage (default: `2`)
- `NUM_EMBEDDINGS` - Codebook size (default: `512`)
- `COMMITMENT_COST` - Commitment loss weight (default: `0.25`)
- `SEQ_LENGTH` - ECG sequence length (default: `5000`)

### Model Settings (Stage 2)
- `HIDDEN_DIM` - Hidden dimension (default: `128`)
- `NUM_LAYERS` - Number of gated conv layers (default: `3`)

### Training Settings
- `LR_STAGE1` - Stage 1 learning rate (default: `0.0001`)
- `LR_STAGE2` - Stage 2 learning rate (default: `0.001`)
- `MAX_EPOCHS_STAGE1` - Stage 1 max epochs (default: `10`)
- `MAX_EPOCHS_STAGE2` - Stage 2 max epochs (default: `10`)
- `ACCELERATOR` - Device type (default: `gpu`)
- `DEVICES` - Device IDs (default: `0`)
- `GRADIENT_CLIP` - Gradient clipping value (default: `1.0`)
- `PATIENCE` - Early stopping patience (default: `10`)
- `SAVE_TOP_K` - Number of best checkpoints to save (default: `3`)

### Weights & Biases
- `WANDB_ENABLED` - Enable W&B logging (default: `true`)
- `WANDB_PROJECT` - W&B project name (default: `ecg-vqvae`)
- `WANDB_ENTITY` - W&B username/team (default: empty)
- `WANDB_RUN_NAME` - W&B run name (default: auto-generated)
- `WANDB_TAGS` - W&B tags (default: empty)

## Finding Checkpoints

If you're not sure which checkpoint to use for Stage 2, the script will automatically search for available checkpoints when you run:

```bash
./run_train_vqvae.sh 2
```

It will display a list of found checkpoints and suggest a command to use them.

Alternatively, you can manually search:

```bash
# Find all VQ-VAE checkpoints
find runs -path "*/vqvae*/seed_*/checkpoints/*.ckpt" -type f

# Find the most recent checkpoint
find runs -path "*/vqvae*/seed_*/checkpoints/*.ckpt" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir=runs
```

Then open http://localhost:6006 in your browser.

### Weights & Biases

If W&B is enabled, the script will print a link to your run dashboard.

## Troubleshooting

### "VQ-VAE checkpoint not specified"

You're trying to run Stage 2 without providing a checkpoint. Use one of these:

```bash
./run_train_vqvae.sh 2 path/to/checkpoint.ckpt
# or
VQVAE_CHECKPOINT=path/to/checkpoint.ckpt ./run_train_vqvae.sh 2
```

### "Data directory not found"

Set the correct path to your MIMIC-IV-ECG dataset:

```bash
DATA_DIR=/path/to/your/mimic-iv-ecg ./run_train_vqvae.sh 1
```

### Out of Memory

Reduce batch size:

```bash
BATCH_SIZE=16 ./run_train_vqvae.sh 1
```

### Training Too Slow

Increase number of data loading workers:

```bash
NUM_WORKERS=8 ./run_train_vqvae.sh 1
```

## Tips

1. **Start small**: Use `MAX_SAMPLES=100` to quickly test your configuration
2. **Save checkpoints frequently**: The default `SAVE_TOP_K=3` keeps the 3 best models
3. **Use descriptive experiment names**: Makes it easier to track different runs
4. **Monitor with W&B**: Easier than TensorBoard for comparing multiple runs
5. **Try different seeds**: Train with multiple seeds for more robust results
