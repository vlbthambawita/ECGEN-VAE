# Weights & Biases Integration for VQ-VAE-2 Prior Training

## Overview

The transformer prior training now supports Weights & Biases (wandb) for experiment tracking and visualization. Wandb logging is **enabled by default** and runs alongside CSV logging for maximum flexibility.

## Features

- **Automatic metric logging**: Loss, accuracy, and learning rate tracked in real-time
- **Dual logging**: Both wandb and CSV logs are maintained
- **Easy configuration**: Control via environment variables or CLI arguments
- **Graceful fallback**: If wandb is not installed, training continues with CSV logging only

## Quick Start

### Enable Wandb (Default)

```bash
# Using shell script (wandb enabled by default)
./run_train_prior.sh fit_top
./run_train_prior.sh fit_bot

# Using Python directly
python transformer_prior.py fit_top \
    --codes-dir codes/vqvae2_mimic \
    --wandb \
    --wandb-project vqvae2-prior \
    --wandb-run-name top_prior_exp1
```

### Disable Wandb

```bash
# Using shell script
WANDB_ENABLED=false ./run_train_prior.sh fit_top

# Using Python directly (omit --wandb flag)
python transformer_prior.py fit_top \
    --codes-dir codes/vqvae2_mimic
```

## Configuration

### Environment Variables (Shell Script)

| Variable | Default | Description |
|----------|---------|-------------|
| `WANDB_ENABLED` | `true` | Enable/disable wandb logging |
| `WANDB_PROJECT` | `vqvae2-prior` | Wandb project name |
| `WANDB_ENTITY` | `` | Wandb entity (username or team) |
| `WANDB_RUN_NAME` | `` | Custom run name (optional) |

### CLI Arguments (Python)

| Argument | Type | Description |
|----------|------|-------------|
| `--wandb` | flag | Enable wandb logging |
| `--wandb-project` | str | Wandb project name |
| `--wandb-entity` | str | Wandb entity (username/team) |
| `--wandb-run-name` | str | Custom run name |

## Usage Examples

### Basic Training with Wandb

```bash
# Top prior
./run_train_prior.sh fit_top

# Bottom prior
./run_train_prior.sh fit_bot
```

### Custom Project Name

```bash
WANDB_PROJECT=my-ecg-experiments ./run_train_prior.sh fit_top
```

### Custom Run Name

```bash
WANDB_RUN_NAME=top_prior_large_model ./run_train_prior.sh fit_top
```

### Specify Entity (Team/Username)

```bash
WANDB_ENTITY=my-team WANDB_PROJECT=ecg-generation ./run_train_prior.sh fit_top
```

### Complete Custom Configuration

```bash
WANDB_ENABLED=true \
WANDB_PROJECT=ecg-vqvae2 \
WANDB_ENTITY=research-lab \
WANDB_RUN_NAME=top_prior_v1 \
./run_train_prior.sh fit_top
```

### Python CLI Examples

```bash
# Basic with wandb
python transformer_prior.py fit_top \
    --codes-dir codes/vqvae2_mimic \
    --wandb

# Custom project and run name
python transformer_prior.py fit_top \
    --codes-dir codes/vqvae2_mimic \
    --wandb \
    --wandb-project my-ecg-project \
    --wandb-run-name experiment_001

# With entity
python transformer_prior.py fit_bot \
    --codes-dir codes/vqvae2_mimic \
    --wandb \
    --wandb-project ecg-generation \
    --wandb-entity my-team \
    --wandb-run-name bot_prior_baseline
```

## Logged Metrics

### Training Metrics

- `train/loss` - Cross-entropy loss during training
- `train/acc` - Token prediction accuracy during training
- `epoch` - Current epoch number

### Validation Metrics

- `val/loss` - Validation loss
- `val/acc` - Validation accuracy

### System Metrics

- Learning rate (via LearningRateMonitor callback)
- Epoch timing
- GPU utilization (automatic)

## Viewing Results

### Wandb Dashboard

1. Visit https://wandb.ai/
2. Navigate to your project: `https://wandb.ai/<entity>/<project>`
3. View runs, compare experiments, and analyze metrics

### Local CSV Logs

CSV logs are always saved regardless of wandb status:

```bash
# Top prior logs
cat logs/top_prior/version_0/metrics.csv

# Bottom prior logs
cat logs/bot_prior/version_0/metrics.csv
```

## Installation

If wandb is not installed:

```bash
pip install wandb
```

First time setup:

```bash
wandb login
```

## Troubleshooting

### "wandb not installed" Warning

**Issue**: Warning message about wandb not being installed

**Solution**: 
```bash
pip install wandb
wandb login
```

Or disable wandb:
```bash
WANDB_ENABLED=false ./run_train_prior.sh fit_top
```

### Authentication Error

**Issue**: Wandb authentication fails

**Solution**:
```bash
wandb login
# Follow prompts to enter your API key
```

### Offline Mode

To run without internet connection:

```bash
export WANDB_MODE=offline
./run_train_prior.sh fit_top
```

Later sync:
```bash
wandb sync logs/top_prior/version_0/wandb/
```

## Best Practices

### 1. Consistent Naming

Use descriptive run names:
```bash
WANDB_RUN_NAME="top_prior_d256_l8_lr3e4" ./run_train_prior.sh fit_top
```

### 2. Project Organization

Group related experiments:
```bash
WANDB_PROJECT="vqvae2-ablation" ./run_train_prior.sh fit_top
```

### 3. Tags

Add tags via wandb config (in code) for better organization:
```python
wandb_logger = WandbLogger(
    project="vqvae2-prior",
    tags=["baseline", "top-prior", "d256"]
)
```

### 4. Hyperparameter Tracking

All hyperparameters are automatically logged via PyTorch Lightning's `save_hyperparameters()`

## Comparison with CSV Logging

| Feature | Wandb | CSV |
|---------|-------|-----|
| Real-time visualization | ✓ | ✗ |
| Experiment comparison | ✓ | Manual |
| Hyperparameter tracking | ✓ | Limited |
| Model artifacts | ✓ | ✗ |
| Collaboration | ✓ | ✗ |
| Offline access | ✓ | ✓ |
| No dependencies | ✗ | ✓ |

**Recommendation**: Use both (default behavior) for maximum flexibility.

## Advanced Usage

### Custom Logging

To log additional metrics, modify the Lightning modules:

```python
class TopPriorLightning(pl.LightningModule):
    def training_step(self, batch, _):
        loss = self._step(batch, "train")
        
        # Log custom metrics
        self.log("custom/metric", value)
        
        return loss
```

### Artifact Logging

Save model artifacts to wandb:

```python
# In trainer callback
wandb.save(checkpoint_path)
```

### Sweeps (Hyperparameter Search)

Create a sweep configuration:

```yaml
# sweep.yaml
program: transformer_prior.py
method: bayes
metric:
  name: val/loss
  goal: minimize
parameters:
  lr:
    min: 0.0001
    max: 0.001
  d_model:
    values: [128, 256, 512]
  n_layers:
    values: [6, 8, 12]
```

Run sweep:
```bash
wandb sweep sweep.yaml
wandb agent <sweep-id>
```

## Integration Details

### Code Structure

1. **Import**: Optional wandb import with fallback
2. **Logger Setup**: `build_trainer()` creates both CSV and wandb loggers
3. **Automatic Logging**: PyTorch Lightning handles all metric logging
4. **Graceful Degradation**: Training continues if wandb unavailable

### Files Modified

- `transformer_prior.py`: Added wandb support
- `run_train_prior.sh`: Added wandb environment variables
- `PRIOR_README.md`: Updated documentation
- `PRIOR_QUICK_START.md`: Added wandb examples

## Summary

Weights & Biases integration provides:
- ✓ Real-time experiment tracking
- ✓ Easy experiment comparison
- ✓ Automatic hyperparameter logging
- ✓ Collaborative features
- ✓ Graceful fallback to CSV
- ✓ Zero code changes required for basic usage

**Default behavior**: Wandb enabled, CSV always available as backup.
