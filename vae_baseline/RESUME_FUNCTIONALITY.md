# Resume Functionality Implementation

## Overview

Resume functionality has been added to both the Python training script and the shell wrapper script, allowing training to be interrupted and continued from checkpoints.

## Changes Made

### 1. Python Script (`train_vae_baseline.py`)

#### Added Argument
```python
parser.add_argument("--resume", type=str, default=None, 
                   help="Path to checkpoint to resume training from")
```

#### Modified Training Function
```python
# Check for resume checkpoint
ckpt_path = None
if args.resume:
    if os.path.exists(args.resume):
        ckpt_path = args.resume
        print(f"Resuming from checkpoint: {ckpt_path}")
    else:
        print(f"WARNING: Resume checkpoint not found: {args.resume}")
        print("Starting fresh training instead")

trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
```

### 2. Shell Script (`run_train_vae_baseline.sh`)

The shell script already had comprehensive resume support with three modes:
- **Auto mode**: Automatically detects and resumes from `last.ckpt` if it exists
- **Train mode**: Always starts fresh training
- **Resume mode**: Explicitly resumes from checkpoint

## Usage

### Python Script Direct Usage

#### Resume from specific checkpoint
```bash
python train_vae_baseline.py \
  --data-dir /path/to/mimic-iv-ecg \
  --exp-name my_experiment \
  --resume runs/my_experiment/seed_42/checkpoints/last.ckpt
```

#### Resume from epoch checkpoint
```bash
python train_vae_baseline.py \
  --data-dir /path/to/mimic-iv-ecg \
  --exp-name my_experiment \
  --resume runs/my_experiment/seed_42/checkpoints/epoch050-step005000.ckpt
```

### Shell Script Usage

#### Auto-resume (recommended)
```bash
./run_train_vae_baseline.sh
```
Automatically resumes if `last.ckpt` exists, otherwise starts fresh.

#### Explicit resume
```bash
./run_train_vae_baseline.sh resume
```
Always tries to resume, fails if no checkpoint found.

#### Fresh training
```bash
./run_train_vae_baseline.sh train
```
Always starts fresh, ignoring any existing checkpoints.

#### Resume from specific checkpoint
```bash
RESUME_CHECKPOINT=runs/exp/seed_42/checkpoints/epoch050.ckpt \
./run_train_vae_baseline.sh resume
```

## Checkpoint Structure

PyTorch Lightning automatically saves checkpoints in the following structure:
```
runs/
└── {exp_name}/
    └── seed_{seed}/
        └── checkpoints/
            ├── epoch001-step000100.ckpt
            ├── epoch002-step000200.ckpt
            ├── ...
            └── last.ckpt  (most recent checkpoint)
```

## What Gets Resumed

When resuming from a checkpoint, PyTorch Lightning restores:
- Model weights and biases
- Optimizer state (Adam momentum, etc.)
- Learning rate scheduler state
- Current epoch number
- Global step count
- Training/validation metrics history

This ensures seamless continuation of training.

## Error Handling

### Python Script
- If `--resume` is specified but file doesn't exist: Prints warning and starts fresh training
- If `--resume` is not specified: Normal fresh training

### Shell Script
- **Auto mode**: Gracefully handles missing checkpoint (starts fresh)
- **Resume mode**: Exits with error if checkpoint not found, provides helpful message
- **Train mode**: Ignores checkpoints completely

## Best Practices

### 1. Use Auto Mode for Convenience
```bash
# First run - starts training
./run_train_vae_baseline.sh

# If interrupted, just run again - automatically resumes
./run_train_vae_baseline.sh
```

### 2. Use last.ckpt for Latest State
```bash
python train_vae_baseline.py \
  --data-dir /path/to/data \
  --exp-name my_exp \
  --resume runs/my_exp/seed_42/checkpoints/last.ckpt
```

### 3. Use Epoch Checkpoints for Specific Points
```bash
# Resume from a specific epoch (e.g., before overfitting started)
python train_vae_baseline.py \
  --data-dir /path/to/data \
  --exp-name my_exp \
  --resume runs/my_exp/seed_42/checkpoints/epoch030-step003000.ckpt
```

### 4. Combine with Different Hyperparameters
```bash
# Resume but with different learning rate
python train_vae_baseline.py \
  --data-dir /path/to/data \
  --exp-name my_exp \
  --resume runs/my_exp/seed_42/checkpoints/last.ckpt \
  --lr 0.00005  # Lower learning rate for fine-tuning
```

## Common Scenarios

### Scenario 1: Training Interrupted by System
```bash
# Training was running
./run_train_vae_baseline.sh

# System crashed/restarted
# Just run again - auto-resumes from last.ckpt
./run_train_vae_baseline.sh
```

### Scenario 2: Want to Train Longer
```bash
# Original training with 100 epochs
./run_train_vae_baseline.sh train

# Resume and train for 50 more epochs
MAX_EPOCHS=150 ./run_train_vae_baseline.sh resume
```

### Scenario 3: Fine-tune with Different Settings
```bash
# Initial training
./run_train_vae_baseline.sh train

# Fine-tune with lower learning rate
LR=0.00005 MAX_EPOCHS=150 ./run_train_vae_baseline.sh resume
```

### Scenario 4: Resume from Specific Checkpoint
```bash
# Resume from epoch 30 instead of last
RESUME_CHECKPOINT=runs/vae_baseline_mimic/seed_42/checkpoints/epoch030.ckpt \
./run_train_vae_baseline.sh resume
```

## Verification

To verify resume functionality works:

1. **Start training**:
```bash
MAX_EPOCHS=5 ./run_train_vae_baseline.sh train
```

2. **Check checkpoint was created**:
```bash
ls runs/vae_baseline_mimic/seed_42/checkpoints/
```

3. **Resume training**:
```bash
MAX_EPOCHS=10 ./run_train_vae_baseline.sh resume
```

4. **Verify epoch continues** from where it left off (check logs)

## Integration with PyTorch Lightning

The resume functionality leverages PyTorch Lightning's built-in checkpoint system:
- `trainer.fit(model, datamodule, ckpt_path=path)` handles all resume logic
- Automatically restores model, optimizer, and training state
- Continues from the exact step where training stopped
- Preserves all metrics and logging history

## Compatibility

- Compatible with all model configurations
- Works with both single-GPU and multi-GPU training
- Supports resuming with different batch sizes (may affect learning dynamics)
- Compatible with both TensorBoard and W&B logging
- Checkpoint format is PyTorch Lightning standard (.ckpt files)

## Troubleshooting

### Issue: "Checkpoint not found"
**Solution**: Check the path is correct
```bash
ls runs/*/seed_*/checkpoints/
```

### Issue: "Incompatible checkpoint"
**Solution**: Checkpoint may be from different model architecture. Start fresh training.

### Issue: "Training starts from epoch 0"
**Solution**: Checkpoint path not passed correctly. Verify `--resume` argument.

### Issue: "Different results after resume"
**Solution**: This is normal due to:
- Different data shuffling order
- Randomness in dropout/augmentation
- Results should converge to similar final performance

## Summary

Resume functionality is now fully integrated into both the Python script and shell wrapper:
- ✅ Python script supports `--resume` argument
- ✅ Shell script has auto-resume, train, and resume modes
- ✅ Comprehensive error handling
- ✅ Documentation updated
- ✅ Compatible with PyTorch Lightning checkpoint system

The implementation provides flexible options for resuming training while maintaining simplicity for common use cases.
