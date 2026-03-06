# ECG Sample Generation Guide

This guide explains how to generate synthetic ECG samples from your trained VQ-VAE + Prior models.

## Quick Start

### Method 1: Using the Shell Script (Easiest)

```bash
# Generate 16 samples with default temperature (1.0)
./generate.sh runs/prior_mimic_standalone/seed_42/checkpoints/best.ckpt

# Generate 32 samples
./generate.sh runs/prior_mimic_standalone/seed_42/checkpoints/best.ckpt 32

# Generate 32 samples with temperature 0.8
./generate.sh runs/prior_mimic_standalone/seed_42/checkpoints/best.ckpt 32 0.8
```

### Method 2: Using the Python Script Directly

```bash
# Basic generation
python generate_samples.py --prior-checkpoint runs/prior_mimic_standalone/seed_42/checkpoints/best.ckpt

# With custom settings
python generate_samples.py \
    --prior-checkpoint runs/prior_mimic_standalone/seed_42/checkpoints/best.ckpt \
    --n-samples 32 \
    --temperature 0.8 \
    --output-dir my_generated_samples
```

## Understanding Temperature

Temperature controls the diversity of generated samples:

- **T = 0.5**: More conservative, samples closer to training data (less diverse)
- **T = 1.0**: Balanced (default, recommended for most cases)
- **T = 1.5**: More creative, higher diversity (may be less realistic)
- **T = 2.0**: Very diverse (may produce artifacts)

### Comparing Different Temperatures

Generate samples with multiple temperatures to find the best balance:

```bash
python generate_samples.py \
    --prior-checkpoint runs/prior_mimic_standalone/seed_42/checkpoints/epoch098-step486981.ckpt \
    --n-samples 16 \
    --temperatures 0.5 0.8 1.0 1.2 1.5
```

This will create separate outputs for each temperature.

## Output Files

After generation, you'll find these files in the output directory:

```
generated_samples/
├── samples_grid_n16_t1.00.png          # Grid visualization of all samples
├── samples_n16_t1.00.npy               # Raw samples as numpy array [16, 12, 5000]
└── individual/
    ├── sample_000_t1.00.png            # Individual sample plots
    ├── sample_001_t1.00.png
    ├── sample_002_t1.00.png
    └── ...
```

### File Descriptions

1. **Grid Plot** (`samples_grid_n16_t1.00.png`):
   - Shows all samples in a grid layout
   - Each subplot shows all 12 ECG leads stacked vertically
   - Good for quick overview of sample quality

2. **Numpy Array** (`samples_n16_t1.00.npy`):
   - Raw sample data: shape `[N, 12, 5000]`
   - Can be loaded with `np.load()` for further analysis
   - Useful for quantitative evaluation

3. **Individual Plots** (`individual/sample_*.png`):
   - Detailed view of each sample
   - All 12 leads plotted separately
   - Good for detailed quality inspection

## Common Use Cases

### 1. Quick Quality Check

Generate a few samples to check if the model is working:

```bash
./generate.sh runs/prior_mimic_standalone/seed_42/checkpoints/best.ckpt 4 1.0
```

### 2. Generate Dataset for Evaluation

Generate many samples for quantitative evaluation:

```bash
python generate_samples.py \
    --prior-checkpoint runs/prior_mimic_standalone/seed_42/checkpoints/best.ckpt \
    --n-samples 1000 \
    --temperature 1.0 \
    --output-dir evaluation_samples \
    --no-individual  # Skip individual plots to save time
```

### 3. Temperature Sweep

Find the optimal temperature for your use case:

```bash
# Generate with multiple temperatures
for temp in 0.5 0.7 0.9 1.0 1.1 1.3 1.5; do
    TEMPERATURE=$temp OUTPUT_DIR="temp_sweep/t${temp}" \
    ./generate.sh runs/prior_mimic_standalone/seed_42/checkpoints/best.ckpt 16
done
```

### 4. Compare Different Model Checkpoints

Compare samples from different training epochs:

```bash
# Early checkpoint
python generate_samples.py \
    --prior-checkpoint runs/prior_mimic_standalone/seed_42/checkpoints/epoch010-step000320.ckpt \
    --n-samples 16 \
    --output-dir comparison/epoch010

# Middle checkpoint
python generate_samples.py \
    --prior-checkpoint runs/prior_mimic_standalone/seed_42/checkpoints/epoch050-step001600.ckpt \
    --n-samples 16 \
    --output-dir comparison/epoch050

# Best checkpoint
python generate_samples.py \
    --prior-checkpoint runs/prior_mimic_standalone/seed_42/checkpoints/best.ckpt \
    --n-samples 16 \
    --output-dir comparison/best
```

### 5. Generate Samples on CPU

If you don't have GPU access:

```bash
python generate_samples.py \
    --prior-checkpoint runs/prior_mimic_standalone/seed_42/checkpoints/best.ckpt \
    --n-samples 16 \
    --device cpu
```

## Advanced Usage

### Disable Specific Outputs

```bash
# Only save numpy array (no plots)
python generate_samples.py \
    --prior-checkpoint runs/prior_mimic_standalone/seed_42/checkpoints/best.ckpt \
    --n-samples 100 \
    --no-grid \
    --no-individual

# Only save grid plot (no numpy or individual plots)
python generate_samples.py \
    --prior-checkpoint runs/prior_mimic_standalone/seed_42/checkpoints/best.ckpt \
    --n-samples 16 \
    --no-npy \
    --no-individual
```

### Custom Sequence Length

Generate ECGs with different lengths:

```bash
python generate_samples.py \
    --prior-checkpoint runs/prior_mimic_standalone/seed_42/checkpoints/best.ckpt \
    --n-samples 16 \
    --seq-length 2500  # Half length
```

**Note**: The model was trained on 5000-sample ECGs, so using different lengths may affect quality.

## Analyzing Generated Samples

### Load Samples in Python

```python
import numpy as np
import matplotlib.pyplot as plt

# Load samples
samples = np.load('generated_samples/samples_n16_t1.00.npy')
print(f"Shape: {samples.shape}")  # [16, 12, 5000]

# Access individual samples
sample_0 = samples[0]  # First sample, all 12 leads
lead_ii = samples[:, 1, :]  # Lead II from all samples

# Plot a specific lead
plt.figure(figsize=(15, 3))
plt.plot(samples[0, 1, :])  # Lead II of first sample
plt.title("Generated ECG - Lead II")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)
plt.show()
```

### Compute Statistics

```python
import numpy as np

samples = np.load('generated_samples/samples_n16_t1.00.npy')

# Overall statistics
print(f"Mean: {np.mean(samples):.4f}")
print(f"Std:  {np.std(samples):.4f}")
print(f"Min:  {np.min(samples):.4f}")
print(f"Max:  {np.max(samples):.4f}")

# Per-lead statistics
lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
for i, lead in enumerate(lead_names):
    lead_data = samples[:, i, :]
    print(f"{lead:3s}: mean={np.mean(lead_data):6.3f}, std={np.std(lead_data):6.3f}")
```

### Compare with Real ECGs

```python
import numpy as np
import matplotlib.pyplot as plt

# Load generated and real samples
generated = np.load('generated_samples/samples_n16_t1.00.npy')
real = np.load('path/to/real_ecgs.npy')  # Your real ECG data

# Compare distributions
fig, axes = plt.subplots(2, 6, figsize=(18, 6))
lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

for i, (ax, lead) in enumerate(zip(axes.flat, lead_names)):
    ax.hist(real[:, i, :].flatten(), bins=50, alpha=0.5, label='Real', density=True)
    ax.hist(generated[:, i, :].flatten(), bins=50, alpha=0.5, label='Generated', density=True)
    ax.set_title(f"Lead {lead}")
    ax.legend()
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Density")

plt.tight_layout()
plt.savefig('distribution_comparison.png', dpi=150)
```

## Quality Assessment Checklist

When evaluating generated samples, check for:

### Visual Inspection

- [ ] **P waves**: Are they present and realistic?
- [ ] **QRS complex**: Is it well-formed and sharp?
- [ ] **T waves**: Are they present with appropriate amplitude?
- [ ] **Baseline**: Is it stable or does it wander?
- [ ] **Noise**: Are there artifacts or unrealistic patterns?
- [ ] **Lead relationships**: Do leads show expected correlations?

### Quantitative Metrics

- [ ] **Amplitude range**: Similar to real ECGs (-1 to 1 after normalization)?
- [ ] **Heart rate**: Reasonable (40-180 bpm)?
- [ ] **PR interval**: Typical duration (120-200 ms)?
- [ ] **QRS duration**: Normal range (60-100 ms)?
- [ ] **QT interval**: Within normal limits?

### Statistical Comparison

- [ ] **Mean and std**: Close to real ECG statistics?
- [ ] **Distribution**: Similar to real ECG distribution?
- [ ] **Frequency content**: Appropriate power spectrum?
- [ ] **Diversity**: Samples are different from each other?

## Troubleshooting

### "Prior checkpoint not found"

**Problem**: The checkpoint file doesn't exist.

**Solution**: 
```bash
# Find available checkpoints
find runs -path "*/prior*/seed_*/checkpoints/*.ckpt" -type f

# Or just run without arguments to see suggestions
./generate.sh
```

### "VQ-VAE not loaded in Prior model"

**Problem**: The Prior checkpoint doesn't have the VQ-VAE loaded.

**Solution**: Make sure you're using a Prior checkpoint (Stage 2), not a VQ-VAE checkpoint (Stage 1). The Prior checkpoint should have been trained with `--vqvae-checkpoint` pointing to a VQ-VAE model.

### Samples Look Unrealistic

**Problem**: Generated samples don't look like real ECGs.

**Solutions**:
1. **Try different temperature**: Start with 0.8 or 1.0
2. **Check training**: Was the model trained long enough?
3. **Use best checkpoint**: Use the checkpoint with lowest validation loss
4. **Inspect training samples**: Look at reconstruction quality from Stage 1

### Out of Memory

**Problem**: GPU runs out of memory during generation.

**Solutions**:
```bash
# Generate fewer samples at a time
python generate_samples.py --prior-checkpoint <path> --n-samples 8

# Use CPU instead
python generate_samples.py --prior-checkpoint <path> --device cpu

# Or generate in batches
for i in {1..10}; do
    python generate_samples.py \
        --prior-checkpoint <path> \
        --n-samples 10 \
        --output-dir batch_${i}
done
```

### Generation is Slow

**Problem**: Generating samples takes too long.

**Solutions**:
1. **Use GPU**: Make sure `--device cuda` is set
2. **Reduce samples**: Generate fewer samples
3. **Skip individual plots**: Use `--no-individual` flag
4. **Batch generation**: Generate many samples in one call instead of multiple small batches

## Tips for Best Results

1. **Start with default temperature (1.0)**: Adjust only if needed
2. **Use the best checkpoint**: Lowest validation loss usually gives best results
3. **Generate multiple batches**: Check consistency across different generations
4. **Compare with training data**: Ensure generated samples match the style
5. **Try temperature sweep**: Find the sweet spot for your use case
6. **Save raw numpy arrays**: Useful for quantitative evaluation later
7. **Visual inspection is key**: Plots are more informative than metrics alone

## Next Steps

After generating samples:

1. **Visual Quality Check**: Review the grid and individual plots
2. **Quantitative Evaluation**: Compute metrics comparing to real ECGs
3. **Clinical Validation**: Have domain experts review the samples
4. **Use Cases**: Apply generated ECGs to your downstream tasks
5. **Model Refinement**: If quality is poor, retrain with adjusted hyperparameters

## Example Workflow

Complete workflow from training to generation:

```bash
# 1. Train VQ-VAE (Stage 1)
./run_train_vqvae.sh 1

# 2. Train Prior (Stage 2)
./run_train_vqvae.sh 2 runs/vqvae_mimic_standalone/seed_42/checkpoints/best.ckpt

# 3. Generate samples for quick check
./generate.sh runs/prior_mimic_standalone/seed_42/checkpoints/best.ckpt 16 1.0

# 4. If quality is good, generate larger dataset
python generate_samples.py \
    --prior-checkpoint runs/prior_mimic_standalone/seed_42/checkpoints/best.ckpt \
    --n-samples 1000 \
    --temperature 1.0 \
    --output-dir final_samples \
    --no-individual

# 5. Analyze results
python -c "
import numpy as np
samples = np.load('final_samples/samples_n1000_t1.00.npy')
print(f'Generated {samples.shape[0]} samples')
print(f'Shape: {samples.shape}')
print(f'Mean: {np.mean(samples):.4f}')
print(f'Std: {np.std(samples):.4f}')
"
```

## References

- See `README.md` for model architecture details
- See `USAGE_EXAMPLES.md` for training examples
- See `STAGE2_ONLY.md` for Stage 2 training guide
