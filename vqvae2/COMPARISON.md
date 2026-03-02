# VQ-VAE vs VQ-VAE-2: Key Differences

## Architecture Comparison

### VQ-VAE (Single-Level)

```
Input (12 × 5000)
    ↓
Encoder (stride 8)
    ↓
Latent (64 × 625)
    ↓
Vector Quantizer
    ↓
Quantized (64 × 625)
    ↓
Decoder (stride 8)
    ↓
Output (12 × 5000)
```

**Characteristics:**
- Single codebook (512 codes)
- Single resolution latent space
- Direct reconstruction from quantized codes
- Good for local features

### VQ-VAE-2 (Two-Level Hierarchical)

```
Input (12 × 5000)
    ↓
Bottom Encoder (stride 8)
    ↓
Bottom Latent (64 × 625) ──────────┐
    ↓                               │
Top Encoder (stride 8)              │
    ↓                               │
Top Latent (64 × 79)                │
    ↓                               │
Top VQ                              │
    ↓                               │
Top Quantized (64 × 79)             │
    ↓                               │
Top Decoder (upsample)              │
    ↓                               │
Top Upsampled (64 × 625) ───────►  │
    ↓                               │
Bottom Latent + Top (conditioning)  │
    ↓                               │
Bottom VQ                           │
    ↓                               │
Bottom Quantized (64 × 625) ────────┘
    ↓
Bottom Decoder (with top conditioning)
    ↓
Output (12 × 5000)
```

**Characteristics:**
- Two codebooks (512 top + 512 bottom)
- Hierarchical latent space (coarse + fine)
- Top features condition bottom quantization
- Better for both local and global features

## Code Structure Comparison

### VQ-VAE Implementation

```python
class VQVAE1D(nn.Module):
    def __init__(self):
        self.encoder = Encoder1D(...)
        self.vq = VectorQuantizer(...)
        self.decoder = Decoder1D(...)
    
    def forward(self, x):
        z = self.encoder(x)
        vq_loss, z_q, indices = self.vq(z)
        recon = self.decoder(z_q)
        return recon, vq_loss, indices
```

### VQ-VAE-2 Implementation

```python
class VQVAE2(nn.Module):
    def __init__(self):
        # Bottom level (local features)
        self.enc_bot = Encoder1d(...)
        self.vq_bot = VectorQuantiser(...)
        self.dec_bot = Decoder1d(..., cond_channels=D)
        
        # Top level (global features)
        self.enc_top = Encoder1d(...)
        self.vq_top = VectorQuantiser(...)
        self.dec_top = Decoder1d(...)
    
    def encode(self, x):
        # Encode bottom
        z_bot = self.enc_bot(x)
        
        # Encode and quantize top
        z_top = self.enc_top(z_bot)
        z_q_top, loss_top, codes_top = self.vq_top(z_top)
        
        # Condition bottom on top
        top_up = self.dec_top(z_q_top)
        z_bot_cond = z_bot + top_up
        z_q_bot, loss_bot, codes_bot = self.vq_bot(z_bot_cond)
        
        return z_q_bot, z_q_top, loss_bot, loss_top, codes_bot, codes_top
    
    def decode(self, z_q_bot, z_q_top):
        top_up = self.dec_top(z_q_top)
        x_recon = self.dec_bot(z_q_bot, cond=top_up)
        return x_recon
```

## Loss Function Comparison

### VQ-VAE Loss

```python
recon_loss = MSE(x_recon, x)
vq_loss = VQ_loss(z, z_q)
total_loss = recon_loss + vq_loss
```

### VQ-VAE-2 Loss

```python
recon_loss = L1(x_recon, x)  # More robust for ECG
vq_loss_bot = VQ_loss(z_bot, z_q_bot)
vq_loss_top = VQ_loss(z_top, z_q_top)
total_loss = recon_loss + vq_loss_bot + vq_loss_top
```

## Performance Characteristics

| Metric | VQ-VAE | VQ-VAE-2 |
|--------|--------|----------|
| **Parameters** | ~2.5M | ~4.5M |
| **Training Time** | 1x | ~1.5x |
| **Codebook Size** | 512 | 512 + 512 = 1024 |
| **Latent Compression** | 8x (5000→625) | 64x (5000→79) + 8x (5000→625) |
| **Reconstruction Quality** | Good | Better |
| **Global Structure** | Limited | Excellent |
| **Local Details** | Good | Excellent |

## When to Use Each

### Use VQ-VAE When:
- Simpler model is sufficient
- Training time is critical
- Memory is limited
- Local features are most important
- Faster inference is needed

### Use VQ-VAE-2 When:
- Best reconstruction quality is needed
- Both local and global features are important
- Modeling long-range dependencies
- Sufficient computational resources
- Hierarchical generation is desired

## Codebook Usage

### VQ-VAE
- Single codebook learns all features
- May struggle with multi-scale features
- Codebook usage typically 60-80%

### VQ-VAE-2
- Top codebook: Global patterns (rhythm, rate)
- Bottom codebook: Local patterns (QRS, P/T waves)
- Better codebook utilization (70-90%)
- More interpretable latent space

## Training Considerations

### VQ-VAE
```bash
# Typical training
python train_vqvae_standalone.py --stage 1 \
    --data-dir /path/to/mimic \
    --batch-size 128 \
    --max-epochs 100
```

### VQ-VAE-2
```bash
# Requires more memory, smaller batch size
python vqvae2.py fit \
    --data-dir /path/to/mimic \
    --batch-size 32 \
    --max-epochs 200
```

## Sampling Comparison

### VQ-VAE Sampling
```python
# Need prior model (PixelCNN)
prior = PixelCNNPrior(...)
codes = prior.sample(batch_size=16)
samples = vqvae.decode_from_indices(codes)
```

### VQ-VAE-2 Sampling
```python
# Need two prior models (hierarchical PixelCNN)
prior_top = PixelCNNPrior(...)
prior_bot = PixelCNNPrior(..., cond_on_top=True)

# Sample top first
codes_top = prior_top.sample(batch_size=16)

# Sample bottom conditioned on top
codes_bot = prior_bot.sample(batch_size=16, cond=codes_top)

# Decode
samples = vqvae2.decode_codes(codes_bot, codes_top)
```

## ECG-Specific Considerations

### VQ-VAE
- **Strengths**: 
  - Good at capturing QRS morphology
  - Fast training and inference
  - Sufficient for beat-level generation
  
- **Limitations**:
  - May miss long-range rhythm patterns
  - Struggles with segment-level coherence
  - Limited control over global structure

### VQ-VAE-2
- **Strengths**:
  - Captures both beat and rhythm
  - Better segment-level coherence
  - Top level can control rhythm/rate
  - Bottom level handles morphology
  - More realistic 10-second ECGs
  
- **Limitations**:
  - Slower training
  - More complex to tune
  - Requires more memory

## Recommended Usage

### For Research/Experimentation
Start with VQ-VAE for faster iteration, then move to VQ-VAE-2 for better quality.

### For Production/Publication
Use VQ-VAE-2 for best results, especially for:
- Full 10-second ECG generation
- Rhythm modeling
- Multi-lead coherence
- Clinical applications

## File Comparison

| File | VQ-VAE | VQ-VAE-2 |
|------|--------|----------|
| Main Script | `train_vqvae_standalone.py` (1321 lines) | `vqvae2.py` (1141 lines) |
| Shell Script | `run_train_vqvae.sh` (332 lines) | `run_train_vqvae2.sh` (326 lines) |
| Model Class | `VQVAE1D` | `VQVAE2` |
| Lightning Module | `VQVAELightning` | `VQVAE2Lightning` |
| Data Module | `VQVAEMIMICDataModule` | `VQVAE2MIMICDataModule` |

## Migration Guide

If you have a trained VQ-VAE model and want to try VQ-VAE-2:

1. **Keep the same data preprocessing**
   - Both use identical dataset loaders
   - Same normalization strategy

2. **Adjust hyperparameters**
   - Reduce batch size (128 → 32)
   - Increase epochs (100 → 200)
   - Consider larger codebooks (512 → 1024)

3. **Monitor new metrics**
   - `codebook_usage_top`
   - `codebook_usage_bot`
   - Both should be > 0.7 for good performance

4. **Visualization**
   - Check both local (QRS) and global (rhythm) features
   - Compare 10-second reconstructions, not just beats

## Summary

**VQ-VAE**: Simpler, faster, good for local features
**VQ-VAE-2**: More complex, better quality, captures both local and global features

For 12-lead ECG generation, VQ-VAE-2 is recommended for production use due to its superior ability to model both morphology and rhythm.
