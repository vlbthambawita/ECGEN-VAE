# VQ-VAE-2 Architecture & Training Process

VQ-VAE-2 for 12-lead ECG (Razavi et al., NeurIPS 2019), adapted for signals of shape **B × 12 × 5000** (12 leads, 10 s @ 500 Hz).

---

## 1. High-Level Model Architecture

Two-level hierarchical vector quantization: **bottom** level captures local morphology (QRS, P/T waves); **top** level captures global structure (rhythm, segments).

```mermaid
flowchart TB
    subgraph input[" "]
        X["x: ECG<br/>B × 12 × 5000"]
    end

    subgraph bottom["Bottom level (local)"]
        enc_bot["Encoder Bottom<br/>enc_bot"]
        vq_bot["VectorQuantiser<br/>vq_bot"]
        z_bot["z_bot: B×64×625"]
        z_q_bot["z_q_bot"]
    end

    subgraph top["Top level (global)"]
        enc_top["Encoder Top<br/>enc_top"]
        vq_top["VectorQuantiser<br/>vq_top"]
        z_top["z_top_pre: B×64×79"]
        z_q_top["z_q_top"]
        dec_top["Decoder Top<br/>dec_top"]
        top_up["top_up (conditioning)"]
    end

    subgraph decode["Reconstruction"]
        dec_bot["Decoder Bottom<br/>dec_bot"]
        x_recon["x_recon: B×12×5000"]
    end

    X --> enc_bot
    enc_bot --> z_bot
    z_bot --> enc_top
    enc_top --> z_top
    z_top --> vq_top
    vq_top --> z_q_top
    z_q_top --> dec_top
    dec_top --> top_up
    top_up --> dec_bot
    z_bot --> add["+"]
    top_up --> add
    add --> vq_bot
    vq_bot --> z_q_bot
    z_q_bot --> dec_bot
    dec_bot --> x_recon
```

---

## 2. Temporal Dimensions (Default Strides ×8 per Level)

```mermaid
flowchart LR
    subgraph dims["Temporal dimensions"]
        A["5000"]
        B["625"]
        C["79"]
    end
    A -->|"÷8 (stride 2,2,2)"| B
    B -->|"÷8 (stride 2,2,2)"| C
```

| Stage           | Shape        | Description                    |
|----------------|-------------|--------------------------------|
| Input          | B × 12 × 5000 | 12-lead ECG, 10 s             |
| After enc_bot  | B × 64 × 625  | Bottom latent (stride 8)      |
| After enc_top  | B × 64 × 79   | Top latent (stride ~8)        |

---

## 3. Building Blocks

### 3.1 Residual Block (1-D)

Pre-activation residual block used in both encoder and decoder.

```mermaid
flowchart LR
    x["x"] --> add["+"]
    x --> norm["GroupNorm(8)"]
    norm --> silu["SiLU"]
    silu --> conv1["Conv1d(3)"]
    conv1 --> norm2["GroupNorm(8)"]
    norm2 --> silu2["SiLU"]
    silu2 --> conv2["Conv1d(1)"]
    conv2 --> add
    add --> out["out"]
```

### 3.2 Encoder (1-D)

Strided convolutions halve the time dimension at each stride; then residual blocks.

```mermaid
flowchart LR
    in["Input"] --> c0["Conv1d(in→H)"]
    c0 --> stride["For each stride: GroupNorm→SiLU→Conv1d(stride×2, stride)"]
    stride --> res["N × ResidualBlock1d"]
    res --> out_block["GroupNorm→SiLU→Conv1d(H→D)"]
    out_block --> z["z (latent)"]
```

### 3.3 Decoder (1-D)

Mirrors encoder: residual blocks first, then transposed convolutions; optional conditioning channel-wise.

```mermaid
flowchart LR
    z["z (latent)"] --> cat["Optional: concat cond"]
    cat --> c0["Conv1d(D+cond→H)"]
    c0 --> res["N × ResidualBlock1d"]
    res --> tconv["For each stride (reversed): GroupNorm→SiLU→ConvTranspose1d"]
    tconv --> out_block["GroupNorm→SiLU→Conv1d(H→out)"]
    out_block --> x_recon["Reconstruction"]
```

### 3.4 Vector Quantiser (EMA + Straight-Through)

```mermaid
flowchart TB
    z["z: (B,D,L)"] --> flat["Reshape to (B×L, D)"]
    flat --> dist["Distances to codebook"]
    dist --> codes["argmin → codes (B,L)"]
    codes --> lookup["Embedding lookup → z_q"]
    lookup --> st["Straight-through: grad flows as z"]
    st --> z_q["z_q (B,D,L)"]

    subgraph loss["VQ loss"]
        vq_loss["VQ loss (MSE or EMA)"]
        commit["Commitment: β·MSE(z, z_q.detach)"]
        vq_loss --> total["loss"]
        commit --> total
    end

    codes -.->|"if EMA"| ema["Update EMA: cluster_size, ema_dw"]
    ema -.->|"Update codebook"| lookup
```

- **EMA**: exponential moving average of cluster assignments and embedding weights (no gradient through codebook when EMA is used).
- **Commitment cost** β = 0.25 encourages encoder to output close to chosen codebook vectors.

---

## 4. Encode Path (Step-by-Step)

```mermaid
sequenceDiagram
    participant x as ECG x
    participant enc_bot as enc_bot
    participant enc_top as enc_top
    participant vq_top as vq_top
    participant dec_top as dec_top
    participant vq_bot as vq_bot

    x->>enc_bot: B×12×5000
    enc_bot->>enc_bot: z_bot B×64×625
    enc_bot->>enc_top: z_bot
    enc_top->>enc_top: z_top_pre B×64×79
    enc_top->>vq_top: z_top_pre
    vq_top->>vq_top: z_q_top, loss_top, codes_top
    vq_top->>dec_top: z_q_top
    dec_top->>dec_top: top_up (interpolate to 625 if needed)
    dec_top->>vq_bot: z_bot + top_up → z_bot_cond
    vq_bot->>vq_bot: z_q_bot, loss_bot, codes_bot
```

---

## 5. Decode Path (Reconstruction)

```mermaid
flowchart LR
    z_q_top["z_q_top"] --> dec_top["dec_top"]
    z_q_bot["z_q_bot"] --> dec_bot["dec_bot"]
    dec_top --> top_up["top_up"]
    top_up --> dec_bot
    dec_bot --> x_recon["x_recon B×12×5000"]
```

Decoder bottom takes **z_q_bot** and conditions on **top_up** (decoder-top output), then outputs **x_recon**.

---

## 6. Training Process (Step-by-Step)

End-to-end training flow with PyTorch Lightning.

```mermaid
flowchart TB
    subgraph data["Data"]
        MIMIC["MIMIC-IV-ECG Dataset"]
        batch["Batch: (ecgs, features)"]
        MIMIC --> batch
    end

    subgraph forward["Forward"]
        batch --> model["VQVAE2(ecgs)"]
        model --> encode["encode: z_q_bot, z_q_top, loss_bot, loss_top"]
        encode --> decode["decode(z_q_bot, z_q_top)"]
        decode --> x_recon["x_recon"]
    end

    subgraph loss["Loss"]
        x_recon --> recon_loss["recon_loss = L1(x_recon, ecgs)"]
        encode --> vq_loss["vq_loss = loss_bot + loss_top"]
        recon_loss --> total["total_loss = recon_loss + vq_loss"]
        vq_loss --> total
    end

    subgraph backward["Backward & Optimizer"]
        total --> backward_op["backward()"]
        backward_op --> optimizer["AdamW step"]
        optimizer --> scheduler["CosineAnnealingLR (per epoch)"]
    end

    subgraph logging["Logging"]
        total --> log["Log: total_loss, recon_loss, vq_loss"]
        encode --> codes_log["Log: unique_codes_bot/top, codebook_usage"]
    end

    data --> forward
    forward --> loss
    loss --> backward
    loss --> logging
```

---

## 7. Training Loop (Epoch / Batch)

```mermaid
flowchart TB
    start(["Start training"]) --> setup["DataModule.setup()"]
    setup --> train_dl["train_dataloader()"]
    train_dl --> epoch["For each epoch"]
    epoch --> batch["For each batch"]
    batch --> load["Load batch (ecgs, features)"]
    load --> step["_step(batch, 'train')"]
    step --> forward["model(ecgs) → x_recon, vq_loss, codes"]
    forward --> recon["recon_loss = L1(x_recon, ecgs)"]
    recon --> total["total_loss = recon + vq_loss"]
    total --> backward["loss.backward()"]
    backward --> optimizer["optimizer.step()"]
    optimizer --> log_train["Log train/total_loss, recon_loss, vq_loss"]
    log_train --> batch

    batch --> val_epoch{"Validation epoch?"}
    val_epoch -->|Yes| val_loop["For each val batch"]
    val_loop --> val_step["_step(batch, 'val')"]
    val_step --> val_log["Log val_loss, unique_codes, codebook_usage"]
    val_log --> val_sample["Store val sample (real vs recon)"]
    val_sample --> on_val_end["on_validation_epoch_end (e.g. clear cache)"]
    on_val_end --> scheduler_step["scheduler.step()"]
    val_epoch -->|No| scheduler_step
    scheduler_step --> epoch

    epoch --> done(["Next epoch or end"])
```

---

## 8. Loss Summary

| Loss            | Formula / role |
|-----------------|----------------|
| **Reconstruction** | L1(x_recon, x) |
| **VQ (top)**    | VQ loss from vq_top + β × commitment (z_top, z_q_top) |
| **VQ (bottom)** | VQ loss from vq_bot + β × commitment (z_bot_cond, z_q_bot) |
| **Total**       | recon_loss + vq_loss (vq_loss = loss_bot + loss_top) |

Optimizer: **AdamW** (lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-4).  
Scheduler: **CosineAnnealingLR** (T_max=max_epochs, eta_min=0.1×lr).

---

## 9. Sampling (Decode from Codes)

For generation, the model can decode from discrete codes without running the encoder:

```mermaid
flowchart LR
    codes_bot["codes_bot (B,L_bot)"] --> vq_bot_decode["vq_bot.decode_codes"]
    codes_top["codes_top (B,L_top)"] --> vq_top_decode["vq_top.decode_codes"]
    vq_bot_decode --> z_q_bot["z_q_bot"]
    vq_top_decode --> z_q_top["z_q_top"]
    z_q_bot --> decode["decode(z_q_bot, z_q_top)"]
    z_q_top --> decode
    decode --> samples["Generated ECG samples"]
```

A prior (e.g. PixelSNAIL, Transformer) can be trained on **(codes_bot, codes_top)** to sample new codes; this module then decodes them to ECG with **decode_codes**.
