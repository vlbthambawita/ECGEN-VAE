#!/usr/bin/env python3
"""
Conditional VQ-VAE-2 for 12-Lead ECG Generation
=================================================
Extends the base VQ-VAE-2 (vqvae2.py) with conditioning on 9 clinical ECG features:

  RR Interval | P Onset | P End | QRS Onset | QRS End | T End |
  P Axis | QRS Axis | T Axis  (all normalized)

Architecture (Razavi et al., NeurIPS 2019 – conditioned variant)
----------------------------------------------------------------
  - Condition vector  c ∈ R^9 is projected to a dense embedding
  - The embedding is broadcast along the time axis and injected into:
      * both decoders (channel-wise concatenation) – local conditioning
      * the top encoder (FiLM: scale + shift) – global conditioning
  - Everything else is identical to the unconditioned VQVAE2

Training
--------
  python vqvae2_conditional.py fit   --data-dir /path/to/mimic-iv-ecg
  python vqvae2_conditional.py test  --data-dir /path/to/mimic-iv-ecg --ckpt-path last.ckpt
  python vqvae2_conditional.py sample --ckpt-path last.ckpt \\
         --rr-interval 0.3421 --p-onset -0.5632 --p-end 0.1234 \\
         --qrs-onset -0.2341 --qrs-end 0.4523 --t-end 1.2345 \\
         --p-axis -0.7654 --qrs-axis 0.8765 --t-axis -0.3421 \\
         --n-samples 8 --out cond_samples.npy
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import Tensor
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import (
    Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

try:
    from pytorch_lightning.loggers import WandbLogger
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    WandbLogger = None
    wandb = None

# Import shared components from companion file
try:
    from vqvae2 import (
        set_global_seed,
        ResidualBlock1d,
        Encoder1d,
        Decoder1d,
        VectorQuantiser,
        MIMICIVECGDataset,
        VQVAE2MIMICDataModule,
    )
except ImportError:
    raise ImportError(
        "Could not import vqvae2.py. "
        "Place vqvae2_conditional.py in the same directory as vqvae2.py."
    )

# ============================================================================
# Condition Encoder  –  maps feature vector → dense conditioning signal
# ============================================================================

class ConditionEncoder(nn.Module):
    """
    Projects a scalar feature vector (B, n_features) to a latent condition
    embedding (B, cond_dim).

    Also produces FiLM parameters (gamma, beta) for feature-wise linear
    modulation of intermediate encoder activations.
    """

    def __init__(self, n_features: int = 9, cond_dim: int = 128):
        super().__init__()
        self.n_features = n_features
        self.cond_dim = cond_dim

        self.mlp = nn.Sequential(
            nn.Linear(n_features, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        # FiLM parameters for top encoder modulation
        self.film_gamma = nn.Linear(cond_dim, cond_dim)
        self.film_beta  = nn.Linear(cond_dim, cond_dim)

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            features: (B, n_features)  normalized feature values
        Returns:
            cond_emb:  (B, cond_dim)           – dense condition embedding
            film_gamma:(B, cond_dim)           – FiLM scale
            film_beta: (B, cond_dim)           – FiLM bias
        """
        cond_emb   = self.mlp(features)
        film_gamma = self.film_gamma(cond_emb)
        film_beta  = self.film_beta(cond_emb)
        return cond_emb, film_gamma, film_beta


# ============================================================================
# Conditional VQ-VAE-2
# ============================================================================

class ConditionalVQVAE2(nn.Module):
    """
    Two-level conditional VQ-VAE-2 for 1-D multi-channel ECG signals.

    Conditioning strategy
    ---------------------
    1. Bottom decoder:  cond_emb broadcast → (B, cond_dim, L_bot) and
                        concatenated with z_q_bot before decoding.
    2. Top decoder:     same, but broadcast to L_top size.
    3. Top encoder:     FiLM modulation applied after the encoder backbone
                        (scale/shift on the z_top_pre feature map).

    All condition tensors are derived from the clinical feature vector via
    ConditionEncoder.
    """

    def __init__(
        self,
        n_leads: int = 12,
        signal_len: int = 5000,
        hidden_channels: int = 128,
        residual_channels: int = 64,
        n_res_blocks: int = 4,
        n_embeddings_top: int = 512,
        n_embeddings_bot: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        enc_bot_strides: Tuple[int, ...] = (2, 2, 2),
        enc_top_strides: Tuple[int, ...] = (2, 2, 2),
        n_features: int = 9,
        cond_dim: int = 128,
    ):
        super().__init__()
        self.n_leads = n_leads
        self.signal_len = signal_len
        self.n_embeddings_top = n_embeddings_top
        self.n_embeddings_bot = n_embeddings_bot
        self.embedding_dim = embedding_dim
        self.enc_bot_strides = enc_bot_strides
        self.enc_top_strides = enc_top_strides
        self.cond_dim = cond_dim

        D = embedding_dim
        H = hidden_channels
        R = residual_channels
        N = n_res_blocks

        # Condition encoder
        self.cond_enc = ConditionEncoder(n_features=n_features, cond_dim=cond_dim)

        # FiLM projection to match embedding_dim (applied on top-enc output)
        self.film_proj_gamma = nn.Linear(cond_dim, D)
        self.film_proj_beta  = nn.Linear(cond_dim, D)

        # ---- Encoder hierarchy ----
        self.enc_bot = Encoder1d(
            in_channels=n_leads, hidden_channels=H,
            residual_channels=R, n_res_blocks=N,
            strides=enc_bot_strides, out_channels=D,
        )
        self.enc_top = Encoder1d(
            in_channels=D, hidden_channels=H,
            residual_channels=R, n_res_blocks=N,
            strides=enc_top_strides, out_channels=D,
        )

        # ---- Vector quantisers ----
        self.vq_top = VectorQuantiser(n_embeddings_top, D, commitment_cost, ema_decay)
        self.vq_bot = VectorQuantiser(n_embeddings_bot, D, commitment_cost, ema_decay)

        # ---- Decoder hierarchy ----
        # Top decoder: receives z_q_top + cond_dim (condition concat)
        self.dec_top = Decoder1d(
            in_channels=D, hidden_channels=H,
            residual_channels=R, n_res_blocks=N,
            strides=enc_top_strides, out_channels=D,
            cond_channels=cond_dim,
        )

        # Bottom decoder: receives (z_q_bot + top_up) + cond_dim from top
        self.dec_bot = Decoder1d(
            in_channels=D, hidden_channels=H,
            residual_channels=R, n_res_blocks=N,
            strides=enc_bot_strides, out_channels=n_leads,
            cond_channels=D + cond_dim,   # top_up + condition broadcast
        )

    # ------------------------------------------------------------------
    def _broadcast_cond(self, cond_emb: Tensor, length: int) -> Tensor:
        """
        cond_emb: (B, cond_dim)  →  (B, cond_dim, length)
        """
        return cond_emb.unsqueeze(-1).expand(-1, -1, length)

    # ------------------------------------------------------------------
    def encode(self, x: Tensor, cond_emb: Tensor, film_gamma: Tensor, film_beta: Tensor):
        """
        x         : (B, 12, 5000)
        cond_emb  : (B, cond_dim)
        film_gamma: (B, cond_dim)
        film_beta : (B, cond_dim)

        Returns:
            z_q_bot, z_q_top, loss_bot, loss_top, codes_bot, codes_top
        """
        z_bot = self.enc_bot(x)                     # (B, D, L_bot)
        z_top_pre = self.enc_top(z_bot)             # (B, D, L_top)

        # FiLM modulation on z_top_pre  (feature-wise linear modulation)
        gamma = self.film_proj_gamma(film_gamma).unsqueeze(-1)  # (B, D, 1)
        beta  = self.film_proj_beta(film_beta).unsqueeze(-1)    # (B, D, 1)
        z_top_pre = (1 + gamma) * z_top_pre + beta

        z_q_top, loss_top, codes_top = self.vq_top(z_top_pre)

        # Up-sample z_q_top back to bottom level
        top_up = self.dec_top(z_q_top, cond=self._broadcast_cond(cond_emb, z_q_top.shape[-1]))
        if top_up.shape[-1] != z_bot.shape[-1]:
            top_up = F.interpolate(top_up, size=z_bot.shape[-1], mode='nearest')

        z_bot_cond = z_bot + top_up
        z_q_bot, loss_bot, codes_bot = self.vq_bot(z_bot_cond)

        return z_q_bot, z_q_top, loss_bot, loss_top, codes_bot, codes_top

    # ------------------------------------------------------------------
    def decode(self, z_q_bot: Tensor, z_q_top: Tensor, cond_emb: Tensor) -> Tensor:
        """
        Reconstruct the ECG signal from quantised latents + condition.
        """
        # Decode top level (conditioned)
        top_up = self.dec_top(z_q_top, cond=self._broadcast_cond(cond_emb, z_q_top.shape[-1]))
        if top_up.shape[-1] != z_q_bot.shape[-1]:
            top_up = F.interpolate(top_up, size=z_q_bot.shape[-1], mode='nearest')

        # Combine: top_up + broadcast condition → bottom decoder context
        cond_bot = self._broadcast_cond(cond_emb, z_q_bot.shape[-1])   # (B, cond_dim, L_bot)
        joint_cond = torch.cat([top_up, cond_bot], dim=1)               # (B, D+cond_dim, L_bot)

        x_recon = self.dec_bot(z_q_bot, cond=joint_cond)
        return x_recon

    # ------------------------------------------------------------------
    def forward(self, x: Tensor, features: Tensor):
        """
        x        : (B, 12, 5000)
        features : (B, 9)   normalized clinical features

        Returns:
            x_recon  : (B, 12, 5000)
            vq_loss  : scalar
            codes_bot: (B, L_bot)
            codes_top: (B, L_top)
        """
        cond_emb, film_gamma, film_beta = self.cond_enc(features)
        z_q_bot, z_q_top, loss_bot, loss_top, codes_bot, codes_top = \
            self.encode(x, cond_emb, film_gamma, film_beta)
        x_recon = self.decode(z_q_bot, z_q_top, cond_emb)
        vq_loss = loss_bot + loss_top
        return x_recon, vq_loss, codes_bot, codes_top

    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode_codes(
        self,
        codes_bot: Tensor,
        codes_top: Tensor,
        features: Tensor,
    ) -> Tensor:
        """
        Decode from discrete codes + feature condition.
        features: (B, 9)
        """
        cond_emb, _, _ = self.cond_enc(features)
        z_q_bot = self.vq_bot.decode_codes(codes_bot)
        z_q_top = self.vq_top.decode_codes(codes_top)
        return self.decode(z_q_bot, z_q_top, cond_emb)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        features: Tensor,
        temperature: float = 1.0,
    ) -> Tensor:
        """
        Sample ECGs conditioned on feature vectors using random codebook entries.

        features: (B, 9)  normalized clinical features
        Returns : (B, 12, 5000)

        Note: For higher-quality generation, use the conditional transformer
        prior (cond_transformer_prior.py) instead of random sampling.
        """
        self.eval()
        device = next(self.parameters()).device
        B = features.shape[0]

        bot_len = self.signal_len // math.prod(self.enc_bot_strides)
        top_len = bot_len // math.prod(self.enc_top_strides)

        if temperature != 1.0:
            # Temperature-scaled random sampling from codebook
            top_logits = torch.ones(B, self.n_embeddings_top, device=device) / temperature
            bot_logits = torch.ones(B, self.n_embeddings_bot, device=device) / temperature
            codes_top = torch.multinomial(
                F.softmax(top_logits, -1).unsqueeze(1).expand(-1, top_len, -1).reshape(B * top_len, -1),
                1
            ).reshape(B, top_len)
            codes_bot = torch.multinomial(
                F.softmax(bot_logits, -1).unsqueeze(1).expand(-1, bot_len, -1).reshape(B * bot_len, -1),
                1
            ).reshape(B, bot_len)
        else:
            codes_top = torch.randint(0, self.n_embeddings_top, (B, top_len), device=device)
            codes_bot = torch.randint(0, self.n_embeddings_bot, (B, bot_len), device=device)

        return self.decode_codes(codes_bot, codes_top, features.to(device))


# ============================================================================
# Config dataclass
# ============================================================================

@dataclass
class CondVQVAE2Config:
    n_leads: int = 12
    signal_len: int = 5000
    hidden_channels: int = 128
    residual_channels: int = 64
    n_res_blocks: int = 4
    n_embeddings_top: int = 512
    n_embeddings_bot: int = 512
    embedding_dim: int = 64
    commitment_cost: float = 0.25
    ema_decay: float = 0.99
    enc_bot_strides: Tuple[int, ...] = (2, 2, 2)
    enc_top_strides: Tuple[int, ...] = (2, 2, 2)
    n_features: int = 9
    cond_dim: int = 128
    lr: float = 3e-4
    b1: float = 0.9
    b2: float = 0.999


# ============================================================================
# PyTorch Lightning Module
# ============================================================================

class CondVQVAE2Lightning(pl.LightningModule):
    """PyTorch Lightning wrapper for Conditional VQ-VAE-2."""

    FEATURE_NAMES = [
        "rr_interval", "p_onset", "p_end", "qrs_onset", "qrs_end", "t_end",
        "p_axis", "qrs_axis", "t_axis",
    ]

    def __init__(self, config: CondVQVAE2Config | dict | None = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = CondVQVAE2Config(**kwargs)
        elif isinstance(config, dict):
            config = CondVQVAE2Config(**config)

        self.save_hyperparameters(config.__dict__)
        self.config = config

        self.model = ConditionalVQVAE2(
            n_leads=config.n_leads,
            signal_len=config.signal_len,
            hidden_channels=config.hidden_channels,
            residual_channels=config.residual_channels,
            n_res_blocks=config.n_res_blocks,
            n_embeddings_top=config.n_embeddings_top,
            n_embeddings_bot=config.n_embeddings_bot,
            embedding_dim=config.embedding_dim,
            commitment_cost=config.commitment_cost,
            ema_decay=config.ema_decay,
            enc_bot_strides=config.enc_bot_strides,
            enc_top_strides=config.enc_top_strides,
            n_features=config.n_features,
            cond_dim=config.cond_dim,
        )

        self._val_real: Tensor | None = None
        self._val_recon: Tensor | None = None
        self._val_features: Tensor | None = None

    # ------------------------------------------------------------------
    def forward(self, x: Tensor, features: Tensor):
        return self.model(x, features)

    # ------------------------------------------------------------------
    def _step(self, batch, stage: str):
        ecgs, features = batch                          # (B,12,5000), (B,9)
        x_recon, vq_loss, codes_bot, codes_top = self.model(ecgs, features)

        recon_loss = F.l1_loss(x_recon, ecgs)
        total_loss = recon_loss + vq_loss

        self.log(f"{stage}/total_loss",  total_loss, prog_bar=True,  on_epoch=True, on_step=False)
        self.log(f"{stage}/recon_loss",  recon_loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log(f"{stage}/vq_loss",     vq_loss,    prog_bar=False, on_epoch=True, on_step=False)

        if stage == "val":
            self.log("val_loss", total_loss, prog_bar=True, on_epoch=True, on_step=False)

        unique_bot = torch.unique(codes_bot).numel()
        unique_top = torch.unique(codes_top).numel()
        self.log(f"{stage}/codebook_usage_bot",
                 unique_bot / self.config.n_embeddings_bot, on_epoch=True, on_step=False)
        self.log(f"{stage}/codebook_usage_top",
                 unique_top / self.config.n_embeddings_top, on_epoch=True, on_step=False)

        return total_loss, x_recon

    # ------------------------------------------------------------------
    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        loss, _ = self._step(batch, "train")
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        ecgs, features = batch
        loss, x_recon = self._step(batch, "val")
        if batch_idx == 0:
            self._val_real     = ecgs[0].detach().cpu()
            self._val_recon    = x_recon[0].detach().cpu()
            self._val_features = features[0].detach().cpu()
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> Tensor:
        loss, _ = self._step(batch, "test")
        return loss

    # ------------------------------------------------------------------
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            betas=(self.config.b1, self.config.b2),
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=self.config.lr * 0.1
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


# ============================================================================
# Visualization Callback
# ============================================================================

class CondVQVAE2VisualizationCallback(Callback):
    """Callback to visualize conditioned reconstructions during training."""

    FEATURE_NAMES = [
        "RR Interval", "P Onset", "P End", "QRS Onset", "QRS End", "T End",
        "P Axis", "QRS Axis", "T Axis",
    ]

    def __init__(self, save_dir: Path, log_every_n_epochs: int = 5, num_samples: int = 4):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_every_n_epochs = log_every_n_epochs
        self.num_samples = num_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
        real    = pl_module._val_real
        recon   = pl_module._val_recon
        feats   = pl_module._val_features
        if real is None or recon is None:
            return
        path = self.save_dir / f"epoch_{trainer.current_epoch:04d}.png"
        self._plot(real, recon, feats, path, trainer.current_epoch)

    def _plot(self, real: Tensor, recon: Tensor, feats: Optional[Tensor],
              path: Path, epoch: int):
        real_np  = real.cpu().numpy()
        recon_np = recon.cpu().numpy()
        n_leads  = min(real_np.shape[0], 8)

        fig, axs = plt.subplots(n_leads, 2, figsize=(14, 1.8 * n_leads))
        if n_leads == 1:
            axs = axs.reshape(1, -1)

        for i in range(n_leads):
            axs[i, 0].plot(real_np[i], linewidth=0.6, color='steelblue')
            axs[i, 0].set_title(f"Real – Lead {i}", fontsize=8)
            axs[i, 0].axis('off')
            axs[i, 1].plot(recon_np[i], linewidth=0.6, color='darkorange')
            axs[i, 1].set_title(f"Reconstructed – Lead {i}", fontsize=8)
            axs[i, 1].axis('off')

        cond_str = ""
        if feats is not None:
            vals = feats.numpy()
            parts = [f"{n}: {v:+.4f}" for n, v in zip(self.FEATURE_NAMES, vals)]
            cond_str = "\n" + "  |  ".join(parts[:5]) + "\n" + "  |  ".join(parts[5:])

        plt.suptitle(
            f"Cond-VQVAE2 Reconstruction – Epoch {epoch}{cond_str}",
            fontsize=9, y=1.01
        )
        plt.tight_layout()
        plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  [VIZ] Saved reconstruction plot → {path}")


# ============================================================================
# Training entry-point
# ============================================================================

def train(args):
    print("=" * 80)
    print("Conditional VQ-VAE-2  –  Training")
    print("=" * 80)

    if not os.path.exists(args.data_dir):
        print(f"ERROR: data directory not found: {args.data_dir}")
        sys.exit(1)

    measurements_file = os.path.join(args.data_dir, "machine_measurements.csv")
    if not os.path.exists(measurements_file):
        print(f"ERROR: machine_measurements.csv not found at: {measurements_file}")
        sys.exit(1)

    set_global_seed(args.seed)

    run_dir   = Path(args.runs_root) / args.exp_name / f"seed_{args.seed}"
    ckpt_dir  = run_dir / "checkpoints"
    viz_dir   = run_dir / "samples"
    for d in (ckpt_dir, viz_dir):
        d.mkdir(parents=True, exist_ok=True)

    config = CondVQVAE2Config(
        n_leads=args.n_leads,
        signal_len=args.signal_len,
        hidden_channels=args.hidden_channels,
        residual_channels=args.residual_channels,
        n_res_blocks=args.n_res_blocks,
        n_embeddings_top=args.n_embeddings_top,
        n_embeddings_bot=args.n_embeddings_bot,
        embedding_dim=args.embedding_dim,
        commitment_cost=args.commitment_cost,
        ema_decay=args.ema_decay,
        enc_bot_strides=(2, 2, 2),
        enc_top_strides=(2, 2, 2),
        n_features=9,
        cond_dim=args.cond_dim,
        lr=args.lr,
        b1=args.b1,
        b2=args.b2,
    )

    model = CondVQVAE2Lightning(config)

    datamodule = VQVAE2MIMICDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        test_split=args.test_split,
        max_samples=args.max_samples,
        seed=args.seed,
        skip_missing_check=args.skip_missing_check,
        num_leads=args.n_leads,
        seq_length=args.signal_len,
    )

    loggers_list = [TensorBoardLogger(save_dir=str(run_dir), name="tb")]
    if args.wandb and WANDB_AVAILABLE:
        try:
            run_name = args.wandb_run_name or f"{args.exp_name}_seed{args.seed}"
            loggers_list.append(WandbLogger(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                save_dir=str(run_dir),
                tags=args.wandb_tags or ["cond_vqvae2", "ecg"],
                config=config.__dict__,
            ))
            print(f"✓ W&B logging: {args.wandb_project}/{run_name}")
        except Exception as e:
            print(f"⚠ W&B init failed: {e}. Continuing with TensorBoard only.")
    elif args.wandb:
        print("⚠ wandb not installed. Use: pip install wandb")

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="epoch{epoch:03d}-val{val_loss:.4f}",
            monitor="val_loss", mode="min",
            save_top_k=args.save_top_k, save_last=True,
        ),
        EarlyStopping(monitor="val_loss", patience=args.patience, mode="min", verbose=True),
        LearningRateMonitor(logging_interval="step"),
        CondVQVAE2VisualizationCallback(
            save_dir=viz_dir,
            log_every_n_epochs=args.viz_every_n_epochs,
            num_samples=args.viz_num_samples,
        ),
    ]

    trainer = pl.Trainer(
        default_root_dir=str(run_dir),
        logger=loggers_list,
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        gradient_clip_val=args.gradient_clip,
        deterministic=True,
    )

    print(f"Run directory   : {run_dir}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    trainer.fit(model=model, datamodule=datamodule)

    print()
    print("=" * 80)
    print("Conditional VQ-VAE-2 training finished.")
    print(f"Best checkpoint : {callbacks[0].best_model_path}")
    print(f"Best val loss   : {callbacks[0].best_model_score:.4f}")
    print("=" * 80)


# ============================================================================
# Test entry-point
# ============================================================================

def test(args):
    print("=" * 80)
    print("Conditional VQ-VAE-2  –  Testing")
    print("=" * 80)

    model = CondVQVAE2Lightning.load_from_checkpoint(args.ckpt_path)
    model.eval()

    datamodule = VQVAE2MIMICDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices)
    trainer.test(model, datamodule=datamodule)


# ============================================================================
# Sample entry-point
# ============================================================================

def sample(args):
    """
    Generate ECG samples conditioned on user-provided clinical features.
    Features are expected to be already normalized (z-score or min-max).
    """
    print("=" * 80)
    print("Conditional VQ-VAE-2  –  Conditional Sampling")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CondVQVAE2Lightning.load_from_checkpoint(args.ckpt_path, map_location=device)
    model.eval().to(device)

    # Build feature vector from CLI args
    features_raw = np.array([
        args.rr_interval,
        args.p_onset,
        args.p_end,
        args.qrs_onset,
        args.qrs_end,
        args.t_end,
        args.p_axis,
        args.qrs_axis,
        args.t_axis,
    ], dtype=np.float32)

    print("\nInput condition (normalized features):")
    feature_names = [
        "RR Interval", "P Onset",  "P End",
        "QRS Onset",   "QRS End",  "T End",
        "P Axis",      "QRS Axis", "T Axis",
    ]
    for name, val in zip(feature_names, features_raw):
        print(f"  {name:<15}: {val:>8.4f}")
    print()

    # Replicate the condition for all samples
    features_tensor = torch.from_numpy(features_raw).unsqueeze(0).expand(args.n_samples, -1)

    print(f"Generating {args.n_samples} conditioned ECG samples...")
    with torch.no_grad():
        samples = model.model.sample(
            features=features_tensor.to(device),
            temperature=args.temperature,
        )
    samples_np = samples.cpu().numpy()  # (N, 12, 5000)

    out_path = Path(args.out)
    np.save(out_path, samples_np)
    print(f"\nSaved {args.n_samples} samples → {out_path}  shape: {samples_np.shape}")

    if args.plot:
        _plot_samples(samples_np, out_path.with_suffix(".png"), features_raw, feature_names)

    print("=" * 80)


def _plot_samples(
    samples: np.ndarray,
    out_path: Path,
    features_raw: np.ndarray,
    feature_names: list,
    n_show: int = 4,
):
    """Plot a grid of generated ECG samples."""
    B, L, T = samples.shape
    n_show = min(n_show, B)
    n_leads = min(L, 6)

    fig, axs = plt.subplots(n_show * n_leads, 1, figsize=(14, 1.5 * n_show * n_leads))
    if n_show * n_leads == 1:
        axs = [axs]

    ax_idx = 0
    colors = plt.cm.tab10.colors
    for s_idx in range(n_show):
        for lead_idx in range(n_leads):
            ax = axs[ax_idx]
            ax.plot(samples[s_idx, lead_idx], linewidth=0.7,
                    color=colors[lead_idx % len(colors)])
            ax.set_ylabel(f"S{s_idx+1} L{lead_idx}", fontsize=7, rotation=0, labelpad=30)
            ax.set_xticks([])
            ax.set_yticks([])
            ax_idx += 1

    cond_str = "  |  ".join(f"{n}: {v:+.4f}" for n, v in zip(feature_names[:5], features_raw[:5]))
    cond_str += "\n" + "  |  ".join(f"{n}: {v:+.4f}" for n, v in zip(feature_names[5:], features_raw[5:]))
    plt.suptitle(f"Conditional ECG Samples\n{cond_str}", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved sample plot → {out_path}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Conditional VQ-VAE-2 for ECG generation conditioned on clinical features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # ------------------------------------------------------------------ fit
    fp = sub.add_parser("fit", help="Train the conditional VQ-VAE-2")

    fp.add_argument("--data-dir",  type=str, required=True,  help="MIMIC-IV-ECG data directory")
    fp.add_argument("--exp-name",  type=str, default="cond_vqvae2_mimic")
    fp.add_argument("--seed",      type=int, default=42)
    fp.add_argument("--runs-root", type=str, default="runs")

    fp.add_argument("--batch-size",    type=int,   default=32)
    fp.add_argument("--num-workers",   type=int,   default=4)
    fp.add_argument("--max-samples",   type=int,   default=None)
    fp.add_argument("--val-split",     type=float, default=0.1)
    fp.add_argument("--test-split",    type=float, default=0.1)
    fp.add_argument("--skip-missing-check", action="store_true")

    fp.add_argument("--n-leads",           type=int,   default=12)
    fp.add_argument("--signal-len",        type=int,   default=5000)
    fp.add_argument("--hidden-channels",   type=int,   default=128)
    fp.add_argument("--residual-channels", type=int,   default=64)
    fp.add_argument("--n-res-blocks",      type=int,   default=4)
    fp.add_argument("--n-embeddings-top",  type=int,   default=512)
    fp.add_argument("--n-embeddings-bot",  type=int,   default=512)
    fp.add_argument("--embedding-dim",     type=int,   default=64)
    fp.add_argument("--commitment-cost",   type=float, default=0.25)
    fp.add_argument("--ema-decay",         type=float, default=0.99)
    fp.add_argument("--cond-dim",          type=int,   default=128,
                    help="Dimension of condition embedding MLP")

    fp.add_argument("--lr",              type=float, default=3e-4)
    fp.add_argument("--b1",              type=float, default=0.9)
    fp.add_argument("--b2",              type=float, default=0.999)
    fp.add_argument("--max-epochs",      type=int,   default=200)
    fp.add_argument("--accelerator",     type=str,   default="gpu")
    fp.add_argument("--devices",         type=int, nargs="+", default=[0])
    fp.add_argument("--log-every-n-steps",        type=int,   default=50)
    fp.add_argument("--check-val-every-n-epoch",  type=int,   default=1)
    fp.add_argument("--gradient-clip",            type=float, default=1.0)
    fp.add_argument("--patience",                 type=int,   default=15)
    fp.add_argument("--save-top-k",               type=int,   default=3)
    fp.add_argument("--viz-every-n-epochs",       type=int,   default=5)
    fp.add_argument("--viz-num-samples",          type=int,   default=4)

    fp.add_argument("--wandb",            action="store_true")
    fp.add_argument("--wandb-project",    type=str, default="ecg-cond-vqvae2")
    fp.add_argument("--wandb-entity",     type=str, default=None)
    fp.add_argument("--wandb-run-name",   type=str, default=None)
    fp.add_argument("--wandb-tags",       type=str, nargs="*", default=None)

    # ------------------------------------------------------------------ test
    tp = sub.add_parser("test", help="Evaluate a trained checkpoint")
    tp.add_argument("--data-dir",   type=str, required=True)
    tp.add_argument("--ckpt-path",  type=str, required=True)
    tp.add_argument("--batch-size", type=int, default=32)
    tp.add_argument("--num-workers",type=int, default=4)
    tp.add_argument("--seed",       type=int, default=42)
    tp.add_argument("--accelerator",type=str, default="gpu")
    tp.add_argument("--devices",    type=int, nargs="+", default=[0])

    # ------------------------------------------------------------------ sample
    sp = sub.add_parser("sample", help="Generate ECGs for given feature conditions")
    sp.add_argument("--ckpt-path",   type=str, required=True)
    sp.add_argument("--n-samples",   type=int,   default=8)
    sp.add_argument("--temperature", type=float, default=1.0,
                    help="Sampling temperature (< 1 = sharper, > 1 = more random)")
    sp.add_argument("--out",         type=str,   default="cond_samples.npy")
    sp.add_argument("--plot",        action="store_true",
                    help="Also save a PNG visualization of the samples")

    # --- Clinical feature inputs (normalized) ---
    sp.add_argument("--rr-interval", type=float, required=True, metavar="NORM_VAL",
                    help="Normalized RR Interval")
    sp.add_argument("--p-onset",     type=float, required=True, metavar="NORM_VAL",
                    help="Normalized P Onset")
    sp.add_argument("--p-end",       type=float, required=True, metavar="NORM_VAL",
                    help="Normalized P End")
    sp.add_argument("--qrs-onset",   type=float, required=True, metavar="NORM_VAL",
                    help="Normalized QRS Onset")
    sp.add_argument("--qrs-end",     type=float, required=True, metavar="NORM_VAL",
                    help="Normalized QRS End")
    sp.add_argument("--t-end",       type=float, required=True, metavar="NORM_VAL",
                    help="Normalized T End")
    sp.add_argument("--p-axis",      type=float, required=True, metavar="NORM_VAL",
                    help="Normalized P Axis")
    sp.add_argument("--qrs-axis",    type=float, required=True, metavar="NORM_VAL",
                    help="Normalized QRS Axis")
    sp.add_argument("--t-axis",      type=float, required=True, metavar="NORM_VAL",
                    help="Normalized T Axis")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    return args


def main():
    args = parse_args()
    if args.command == "fit":
        train(args)
    elif args.command == "test":
        test(args)
    elif args.command == "sample":
        sample(args)


if __name__ == "__main__":
    main()
