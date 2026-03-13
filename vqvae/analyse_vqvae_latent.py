#!/usr/bin/env python3
"""
VQ-VAE Latent Space Analysis Script
=====================================
Loads a pretrained VQ-VAE checkpoint and runs a suite of visual analyses
on the latent space for a (configurable) subset of the test split.

Analyses produced
-----------------
1.  Codebook usage histogram       — which codes are used and how often
2.  Codebook usage CDF             — cumulative utilisation curve
3.  Codebook embedding PCA (2-D)   — structure of the 1024-vector codebook
4.  Codebook embedding UMAP (2-D)  — non-linear layout of codebook vectors
5.  Per-sample code-sequence heatmap — codes over time for N example ECGs
6.  Code-frequency per lead position — where in the sequence each code fires
7.  PCA of per-sample mean latent   — sample-level latent cloud
8.  UMAP of per-sample mean latent  — non-linear sample cloud
9.  Reconstruction overlay grid     — real vs reconstructed waveform for N samples
10. VQ loss / reconstruction loss   — per-batch scatter over the test set

All figures are saved to --output-dir as high-res PNGs.
A compact JSON summary (stats.json) is also written.

Usage
-----
    python analyse_vqvae_latent.py \
        --checkpoint runs/vqvae_mimic/seed_42/checkpoints/best.ckpt \
        --data-dir /path/to/mimic-iv-ecg \
        --max-samples 2000 \
        --output-dir latent_analysis
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wfdb
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("[WARN] umap-learn not installed — UMAP plots will be skipped. "
          "Install with: pip install umap-learn")

import pytorch_lightning as pl

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================================
# Model definitions (copied verbatim from train_vqvae_standalone.py so this
# script is fully self-contained and can load the saved checkpoint)
# ============================================================================

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        h = F.silu(self.norm1(self.conv1(x)))
        h = F.silu(self.norm2(self.conv2(h)))
        return h + residual


class Encoder1D(nn.Module):
    def __init__(self, in_channels=12, base_channels=64, latent_channels=8,
                 channel_multipliers=(1,2,4,4), num_res_blocks=2) -> None:
        super().__init__()
        self.conv_in = nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3)
        self.down_blocks = nn.ModuleList()
        ch = base_channels
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock1D(ch, out_ch))
                ch = out_ch
            if i < len(channel_multipliers) - 1:
                self.down_blocks.append(nn.Conv1d(ch, ch, kernel_size=4, stride=2, padding=1))
        self.mid_block1 = ResidualBlock1D(ch, ch)
        self.mid_block2 = ResidualBlock1D(ch, ch)
        self.norm_out = nn.GroupNorm(8, ch)
        self.conv_out = nn.Conv1d(ch, latent_channels * 2, kernel_size=3, padding=1)

    def forward(self, x: Tensor):
        h = self.conv_in(x)
        for block in self.down_blocks:
            h = block(h)
        h = self.mid_block2(self.mid_block1(h))
        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)
        mean, logvar = torch.chunk(h, 2, dim=1)
        return mean, logvar


class Decoder1D(nn.Module):
    def __init__(self, out_channels=12, base_channels=64, latent_channels=8,
                 channel_multipliers=(1,2,4,4), num_res_blocks=2) -> None:
        super().__init__()
        channel_multipliers = tuple(reversed(channel_multipliers))
        ch = base_channels * channel_multipliers[0]
        self.conv_in = nn.Conv1d(latent_channels, ch, kernel_size=3, padding=1)
        self.mid_block1 = ResidualBlock1D(ch, ch)
        self.mid_block2 = ResidualBlock1D(ch, ch)
        self.up_blocks = nn.ModuleList()
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.up_blocks.append(ResidualBlock1D(ch, out_ch))
                ch = out_ch
            if i < len(channel_multipliers) - 1:
                self.up_blocks.append(
                    nn.ConvTranspose1d(ch, ch, kernel_size=4, stride=2, padding=1))
        self.norm_out = nn.GroupNorm(8, ch)
        self.conv_out = nn.Conv1d(ch, out_channels, kernel_size=7, padding=3)

    def forward(self, z: Tensor) -> Tensor:
        h = self.conv_in(z)
        h = self.mid_block2(self.mid_block1(h))
        for block in self.up_blocks:
            h = block(h)
        return self.conv_out(F.silu(self.norm_out(h)))


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z: Tensor):
        z = z.permute(0, 2, 1).contiguous()
        z_flat = z.view(-1, self.embedding_dim)
        dist = (z_flat**2).sum(1, keepdim=True) + (self.embedding.weight**2).sum(1) \
               - 2 * z_flat @ self.embedding.weight.t()
        idx = dist.argmin(1).unsqueeze(1)
        enc = torch.zeros(idx.shape[0], self.num_embeddings, device=z.device)
        enc.scatter_(1, idx, 1)
        q = (enc @ self.embedding.weight).view(z.shape)
        vq_loss = F.mse_loss(q.detach(), z) + self.commitment_cost * F.mse_loss(q, z.detach())
        q = z + (q - z).detach()
        return vq_loss, q.permute(0, 2, 1).contiguous(), idx.view(z.shape[0], -1)

    @torch.no_grad()
    def get_codebook_entry(self, indices: Tensor) -> Tensor:
        q = self.embedding(indices).permute(0, 2, 1).contiguous()
        return q


class VQVAE1D(nn.Module):
    def __init__(self, in_channels=12, base_channels=64, latent_channels=64,
                 channel_multipliers=(1,2,4,4), num_res_blocks=2,
                 num_embeddings=512, commitment_cost=0.25) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.encoder = Encoder1D(in_channels, base_channels, latent_channels,
                                 channel_multipliers, num_res_blocks)
        self.vq = VectorQuantizer(num_embeddings, latent_channels, commitment_cost)
        self.decoder = Decoder1D(in_channels, base_channels, latent_channels,
                                 channel_multipliers, num_res_blocks)

    def encode(self, x):
        z, _ = self.encoder(x)
        return z, z

    def forward(self, x):
        z, _ = self.encode(x)
        vq_loss, z_q, indices = self.vq(z)
        return self.decoder(z_q), vq_loss, indices

    @torch.no_grad()
    def encode_to_indices(self, x):
        z, _ = self.encode(x)
        _, _, idx = self.vq(z)
        return idx

    @torch.no_grad()
    def encode_to_latent(self, x):
        """Return continuous latent z (before quantisation)."""
        z, _ = self.encode(x)
        return z  # [B, latent_channels, L/16]

    @torch.no_grad()
    def decode_from_indices(self, indices):
        z_q = self.vq.get_codebook_entry(indices)
        return self.decoder(z_q)


@dataclass
class VQVAEConfig:
    in_channels: int = 12
    base_channels: int = 64
    latent_channels: int = 64
    channel_multipliers: tuple = (1, 2, 4, 4)
    num_res_blocks: int = 2
    num_embeddings: int = 512
    commitment_cost: float = 0.25
    lr: float = 1e-4
    b1: float = 0.9
    b2: float = 0.999


class VQVAELightning(pl.LightningModule):
    def __init__(self, config=None, **kwargs):
        super().__init__()
        if config is None:
            config = VQVAEConfig(**kwargs)
        elif isinstance(config, dict):
            config = VQVAEConfig(**config)
        self.save_hyperparameters(config.__dict__)
        self.config = config
        self.vqvae = VQVAE1D(
            in_channels=config.in_channels,
            base_channels=config.base_channels,
            latent_channels=config.latent_channels,
            channel_multipliers=config.channel_multipliers,
            num_res_blocks=config.num_res_blocks,
            num_embeddings=config.num_embeddings,
            commitment_cost=config.commitment_cost,
        )

    def forward(self, x):
        return self.vqvae(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.vqvae.parameters(), lr=self.config.lr,
                                betas=(self.config.b1, self.config.b2))


# ============================================================================
# Dataset (minimal — test split only)
# ============================================================================

FEATURE_NAMES = [
    "rr_interval", "p_onset", "p_end", "qrs_onset", "qrs_end", "t_end",
    "p_axis", "qrs_axis", "t_axis",
]

class MIMICTestDataset(Dataset):
    """Loads the test split from MIMIC-IV-ECG."""

    def __init__(self, data_dir: str, max_samples: Optional[int],
                 seq_length: int, seed: int,
                 val_split: float = 0.1, test_split: float = 0.1) -> None:
        self.data_dir = Path(data_dir)
        self.seq_length = seq_length

        records_csv = self.data_dir / "record_list.csv"
        if not records_csv.exists():
            raise FileNotFoundError(f"record_list.csv not found in {data_dir}")

        df = pd.read_csv(records_csv)

        # Try to load machine measurements
        machine_csv = self.data_dir / "machine_measurements.csv"
        if machine_csv.exists():
            meas = pd.read_csv(machine_csv)
            df = df.merge(meas, on="study_id", how="left") if "study_id" in df.columns \
                 else df.merge(meas, left_index=True, right_index=True, how="left")
        for fn in FEATURE_NAMES:
            if fn not in df.columns:
                df[fn] = 0.0

        # 80 / 10 / 10 split — reproduce training split exactly
        train_df, temp_df = train_test_split(df, test_size=val_split + test_split,
                                             random_state=seed)
        val_df, test_df = train_test_split(temp_df,
                                           test_size=test_split / (val_split + test_split),
                                           random_state=seed)

        if max_samples:
            test_df = test_df.head(max_samples)

        self.df = test_df.reset_index(drop=True)

        # Feature stats from train set
        feat = train_df[FEATURE_NAMES].values.astype(np.float32)
        self.feature_mean = np.nanmean(feat, axis=0)
        self.feature_std  = np.nanstd(feat, axis=0) + 1e-6

        print(f"[Dataset] Test samples: {len(self.df)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = str(self.data_dir / row["path"])
        try:
            record = wfdb.rdrecord(path)
            ecg = record.p_signal.T.astype(np.float32)      # [12, T]
        except Exception:
            ecg = np.zeros((12, self.seq_length), dtype=np.float32)

        # Pad / crop
        T = ecg.shape[1]
        if T >= self.seq_length:
            ecg = ecg[:, :self.seq_length]
        else:
            ecg = np.pad(ecg, ((0, 0), (0, self.seq_length - T)))

        # Normalise per sample
        std = ecg.std() + 1e-6
        ecg = (ecg - ecg.mean()) / std

        features = row[FEATURE_NAMES].values.astype(np.float32)
        features = (features - self.feature_mean) / self.feature_std

        return torch.from_numpy(ecg), torch.from_numpy(features)


# ============================================================================
# Plotting helpers
# ============================================================================

PALETTE = plt.cm.tab20.colors
FIG_DPI = 150
LEAD_NAMES = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path.name}")


# ---------------------------------------------------------------------------
# 1 & 2  Codebook usage histogram + CDF
# ---------------------------------------------------------------------------

def plot_codebook_usage(counts: np.ndarray, out_dir: Path) -> dict:
    num_embeddings = len(counts)
    used = int((counts > 0).sum())
    usage_pct = used / num_embeddings * 100

    # Sort descending
    sorted_counts = np.sort(counts)[::-1]
    total = counts.sum()
    cum = np.cumsum(sorted_counts) / total * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    ax.bar(np.arange(num_embeddings), sorted_counts, width=1.0, color="#5B8DB8", linewidth=0)
    ax.set_xlabel("Codebook index (sorted by frequency)")
    ax.set_ylabel("Count")
    ax.set_title(f"Codebook usage  ({used}/{num_embeddings} codes used, {usage_pct:.1f}%)")
    ax.axhline(counts.mean(), color="#E05C5C", linestyle="--", linewidth=1,
               label=f"mean = {counts.mean():.1f}")
    ax.legend(fontsize=9)

    # CDF
    ax = axes[1]
    ax.plot(np.arange(1, num_embeddings + 1), cum, color="#5B8DB8", linewidth=1.5)
    ax.axhline(80, color="#E05C5C", linestyle="--", linewidth=1, label="80 %")
    ax.axhline(95, color="#F0A500", linestyle="--", linewidth=1, label="95 %")
    top20_pct = float(cum[max(0, int(num_embeddings * 0.2) - 1)])
    ax.set_xlabel("Top-k codes")
    ax.set_ylabel("Cumulative % of all code assignments")
    ax.set_title("Cumulative code-frequency distribution")
    ax.legend(fontsize=9)

    fig.tight_layout()
    save(fig, out_dir / "01_codebook_usage.png")

    return {"used_codes": used, "total_codes": num_embeddings,
            "usage_pct": round(usage_pct, 2),
            "top20pct_coverage": round(top20_pct, 2)}


# ---------------------------------------------------------------------------
# 3 & 4  Codebook embedding PCA / UMAP
# ---------------------------------------------------------------------------

def plot_codebook_embeddings(embedding_matrix: np.ndarray,
                             counts: np.ndarray, out_dir: Path) -> None:
    """embedding_matrix: [K, D]"""
    freq = counts / (counts.max() + 1e-8)

    # PCA
    pca = PCA(n_components=2)
    emb2 = pca.fit_transform(embedding_matrix)

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(emb2[:, 0], emb2[:, 1], c=freq, cmap="viridis",
                    s=8, alpha=0.7, linewidths=0)
    plt.colorbar(sc, ax=ax, label="Relative frequency")
    ax.set_title(f"Codebook embeddings — PCA 2D\n"
                 f"(variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    save(fig, out_dir / "03_codebook_pca.png")

    # UMAP
    if UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        emb_umap = reducer.fit_transform(embedding_matrix)
        fig, ax = plt.subplots(figsize=(8, 7))
        sc = ax.scatter(emb_umap[:, 0], emb_umap[:, 1], c=freq, cmap="viridis",
                        s=8, alpha=0.7, linewidths=0)
        plt.colorbar(sc, ax=ax, label="Relative frequency")
        ax.set_title("Codebook embeddings — UMAP 2D")
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        save(fig, out_dir / "04_codebook_umap.png")


# ---------------------------------------------------------------------------
# 5  Per-sample code-sequence heatmap
# ---------------------------------------------------------------------------

def plot_code_heatmap(all_indices: np.ndarray, n_samples: int,
                      num_embeddings: int, out_dir: Path) -> None:
    """all_indices: [N, L] integer array of code indices."""
    n = min(n_samples, len(all_indices))
    subset = all_indices[:n]                          # [n, L]

    fig, ax = plt.subplots(figsize=(14, max(4, n * 0.35)))
    im = ax.imshow(subset, aspect="auto", cmap="tab20",
                   vmin=0, vmax=num_embeddings - 1,
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Code index")
    ax.set_xlabel("Latent position (time)")
    ax.set_ylabel("Sample index")
    ax.set_title(f"Discrete code sequences — first {n} test samples")
    fig.tight_layout()
    save(fig, out_dir / "05_code_heatmap.png")


# ---------------------------------------------------------------------------
# 6  Code frequency at each latent position
# ---------------------------------------------------------------------------

def plot_positional_code_frequency(all_indices: np.ndarray,
                                   num_embeddings: int, out_dir: Path) -> None:
    """Show which codes dominate at each latent time step."""
    N, L = all_indices.shape
    # Top-5 codes across whole sequence
    flat_counts = np.bincount(all_indices.ravel(), minlength=num_embeddings)
    top_codes = flat_counts.argsort()[-5:][::-1]

    freq_map = np.zeros((5, L), dtype=np.float32)
    for pos in range(L):
        col_counts = np.bincount(all_indices[:, pos], minlength=num_embeddings)
        for k, code in enumerate(top_codes):
            freq_map[k, pos] = col_counts[code] / N

    fig, ax = plt.subplots(figsize=(14, 4))
    for k in range(5):
        ax.plot(freq_map[k], linewidth=0.8, label=f"Code {top_codes[k]}")
    ax.set_xlabel("Latent position")
    ax.set_ylabel("Fraction of samples using this code")
    ax.set_title("Top-5 codes: frequency at each latent position")
    ax.legend(fontsize=9, ncol=5)
    fig.tight_layout()
    save(fig, out_dir / "06_positional_code_frequency.png")


# ---------------------------------------------------------------------------
# 7 & 8  Sample-level latent cloud (mean over time axis)
# ---------------------------------------------------------------------------

def plot_sample_latent_cloud(all_latents: np.ndarray, out_dir: Path) -> None:
    """all_latents: [N, D, L] — we take mean over L → [N, D]"""
    z_mean = all_latents.mean(axis=2)          # [N, D]

    # Drop rows with NaN/Inf; PCA and UMAP do not accept them
    valid = np.isfinite(z_mean).all(axis=1)
    n_bad = int((~valid).sum())
    if n_bad > 0:
        print(f"      Dropping {n_bad} samples with NaN/Inf in latent mean")
    z_clean = z_mean[valid]
    if len(z_clean) < 2:
        print("      Skipping sample latent cloud: too few valid samples after removing NaN/Inf")
        return

    # PCA
    pca = PCA(n_components=2)
    z2 = pca.fit_transform(z_clean)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(z2[:, 0], z2[:, 1], s=6, alpha=0.5, color="#5B8DB8", linewidths=0)
    ax.set_title(f"Sample latent cloud — PCA 2D\n"
                 f"(variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    save(fig, out_dir / "07_sample_latent_pca.png")

    if UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        z_umap = reducer.fit_transform(z_clean)
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(z_umap[:, 0], z_umap[:, 1], s=6, alpha=0.5,
                   color="#5B8DB8", linewidths=0)
        ax.set_title("Sample latent cloud — UMAP 2D")
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        save(fig, out_dir / "08_sample_latent_umap.png")


# ---------------------------------------------------------------------------
# 9  Reconstruction overlay grid
# ---------------------------------------------------------------------------

def plot_reconstructions(ecgs: np.ndarray, recons: np.ndarray,
                         n_samples: int, n_leads: int, out_dir: Path) -> None:
    """ecgs / recons: [N, 12, T]"""
    n = min(n_samples, len(ecgs))
    leads = min(n_leads, 12)
    T = ecgs.shape[2]
    t = np.linspace(0, T / 500, T)       # assume 500 Hz

    fig = plt.figure(figsize=(5 * n, 1.6 * leads))
    gs = gridspec.GridSpec(leads, n, figure=fig, hspace=0.05, wspace=0.05)

    for s in range(n):
        for l in range(leads):
            ax = fig.add_subplot(gs[l, s])
            ax.plot(t, ecgs[s, l], color="#444", linewidth=0.6, label="Real" if l == 0 else "")
            ax.plot(t, recons[s, l], color="#D95F02", linewidth=0.6, alpha=0.8,
                    label="Recon" if l == 0 else "")
            ax.set_yticks([])
            if l == 0:
                ax.set_title(f"Sample {s+1}", fontsize=9)
                ax.legend(fontsize=7, loc="upper right", framealpha=0.5)
            if s == 0:
                ax.set_ylabel(LEAD_NAMES[l], fontsize=8, rotation=0, labelpad=20)
            if l < leads - 1:
                ax.set_xticks([])
            else:
                ax.set_xlabel("s", fontsize=8)

    fig.suptitle("Real (grey) vs Reconstructed (orange) ECG", fontsize=11, y=1.01)
    save(fig, out_dir / "09_reconstructions.png")


# ---------------------------------------------------------------------------
# 10  Per-batch loss scatter
# ---------------------------------------------------------------------------

def plot_loss_scatter(recon_losses: list, vq_losses: list, out_dir: Path) -> dict:
    r = np.array(recon_losses)
    v = np.array(vq_losses)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(r, bins=40, color="#5B8DB8", edgecolor="none", alpha=0.85)
    axes[0].set_xlabel("Reconstruction loss (MSE)")
    axes[0].set_ylabel("Batches")
    axes[0].set_title(f"Recon loss  (mean={r.mean():.4f})")

    axes[1].hist(v, bins=40, color="#F0A500", edgecolor="none", alpha=0.85)
    axes[1].set_xlabel("VQ loss")
    axes[1].set_title(f"VQ loss  (mean={v.mean():.4f})")

    axes[2].scatter(r, v, s=6, alpha=0.4, color="#5B8DB8")
    axes[2].set_xlabel("Reconstruction loss")
    axes[2].set_ylabel("VQ loss")
    axes[2].set_title("Recon vs VQ loss per batch")

    fig.tight_layout()
    save(fig, out_dir / "10_loss_scatter.png")

    return {"mean_recon_loss": round(float(r.mean()), 6),
            "std_recon_loss":  round(float(r.std()),  6),
            "mean_vq_loss":    round(float(v.mean()), 6),
            "std_vq_loss":     round(float(v.std()),  6)}


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="VQ-VAE latent space analysis")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to VQ-VAE Lightning checkpoint (best.ckpt)")
    p.add_argument("--data-dir", type=str, required=True,
                   help="Path to MIMIC-IV-ECG dataset root")
    p.add_argument("--output-dir", type=str, default="latent_analysis",
                   help="Directory to write all output figures")
    p.add_argument("--max-samples", type=int, default=1000,
                   help="Max test samples to analyse (set higher for more coverage)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seq-length", type=int, default=5000)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--test-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-recon-samples", type=int, default=8,
                   help="Number of ECGs shown in reconstruction grid")
    p.add_argument("--n-recon-leads", type=int, default=6,
                   help="Number of leads shown in reconstruction grid (max 12)")
    p.add_argument("--n-heatmap-samples", type=int, default=50,
                   help="Rows in code sequence heatmap")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"\n[1/4] Loading checkpoint: {args.checkpoint}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lightning_model = VQVAELightning.load_from_checkpoint(
            args.checkpoint, map_location="cpu")
    model = lightning_model.vqvae.to(args.device).eval()
    num_embeddings = model.num_embeddings
    codebook_matrix = model.vq.embedding.weight.detach().cpu().numpy()  # [K, D]
    print(f"   Codebook: {num_embeddings} × {codebook_matrix.shape[1]}")

    # ------------------------------------------------------------------
    # Build dataset / dataloader
    # ------------------------------------------------------------------
    print(f"\n[2/4] Building test dataset (max_samples={args.max_samples})")
    dataset = MIMICTestDataset(
        data_dir=args.data_dir,
        max_samples=args.max_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        val_split=args.val_split,
        test_split=args.test_split,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers,
                        pin_memory=False)

    # ------------------------------------------------------------------
    # Inference pass — collect latents, indices, losses, ECGs, recons
    # ------------------------------------------------------------------
    print(f"\n[3/4] Running inference over {len(dataset)} samples …")

    all_indices  = []    # list of [B, L] int tensors
    all_latents  = []    # list of [B, D, L] float tensors (continuous z)
    all_ecgs     = []    # list of [B, 12, T] tensors
    all_recons   = []    # list of [B, 12, T] tensors
    recon_losses = []
    vq_losses    = []

    with torch.no_grad():
        for batch_idx, (ecgs, _) in enumerate(loader):
            ecgs = ecgs.to(args.device)

            # Continuous latent z
            z = model.encode_to_latent(ecgs)       # [B, D, L]
            # Discrete indices
            idx = model.encode_to_indices(ecgs)     # [B, L]
            # Reconstruction (through VQ forward)
            recon, vq_loss, _ = model(ecgs)

            recon_loss = F.mse_loss(recon, ecgs).item()
            recon_losses.append(recon_loss)
            vq_losses.append(vq_loss.item())

            all_indices.append(idx.cpu().numpy())
            all_latents.append(z.cpu().numpy())
            all_ecgs.append(ecgs.cpu().numpy())
            all_recons.append(recon.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"   batch {batch_idx+1}/{len(loader)}  "
                      f"recon={recon_loss:.4f}  vq={vq_loss.item():.4f}")

    all_indices = np.concatenate(all_indices, axis=0)   # [N, L]
    all_latents = np.concatenate(all_latents, axis=0)   # [N, D, L]
    all_ecgs    = np.concatenate(all_ecgs,    axis=0)   # [N, 12, T]
    all_recons  = np.concatenate(all_recons,  axis=0)   # [N, 12, T]

    print(f"   Done. Shapes — indices: {all_indices.shape}, "
          f"latents: {all_latents.shape}")

    # ------------------------------------------------------------------
    # Compute global code counts
    # ------------------------------------------------------------------
    counts = np.bincount(all_indices.ravel(), minlength=num_embeddings).astype(np.float64)

    # ------------------------------------------------------------------
    # Generate plots
    # ------------------------------------------------------------------
    print(f"\n[4/4] Generating plots → {out_dir}/")

    stats = {}

    print("  [1/7] Codebook usage histogram + CDF")
    stats.update(plot_codebook_usage(counts, out_dir))

    print("  [2/7] Codebook embedding PCA / UMAP")
    plot_codebook_embeddings(codebook_matrix, counts, out_dir)

    print("  [3/7] Code sequence heatmap")
    plot_code_heatmap(all_indices, args.n_heatmap_samples, num_embeddings, out_dir)

    print("  [4/7] Positional code frequency")
    plot_positional_code_frequency(all_indices, num_embeddings, out_dir)

    print("  [5/7] Sample latent cloud PCA / UMAP")
    plot_sample_latent_cloud(all_latents, out_dir)

    print("  [6/7] Reconstruction overlay grid")
    plot_reconstructions(all_ecgs, all_recons,
                         args.n_recon_samples, args.n_recon_leads, out_dir)

    print("  [7/7] Loss scatter")
    stats.update(plot_loss_scatter(recon_losses, vq_losses, out_dir))

    # ------------------------------------------------------------------
    # Write summary JSON
    # ------------------------------------------------------------------
    stats["n_samples_analysed"] = len(dataset)
    stats["latent_length"] = int(all_indices.shape[1])
    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  LATENT SPACE SUMMARY")
    print("=" * 60)
    print(f"  Samples analysed      : {stats['n_samples_analysed']}")
    print(f"  Latent length (L/16)  : {stats['latent_length']}")
    print(f"  Codebook utilisation  : {stats['used_codes']}/{stats['total_codes']} "
          f"({stats['usage_pct']:.1f}%)")
    print(f"  Top-20% code coverage : {stats['top20pct_coverage']:.1f}% of assignments")
    print(f"  Mean recon loss (MSE) : {stats['mean_recon_loss']:.6f} "
          f"± {stats['std_recon_loss']:.6f}")
    print(f"  Mean VQ loss          : {stats['mean_vq_loss']:.6f} "
          f"± {stats['std_vq_loss']:.6f}")
    print("=" * 60)
    print(f"\n  All figures and stats.json written to: {out_dir}/\n")


if __name__ == "__main__":
    main()