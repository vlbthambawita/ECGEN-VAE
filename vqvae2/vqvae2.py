#!/usr/bin/env python3
"""
VQ-VAE-2 for 12-Lead ECG Generation
=====================================
Architecture based on:
  "Generating Diverse High-Fidelity Images with VQ-VAE-2"
  Razavi et al., NeurIPS 2019 (https://arxiv.org/abs/1906.00446)

Adapted for 12-lead, 10s, 500Hz ECG signals (shape: B x 12 x 5000).

Hierarchy:
  - Bottom level:  encodes local morphology  (QRS, P/T waves)  → 5000 → 625  (stride 8)
  - Top level:     encodes global structure   (rhythm, segment) → 625  → 79   (stride ~8)

Usage
-----
  python vqvae2.py fit   --data-dir /path/to/mimic  [--devices 0]
  python vqvae2.py test  --ckpt-path last.ckpt --data-dir /path/to/mimic
  python vqvae2.py sample --ckpt-path last.ckpt --n-samples 16 --out samples.npy
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wfdb
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)


class ResetEarlyStoppingOnResume(Callback):
    """Reset EarlyStopping wait counter when resuming so training gets a full patience window."""

    def on_fit_start(self, trainer, pl_module):
        if trainer.ckpt_path is None:
            return
        for cb in trainer.callbacks:
            if isinstance(cb, EarlyStopping):
                if hasattr(cb, "wait_count"):
                    cb.wait_count = 0
                if hasattr(cb, "_wait_count"):
                    cb._wait_count = 0
                if hasattr(cb, "stopped_epoch") and cb.stopped_epoch is not None:
                    cb.stopped_epoch = None
                break
from pytorch_lightning.loggers import TensorBoardLogger

# Optional wandb import
try:
    from pytorch_lightning.loggers import WandbLogger
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    WandbLogger = None
    wandb = None


# ============================================================================
# Utility Functions
# ============================================================================

def set_global_seed(seed: int, deterministic: Optional[bool] = None) -> None:
    """Set seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic is True:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================================
# Building Blocks
# ============================================================================

class ResidualBlock1d(nn.Module):
    """Pre-activation residual block for 1-D sequences."""

    def __init__(self, channels: int, res_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, res_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, res_channels),
            nn.SiLU(),
            nn.Conv1d(res_channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class Encoder1d(nn.Module):
    """
    Strided 1-D convolutional encoder.
    Each stride-2 layer halves the temporal dimension.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        residual_channels: int,
        n_res_blocks: int,
        strides: Tuple[int, ...],
        out_channels: int,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)
        ]
        for stride in strides:
            layers += [
                nn.GroupNorm(8, hidden_channels),
                nn.SiLU(),
                nn.Conv1d(hidden_channels, hidden_channels,
                          kernel_size=stride * 2, stride=stride, padding=stride // 2),
            ]
        for _ in range(n_res_blocks):
            layers.append(ResidualBlock1d(hidden_channels, residual_channels))
        layers += [nn.GroupNorm(8, hidden_channels), nn.SiLU(),
                   nn.Conv1d(hidden_channels, out_channels, kernel_size=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder1d(nn.Module):
    """
    Transposed-conv decoder – mirrors the encoder.
    Accepts an optional condition tensor (same length as input) added channel-wise.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        residual_channels: int,
        n_res_blocks: int,
        strides: Tuple[int, ...],
        out_channels: int,
        cond_channels: int = 0,
    ):
        super().__init__()
        total_in = in_channels + cond_channels
        layers: list[nn.Module] = [
            nn.Conv1d(total_in, hidden_channels, kernel_size=1)
        ]
        for _ in range(n_res_blocks):
            layers.append(ResidualBlock1d(hidden_channels, residual_channels))
        for stride in reversed(strides):
            layers += [
                nn.GroupNorm(8, hidden_channels),
                nn.SiLU(),
                nn.ConvTranspose1d(hidden_channels, hidden_channels,
                                   kernel_size=stride * 2, stride=stride,
                                   padding=stride // 2),
            ]
        layers += [nn.GroupNorm(8, hidden_channels), nn.SiLU(),
                   nn.Conv1d(hidden_channels, out_channels, kernel_size=3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        if cond is not None:
            if cond.shape[-1] != x.shape[-1]:
                cond = F.interpolate(cond, size=x.shape[-1], mode='nearest')
            x = torch.cat([x, cond], dim=1)
        return self.net(x)


# ============================================================================
# Vector Quantiser (EMA + straight-through)
# ============================================================================

class VectorQuantiser(nn.Module):
    """
    VQ layer with optional exponential moving average (EMA) codebook updates.
    EMA updates (Oord et al. 2017 – Appendix A) are more stable than gradient updates.
    """

    def __init__(
        self,
        n_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.use_ema = ema_decay > 0

        self.embedding = nn.Embedding(n_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1 / n_embeddings, 1 / n_embeddings)

        if self.use_ema:
            self.ema_decay = ema_decay
            self.register_buffer('ema_cluster_size', torch.zeros(n_embeddings))
            self.register_buffer('ema_dw', self.embedding.weight.data.clone())

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, D, L)  pre-quantised continuous latents
        Returns:
            z_q:   (B, D, L)  quantised latents (straight-through gradient)
            loss:  scalar VQ loss
            codes: (B, L)     discrete code indices
        """
        B, D, L = z.shape
        z_flat = z.permute(0, 2, 1).reshape(-1, D)

        distances = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.embedding.weight.T
            + self.embedding.weight.pow(2).sum(1)
        )

        codes = distances.argmin(1)
        z_q_flat = self.embedding(codes)

        if self.use_ema and self.training:
            one_hot = F.one_hot(codes.detach(), self.n_embeddings).float()
            self.ema_cluster_size.mul_(self.ema_decay).add_(
                one_hot.sum(0), alpha=1 - self.ema_decay)
            dw = one_hot.T @ z_flat.detach()
            self.ema_dw.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)
            n = self.ema_cluster_size.sum()
            smoothed = (self.ema_cluster_size + 1e-5) / (n + self.n_embeddings * 1e-5) * n
            self.embedding.weight.data.copy_(self.ema_dw / smoothed.unsqueeze(1))
            vq_loss = torch.tensor(0.0, device=z.device)
        else:
            vq_loss = F.mse_loss(z_q_flat, z_flat.detach())

        commitment_loss = F.mse_loss(z_flat, z_q_flat.detach())
        loss = vq_loss + self.commitment_cost * commitment_loss

        z_q_flat = z_flat + (z_q_flat - z_flat).detach()
        z_q = z_q_flat.reshape(B, L, D).permute(0, 2, 1)

        return z_q, loss, codes.reshape(B, L)

    @torch.no_grad()
    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (B, L) → (B, D, L)"""
        return self.embedding(codes).permute(0, 2, 1)


# ============================================================================
# VQ-VAE-2 – two-level hierarchy
# ============================================================================

class VQVAE2(nn.Module):
    """
    Two-level VQ-VAE-2 for 1-D multi-channel signals.

    Temporal dimensions (with default strides ×8 each level):
      Input:       B × 12 × 5000
      Bottom enc:  B × D  × 625
      Top enc:     B × D  × 79    (5000 // 64 ≈ 79)
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
    ):
        super().__init__()
        self.n_leads = n_leads
        self.signal_len = signal_len
        self.n_embeddings_top = n_embeddings_top
        self.n_embeddings_bot = n_embeddings_bot
        self.embedding_dim = embedding_dim
        self.enc_bot_strides = enc_bot_strides
        self.enc_top_strides = enc_top_strides

        D = embedding_dim
        H = hidden_channels
        R = residual_channels
        N = n_res_blocks

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

        self.vq_top = VectorQuantiser(
            n_embeddings_top, D, commitment_cost, ema_decay)
        self.vq_bot = VectorQuantiser(
            n_embeddings_bot, D, commitment_cost, ema_decay)

        self.dec_top = Decoder1d(
            in_channels=D, hidden_channels=H,
            residual_channels=R, n_res_blocks=N,
            strides=enc_top_strides, out_channels=D,
        )

        self.dec_bot = Decoder1d(
            in_channels=D, hidden_channels=H,
            residual_channels=R, n_res_blocks=N,
            strides=enc_bot_strides, out_channels=n_leads,
            cond_channels=D,
        )

    def encode(self, x: torch.Tensor):
        """
        x: (B, 12, 5000)
        Returns (z_q_bot, z_q_top, loss_bot, loss_top, codes_bot, codes_top)
        """
        z_bot = self.enc_bot(x)
        z_top_pre = self.enc_top(z_bot)

        z_q_top, loss_top, codes_top = self.vq_top(z_top_pre)

        top_up = self.dec_top(z_q_top)
        
        # Ensure top_up matches z_bot dimensions
        if top_up.shape[-1] != z_bot.shape[-1]:
            top_up = F.interpolate(top_up, size=z_bot.shape[-1], mode='nearest')
        
        z_bot_cond = z_bot + top_up
        z_q_bot, loss_bot, codes_bot = self.vq_bot(z_bot_cond)

        return z_q_bot, z_q_top, loss_bot, loss_top, codes_bot, codes_top

    def decode(self, z_q_bot: torch.Tensor, z_q_top: torch.Tensor) -> torch.Tensor:
        top_up = self.dec_top(z_q_top)
        
        # Ensure top_up matches z_q_bot dimensions
        if top_up.shape[-1] != z_q_bot.shape[-1]:
            top_up = F.interpolate(top_up, size=z_q_bot.shape[-1], mode='nearest')
        
        x_recon = self.dec_bot(z_q_bot, cond=top_up)
        return x_recon

    def forward(self, x: torch.Tensor):
        z_q_bot, z_q_top, loss_bot, loss_top, codes_bot, codes_top = self.encode(x)
        x_recon = self.decode(z_q_bot, z_q_top)
        vq_loss = loss_bot + loss_top
        return x_recon, vq_loss, codes_bot, codes_top

    @torch.no_grad()
    def decode_codes(self, codes_bot: torch.Tensor, codes_top: torch.Tensor) -> torch.Tensor:
        z_q_bot = self.vq_bot.decode_codes(codes_bot)
        z_q_top = self.vq_top.decode_codes(codes_top)
        return self.decode(z_q_bot, z_q_top)


# ============================================================================
# Dataset
# ============================================================================

class MIMICIVECGDataset(Dataset):
    """MIMIC-IV-ECG dataset."""

    FEATURE_NAMES = [
        "rr_interval", "p_onset", "p_end", "qrs_onset", "qrs_end", "t_end",
        "p_axis", "qrs_axis", "t_axis",
    ]

    def __init__(
        self,
        mimic_path: str,
        split: str = "train",
        val_split: float = 0.1,
        test_split: float = 0.1,
        max_samples: Optional[int] = None,
        seed: int = 42,
        skip_missing_check: bool = False,
        num_leads: int = 12,
        seq_length: int = 5000,
    ) -> None:
        self.mimic_path = mimic_path
        self.split = split
        self.seed = seed
        self.skip_missing_check = skip_missing_check
        self.num_leads = num_leads
        self.seq_length = seq_length

        self.load_measurements()
        self.create_splits(val_split, test_split)
        self.filter_by_split()

        if not skip_missing_check:
            self.filter_missing_files()

        if max_samples is not None:
            self.measurements = self.measurements.head(max_samples).reset_index(drop=True)

        if len(self.measurements) == 0:
            raise ValueError(
                f"Dataset is empty after filtering! "
                f"Split: {self.split}, Data path: {self.mimic_path}. "
                "Please check:\n"
                "1. Data path is correct\n"
                "2. machine_measurements.csv exists\n"
                "3. ECG files (.hea/.dat) are present\n"
                "4. Try using --skip-missing-check flag"
            )

        self.compute_feature_stats()
        print(f"Dataset loaded: {len(self.measurements)} samples for split '{self.split}'")

    def load_measurements(self) -> None:
        path = os.path.join(self.mimic_path, "machine_measurements.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"machine_measurements.csv not found at {path}")
        self.measurements = pd.read_csv(path)

    def create_splits(self, val_split: float, test_split: float) -> None:
        subjects = self.measurements["subject_id"].unique()
        train_subjects, test_subjects = train_test_split(
            subjects, test_size=test_split, random_state=self.seed
        )
        train_subjects, val_subjects = train_test_split(
            train_subjects, test_size=val_split / (1 - test_split), random_state=self.seed
        )
        self.train_subjects = set(train_subjects)
        self.val_subjects = set(val_subjects)
        self.test_subjects = set(test_subjects)

    def filter_by_split(self) -> None:
        if self.split == "train":
            subjects = self.train_subjects
        elif self.split == "val":
            subjects = self.val_subjects
        elif self.split == "test":
            subjects = self.test_subjects
        else:
            raise ValueError(f"Unknown split: {self.split}")
        self.measurements = self.measurements[
            self.measurements["subject_id"].isin(subjects)
        ].reset_index(drop=True)

    def filter_missing_files(self) -> None:
        valid_indices = []
        print(f"Checking {len(self.measurements)} files for existence...")
        for idx in range(len(self.measurements)):
            row = self.measurements.iloc[idx]
            path = self._get_wfdb_path(row)
            if os.path.exists(path + ".hea") and os.path.exists(path + ".dat"):
                valid_indices.append(idx)
        
        print(f"Found {len(valid_indices)} valid files out of {len(self.measurements)}")
        
        if len(valid_indices) == 0:
            import warnings
            warnings.warn(
                f"No valid ECG files found in {self.mimic_path}. "
                "Please check your data path or use --skip-missing-check to skip validation.",
                UserWarning
            )
        
        self.measurements = self.measurements.iloc[valid_indices].reset_index(drop=True)

    def compute_feature_stats(self) -> None:
        features = self.measurements[self.FEATURE_NAMES].values.astype(np.float32)
        self.feature_mean = np.nanmean(features, axis=0)
        self.feature_std = np.nanstd(features, axis=0) + 1e-6

    def _get_wfdb_path(self, row: pd.Series) -> str:
        subject_id = int(row["subject_id"])
        study_id = int(row["study_id"])
        subject_str = str(subject_id)
        if len(subject_str) >= 4:
            p_dir = f"p{subject_str[:4]}"
        else:
            p_dir = f"p{subject_str[:2]}"
        p_subdir = f"p{subject_id}"
        s_dir = f"s{study_id}"
        return os.path.join(self.mimic_path, "files", p_dir, p_subdir, s_dir, str(study_id))

    def __len__(self) -> int:
        return len(self.measurements)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        row = self.measurements.iloc[idx]
        path = self._get_wfdb_path(row)

        try:
            record = wfdb.rdrecord(path)
            ecg = record.p_signal.T.astype(np.float32)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            ecg = np.zeros((self.num_leads, self.seq_length), dtype=np.float32)

        if ecg.shape[0] != self.num_leads:
            ecg = ecg[: self.num_leads]
        if ecg.shape[1] < self.seq_length:
            pad_width = ((0, 0), (0, self.seq_length - ecg.shape[1]))
            ecg = np.pad(ecg, pad_width, mode="constant")
        elif ecg.shape[1] > self.seq_length:
            ecg = ecg[:, : self.seq_length]

        ecg = (ecg - ecg.mean()) / (ecg.std() + 1e-6)

        features = row[self.FEATURE_NAMES].values.astype(np.float32)
        features = (features - self.feature_mean) / self.feature_std

        return torch.from_numpy(ecg), torch.from_numpy(features)


# ============================================================================
# Data Module
# ============================================================================

class VQVAE2MIMICDataModule(pl.LightningDataModule):
    """Data module for VQ-VAE-2 training with MIMIC-IV-ECG."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.1,
        test_split: float = 0.1,
        max_samples: int = None,
        seed: int = 42,
        skip_missing_check: bool = False,
        num_leads: int = 12,
        seq_length: int = 5000,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.max_samples = max_samples
        self.seed = seed
        self.skip_missing_check = skip_missing_check
        self.num_leads = num_leads
        self.seq_length = seq_length

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = MIMICIVECGDataset(
                mimic_path=self.data_dir,
                split="train",
                val_split=self.val_split,
                test_split=self.test_split,
                max_samples=self.max_samples,
                seed=self.seed,
                skip_missing_check=self.skip_missing_check,
                num_leads=self.num_leads,
                seq_length=self.seq_length,
            )
            self.val_dataset = MIMICIVECGDataset(
                mimic_path=self.data_dir,
                split="val",
                val_split=self.val_split,
                test_split=self.test_split,
                max_samples=self.max_samples,
                seed=self.seed,
                skip_missing_check=self.skip_missing_check,
                num_leads=self.num_leads,
                seq_length=self.seq_length,
            )
        
        if stage == "test":
            self.test_dataset = MIMICIVECGDataset(
                mimic_path=self.data_dir,
                split="test",
                val_split=self.val_split,
                test_split=self.test_split,
                max_samples=self.max_samples,
                seed=self.seed,
                skip_missing_check=self.skip_missing_check,
                num_leads=self.num_leads,
                seq_length=self.seq_length,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )


# ============================================================================
# PyTorch Lightning Module
# ============================================================================

@dataclass
class VQVAE2Config:
    """Configuration for VQ-VAE-2 model."""
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
    lr: float = 3e-4
    b1: float = 0.9
    b2: float = 0.999


class VQVAE2Lightning(pl.LightningModule):
    """PyTorch Lightning wrapper for VQ-VAE-2 model."""

    def __init__(self, config: VQVAE2Config | dict | None = None, **kwargs) -> None:
        super().__init__()
        
        if config is None:
            config = VQVAE2Config(**kwargs)
        elif isinstance(config, dict):
            config = VQVAE2Config(**config)
        
        self.save_hyperparameters(config.__dict__)
        self.config = config

        self.model = VQVAE2(
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
        )

        self._val_real_sample: Tensor | None = None
        self._val_recon_sample: Tensor | None = None

    def forward(self, x: Tensor):
        return self.model(x)

    def _step(self, batch, stage: str):
        ecgs, _ = batch
        x_recon, vq_loss, codes_bot, codes_top = self.model(ecgs)

        recon_loss = F.l1_loss(x_recon, ecgs)
        total_loss = recon_loss + vq_loss

        self.log(f"{stage}/total_loss", total_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}/recon_loss", recon_loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log(f"{stage}/vq_loss", vq_loss, prog_bar=False, on_epoch=True, on_step=False)
        
        if stage == "val":
            self.log("val_loss", total_loss, prog_bar=True, on_epoch=True, on_step=False)

        # Compute codebook usage less frequently to save memory and compute
        should_log_codes = (stage == "val") or (self.trainer.global_step % 50 == 0)
        if should_log_codes:
            unique_codes_bot = torch.unique(codes_bot.detach()).numel()
            unique_codes_top = torch.unique(codes_top.detach()).numel()
            self.log(f"{stage}/unique_codes_bot", float(unique_codes_bot), on_epoch=True, on_step=False)
            self.log(f"{stage}/unique_codes_top", float(unique_codes_top), on_epoch=True, on_step=False)
            self.log(f"{stage}/codebook_usage_bot", 
                     float(unique_codes_bot) / self.config.n_embeddings_bot, on_epoch=True, on_step=False)
            self.log(f"{stage}/codebook_usage_top", 
                     float(unique_codes_top) / self.config.n_embeddings_top, on_epoch=True, on_step=False)

        return total_loss, x_recon

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        total_loss, _ = self._step(batch, "train")
        return total_loss

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        ecgs, _ = batch
        total_loss, x_recon = self._step(batch, "val")

        if batch_idx == 0:
            self._val_real_sample = ecgs[0].detach().cpu()
            self._val_recon_sample = x_recon[0].detach().cpu()

        return total_loss

    def test_step(self, batch: Any, batch_idx: int) -> Tensor:
        total_loss, _ = self._step(batch, "test")
        return total_loss

    def on_validation_epoch_end(self):
        """Clear CUDA cache after validation to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def sample(
        self,
        n_samples: int = 16,
        temperature: float = 1.0,
    ) -> Tensor:
        """Sample from random codes (placeholder for prior-based sampling)."""
        self.model.eval()
        device = next(self.model.parameters()).device
        
        bot_len = self.config.signal_len // math.prod(self.config.enc_bot_strides)
        top_len = bot_len // math.prod(self.config.enc_top_strides)
        
        codes_top = torch.randint(0, self.config.n_embeddings_top, (n_samples, top_len), device=device)
        codes_bot = torch.randint(0, self.config.n_embeddings_bot, (n_samples, bot_len), device=device)
        
        samples = self.model.decode_codes(codes_bot, codes_top)
        return samples

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

class VQVAE2VisualizationCallback(Callback):
    """Callback to visualize reconstructions during training."""

    def __init__(
        self,
        save_dir: Path,
        log_every_n_epochs: int = 5,
        num_samples: int = 4,
    ) -> None:
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_every_n_epochs = log_every_n_epochs
        self.num_samples = num_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        if not hasattr(pl_module, "_val_real_sample") or not hasattr(pl_module, "_val_recon_sample"):
            return

        real = pl_module._val_real_sample
        recon = pl_module._val_recon_sample

        if real is None or recon is None:
            return

        save_path = self.save_dir / f"epoch_{trainer.current_epoch:04d}.png"
        self._plot_comparison(real, recon, save_path, epoch=trainer.current_epoch)

    def _plot_comparison(self, real: Tensor, recon: Tensor, path: Path, epoch: int):
        real_np = real.cpu().numpy()
        recon_np = recon.cpu().numpy()
        n_leads = min(real_np.shape[0], 8)

        fig, axs = plt.subplots(n_leads, 2, figsize=(12, 1.5 * n_leads))
        if n_leads == 1:
            axs = axs.reshape(1, -1)

        for i in range(n_leads):
            axs[i, 0].plot(real_np[i], linewidth=0.7)
            axs[i, 0].set_title(f"Real - Lead {i}")
            axs[i, 1].plot(recon_np[i], linewidth=0.7)
            axs[i, 1].set_title(f"Reconstructed - Lead {i}")

        plt.suptitle(f"VQ-VAE-2 Reconstruction - Epoch {epoch}")
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        plt.close()


# ============================================================================
# Training Functions
# ============================================================================

def train_vqvae2(args):
    """Train VQ-VAE-2."""
    print("=" * 80)
    print("Training VQ-VAE-2 for ECG Generation")
    print("=" * 80)
    
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory does not exist: {args.data_dir}")
        print("Please set the correct path using --data-dir")
        sys.exit(1)
    
    measurements_file = os.path.join(args.data_dir, "machine_measurements.csv")
    if not os.path.exists(measurements_file):
        print(f"ERROR: machine_measurements.csv not found at: {measurements_file}")
        print("Please ensure you have the correct MIMIC-IV-ECG dataset path")
        sys.exit(1)
    
    print(f"Data directory: {args.data_dir}")
    print(f"Measurements file: {measurements_file}")

    set_global_seed(args.seed)

    run_dir = Path(args.runs_root) / args.exp_name / f"seed_{args.seed}"
    checkpoints_dir = run_dir / "checkpoints"
    samples_dir = run_dir / "samples"
    for d in (checkpoints_dir, samples_dir):
        d.mkdir(parents=True, exist_ok=True)

    model_config = VQVAE2Config(
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
        lr=args.lr,
        b1=args.b1,
        b2=args.b2,
    )

    model = VQVAE2Lightning(model_config)

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

    loggers_list = []
    
    tb_logger = TensorBoardLogger(save_dir=str(run_dir), name="tb")
    loggers_list.append(tb_logger)
    
    if args.wandb and WANDB_AVAILABLE:
        wandb_run_name = args.wandb_run_name or f"{args.exp_name}_seed{args.seed}"
        
        try:
            wandb_logger = WandbLogger(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=wandb_run_name,
                save_dir=str(run_dir),
                tags=args.wandb_tags if args.wandb_tags else ["vqvae2", "ecg"],
                config={
                    "n_leads": args.n_leads,
                    "signal_len": args.signal_len,
                    "hidden_channels": args.hidden_channels,
                    "n_embeddings_top": args.n_embeddings_top,
                    "n_embeddings_bot": args.n_embeddings_bot,
                    "embedding_dim": args.embedding_dim,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "max_epochs": args.max_epochs,
                    "seed": args.seed,
                },
            )
            loggers_list.append(wandb_logger)
            print(f"✓ Weights & Biases logging enabled: {args.wandb_project}/{wandb_run_name}")
        except Exception as e:
            print(f"⚠ Warning: Failed to initialize wandb: {e}")
            print("  Continuing with TensorBoard only")
    elif args.wandb and not WANDB_AVAILABLE:
        print("⚠ Warning: wandb requested but not installed")
        print("  Install with: pip install wandb")
        print("  Continuing with TensorBoard only")

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoints_dir),
        filename="epoch{epoch:03d}-step{step:06d}",
        save_last=True,
        save_top_k=args.save_top_k,
        monitor="val_loss",
        mode="min",
        auto_insert_metric_name=False,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        mode="min",
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    viz_callback = VQVAE2VisualizationCallback(
        save_dir=samples_dir,
        log_every_n_epochs=args.viz_every_n_epochs,
        num_samples=args.viz_num_samples,
    )

    reset_early_stop = ResetEarlyStoppingOnResume()
    callbacks = [checkpoint_callback, early_stop_callback, lr_monitor, viz_callback, reset_early_stop]

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

    print(f"Run directory: {run_dir}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    if getattr(args, "ckpt_path", None):
        print(f"Resuming from checkpoint: {args.ckpt_path}")
    print()

    ckpt_path = getattr(args, "ckpt_path", None)
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    print()
    print("=" * 80)
    print("VQ-VAE-2 training finished.")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")
    print("=" * 80)


def test_vqvae2(args):
    """Test VQ-VAE-2."""
    print("=" * 80)
    print("Testing VQ-VAE-2")
    print("=" * 80)

    if not Path(args.ckpt_path).exists():
        print(f"ERROR: Checkpoint not found: {args.ckpt_path}")
        sys.exit(1)

    model = VQVAE2Lightning.load_from_checkpoint(args.ckpt_path)
    model.eval()

    datamodule = VQVAE2MIMICDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
    )

    trainer.test(model, datamodule=datamodule)


def sample_vqvae2(args):
    """Sample from VQ-VAE-2 using random codes."""
    print("=" * 80)
    print("Sampling from VQ-VAE-2")
    print("=" * 80)

    if not Path(args.ckpt_path).exists():
        print(f"ERROR: Checkpoint not found: {args.ckpt_path}")
        sys.exit(1)

    model = VQVAE2Lightning.load_from_checkpoint(args.ckpt_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"Generating {args.n_samples} samples...")
    samples = model.sample(n_samples=args.n_samples, temperature=args.temperature)
    samples_np = samples.cpu().numpy()

    np.save(args.out, samples_np)
    print(f"Saved {args.n_samples} samples to {args.out}")
    print(f"Shape: {samples_np.shape}")
    print("=" * 80)


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="VQ-VAE-2 for 12-lead ECG")
    subparsers = parser.add_subparsers(dest="command")

    fit_p = subparsers.add_parser("fit", help="Train the model")
    fit_p.add_argument("--data-dir", type=str, required=True,
                       help="Path to MIMIC-IV-ECG dataset")
    fit_p.add_argument("--exp-name", type=str, default="vqvae2_mimic",
                       help="Experiment name")
    fit_p.add_argument("--seed", type=int, default=42, help="Random seed")
    fit_p.add_argument("--runs-root", type=str, default="runs",
                       help="Root directory for runs")
    fit_p.add_argument("--ckpt-path", type=str, default=None,
                       help="Resume training from checkpoint (e.g. last.ckpt)")
    
    fit_p.add_argument("--batch-size", type=int, default=32, help="Batch size")
    fit_p.add_argument("--num-workers", type=int, default=4,
                       help="Number of data loading workers")
    fit_p.add_argument("--max-samples", type=int, default=None,
                       help="Max samples (for debugging)")
    fit_p.add_argument("--skip-missing-check", action="store_true",
                       help="Skip missing file check")
    fit_p.add_argument("--val-split", type=float, default=0.1,
                       help="Validation split ratio")
    fit_p.add_argument("--test-split", type=float, default=0.1,
                       help="Test split ratio")
    
    fit_p.add_argument("--n-leads", type=int, default=12, help="Number of ECG leads")
    fit_p.add_argument("--signal-len", type=int, default=5000,
                       help="ECG sequence length")
    fit_p.add_argument("--hidden-channels", type=int, default=128,
                       help="Hidden channels")
    fit_p.add_argument("--residual-channels", type=int, default=64,
                       help="Residual channels")
    fit_p.add_argument("--n-res-blocks", type=int, default=4,
                       help="Number of residual blocks")
    fit_p.add_argument("--n-embeddings-top", type=int, default=512,
                       help="Top codebook size")
    fit_p.add_argument("--n-embeddings-bot", type=int, default=512,
                       help="Bottom codebook size")
    fit_p.add_argument("--embedding-dim", type=int, default=64,
                       help="Embedding dimension")
    fit_p.add_argument("--commitment-cost", type=float, default=0.25,
                       help="Commitment cost")
    fit_p.add_argument("--ema-decay", type=float, default=0.99,
                       help="EMA decay (0 to disable)")
    
    fit_p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    fit_p.add_argument("--b1", type=float, default=0.9, help="Adam beta1")
    fit_p.add_argument("--b2", type=float, default=0.999, help="Adam beta2")
    fit_p.add_argument("--max-epochs", type=int, default=200,
                       help="Maximum number of epochs")
    fit_p.add_argument("--accelerator", type=str, default="gpu",
                       help="Accelerator type (gpu/cpu)")
    fit_p.add_argument("--devices", type=int, nargs="+", default=[0],
                       help="Device IDs")
    fit_p.add_argument("--log-every-n-steps", type=int, default=50,
                       help="Log every N steps")
    fit_p.add_argument("--check-val-every-n-epoch", type=int, default=1,
                       help="Validate every N epochs")
    fit_p.add_argument("--gradient-clip", type=float, default=1.0,
                       help="Gradient clipping value")
    fit_p.add_argument("--patience", type=int, default=10,
                       help="Early stopping patience")
    fit_p.add_argument("--save-top-k", type=int, default=3,
                       help="Save top k checkpoints")
    
    fit_p.add_argument("--viz-every-n-epochs", type=int, default=5,
                       help="Generate visualizations every N epochs")
    fit_p.add_argument("--viz-num-samples", type=int, default=4,
                       help="Number of samples to visualize")
    
    fit_p.add_argument("--wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    fit_p.add_argument("--wandb-project", type=str, default="ecg-vqvae2",
                       help="W&B project name")
    fit_p.add_argument("--wandb-entity", type=str, default=None,
                       help="W&B entity (username/team)")
    fit_p.add_argument("--wandb-run-name", type=str, default=None,
                       help="W&B run name")
    fit_p.add_argument("--wandb-tags", type=str, nargs="*", default=None,
                       help="W&B tags")

    test_p = subparsers.add_parser("test", help="Evaluate a checkpoint")
    test_p.add_argument("--data-dir", type=str, required=True)
    test_p.add_argument("--ckpt-path", type=str, required=True)
    test_p.add_argument("--batch-size", type=int, default=32)
    test_p.add_argument("--num-workers", type=int, default=4)
    test_p.add_argument("--seed", type=int, default=42)
    test_p.add_argument("--accelerator", type=str, default="gpu")
    test_p.add_argument("--devices", type=int, nargs="+", default=[0])

    sample_p = subparsers.add_parser("sample", help="Sample from a trained model")
    sample_p.add_argument("--ckpt-path", type=str, required=True)
    sample_p.add_argument("--n-samples", type=int, default=16)
    sample_p.add_argument("--temperature", type=float, default=1.0)
    sample_p.add_argument("--out", type=str, default="samples.npy")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    return args


def main():
    args = parse_args()

    if args.command == "fit":
        train_vqvae2(args)
    elif args.command == "test":
        test_vqvae2(args)
    elif args.command == "sample":
        sample_vqvae2(args)


if __name__ == "__main__":
    main()
