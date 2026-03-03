#!/usr/bin/env python3
"""
Standalone VAE Training Script for ECG Generation
==================================================

This script trains a basic VAE (Variational Autoencoder) model with KL divergence
for ECG signal reconstruction and generation.

All model components are included in this single file for easy deployment.

Usage:
    python train_vae_baseline.py --data-dir /path/to/mimic --exp-name vae_exp
"""

from __future__ import annotations

import argparse
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import wfdb
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
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
# Loss Functions
# ============================================================================

def vae_loss(
    recon: Tensor,
    x: Tensor,
    mean: Tensor,
    logvar: Tensor,
    kl_weight: float = 0.0001,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    VAE loss function: reconstruction + KL divergence.
    
    Args:
        recon: Reconstructed signal
        x: Original signal
        mean: Latent mean
        logvar: Latent log variance
        kl_weight: Weight for KL divergence term
    
    Returns:
        total_loss, recon_loss, kl_loss
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    kl_loss = kl_loss / (x.size(0) * x.size(1) * x.size(2))
    
    # Total loss
    total_loss = recon_loss + kl_weight * kl_loss
    
    return total_loss, recon_loss, kl_loss


# ============================================================================
# Model Components
# ============================================================================

class ResidualBlock1D(nn.Module):
    """1D Residual block with group normalization."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        return h + residual


class Encoder1D(nn.Module):
    """1D Encoder for ECG signals."""

    def __init__(
        self,
        in_channels: int = 12,
        base_channels: int = 64,
        latent_channels: int = 8,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels

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

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.conv_in(x)
        for block in self.down_blocks:
            h = block(h)
        h = self.mid_block1(h)
        h = self.mid_block2(h)
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        mean, logvar = torch.chunk(h, 2, dim=1)
        return mean, logvar


class Decoder1D(nn.Module):
    """1D Decoder for ECG signals."""

    def __init__(
        self,
        out_channels: int = 12,
        base_channels: int = 64,
        latent_channels: int = 8,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.latent_channels = latent_channels

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
                self.up_blocks.append(nn.ConvTranspose1d(ch, ch, kernel_size=4, stride=2, padding=1))

        self.norm_out = nn.GroupNorm(8, ch)
        self.conv_out = nn.Conv1d(ch, out_channels, kernel_size=7, padding=3)

    def forward(self, z: Tensor) -> Tensor:
        h = self.conv_in(z)
        h = self.mid_block1(h)
        h = self.mid_block2(h)
        for block in self.up_blocks:
            h = block(h)
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


class VAE1D(nn.Module):
    """Variational Autoencoder for ECG signals."""

    def __init__(
        self,
        in_channels: int = 12,
        base_channels: int = 64,
        latent_channels: int = 8,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels

        self.encoder = Encoder1D(
            in_channels=in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
        )

        self.decoder = Decoder1D(
            out_channels=in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.encoder(x)

    def reparameterize(self, mean: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar

    @torch.no_grad()
    def encode_to_latent(self, x: Tensor) -> Tensor:
        mean, _ = self.encode(x)
        return mean

    @torch.no_grad()
    def decode_from_latent(self, z: Tensor) -> Tensor:
        return self.decode(z)


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
# Lightning Modules
# ============================================================================

@dataclass
class VAEConfig:
    """Configuration for VAE model."""
    in_channels: int = 12
    base_channels: int = 64
    latent_channels: int = 8
    channel_multipliers: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    kl_weight: float = 0.0001
    lr: float = 1e-4
    b1: float = 0.9
    b2: float = 0.999


class VAELightningModule(pl.LightningModule):
    """PyTorch Lightning wrapper for VAE model."""

    def __init__(self, config: VAEConfig | dict | None = None, **kwargs) -> None:
        super().__init__()
        
        if config is None:
            config = VAEConfig(**kwargs)
        elif isinstance(config, dict):
            config = VAEConfig(**config)
        
        self.save_hyperparameters(config.__dict__)
        self.config = config

        self.vae = VAE1D(
            in_channels=config.in_channels,
            base_channels=config.base_channels,
            latent_channels=config.latent_channels,
            channel_multipliers=config.channel_multipliers,
            num_res_blocks=config.num_res_blocks,
        )

        self._val_real_sample: Tensor | None = None
        self._val_recon_sample: Tensor | None = None

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.vae(x)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        ecgs, _ = batch
        recon, mean, logvar = self.vae(ecgs)
        total_loss, recon_loss, kl_loss_val = vae_loss(recon, ecgs, mean, logvar, self.config.kl_weight)

        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/recon_loss", recon_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/kl_loss", kl_loss_val, on_step=True, on_epoch=True, prog_bar=False)

        return total_loss

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        ecgs, _ = batch
        recon, mean, logvar = self.vae(ecgs)
        total_loss, recon_loss, kl_loss_val = vae_loss(recon, ecgs, mean, logvar, self.config.kl_weight)

        self.log("val/total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recon_loss", recon_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/kl_loss", kl_loss_val, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            self._val_real_sample = ecgs[0].detach().cpu()
            self._val_recon_sample = recon[0].detach().cpu()

        return total_loss

    @torch.no_grad()
    def encode_to_latent(self, x: Tensor) -> Tensor:
        self.vae.eval()
        return self.vae.encode_to_latent(x)

    @torch.no_grad()
    def decode_from_latent(self, z: Tensor) -> Tensor:
        self.vae.eval()
        return self.vae.decode_from_latent(z)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(
            self.vae.parameters(),
            lr=self.config.lr,
            betas=(self.config.b1, self.config.b2),
        )
        return optimizer


# ============================================================================
# Data Module
# ============================================================================

class VAEMIMICDataModule(pl.LightningDataModule):
    """Data module for VAE training with MIMIC-IV-ECG."""

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

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )


# ============================================================================
# Visualization Callback
# ============================================================================

class VAEVisualizationCallback(Callback):
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

        plt.suptitle(f"Epoch {epoch}")
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        plt.close()


# ============================================================================
# Training Functions
# ============================================================================

def train_vae(args):
    """Train VAE model."""
    print("=" * 80)
    print("Training VAE for ECG Generation")
    print("=" * 80)
    
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory does not exist: {args.data_dir}")
        print("Please set the correct path using --data-dir or DATA_DIR environment variable")
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

    model_config = VAEConfig(
        in_channels=args.in_channels,
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
        channel_multipliers=(1, 2, 4, 4),
        num_res_blocks=args.num_res_blocks,
        kl_weight=args.kl_weight,
        lr=args.lr,
        b1=args.b1,
        b2=args.b2,
    )

    model = VAELightningModule(model_config)

    datamodule = VAEMIMICDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        test_split=args.test_split,
        max_samples=args.max_samples,
        seed=args.seed,
        skip_missing_check=args.skip_missing_check,
        num_leads=args.in_channels,
        seq_length=args.seq_length,
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
                tags=args.wandb_tags if args.wandb_tags else ["vae", "ecg", "baseline"],
                config={
                    "in_channels": args.in_channels,
                    "base_channels": args.base_channels,
                    "latent_channels": args.latent_channels,
                    "num_res_blocks": args.num_res_blocks,
                    "kl_weight": args.kl_weight,
                    "seq_length": args.seq_length,
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

    viz_callback = VAEVisualizationCallback(
        save_dir=samples_dir,
        log_every_n_epochs=args.viz_every_n_epochs,
        num_samples=args.viz_num_samples,
    )

    callbacks = [checkpoint_callback, early_stop_callback, lr_monitor, viz_callback]

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
    
    # Check for resume checkpoint
    ckpt_path = None
    if args.resume:
        if os.path.exists(args.resume):
            ckpt_path = args.resume
            print(f"Resuming from checkpoint: {ckpt_path}")
        else:
            print(f"WARNING: Resume checkpoint not found: {args.resume}")
            print("Starting fresh training instead")
    
    print()

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    print()
    print("=" * 80)
    print("VAE training finished.")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")
    print("=" * 80)


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE for ECG generation")

    # Experiment
    parser.add_argument("--exp-name", type=str, default="vae_baseline", help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--runs-root", type=str, default="runs", help="Root directory for runs")

    # Data
    parser.add_argument("--data-dir", type=str, required=True, help="Path to MIMIC-IV-ECG dataset")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples (for debugging)")
    parser.add_argument("--skip-missing-check", action="store_true", help="Skip missing file check")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test-split", type=float, default=0.1, help="Test split ratio")

    # Model
    parser.add_argument("--in-channels", type=int, default=12, help="Number of ECG leads")
    parser.add_argument("--base-channels", type=int, default=64, help="Base number of channels")
    parser.add_argument("--latent-channels", type=int, default=8, help="Latent channels")
    parser.add_argument("--num-res-blocks", type=int, default=2, help="Number of residual blocks")
    parser.add_argument("--kl-weight", type=float, default=0.0001, help="KL divergence weight")
    parser.add_argument("--seq-length", type=int, default=5000, help="ECG sequence length")

    # Training
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--b2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--max-epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--accelerator", type=str, default="gpu", help="Accelerator type (gpu/cpu)")
    parser.add_argument("--devices", type=int, nargs="+", default=[0], help="Device IDs")
    parser.add_argument("--log-every-n-steps", type=int, default=50, help="Log every N steps")
    parser.add_argument("--check-val-every-n-epoch", type=int, default=1, help="Validate every N epochs")
    parser.add_argument("--gradient-clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--save-top-k", type=int, default=3, help="Save top k checkpoints")

    # Visualization
    parser.add_argument("--viz-every-n-epochs", type=int, default=5, help="Generate visualizations every N epochs")
    parser.add_argument("--viz-num-samples", type=int, default=4, help="Number of samples to visualize")

    # Resume training
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="ecg-vae", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity (username/team)")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name (auto-generated if not set)")
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=None, help="W&B tags")

    return parser.parse_args()


def main():
    args = parse_args()
    train_vae(args)


if __name__ == "__main__":
    main()


