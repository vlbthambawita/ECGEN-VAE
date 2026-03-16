"""
nvae_ecg_lightning.py
=====================
PyTorch Lightning NVAE adapted for 12-lead ECG signals.

Input  : (B, 12, 5000)  — 12 leads × 5000 samples @ 500 Hz (10 s)
Latents: hierarchical Normal ladder (same topology as image NVAE)
Output : Gaussian NLL decoder with learned per-timestep σ

Dataset: PTB-XL  (https://physionet.org/content/ptb-xl/1.0.3/)
         Loaded via wfdb from the raw .hea/.dat files.

Architecture changes vs. image NVAE
------------------------------------
  • All Conv2d  → Conv1d   (time is the single spatial axis)
  • All BN2d    → BN1d
  • SE pools over dim=2 (time) only
  • h_prior shape: (1, C, L_coarsest)
  • DiscMixLogistic / Bernoulli → NormalDecoder (μ, log σ per sample)
  • Reconstruction loss: Gaussian NLL
  • DataModule: wfdb reader, per-lead z-score, 4 augmentations

Quick-start
-----------
# Install deps
pip install pytorch-lightning wfdb pandas

# Download PTB-XL (≈2 GB) to ./data/ptbxl/
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/ -P ./data/ptbxl

# Train (MNIST-scale architecture, single GPU)
python nvae_ecg_lightning.py fit \\
    --ptbxl_dir ./data/ptbxl \\
    --num_channels_enc 32 --num_channels_dec 32 \\
    --num_latent_scales 3 --num_groups_per_scale 8 \\
    --num_latent_per_group 16 \\
    --num_preprocess_blocks 2 --num_postprocess_blocks 2 \\
    --num_preprocess_cells 2 --num_postprocess_cells 2 \\
    --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \\
    --ada_groups --use_se --res_dist \\
    --batch_size 32 --max_epochs 200 --devices 1

# Evaluate with IW-ELBO
python nvae_ecg_lightning.py test \\
    --ckpt_path outputs/last.ckpt \\
    --ptbxl_dir ./data/ptbxl \\
    --num_iw_samples 200
"""

from __future__ import annotations

import io
import math
import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import wandb
except ImportError:
    wandb = None  # type: ignore
try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

# Allow importing shared dataset from project root (run from nvae/ or ECGEN-VAE/)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from data.dataset import (  # noqa: E402
    PTBXLDataset as SharedPTBXLDataset,
    MIMICIVECGDataset,
)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from torch.utils.data import Dataset, DataLoader

# ──────────────────────────────────────────────────────────────────────────────
# 1.  UTILITY HELPERS
# ──────────────────────────────────────────────────────────────────────────────

BN_EPS   = 1e-5
BN_MOM   = 0.05
CHANNEL_MULT = 2          # channels double per decoder upscale / halve per encoder downscale
LOG2PI   = math.log(2 * math.pi)

ECG_LEADS      = 12
ECG_SAMPLES    = 5000     # 500 Hz × 10 s
ECG_FS         = 500      # Hz


def soft_clamp5(x: Tensor) -> Tensor:
    return x.div(5.0).tanh().mul(5.0)


def soft_clamp(x: Tensor, c: float) -> Tensor:
    return x.div(c).tanh().mul(c)


def groups_per_scale(num_scales: int,
                     num_groups: int,
                     ada_groups: bool = False,
                     minimum_groups: int = 1) -> List[int]:
    """Group counts per scale, finest first."""
    if not ada_groups:
        return [num_groups] * num_scales
    g = num_groups
    result = []
    for _ in range(num_scales):
        result.append(max(g, minimum_groups))
        g = g // 2
    return list(reversed(result))


# ──────────────────────────────────────────────────────────────────────────────
# 2.  DISTRIBUTIONS
# ──────────────────────────────────────────────────────────────────────────────

class Normal:
    """
    Diagonal Gaussian parameterised by (μ, log σ).
    Works on tensors of any shape.
    """

    def __init__(self, mu: Tensor, log_sigma: Tensor):
        self.mu        = mu
        self.log_sigma = soft_clamp5(log_sigma)

    @property
    def sigma(self) -> Tensor:
        return self.log_sigma.exp()

    def sample(self) -> Tensor:
        return self.mu + self.sigma * torch.randn_like(self.mu)

    def log_p(self, x: Tensor) -> Tensor:
        """Element-wise log-probability."""
        return -0.5 * (LOG2PI + 2.0 * self.log_sigma
                       + ((x - self.mu) / self.sigma) ** 2)

    def kl(self, other: "Normal") -> Tensor:
        """KL( self || other ) element-wise."""
        return (
            other.log_sigma - self.log_sigma
            + 0.5 * (self.sigma ** 2 + (self.mu - other.mu) ** 2)
              / (other.sigma ** 2)
            - 0.5
        )

    def kl_prior(self) -> Tensor:
        """KL( self || N(0,1) ) element-wise."""
        return -0.5 * (1.0 + 2.0 * self.log_sigma
                       - self.mu ** 2 - self.sigma ** 2)


class NormalDecoder:
    """
    Continuous output distribution for ECG reconstruction.

    logits : (B, 2*C_in, L) — first half = μ, second half = log σ
    The log σ is clamped to [-4, 4] to keep the NLL numerically stable.
    """

    LOG_SIGMA_MIN = -4.0
    LOG_SIGMA_MAX =  4.0

    def __init__(self, logits: Tensor):
        mu, log_sigma = logits.chunk(2, dim=1)
        self.mu        = mu
        self.log_sigma = log_sigma.clamp(self.LOG_SIGMA_MIN, self.LOG_SIGMA_MAX)

    @property
    def sigma(self) -> Tensor:
        return self.log_sigma.exp()

    @property
    def mean(self) -> Tensor:
        return self.mu

    def log_p(self, x: Tensor) -> Tensor:
        """
        Gaussian NLL, element-wise. Returns (B, C, L).
        Sum over C and L, mean over B for the scalar loss.
        """
        return -0.5 * (LOG2PI + 2.0 * self.log_sigma
                       + ((x - self.mu) / self.sigma) ** 2)

    def nll(self, x: Tensor) -> Tensor:
        """Scalar: mean over batch, sum over leads & time."""
        return -self.log_p(x).sum(dim=[1, 2]).mean()


# ──────────────────────────────────────────────────────────────────────────────
# 3.  1-D NEURAL OPERATIONS
# ──────────────────────────────────────────────────────────────────────────────

class SpectralNormConv1d(nn.Module):
    """
    Conv1d with optional spectral-norm power iteration.
    Tracks running (u, v) vectors for the dominant singular value.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 weight_norm: bool = True,
                 num_power_iter: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.weight_norm    = weight_norm
        self.num_power_iter = num_power_iter

        if weight_norm:
            rows = out_channels
            cols = in_channels * kernel_size // groups
            self.register_buffer("sr_u", F.normalize(
                torch.ones(rows).normal_(0, 1), dim=0, eps=1e-3))
            self.register_buffer("sr_v", F.normalize(
                torch.ones(cols).normal_(0, 1), dim=0, eps=1e-3))
            # warm-up iterations
            w = self.conv.weight.view(rows, -1)
            with torch.no_grad():
                for _ in range(20):
                    vn = F.normalize(w.t() @ self.sr_u, dim=0, eps=1e-3)
                    un = F.normalize(w   @ vn,          dim=0, eps=1e-3)
                self.sr_v.copy_(vn)
                self.sr_u.copy_(un)

    def spectral_norm_loss(self) -> Tensor:
        """σ_max² for spectral regularisation."""
        if not self.weight_norm:
            return torch.tensor(0.0, device=self.conv.weight.device)
        rows = self.conv.weight.size(0)
        w = self.conv.weight.view(rows, -1)
        with torch.no_grad():
            for _ in range(self.num_power_iter):
                vn = F.normalize(w.t() @ self.sr_u, dim=0, eps=1e-3)
                un = F.normalize(w   @ vn,          dim=0, eps=1e-3)
            self.sr_v.copy_(vn)
            self.sr_u.copy_(un)
        sigma = self.sr_u @ (w @ self.sr_v)
        return sigma ** 2

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


# Short alias
Conv1D = SpectralNormConv1d


class BNSwishConv1d(nn.Module):
    """BN1d → Swish (SiLU) → Conv1d."""

    def __init__(self, Cin: int, Cout: int, k: int = 5,
                 stride: int = 1, padding: int = 2,
                 groups: int = 1, dilation: int = 1,
                 weight_norm: bool = True):
        super().__init__()
        self.bn   = nn.BatchNorm1d(Cin, eps=BN_EPS, momentum=BN_MOM)
        self.conv = Conv1D(Cin, Cout, k, stride=stride, padding=padding,
                           groups=groups, dilation=dilation, bias=False,
                           weight_norm=weight_norm)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.silu(self.bn(x)))

    def spectral_norm_loss(self) -> Tensor:
        return self.conv.spectral_norm_loss()


class SE1d(nn.Module):
    """Squeeze-and-Excitation over the time axis."""

    def __init__(self, C: int, r: int = 4):
        super().__init__()
        mid = max(C // r, 4)
        self.fc = nn.Sequential(
            nn.Linear(C, mid, bias=True),
            nn.SiLU(),
            nn.Linear(mid, C, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, L)
        s = x.mean(dim=2)                          # global avg pool → (B, C)
        s = self.fc(s).unsqueeze(2)                # (B, C, 1)
        return x * s


class ResidualCell1d(nn.Module):
    """
    Inverted-bottleneck residual cell (1-D).
    Architecture: BN → pw-expand → depthwise(k=5) → pw-contract → SE → residual
    Matches the NVAE image cell but with 1-D convolutions.
    """

    def __init__(self, C: int, use_se: bool = True,
                 exp_factor: int = 6, stride: int = 1,
                 weight_norm: bool = True):
        super().__init__()
        Cmid = C * exp_factor
        self.bn    = nn.BatchNorm1d(C, eps=BN_EPS, momentum=BN_MOM)
        # point-wise expand
        self.conv1 = Conv1D(C, Cmid, 1, bias=False, weight_norm=weight_norm)
        # depth-wise temporal convolution (k=5, causal-friendly padding)
        self.dw    = BNSwishConv1d(Cmid, Cmid, k=5, stride=stride, padding=2,
                                   groups=Cmid, weight_norm=weight_norm)
        # point-wise contract
        self.conv2 = Conv1D(Cmid, C, 1, bias=False, weight_norm=weight_norm)
        self.se    = SE1d(C) if use_se else nn.Identity()
        # skip when strided
        self.skip  = (Conv1D(C, C, 1, stride=stride, bias=False,
                             weight_norm=weight_norm)
                      if stride > 1 else None)

    def forward(self, x: Tensor) -> Tensor:
        residual = x if self.skip is None else self.skip(x)
        out = F.silu(self.bn(x))
        out = F.silu(self.conv1(out))
        out = self.dw(out)
        out = self.conv2(out)
        out = self.se(out)
        return out + residual

    def spectral_norm_losses(self) -> List[Tensor]:
        losses = [self.conv1.spectral_norm_loss(),
                  self.dw.spectral_norm_loss(),
                  self.conv2.spectral_norm_loss()]
        if self.skip is not None:
            losses.append(self.skip.spectral_norm_loss())
        return losses


class EncCombinerCell1d(nn.Module):
    """
    Fuse bottom-up encoder feature with top-down decoder context.

    x_enc : (B, enc_ch, L_enc)  — bottom-up feature at this group
    x_dec : (B, dec_ch, L_dec)  — top-down context at this scale

    The projection conv maps dec_ch → enc_ch so the residual addition is valid.
    Length alignment uses adaptive_avg_pool1d (parameter-free, exact output size).
    """

    def __init__(self, enc_ch: int, dec_ch: int):
        super().__init__()
        # Projects from dec_ch to enc_ch — crucial when enc_ch != dec_ch
        self.conv = Conv1D(dec_ch, enc_ch, 1, weight_norm=False)

    def forward(self, x_enc: Tensor, x_dec: Tensor) -> Tensor:
        x_dec_proj = self.conv(x_dec)                              # (B, enc_ch, L_dec)
        if x_dec_proj.size(2) != x_enc.size(2):
            # adaptive_avg_pool1d is exact and parameter-free — no learned interpolation
            x_dec_proj = F.adaptive_avg_pool1d(x_dec_proj, x_enc.size(2))
        return x_enc + x_dec_proj                                  # (B, enc_ch, L_enc)


class DecCombinerCell1d(nn.Module):
    """Concatenate sampled latent z onto decoder feature map."""

    def __init__(self, Cin: int, Cz: int, Cout: int):
        super().__init__()
        self.conv = Conv1D(Cin + Cz, Cout, 1, weight_norm=False)

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        return self.conv(torch.cat([x, z], dim=1))


class _ReflectPad1d(nn.Module):
    """
    Reflect-pad a 1-D signal (B, C, L) to (B, C, L + pad_amount).
    Pads symmetrically: half on the left, half+1 on the right (for odd amounts).
    Reflect padding is preferred over zero-padding for ECG because it avoids
    artificial discontinuities at the signal boundary.
    """

    def __init__(self, pad_amount: int):
        super().__init__()
        self.left  = pad_amount // 2
        self.right = pad_amount - self.left

    def forward(self, x: Tensor) -> Tensor:
        return F.pad(x, (self.left, self.right), mode="reflect")


# ──────────────────────────────────────────────────────────────────────────────
# 4.  NVAE-ECG  MODEL
# ──────────────────────────────────────────────────────────────────────────────

class NVAE_ECG(nn.Module):
    """
    Deep Hierarchical VAE for 12-lead ECG signals.

    Input shape : (B, 12, 5000)
    Latent shape: hierarchical — multiple groups at multiple temporal scales
    Output      : NormalDecoder logits (B, 24, 5000) → μ and log σ per sample

    Architecture (top → bottom)
    ---------------------------
    pre-process  : Conv1d stem + num_preprocess_blocks × (ResCell + stride-2)
                   5000 → 5000 / 2^num_preprocess_blocks
    encoder      : num_latent_scales × (num_groups × ResCell + stride-2)
    top-down dec : coarse → fine, sampling z_l at each group
    post-process : num_postprocess_blocks × (upsample + ResCell)
    output head  : Conv1d → (B, 2*12, 5000)

    Parameters
    ----------
    num_channels_enc / dec : base channel count (doubled each down-scale)
    num_latent_scales      : number of temporal resolution levels
    num_groups_per_scale   : latent groups per scale (or max if ada_groups)
    num_latent_per_group   : dimensionality of each latent group
    ada_groups             : halve group count each coarser scale
    min_groups_per_scale   : floor when ada_groups=True
    use_se                 : Squeeze-and-Excitation in residual cells
    res_dist               : residual posterior parameterisation
    weight_decay_norm      : λ for spectral regularisation
    """

    def __init__(self,
                 num_channels_enc: int   = 32,
                 num_channels_dec: int   = 32,
                 num_preprocess_blocks: int  = 2,
                 num_postprocess_blocks: int = 2,
                 num_preprocess_cells: int   = 2,
                 num_postprocess_cells: int  = 2,
                 num_latent_scales: int  = 3,
                 num_groups_per_scale: int = 8,
                 num_latent_per_group: int = 16,
                 num_cell_per_cond_enc: int = 2,
                 num_cell_per_cond_dec: int = 2,
                 ada_groups: bool        = False,
                 min_groups_per_scale: int = 1,
                 use_se: bool            = True,
                 res_dist: bool          = True,
                 weight_decay_norm: float = 1e-2,
                 ):
        super().__init__()

        self.use_se   = use_se
        self.res_dist = res_dist
        self.weight_decay_norm = weight_decay_norm

        self.num_latent_scales    = num_latent_scales
        self.num_groups_per_scale = num_groups_per_scale
        self.num_latent_per_group = num_latent_per_group
        self.groups_per_scale     = groups_per_scale(
            num_latent_scales, num_groups_per_scale,
            ada_groups, min_groups_per_scale)

        # ── Compute coarsest sequence length BY SIMULATION ────────────────
        # Do NOT use ECG_SAMPLES // 2^N — that is floor division and will
        # give the wrong answer when any intermediate length is odd.
        # Instead simulate the exact same stride-2 chain the encoder runs.
        #
        # Input after reflect-pad: 5120 (guaranteed power-of-2 multiple)
        # Each stride-2 ResidualCell (k=5, p=2) outputs ceil(L/2) = L//2
        # when L is a power-of-2 (always even), so floor==ceil here.
        _L = 5120   # padded length used in _make_preprocess
        for _ in range(num_preprocess_blocks):        # pre-process downs
            _L = (_L + 2*2 - 1*(5-1) - 1) // 2 + 1  # conv1d_out formula
        for _ in range(num_latent_scales - 1):        # encoder between-scale downs
            _L = (_L + 2*2 - 1*(5-1) - 1) // 2 + 1
        self.coarsest_len = _L
        assert self.coarsest_len >= 1, (
            f"Too many downsampling steps: coarsest_len={self.coarsest_len}")

        # ── Pre-processing ────────────────────────────────────────────────
        self.pre_process = self._make_preprocess(
            num_channels_enc, num_preprocess_blocks, num_preprocess_cells)
        # _make_preprocess sets self._pad_to and self._pad_amount as side-effects

        enc_ch = num_channels_enc * (CHANNEL_MULT ** num_preprocess_blocks)

        # ── Pre-compute decoder channel width at each encoder scale ───────
        # Decoder starts at dec_ch (coarsest) and halves each finer scale.
        # Index [s] = decoder channel width when processing encoder scale s.
        _dec_ch_start = num_channels_dec * (
            CHANNEL_MULT ** (num_preprocess_blocks + num_latent_scales - 1))
        dec_ch_per_scale: List[int] = []
        _dc = _dec_ch_start
        for s in range(num_latent_scales - 1, -1, -1):
            dec_ch_per_scale.insert(0, _dc)
            if s > 0:
                _dc = _dc // CHANNEL_MULT

        # ── Bottom-up (encoder) ───────────────────────────────────────────
        self.enc_cells     = nn.ModuleList()
        self.enc_combiners = nn.ModuleList()
        self.enc_sampler   = nn.ModuleList()

        ch = enc_ch
        for s in range(num_latent_scales):
            for g in range(self.groups_per_scale[s]):
                self.enc_cells.append(nn.Sequential(*[
                    ResidualCell1d(ch, use_se=use_se)
                    for _ in range(num_cell_per_cond_enc)]))
                # FIX: pass both enc_ch and dec_ch — projection is dec→enc
                self.enc_combiners.append(
                    EncCombinerCell1d(enc_ch=ch, dec_ch=dec_ch_per_scale[s]))
                self.enc_sampler.append(
                    Conv1D(ch, 2 * num_latent_per_group, 1, weight_norm=False))
            if s < num_latent_scales - 1:
                # stride-2 downsampling between scales (same channels)
                self.enc_cells.append(ResidualCell1d(ch, use_se=use_se, stride=2))
                # double channels for next scale
                self.enc_cells.append(nn.Sequential(
                    Conv1D(ch, ch * CHANNEL_MULT, 1, weight_norm=False)))
                ch = ch * CHANNEL_MULT

        # ── Learnable prior input at coarsest scale ───────────────────────
        dec_ch = num_channels_dec * (CHANNEL_MULT ** (
            num_preprocess_blocks + num_latent_scales - 1))
        self.h_prior = nn.Parameter(
            torch.zeros(1, dec_ch, self.coarsest_len))

        # ── Top-down (decoder) ────────────────────────────────────────────
        self.dec_cells     = nn.ModuleList()
        self.dec_combiners = nn.ModuleList()
        self.dec_sampler   = nn.ModuleList()

        dc = dec_ch
        for s in range(num_latent_scales - 1, -1, -1):
            for g in range(self.groups_per_scale[s]):
                # prior: p(z_l | z_{<l})
                self.dec_sampler.append(nn.Sequential(
                    nn.ELU(),
                    Conv1D(dc, 2 * num_latent_per_group, 1, weight_norm=False)))
                # combiner: concat z → feature
                self.dec_combiners.append(
                    DecCombinerCell1d(dc, num_latent_per_group, dc))
                # residual cells after combiner
                self.dec_cells.append(nn.Sequential(*[
                    ResidualCell1d(dc, use_se=use_se)
                    for _ in range(num_cell_per_cond_dec)]))
            if s > 0:
                # upsample + channel contraction
                self.dec_cells.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="linear",
                                align_corners=False),
                    Conv1D(dc, dc // CHANNEL_MULT, 1, weight_norm=False)))
                dc = dc // CHANNEL_MULT

        # ── Post-processing ───────────────────────────────────────────────
        self.post_process = self._make_postprocess(
            dc, num_postprocess_blocks, num_postprocess_cells)

        final_ch = dc // (CHANNEL_MULT ** num_postprocess_blocks)
        self.bn_out = nn.BatchNorm1d(final_ch, eps=BN_EPS, momentum=BN_MOM)

        # ── Output head: predict (μ, log σ) for each lead × time-step ─────
        # output channels = 2 × ECG_LEADS  (μ and log σ for each of 12 leads)
        self.out_conv = Conv1D(final_ch, 2 * ECG_LEADS, 1, weight_norm=False)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _make_preprocess(self, base_ch: int, num_blocks: int,
                         cells_per_block: int) -> nn.Sequential:
        layers: List[nn.Module] = []
        # ── Pad to nearest power-of-2 length ──────────────────────────────
        # 5000 is not a power of 2. When stride-2 conv (k=5, p=2) hits an
        # odd length (625), PyTorch ceiling-divides: 625→313, not 312.
        # This makes encoder lengths (313, 625, 1250) and decoder lengths
        # (312, 624, 1248) permanently mismatched, crashing at kl() and
        # any tensor addition across encoder/decoder.
        #
        # Fix: pad the input to 5120 (= 512×10, nearest ≥5000 power-of-2
        # factor). All stride-2 steps are now exact: 5120→2560→1280→640→320.
        # We crop back to ECG_SAMPLES at the end of decode().
        #
        # Padding added here as a constant reflect pad (preserves ECG signal
        # at edges better than zero-padding).
        pad_to   = 1 << (ECG_SAMPLES - 1).bit_length()   # 8192 for 5000
        # 8192 is too large; find smallest power-of-2 ≥ ECG_SAMPLES
        # that is still divisible by 2^(num_blocks + num_latent_scales - 1)
        # For 5000 and 4 total downsamples: need divisible by 16 → 5120
        self._pad_to = 5120          # stored so decode() knows the padded length
        self._pad_amount = self._pad_to - ECG_SAMPLES   # 120 samples

        layers.append(_ReflectPad1d(self._pad_amount))   # 5000 → 5120
        # stem: project 12 leads → base channels
        layers.append(Conv1D(ECG_LEADS, base_ch, 7, padding=3,
                             weight_norm=False))
        ch = base_ch
        for _ in range(num_blocks):
            for _ in range(cells_per_block):
                layers.append(ResidualCell1d(ch, use_se=self.use_se))
            # stride-2 downsample (same channels out)
            layers.append(ResidualCell1d(ch, use_se=self.use_se, stride=2))
            # double channels so next block and encoder get enc_ch
            layers.append(Conv1D(ch, ch * CHANNEL_MULT, 1, weight_norm=False))
            ch = ch * CHANNEL_MULT
        return nn.Sequential(*layers)

    def _make_postprocess(self, base_ch: int, num_blocks: int,
                          cells_per_block: int) -> nn.Sequential:
        layers: List[nn.Module] = []
        ch = base_ch
        for _ in range(num_blocks):
            ch_out = ch // CHANNEL_MULT
            layers.append(nn.Upsample(scale_factor=2, mode="linear",
                                      align_corners=False))
            layers.append(Conv1D(ch, ch_out, 1, weight_norm=False))
            ch = ch_out
            for _ in range(cells_per_block):
                layers.append(ResidualCell1d(ch, use_se=self.use_se))
        return nn.Sequential(*layers)

    # ── spectral regularisation ───────────────────────────────────────────────

    def spectral_norm_loss(self) -> Tensor:
        total = torch.tensor(0.0, device=self.h_prior.device)
        for m in self.modules():
            if isinstance(m, SpectralNormConv1d) and m.weight_norm:
                total = total + m.spectral_norm_loss()
        return total

    # ── encode / decode / forward ─────────────────────────────────────────────

    def encode(self, x: Tensor) -> List[Tensor]:
        """
        Bottom-up pass.
        x : (B, 12, 5000)
        Returns list of encoder feature tensors, one per latent group,
        ordered finest-scale-first.
        """
        s = self.pre_process(x)
        enc_feats: List[Tensor] = []
        cell_idx = 0
        for scale_idx in range(self.num_latent_scales):
            for g in range(self.groups_per_scale[scale_idx]):
                s = self.enc_cells[cell_idx](s)
                cell_idx += 1
                enc_feats.append(s)
            if scale_idx < self.num_latent_scales - 1:
                s = self.enc_cells[cell_idx](s)   # stride-2 downsampling cell
                cell_idx += 1
                s = self.enc_cells[cell_idx](s)   # channel-doubling conv
                cell_idx += 1
        return enc_feats  # finest-scale last

    def decode(self, enc_feats: List[Tensor]
               ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        """
        Top-down pass.

        enc_feats ordering (ENCODER order, finest scale LAST):
            enc_feats[0 .. gps[0]-1]          scale 0 (finest)
            enc_feats[gps[0] .. gps[0]+gps[1]] scale 1
            ...
            enc_feats[-gps[-1]:]               scale S-1 (coarsest)

        Decoder iterates COARSE→FINE (scale_idx: S-1 → 0).
        enc_group_idx maps decoder scale_idx+g back to encoder feat position.

        All lengths are now guaranteed to match because the encoder input is
        padded to 5120 (power-of-2). The padded region is cropped at the end.

        Returns
        -------
        x_hat      : (B, 2*ECG_LEADS, ECG_SAMPLES)
        kl_list    : per-group KL  (B, num_latent_per_group, L_scale)
        log_q_list : per-group log q(z|x)  (same shape)
        """
        B = enc_feats[0].size(0)
        s = self.h_prior.expand(B, -1, -1)          # (B, dec_ch, coarsest_L)

        kl_list:    List[Tensor] = []
        log_q_list: List[Tensor] = []

        dec_cell_idx    = 0
        dec_sampler_idx = 0

        for scale_idx in range(self.num_latent_scales - 1, -1, -1):
            for g in range(self.groups_per_scale[scale_idx]):
                # Map decoder iteration → encoder feat index
                enc_group_idx = sum(self.groups_per_scale[:scale_idx]) + g

                # ── prior p(z_l | z_{<l}) ─────────────────────────────────
                # s:            (B, dc,          L_dec)
                # prior_params: (B, 2*num_latent, L_dec)
                prior_params = self.dec_sampler[dec_sampler_idx](s)
                mu_p, log_s_p = prior_params.chunk(2, dim=1)

                # ── posterior q(z_l | x, z_{<l}) ──────────────────────────
                # enc_feats[enc_group_idx]: (B, enc_ch, L_enc)
                # EncCombinerCell projects dec→enc channels and aligns length
                # fused:       (B, enc_ch, L_enc)
                # post_params: (B, 2*num_latent, L_enc)
                fused = self.enc_combiners[enc_group_idx](
                    enc_feats[enc_group_idx], s)
                post_params = self.enc_sampler[enc_group_idx](fused)
                mu_q, log_s_q = post_params.chunk(2, dim=1)

                if self.res_dist:
                    # Residual parameterisation: q = prior + learned offset
                    # prior is at L_dec, posterior at L_enc — align prior first
                    if mu_p.size(2) != mu_q.size(2):
                        mu_p    = F.adaptive_avg_pool1d(mu_p,    mu_q.size(2))
                        log_s_p = F.adaptive_avg_pool1d(log_s_p, mu_q.size(2))
                    mu_q    = mu_p.detach() + mu_q
                    log_s_q = log_s_p.detach() + log_s_q

                posterior = Normal(mu_q, log_s_q)
                z = posterior.sample()              # (B, num_latent, L_enc)

                # ── KL( q || p ) ───────────────────────────────────────────
                # Both must be at the same length before subtraction in kl()
                if mu_p.size(2) != mu_q.size(2):
                    mu_p    = F.adaptive_avg_pool1d(mu_p,    mu_q.size(2))
                    log_s_p = F.adaptive_avg_pool1d(log_s_p, mu_q.size(2))
                prior = Normal(mu_p, log_s_p)
                kl = posterior.kl(prior)            # (B, num_latent, L_enc)
                kl_list.append(kl)
                log_q_list.append(posterior.log_p(z))

                # ── decoder combiner ──────────────────────────────────────
                # Bring s to z's length before concat (in case of residual drift)
                if s.size(2) != z.size(2):
                    s = F.adaptive_avg_pool1d(s, z.size(2))
                s = self.dec_combiners[dec_sampler_idx](s, z)   # (B, dc, L_enc)

                s = self.dec_cells[dec_cell_idx](s)
                dec_cell_idx    += 1
                dec_sampler_idx += 1

            if scale_idx > 0:
                s = self.dec_cells[dec_cell_idx](s)     # upsample block
                dec_cell_idx += 1

        # ── post-process ──────────────────────────────────────────────────
        s = self.post_process(s)
        s = F.silu(self.bn_out(s))

        # Crop from padded length (5120-derived) back to exact ECG_SAMPLES.
        # The reflect-pad added self._pad_amount samples symmetrically;
        # slice off the right portion first, then adaptive-pool for safety.
        pad_left = getattr(self, '_pad_amount', 0) // 2
        if s.size(2) > ECG_SAMPLES:
            s = s[:, :, pad_left: pad_left + ECG_SAMPLES]
        if s.size(2) != ECG_SAMPLES:
            # Fallback: shouldn't be needed with power-of-2 padding
            s = F.interpolate(s, size=ECG_SAMPLES, mode="linear",
                              align_corners=False)

        x_hat = self.out_conv(s)                    # (B, 2*ECG_LEADS, ECG_SAMPLES)
        return x_hat, kl_list, log_q_list

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        return self.decode(self.encode(x))

    # ── ELBO ─────────────────────────────────────────────────────────────────

    def elbo(self, x: Tensor,
             beta: float = 1.0,
             weight_decay_norm: Optional[float] = None
             ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Negative ELBO:  recon_nll + β · KL + λ · spectral_reg

        Returns (loss, recon_nll, kl)  — all scalars.
        """
        x_hat, kl_list, _ = self.forward(x)

        # reconstruction: Gaussian NLL summed over leads & time, mean over batch
        dec = NormalDecoder(x_hat)
        recon = dec.nll(x)

        # KL: mean over latent-dim & time, then mean over batch.
        # Using .mean(dim=[1,2]) rather than .sum() ensures that coarse groups
        # (short L) and fine groups (long L) contribute equally — .sum() would
        # weight fine-scale groups proportional to their sequence length.
        kl = sum(kl.mean(dim=[1, 2]).mean() for kl in kl_list)

        # spectral regularisation
        lam = weight_decay_norm if weight_decay_norm is not None \
              else self.weight_decay_norm
        sr  = lam * self.spectral_norm_loss()

        loss = recon + beta * kl + sr
        return loss, recon, kl

    # ── IW-ELBO ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def iwelbo(self, x: Tensor, num_samples: int = 200) -> Tensor:
        """
        Importance-weighted ELBO (Burda et al., 2016).
        Returns mean log p(x) estimate over the batch.
        """
        enc_feats = self.encode(x)
        log_weights: List[Tensor] = []

        for _ in range(num_samples):
            x_hat, kl_list, log_q_list = self.decode(enc_feats)

            dec = NormalDecoder(x_hat)
            log_px_z = dec.log_p(x).sum(dim=[1, 2])   # (B,)

            log_pz   = sum(-kl.mean(dim=[1, 2]) for kl in kl_list)
            log_qz_x = sum(lq.mean(dim=[1, 2]) for lq in log_q_list)

            log_weights.append(log_px_z + log_pz - log_qz_x)

        log_w = torch.stack(log_weights, dim=0)               # (K, B)
        log_p = torch.logsumexp(log_w, dim=0) - math.log(num_samples)
        return log_p.mean()

    # ── Sampling ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(self, num_samples: int, temp: float = 0.8,
               device: torch.device = torch.device("cpu")) -> Tensor:
        """Generate synthetic ECG signals from the prior."""
        B  = num_samples
        s  = self.h_prior.expand(B, -1, -1).to(device)

        dec_cell_idx    = 0
        dec_sampler_idx = 0

        for scale_idx in range(self.num_latent_scales - 1, -1, -1):
            for g in range(self.groups_per_scale[scale_idx]):
                prior_params = self.dec_sampler[dec_sampler_idx](s)
                mu_p, log_s_p = prior_params.chunk(2, dim=1)
                prior = Normal(mu_p, log_s_p)
                z = mu_p + temp * prior.sigma * torch.randn_like(mu_p)

                s = self.dec_combiners[dec_sampler_idx](s, z)
                s = self.dec_cells[dec_cell_idx](s)
                dec_cell_idx    += 1
                dec_sampler_idx += 1

            if scale_idx > 0:
                s = self.dec_cells[dec_cell_idx](s)
                dec_cell_idx += 1

        s = self.post_process(s)
        s = F.silu(self.bn_out(s))

        # Crop padded length back to ECG_SAMPLES
        pad_left = getattr(self, '_pad_amount', 0) // 2
        if s.size(2) > ECG_SAMPLES:
            s = s[:, :, pad_left: pad_left + ECG_SAMPLES]
        if s.size(2) != ECG_SAMPLES:
            s = F.interpolate(s, size=ECG_SAMPLES, mode="linear",
                              align_corners=False)

        x_hat = self.out_conv(s)
        dec   = NormalDecoder(x_hat)
        return dec.mean   # (B, 12, 5000)


# ──────────────────────────────────────────────────────────────────────────────
# 4b. VALIDATION PLOTTING (for wandb)
# ──────────────────────────────────────────────────────────────────────────────

_LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _fig_to_wandb_image(fig: "plt.Figure"):
    """Convert a matplotlib figure to wandb.Image for logging (or None if wandb/PIL unavailable)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    if wandb is not None and Image is not None:
        return wandb.Image(Image.open(buf).copy())
    return None


def plot_ecg_recon_panels(
    input_ecg: Tensor,
    recon_ecg: Tensor,
    num_examples: int = 4,
    lead_indices: Optional[List[int]] = None,
    max_time: Optional[int] = None,
) -> "wandb.Image":
    """
    Plot input vs reconstruction for a few examples and selected leads.
    input_ecg, recon_ecg: (B, 12, T). Returns wandb.Image.
    """
    if lead_indices is None:
        lead_indices = [0, 1, 6, 10]  # I, II, V1, V5
    max_time = max_time or input_ecg.size(2)
    n = min(num_examples, input_ecg.size(0))
    n_leads = len(lead_indices)
    fig, axes = plt.subplots(n, n_leads * 2, figsize=(2 * n_leads * 2, 2 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    t = torch.arange(max_time, device=input_ecg.device).float()
    for i in range(n):
        for j, lead_idx in enumerate(lead_indices):
            ax_in = axes[i, j * 2]
            ax_in.plot(t.cpu().numpy(), input_ecg[i, lead_idx, :max_time].cpu().numpy(), "b-", linewidth=0.6, label="input")
            ax_in.set_ylabel(_LEAD_NAMES[lead_idx], fontsize=8)
            ax_in.set_ylim(input_ecg[i].min().item() - 0.2, input_ecg[i].max().item() + 0.2)
            ax_in.axis("off")
            ax_re = axes[i, j * 2 + 1]
            ax_re.plot(t.cpu().numpy(), recon_ecg[i, lead_idx, :max_time].cpu().numpy(), "r-", linewidth=0.6, label="recon")
            ax_re.set_ylabel(_LEAD_NAMES[lead_idx], fontsize=8)
            ax_re.set_ylim(recon_ecg[i].min().item() - 0.2, recon_ecg[i].max().item() + 0.2)
            ax_re.axis("off")
    for ax in axes.flat:
        ax.set_xticks([])
    plt.suptitle("Input (blue) vs Reconstruction (red)", fontsize=10)
    plt.tight_layout()
    return _fig_to_wandb_image(fig)


def plot_ecg_samples(samples: Tensor, num_examples: int = 4, lead_indices: Optional[List[int]] = None) -> "wandb.Image":
    """Plot a grid of generated ECG samples. samples: (B, 12, T)."""
    if lead_indices is None:
        lead_indices = [0, 1, 6, 10]
    n = min(num_examples, samples.size(0))
    n_leads = len(lead_indices)
    fig, axes = plt.subplots(n, n_leads, figsize=(2 * n_leads, 2 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    T = samples.size(2)
    t = torch.arange(T, device=samples.device).float().cpu().numpy()
    for i in range(n):
        for j, lead_idx in enumerate(lead_indices):
            axes[i, j].plot(t, samples[i, lead_idx].cpu().numpy(), "g-", linewidth=0.6)
            axes[i, j].set_ylabel(_LEAD_NAMES[lead_idx], fontsize=8)
            axes[i, j].axis("off")
    plt.suptitle("Generated samples (prior)", fontsize=10)
    plt.tight_layout()
    return _fig_to_wandb_image(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  LIGHTNING MODULE
# ──────────────────────────────────────────────────────────────────────────────

class NVAEECGModule(pl.LightningModule):
    """
    Lightning wrapper for NVAE-ECG.

    Training features
    -----------------
    • Linear KL warm-up (β: 0 → 1 over kl_anneal_portion of steps)
    • Optional exponential annealing of spectral-norm weight λ
    • Adamax + linear LR warm-up + cosine decay
    • Logs: loss, recon NLL, KL, per-lead MSE proxy
    • Test: IW-ELBO with configurable num_samples
    """

    def __init__(self,
                 # model
                 num_channels_enc: int   = 32,
                 num_channels_dec: int   = 32,
                 num_preprocess_blocks: int  = 2,
                 num_postprocess_blocks: int = 2,
                 num_preprocess_cells: int   = 2,
                 num_postprocess_cells: int  = 2,
                 num_latent_scales: int  = 3,
                 num_groups_per_scale: int = 8,
                 num_latent_per_group: int = 16,
                 num_cell_per_cond_enc: int = 2,
                 num_cell_per_cond_dec: int = 2,
                 ada_groups: bool        = False,
                 min_groups_per_scale: int = 1,
                 use_se: bool            = True,
                 res_dist: bool          = True,
                 weight_decay_norm: float = 1e-2,
                 # training
                 learning_rate: float   = 1e-3,
                 warmup_epochs: int     = 5,
                 weight_decay: float    = 3e-4,
                 weight_decay_norm_init: float = 10.0,
                 weight_decay_norm_anneal: bool = False,
                 kl_anneal_portion: float = 0.3,
                 kl_const_portion:  float = 0.0001,
                 kl_const_coeff:    float = 0.0001,
                 max_epochs: int        = 200,
                 num_iw_samples: int    = 200,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = NVAE_ECG(
            num_channels_enc=num_channels_enc,
            num_channels_dec=num_channels_dec,
            num_preprocess_blocks=num_preprocess_blocks,
            num_postprocess_blocks=num_postprocess_blocks,
            num_preprocess_cells=num_preprocess_cells,
            num_postprocess_cells=num_postprocess_cells,
            num_latent_scales=num_latent_scales,
            num_groups_per_scale=num_groups_per_scale,
            num_latent_per_group=num_latent_per_group,
            num_cell_per_cond_enc=num_cell_per_cond_enc,
            num_cell_per_cond_dec=num_cell_per_cond_dec,
            ada_groups=ada_groups,
            min_groups_per_scale=min_groups_per_scale,
            use_se=use_se,
            res_dist=res_dist,
            weight_decay_norm=weight_decay_norm,
        )

    # ── KL coefficient ────────────────────────────────────────────────────────

    def _kl_coeff(self) -> float:
        hp = self.hparams
        num_batches = max(getattr(self.trainer, "num_training_batches", 1), 1)
        total  = hp.max_epochs * num_batches
        t0     = int(hp.kl_const_portion  * total)
        t1     = int((hp.kl_const_portion + hp.kl_anneal_portion) * total)
        step   = self.global_step
        if step <= t0:
            return hp.kl_const_coeff
        elif step >= t1:
            return 1.0
        frac = (step - t0) / max(t1 - t0, 1)
        return hp.kl_const_coeff + frac * (1.0 - hp.kl_const_coeff)

    def _weight_decay_norm(self) -> float:
        hp = self.hparams
        if not hp.weight_decay_norm_anneal:
            return hp.weight_decay_norm
        frac    = self.current_epoch / max(hp.max_epochs, 1)
        log_ini = math.log(hp.weight_decay_norm_init)
        log_end = math.log(hp.weight_decay_norm)
        return math.exp(log_ini + frac * (log_end - log_ini))

    # ── steps ─────────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        x = batch["signal"]
        beta = self._kl_coeff()
        lam  = self._weight_decay_norm()
        loss, recon, kl = self.model.elbo(x, beta=beta,
                                          weight_decay_norm=lam)
        self.log_dict({
            "train/loss":  loss,
            "train/recon": recon,
            "train/kl":    kl,
            "train/beta":  beta,
            "train/lam":   lam,
        }, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["signal"]
        loss, recon, kl = self.model.elbo(x, beta=1.0)
        # per-lead MSE (reconstruction quality proxy)
        with torch.no_grad():
            x_hat, _, _ = self.model(x)
            dec = NormalDecoder(x_hat)
            recon_mean = dec.mean
            mse = F.mse_loss(recon_mean, x)
        self.log_dict({
            "val/nelbo": loss,
            "val/recon": recon,
            "val/kl":    kl,
            "val/mse":   mse,
        }, on_epoch=True, prog_bar=True, sync_dist=True)

        # Log example reconstructions and generated samples to wandb (first batch only)
        if (
            batch_idx == 0
            and wandb is not None
            and getattr(self.logger, "experiment", None) is not None
        ):
            recon_img = plot_ecg_recon_panels(x, recon_mean, num_examples=4)
            if recon_img is not None:
                self.logger.experiment.log(
                    {"val/recon_examples": recon_img},
                    step=self.global_step,
                )
            with torch.no_grad():
                samples = self.model.sample(num_samples=4, device=x.device)
            gen_img = plot_ecg_samples(samples, num_examples=4)
            if gen_img is not None:
                self.logger.experiment.log(
                    {"val/generated_examples": gen_img},
                    step=self.global_step,
                )

    def test_step(self, batch, batch_idx):
        x = batch["signal"]
        loss, recon, kl = self.model.elbo(x, beta=1.0)
        log_px = self.model.iwelbo(x, num_samples=self.hparams.num_iw_samples)
        mse = F.mse_loss(NormalDecoder(self.model(x)[0]).mean, x)
        self.log_dict({
            "test/neg_elbo":  loss,
            "test/neg_log_p": -log_px,
            "test/recon":     recon,
            "test/kl":        kl,
            "test/mse":       mse,
        }, on_epoch=True, sync_dist=True)

    # ── optimiser ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        hp  = self.hparams
        opt = torch.optim.Adamax(
            self.parameters(),
            lr=hp.learning_rate,
            weight_decay=hp.weight_decay,
            eps=1e-3,
        )
        warmup = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=lambda e: min(1.0, (e + 1) / max(hp.warmup_epochs, 1)))
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=hp.max_epochs, eta_min=1e-6)
        combined = torch.optim.lr_scheduler.SequentialLR(
            opt, [warmup, cosine], milestones=[hp.warmup_epochs])
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": combined, "interval": "epoch"}}


# ──────────────────────────────────────────────────────────────────────────────
# 6.  PTB-XL  DATASET  &  DATA  MODULE (uses shared data.dataset.PTBXLDataset)
# ──────────────────────────────────────────────────────────────────────────────


class _PTBXLBatchWrapper(Dataset):
    """Wraps shared PTBXLDataset (ecg, features) to return dicts for Lightning."""

    def __init__(self, base: SharedPTBXLDataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        ecg, features = self.base[idx]
        return {"signal": ecg, "features": features}


class PTBXLDataModule(pl.LightningDataModule):
    """
    DataModule for PTB-XL using the shared PTBXLDataset from data.dataset.

    Splits: strat_fold 1–8 train, 9 val, 10 test.
    Batches are dicts with "signal" (B, 12, seq_length) and "features" (B, 9).
    """

    def __init__(
        self,
        ptbxl_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        seq_length: int = ECG_SAMPLES,
        num_leads: int = ECG_LEADS,
        scp_superclass: Optional[str] = None,
        muse_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.ptbxl_path = ptbxl_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seq_length = seq_length
        self.num_leads = num_leads
        self.scp_superclass = scp_superclass
        self.muse_path = muse_path
        self.max_samples = max_samples
        self.seed = seed

    def setup(self, stage: Optional[str] = None):
        common = dict(
            ptbxl_path=self.ptbxl_path,
            seq_length=self.seq_length,
            num_leads=self.num_leads,
            scp_superclass=self.scp_superclass,
            muse_path=self.muse_path,
            max_samples=self.max_samples,
            seed=self.seed,
        )
        self.train_ds = _PTBXLBatchWrapper(
            SharedPTBXLDataset(**common, split="train")
        )
        self.val_ds = _PTBXLBatchWrapper(
            SharedPTBXLDataset(**common, split="val")
        )
        self.test_ds = _PTBXLBatchWrapper(
            SharedPTBXLDataset(**common, split="test")
        )

    def _loader(self, ds: Dataset, shuffle: bool, drop_last: bool = True) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=drop_last,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False, drop_last=False)


class _MIMICBatchWrapper(Dataset):
    """Wraps MIMICIVECGDataset to return dicts for Lightning."""

    def __init__(self, base: MIMICIVECGDataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        ecg, features = self.base[idx]
        return {"signal": ecg, "features": features}


class MIMICDataModule(pl.LightningDataModule):
    """
    DataModule for MIMIC-IV-ECG using MIMICIVECGDataset from data.dataset.

    Uses subject-level splits with 80/10/10 train/val/test.
    """

    def __init__(
        self,
        mimic_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        seq_length: int = ECG_SAMPLES,
        num_leads: int = ECG_LEADS,
        max_samples: Optional[int] = None,
        seed: int = 42,
        val_split: float = 0.1,
        test_split: float = 0.1,
        skip_missing_check: bool = False,
    ):
        super().__init__()
        self.mimic_path = mimic_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seq_length = seq_length
        self.num_leads = num_leads
        self.max_samples = max_samples
        self.seed = seed
        self.val_split = val_split
        self.test_split = test_split
        self.skip_missing_check = skip_missing_check

    def setup(self, stage: Optional[str] = None):
        common = dict(
            mimic_path=self.mimic_path,
            seq_length=self.seq_length,
            num_leads=self.num_leads,
            max_samples=self.max_samples,
            seed=self.seed,
            val_split=self.val_split,
            test_split=self.test_split,
            skip_missing_check=self.skip_missing_check,
        )
        self.train_ds = _MIMICBatchWrapper(
            MIMICIVECGDataset(**common, split="train")
        )
        self.val_ds = _MIMICBatchWrapper(
            MIMICIVECGDataset(**common, split="val")
        )
        self.test_ds = _MIMICBatchWrapper(
            MIMICIVECGDataset(**common, split="test")
        )

    def _loader(self, ds: Dataset, shuffle: bool, drop_last: bool = True) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=drop_last,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False, drop_last=False)


# ──────────────────────────────────────────────────────────────────────────────
# 7.  CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="NVAE-ECG — hierarchical VAE for PTB-XL 12-lead ECG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("mode", choices=["fit", "validate", "test"])

    # ── data: PTB-XL vs MIMIC-IV-ECG ──────────────────────────────────────────
    p.add_argument(
        "--dataset",
        choices=["ptbxl", "mimic"],
        default="ptbxl",
        help="Which ECG dataset to use.",
    )

    # PTB-XL options
    p.add_argument(
        "--ptbxl_dir",
        help="Root directory of PTB-XL download (ptbxl_database.csv here)",
    )
    p.add_argument("--ptbxl_seq_length", type=int, default=ECG_SAMPLES,
                   help="ECG sequence length (time steps)")
    p.add_argument("--ptbxl_num_leads", type=int, default=ECG_LEADS)
    p.add_argument("--ptbxl_scp_superclass", type=str, default=None,
                   help="Optional SCP superclass filter (e.g. HYP)")
    p.add_argument("--ptbxl_muse_path", type=str, default=None,
                   help="Optional path to MUSE reports for conditioning")
    p.add_argument("--ptbxl_max_samples", type=int, default=None,
                   help="Cap samples per split (for debugging)")
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)

    # MIMIC-IV-ECG options
    p.add_argument(
        "--mimic_dir",
        type=str,
        default=None,
        help="Root of MIMIC-IV-ECG (machine_measurements.csv here)",
    )
    p.add_argument(
        "--mimic_max_samples",
        type=int,
        default=None,
        help="Cap MIMIC samples per split (for debugging)",
    )
    p.add_argument(
        "--mimic_skip_missing_check",
        action="store_true",
        default=False,
        help="Skip expensive check that all MIMIC ECG files exist.",
    )

    # ── architecture ──────────────────────────────────────────────────────────
    p.add_argument("--num_channels_enc",  type=int, default=32)
    p.add_argument("--num_channels_dec",  type=int, default=32)
    p.add_argument("--num_preprocess_blocks",  type=int, default=2)
    p.add_argument("--num_postprocess_blocks", type=int, default=2)
    p.add_argument("--num_preprocess_cells",   type=int, default=2)
    p.add_argument("--num_postprocess_cells",  type=int, default=2)
    p.add_argument("--num_latent_scales",  type=int, default=3)
    p.add_argument("--num_groups_per_scale", type=int, default=8)
    p.add_argument("--num_latent_per_group", type=int, default=16)
    p.add_argument("--num_cell_per_cond_enc", type=int, default=2)
    p.add_argument("--num_cell_per_cond_dec", type=int, default=2)
    p.add_argument("--ada_groups",        action="store_true", default=False)
    p.add_argument("--min_groups_per_scale", type=int, default=1)
    p.add_argument("--use_se",            action="store_true", default=False)
    p.add_argument("--res_dist",          action="store_true", default=False)

    # ── regularisation ────────────────────────────────────────────────────────
    p.add_argument("--weight_decay_norm",       type=float, default=1e-2)
    p.add_argument("--weight_decay_norm_init",  type=float, default=10.0)
    p.add_argument("--weight_decay_norm_anneal", action="store_true", default=False)

    # ── optimiser ─────────────────────────────────────────────────────────────
    p.add_argument("--learning_rate",  type=float, default=1e-3)
    p.add_argument("--warmup_epochs",  type=int,   default=5)
    p.add_argument("--weight_decay",   type=float, default=3e-4)

    # ── KL annealing ──────────────────────────────────────────────────────────
    p.add_argument("--kl_anneal_portion", type=float, default=0.3)
    p.add_argument("--kl_const_portion",  type=float, default=0.0001)
    p.add_argument("--kl_const_coeff",    type=float, default=0.0001)

    # ── training ──────────────────────────────────────────────────────────────
    p.add_argument("--max_epochs",  type=int, default=200)
    p.add_argument("--devices",     type=int, default=1)
    p.add_argument("--precision",   default="32",
                   choices=["16-mixed", "bf16-mixed", "32"])
    p.add_argument("--accumulate_grad_batches", type=int, default=1)
    p.add_argument("--gradient_clip_val", type=float, default=200.0,
                   help="Gradient norm clip (important for ECG stability)")

    # ── evaluation ────────────────────────────────────────────────────────────
    p.add_argument("--num_iw_samples", type=int, default=200)

    # ── checkpointing ─────────────────────────────────────────────────────────
    p.add_argument("--output_dir",    default="./outputs")
    p.add_argument("--ckpt_path",     default=None)
    p.add_argument("--cont_training", action="store_true", default=False)

    # ── wandb ─────────────────────────────────────────────────────────────────
    p.add_argument("--wandb_off",       action="store_true", default=False,
                   help="Disable Weights & Biases logging")
    p.add_argument("--wandb_project",   type=str, default="nvae-ecg")
    p.add_argument("--wandb_entity",     type=str, default=None)
    p.add_argument("--wandb_run_name",   type=str, default=None)
    p.add_argument("--wandb_tags",       type=str, nargs="*", default=None,
                   help="Tags for the run (e.g. ptbxl baseline)")

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    # ── data ──────────────────────────────────────────────────────────────────
    if args.dataset == "ptbxl":
        if args.ptbxl_dir is None:
            raise ValueError("--ptbxl_dir must be set when --dataset ptbxl")
        dm = PTBXLDataModule(
            ptbxl_path=args.ptbxl_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seq_length=args.ptbxl_seq_length,
            num_leads=args.ptbxl_num_leads,
            scp_superclass=args.ptbxl_scp_superclass,
            muse_path=args.ptbxl_muse_path,
            max_samples=args.ptbxl_max_samples,
            seed=42,
        )
    else:
        if args.mimic_dir is None:
            raise ValueError("--mimic_dir must be set when --dataset mimic")
        dm = MIMICDataModule(
            mimic_path=args.mimic_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seq_length=args.ptbxl_seq_length,
            num_leads=args.ptbxl_num_leads,
            max_samples=args.mimic_max_samples,
            seed=42,
            skip_missing_check=args.mimic_skip_missing_check,
        )

    # ── model ─────────────────────────────────────────────────────────────────
    model_kwargs = dict(
        num_channels_enc=args.num_channels_enc,
        num_channels_dec=args.num_channels_dec,
        num_preprocess_blocks=args.num_preprocess_blocks,
        num_postprocess_blocks=args.num_postprocess_blocks,
        num_preprocess_cells=args.num_preprocess_cells,
        num_postprocess_cells=args.num_postprocess_cells,
        num_latent_scales=args.num_latent_scales,
        num_groups_per_scale=args.num_groups_per_scale,
        num_latent_per_group=args.num_latent_per_group,
        num_cell_per_cond_enc=args.num_cell_per_cond_enc,
        num_cell_per_cond_dec=args.num_cell_per_cond_dec,
        ada_groups=args.ada_groups,
        min_groups_per_scale=args.min_groups_per_scale,
        use_se=args.use_se,
        res_dist=args.res_dist,
        weight_decay_norm=args.weight_decay_norm,
        learning_rate=args.learning_rate,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        weight_decay_norm_init=args.weight_decay_norm_init,
        weight_decay_norm_anneal=args.weight_decay_norm_anneal,
        kl_anneal_portion=args.kl_anneal_portion,
        kl_const_portion=args.kl_const_portion,
        kl_const_coeff=args.kl_const_coeff,
        max_epochs=args.max_epochs,
        num_iw_samples=args.num_iw_samples,
    )

    if args.mode == "test" and args.ckpt_path:
        module = NVAEECGModule.load_from_checkpoint(
            args.ckpt_path, **model_kwargs)
    else:
        module = NVAEECGModule(**model_kwargs)

    # ── callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            dirpath=args.output_dir,
            filename="nvae-ecg-{epoch:03d}-{val/mse:.5f}",
            monitor="val/mse",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # ── logger (wandb as primary) ────────────────────────────────────────────
    if args.wandb_off or wandb is None:
        logger = True  # Lightning's default CSV/console
    else:
        try:
            logger = WandbLogger(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                tags=args.wandb_tags or [],
                save_dir=args.output_dir,
            )
        except Exception:
            logger = True

    # ── trainer ───────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if args.devices > 0 else "cpu",
        devices=args.devices if args.devices > 0 else None,
        strategy="ddp_find_unused_parameters_false" if args.devices > 1 else "auto",
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=20,
        enable_progress_bar=True,
        detect_anomaly=False,   # set True briefly if debugging NaN
    )

    ckpt = args.ckpt_path if args.cont_training else None

    if args.mode == "fit":
        trainer.fit(module, datamodule=dm, ckpt_path=ckpt)
    elif args.mode == "validate":
        trainer.validate(module, datamodule=dm, ckpt_path=ckpt)
    elif args.mode == "test":
        trainer.test(module, datamodule=dm)


if __name__ == "__main__":
    main()