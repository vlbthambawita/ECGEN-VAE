#!/usr/bin/env python3
"""
Conditional Transformer Prior for VQ-VAE-2 ECG Generation
==========================================================
Stage 2 of the Conditional VQ-VAE-2 pipeline.

Extends the base transformer_prior.py so that BOTH the TopPrior and
BottomPrior are conditioned on the 9 clinical ECG features:

    RR Interval | P Onset | P End | QRS Onset | QRS End | T End |
    P Axis | QRS Axis | T Axis   (normalized)

Conditioning strategy (per Razavi et al. and classifier-free guidance)
-----------------------------------------------------------------------
  * A shared ConditionMLP projects the feature vector to a dense embedding.
  * TopPrior:    condition embedding is prepended as a soft "CLS" token to
                 the input sequence (cross-attention to a single token).
  * BottomPrior: condition embedding is added to the cross-attention context
                 that already contains top codes (simple concatenation along
                 the sequence axis).

This keeps the autoregressive structure intact while injecting condition
information at every transformer layer through cross-attention.

Pipeline
--------
  1. Train Cond-VQVAE2   →  vqvae2_conditional.py fit ...
  2. Extract codes        →  python cond_transformer_prior.py extract ...
  3. Train CondTopPrior   →  python cond_transformer_prior.py fit_top ...
  4. Train CondBotPrior   →  python cond_transformer_prior.py fit_bot ...
  5. Sample conditioned   →  python cond_transformer_prior.py sample  ...
        --rr-interval 0.3421 --p-onset -0.5632 ...

All five steps share this single script.

Sequence lengths (default strides ×8 per level)
------------------------------------------------
  top codes : 5000 // 64  =  78
  bot codes : 5000 //  8  = 625
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

try:
    from pytorch_lightning.loggers import WandbLogger
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    WandbLogger = None
    wandb = None

# Import vqvae2_conditional (which in turn imports vqvae2)
try:
    from vqvae2_conditional import (
        CondVQVAE2Lightning,
        CondVQVAE2Config,
    )
    from vqvae2 import MIMICIVECGDataset
except ImportError:
    raise ImportError(
        "Could not import vqvae2_conditional.py / vqvae2.py. "
        "Place cond_transformer_prior.py in the same directory."
    )


# ============================================================================
# Hyper-parameters
# ============================================================================

class CondPriorHParams:
    # Must match the trained Cond-VQVAE2
    n_embeddings_top: int = 512
    n_embeddings_bot: int = 512
    top_seq_len: int = 78        # 5000 // 64
    bot_seq_len: int = 625       # 5000 //  8
    n_features:  int = 9         # length of condition vector

    # Condition MLP
    cond_hidden: int = 128
    cond_dim:    int = 256       # output dim (matches top d_model)

    # Top Prior Transformer
    top_d_model:  int = 256
    top_n_heads:  int = 8
    top_n_layers: int = 8
    top_d_ff:     int = 1024
    top_dropout:  float = 0.1

    # Bottom Prior Transformer
    bot_d_model:  int = 512
    bot_n_heads:  int = 8
    bot_n_layers: int = 12
    bot_d_ff:     int = 2048
    bot_dropout:  float = 0.1

    # Training
    learning_rate:  float = 3e-4
    warmup_steps:   int   = 2000
    batch_size:     int   = 16
    max_epochs:     int   = 100
    num_workers:    int   = 4
    val_fraction:   float = 0.1
    label_smoothing: float = 0.1

    # Sampling
    top_temperature: float = 1.0
    bot_temperature: float = 1.0
    top_top_k:  int   = 0
    top_top_p:  float = 0.95
    bot_top_k:  int   = 0
    bot_top_p:  float = 0.95


PHP = CondPriorHParams()
BOS = 0   # Beginning-of-sequence token


# ============================================================================
# Shared building blocks
# ============================================================================

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, :x.size(1)]


class CausalTransformerBlock(nn.Module):
    """GPT-style block with optional cross-attention for conditioning."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        cross_attend: bool = False,
        cross_d_model: Optional[int] = None,
    ):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.cross_attend = cross_attend
        if cross_attend:
            kv_dim = cross_d_model or d_model
            self.cross_attn = nn.MultiheadAttention(
                d_model, n_heads, kdim=kv_dim, vdim=kv_dim,
                dropout=dropout, batch_first=True,
            )
            self.norm_cross = nn.LayerNorm(d_model)

    def _causal_mask(self, sz: int, device: torch.device) -> Tensor:
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        L    = x.size(1)
        mask = self._causal_mask(L, x.device)

        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h, attn_mask=mask, is_causal=True)
        x = x + h

        if self.cross_attend and context is not None:
            h = self.norm_cross(x)
            h, _ = self.cross_attn(h, context, context)
            x = x + h

        x = x + self.ff(self.norm2(x))
        return x


# ============================================================================
# Condition MLP  –  shared between top and bottom prior
# ============================================================================

class ConditionMLP(nn.Module):
    """
    Maps the 9-dimensional feature vector to a condition embedding
    that is injected via cross-attention.

    Output shape: (B, 1, cond_dim)   ← single "condition token"
    """

    def __init__(self, n_features: int = 9, hidden: int = 128, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        """
        features: (B, n_features)
        returns : (B, 1, out_dim)  – single condition token for cross-attention
        """
        return self.net(features).unsqueeze(1)   # (B, 1, cond_dim)


# ============================================================================
# Conditional Top Prior
# ============================================================================

class CondTopPrior(nn.Module):
    """
    GPT-style model that learns p(z_top | features).

    Conditioning: the feature condition token (B, 1, cond_dim) is fed into
    every transformer block via cross-attention.
    """

    def __init__(self, hp: CondPriorHParams = PHP):
        super().__init__()
        vocab = hp.n_embeddings_top + 1   # +1 for BOS
        self.embed  = nn.Embedding(vocab, hp.top_d_model)
        self.pe     = SinusoidalPE(hp.top_d_model, max_len=hp.top_seq_len + 1)
        self.drop   = nn.Dropout(hp.top_dropout)

        # Project condition token to top d_model if dimensions differ
        cond_dim = getattr(hp, 'cond_dim', hp.top_d_model)
        self.cond_proj = (
            nn.Linear(cond_dim, hp.top_d_model)
            if cond_dim != hp.top_d_model else nn.Identity()
        )

        self.blocks = nn.ModuleList([
            CausalTransformerBlock(
                hp.top_d_model, hp.top_n_heads, hp.top_d_ff, hp.top_dropout,
                cross_attend=True, cross_d_model=hp.top_d_model,
            )
            for _ in range(hp.top_n_layers)
        ])
        self.norm = nn.LayerNorm(hp.top_d_model)
        self.head = nn.Linear(hp.top_d_model, hp.n_embeddings_top, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, codes: Tensor, cond_token: Tensor) -> Tensor:
        """
        codes     : (B, L_top)   integer top codes (0-indexed, no BOS)
        cond_token: (B, 1, D)    condition token from ConditionMLP
        returns logits (B, L_top, K_top)
        """
        B, L = codes.shape
        bos  = torch.full((B, 1), BOS, dtype=torch.long, device=codes.device)
        x    = torch.cat([bos, codes[:, :-1] + 1], dim=1)   # shift right, +1 for BOS offset
        x    = self.drop(self.pe(self.embed(x)))
        ctx  = self.cond_proj(cond_token)                    # (B, 1, top_d_model)
        for block in self.blocks:
            x = block(x, context=ctx)
        return self.head(self.norm(x))                       # (B, L, K_top)

    @torch.no_grad()
    def sample(
        self,
        cond_token: Tensor,
        seq_len: int,
        hp: CondPriorHParams,
        device: torch.device,
    ) -> Tensor:
        """
        Autoregressively sample top codes conditioned on cond_token.
        cond_token: (B, 1, D)
        Returns  : (B, seq_len) int64
        """
        B      = cond_token.size(0)
        ctx    = self.cond_proj(cond_token.to(device))
        tokens = torch.full((B, 1), BOS, dtype=torch.long, device=device)

        for _ in range(seq_len):
            x    = self.drop(self.pe(self.embed(tokens)))
            for block in self.blocks:
                x = block(x, context=ctx)
            logit = self.head(self.norm(x))[:, -1, :]   # (B, K)
            logit = logit / hp.top_temperature
            logit = _top_p_filter(logit, hp.top_top_p)
            if hp.top_top_k > 0:
                logit = _top_k_filter(logit, hp.top_top_k)
            next_tok = torch.multinomial(F.softmax(logit, -1), 1)
            tokens   = torch.cat([tokens, next_tok + 1], dim=1)

        return tokens[:, 1:] - 1   # strip BOS, un-shift


# ============================================================================
# Conditional Bottom Prior
# ============================================================================

class CondBottomPrior(nn.Module):
    """
    Conditioned autoregressive model for bottom codes given:
      - top codes  z_top  (autoregressive context, as in base prior)
      - feature condition token  (clinical features)

    Both condition signals are concatenated along the sequence axis and
    fed as a unified cross-attention context.
    """

    def __init__(self, hp: CondPriorHParams = PHP):
        super().__init__()
        vocab_bot = hp.n_embeddings_bot + 1
        vocab_top = hp.n_embeddings_top + 1

        self.embed_bot = nn.Embedding(vocab_bot, hp.bot_d_model)
        self.embed_top = nn.Embedding(vocab_top, hp.bot_d_model)
        self.pe_bot    = SinusoidalPE(hp.bot_d_model, max_len=hp.bot_seq_len + 1)
        self.pe_top    = SinusoidalPE(hp.bot_d_model, max_len=hp.top_seq_len + 1)
        self.drop      = nn.Dropout(hp.bot_dropout)

        # Project condition token to bot d_model
        cond_dim = getattr(hp, 'cond_dim', hp.top_d_model)
        self.cond_proj = (
            nn.Linear(cond_dim, hp.bot_d_model)
            if cond_dim != hp.bot_d_model else nn.Identity()
        )

        self.blocks = nn.ModuleList([
            CausalTransformerBlock(
                hp.bot_d_model, hp.bot_n_heads, hp.bot_d_ff, hp.bot_dropout,
                cross_attend=True, cross_d_model=hp.bot_d_model,
            )
            for _ in range(hp.bot_n_layers)
        ])
        self.norm = nn.LayerNorm(hp.bot_d_model)
        self.head = nn.Linear(hp.bot_d_model, hp.n_embeddings_bot, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _encode_top(self, top_codes: Tensor) -> Tensor:
        """top_codes: (B, L_top) → (B, L_top, d_model)"""
        return self.drop(self.pe_top(self.embed_top(top_codes + 1)))

    def forward(
        self,
        bot_codes: Tensor,
        top_codes: Tensor,
        cond_token: Tensor,
    ) -> Tensor:
        """
        bot_codes : (B, L_bot)
        top_codes : (B, L_top)
        cond_token: (B, 1, D)   condition token
        returns logits (B, L_bot, K_bot)
        """
        B, L = bot_codes.shape
        bos  = torch.full((B, 1), BOS, dtype=torch.long, device=bot_codes.device)
        x    = self.drop(self.pe_bot(self.embed_bot(torch.cat([bos, bot_codes[:, :-1] + 1], dim=1))))

        # Build context: [top_tokens | cond_token]
        top_ctx  = self._encode_top(top_codes)                  # (B, L_top, d)
        cond_ctx = self.cond_proj(cond_token)                   # (B, 1, d)
        ctx      = torch.cat([top_ctx, cond_ctx], dim=1)        # (B, L_top+1, d)

        for block in self.blocks:
            x = block(x, context=ctx)
        return self.head(self.norm(x))                          # (B, L_bot, K_bot)

    @torch.no_grad()
    def sample(
        self,
        top_codes: Tensor,
        cond_token: Tensor,
        seq_len: int,
        hp: CondPriorHParams,
        device: torch.device,
    ) -> Tensor:
        """
        Sample bottom codes conditioned on top codes + feature token.
        Returns (B, seq_len).
        """
        B       = top_codes.size(0)
        top_ctx = self._encode_top(top_codes.to(device))
        c_ctx   = self.cond_proj(cond_token.to(device))
        ctx     = torch.cat([top_ctx, c_ctx], dim=1)

        tokens = torch.full((B, 1), BOS, dtype=torch.long, device=device)
        for _ in range(seq_len):
            x = self.drop(self.pe_bot(self.embed_bot(tokens)))
            for block in self.blocks:
                x = block(x, context=ctx)
            logit    = self.head(self.norm(x))[:, -1, :]
            logit    = logit / hp.bot_temperature
            logit    = _top_p_filter(logit, hp.bot_top_p)
            if hp.bot_top_k > 0:
                logit = _top_k_filter(logit, hp.bot_top_k)
            next_tok = torch.multinomial(F.softmax(logit, -1), 1)
            tokens   = torch.cat([tokens, next_tok + 1], dim=1)

        return tokens[:, 1:] - 1


# ============================================================================
# Sampling helpers
# ============================================================================

def _top_k_filter(logits: Tensor, k: int) -> Tensor:
    vals, _ = torch.topk(logits, k)
    return logits.masked_fill(logits < vals[:, -1:], float('-inf'))


def _top_p_filter(logits: Tensor, p: float) -> Tensor:
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cumprobs = torch.cumsum(F.softmax(sorted_logits, -1), -1)
    remove   = (cumprobs - F.softmax(sorted_logits, -1)) > p
    sorted_logits[remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_idx, sorted_logits)


# ============================================================================
# Lightning modules
# ============================================================================

class CondTopPriorLightning(pl.LightningModule):

    def __init__(self, hp: CondPriorHParams = PHP):
        super().__init__()
        self.hp = hp
        self.save_hyperparameters(vars(hp))
        self.cond_mlp = ConditionMLP(
            n_features=hp.n_features,
            hidden=hp.cond_hidden,
            out_dim=getattr(hp, 'cond_dim', hp.top_d_model),
        )
        self.model     = CondTopPrior(hp)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=hp.label_smoothing)

    def _step(self, batch, stage: str):
        top_codes, _, features = batch              # (B, L_top), (B, L_bot), (B, 9)
        cond_token = self.cond_mlp(features)        # (B, 1, D)
        logits     = self.model(top_codes, cond_token)
        B, L, K    = logits.shape
        loss = self.criterion(logits.reshape(B * L, K), top_codes.reshape(B * L))
        acc  = (logits.argmax(-1) == top_codes).float().mean()
        self.log(f"{stage}/loss", loss, prog_bar=True,  on_epoch=True, on_step=False)
        self.log(f"{stage}/acc",  acc,  prog_bar=True,  on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, _):   return self._step(batch, "train")
    def validation_step(self, batch, _): self._step(batch, "val")
    def test_step(self, batch, _):       self._step(batch, "test")

    def configure_optimizers(self):
        opt   = torch.optim.AdamW(self.parameters(), lr=self.hp.learning_rate,
                                  betas=(0.9, 0.98), weight_decay=1e-2)
        sched = _cosine_with_warmup(opt, self.hp.warmup_steps, self.hp.max_epochs * 1000)
        return [opt], [{"scheduler": sched, "interval": "step"}]


class CondBottomPriorLightning(pl.LightningModule):

    def __init__(self, hp: CondPriorHParams = PHP):
        super().__init__()
        self.hp = hp
        self.save_hyperparameters(vars(hp))
        self.cond_mlp = ConditionMLP(
            n_features=hp.n_features,
            hidden=hp.cond_hidden,
            out_dim=getattr(hp, 'cond_dim', hp.top_d_model),
        )
        self.model     = CondBottomPrior(hp)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=hp.label_smoothing)

    def _step(self, batch, stage: str):
        top_codes, bot_codes, features = batch
        cond_token = self.cond_mlp(features)
        logits     = self.model(bot_codes, top_codes, cond_token)
        B, L, K    = logits.shape
        loss = self.criterion(logits.reshape(B * L, K), bot_codes.reshape(B * L))
        acc  = (logits.argmax(-1) == bot_codes).float().mean()
        self.log(f"{stage}/loss", loss, prog_bar=True,  on_epoch=True, on_step=False)
        self.log(f"{stage}/acc",  acc,  prog_bar=True,  on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, _):   return self._step(batch, "train")
    def validation_step(self, batch, _): self._step(batch, "val")
    def test_step(self, batch, _):       self._step(batch, "test")

    def configure_optimizers(self):
        opt   = torch.optim.AdamW(self.parameters(), lr=self.hp.learning_rate,
                                  betas=(0.9, 0.98), weight_decay=1e-2)
        sched = _cosine_with_warmup(opt, self.hp.warmup_steps, self.hp.max_epochs * 1000)
        return [opt], [{"scheduler": sched, "interval": "step"}]


def _cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# Code + Feature Dataset
# ============================================================================

class CondCodeDataset(Dataset):
    """
    Loads pre-extracted codes AND the corresponding normalized features.
    Expects:
        codes_top.npy      (N, L_top)  int16/int32
        codes_bot.npy      (N, L_bot)  int16/int32
        features.npy       (N, 9)      float32  – normalized clinical features
    """

    def __init__(self, codes_dir: str):
        codes_dir = Path(codes_dir)
        top   = np.load(codes_dir / "codes_top.npy").astype(np.int64)
        bot   = np.load(codes_dir / "codes_bot.npy").astype(np.int64)
        feats = np.load(codes_dir / "features.npy").astype(np.float32)
        assert len(top) == len(bot) == len(feats), (
            f"Shape mismatch: top {len(top)}, bot {len(bot)}, feats {len(feats)}"
        )
        self.top   = torch.from_numpy(top)
        self.bot   = torch.from_numpy(bot)
        self.feats = torch.from_numpy(feats)

    def __len__(self):
        return len(self.top)

    def __getitem__(self, idx):
        return self.top[idx], self.bot[idx], self.feats[idx]


class CondCodeDataModule(pl.LightningDataModule):

    def __init__(self, codes_dir: str, hp: CondPriorHParams = PHP):
        super().__init__()
        self.codes_dir = codes_dir
        self.hp = hp

    def setup(self, stage: Optional[str] = None):
        full  = CondCodeDataset(self.codes_dir)
        n_val = max(1, int(len(full) * self.hp.val_fraction))
        self.train_ds, self.val_ds = random_split(
            full, [len(full) - n_val, n_val],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hp.batch_size,
                          shuffle=True, num_workers=self.hp.num_workers,
                          pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hp.batch_size,
                          shuffle=False, num_workers=self.hp.num_workers, pin_memory=True)

    def test_dataloader(self):
        return self.val_dataloader()


# ============================================================================
# Code extraction  (saves codes + features)
# ============================================================================

def extract_codes(
    vqvae_ckpt: str,
    data_dir: str,
    out_dir: str,
    batch_size: int = 32,
    max_samples: Optional[int] = None,
):
    """
    Encode all ECG samples through the Conditional VQ-VAE-2 and save:
        codes_top.npy   (N, L_top)
        codes_bot.npy   (N, L_bot)
        features.npy    (N, 9)        ← normalized clinical features
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Cond-VQVAE2 from: {vqvae_ckpt}")
    vqvae_pl = CondVQVAE2Lightning.load_from_checkpoint(vqvae_ckpt, map_location=device)
    vqvae_pl.eval()
    vqvae = vqvae_pl.model.to(device)

    FEATURE_NAMES = [
        "rr_interval", "p_onset", "p_end", "qrs_onset", "qrs_end",
        "t_end", "p_axis", "qrs_axis", "t_axis",
    ]

    dataset = MIMICIVECGDataset(
        mimic_path=data_dir,
        split="train",
        val_split=0.1,
        test_split=0.1,
        max_samples=max_samples,
        seed=42,
        skip_missing_check=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    all_top, all_bot, all_feats = [], [], []
    print(f"Extracting codes + features from {len(dataset)} samples...")

    with torch.no_grad():
        for i, (x, features) in enumerate(loader):
            x        = x.to(device)
            features = features.to(device)

            cond_emb, film_gamma, film_beta = vqvae.cond_enc(features)
            _, _, _, _, codes_bot, codes_top = vqvae.encode(x, cond_emb, film_gamma, film_beta)

            all_top.append(codes_top.cpu().short())
            all_bot.append(codes_bot.cpu().short())
            all_feats.append(features.cpu().float())

            if (i + 1) % 20 == 0:
                print(f"  batch {i+1}/{len(loader)}")

    codes_top_np = torch.cat(all_top,   0).numpy()
    codes_bot_np = torch.cat(all_bot,   0).numpy()
    features_np  = torch.cat(all_feats, 0).numpy()

    np.save(out_dir / "codes_top.npy", codes_top_np)
    np.save(out_dir / "codes_bot.npy", codes_bot_np)
    np.save(out_dir / "features.npy",  features_np)

    print(f"Saved codes_top {codes_top_np.shape}, codes_bot {codes_bot_np.shape}, "
          f"features {features_np.shape} → {out_dir}")


# ============================================================================
# End-to-end conditional generation
# ============================================================================

@torch.no_grad()
def generate_ecgs(
    vqvae_ckpt:     str,
    top_prior_ckpt: str,
    bot_prior_ckpt: str,
    features_raw:   np.ndarray,          # shape (9,)  normalized
    n_samples:      int,
    out_path:       str,
    hp:             CondPriorHParams = PHP,
    plot:           bool = False,
):
    """
    Full conditional generation pipeline:
        1. Encode condition features → condition token
        2. Sample top codes  p(z_top | features)
        3. Sample bot codes  p(z_bot | z_top, features)
        4. Decode codes → ECG signals
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load models ----
    print(f"Loading Cond-VQVAE2         : {vqvae_ckpt}")
    vqvae_pl = CondVQVAE2Lightning.load_from_checkpoint(vqvae_ckpt, map_location=device)
    vqvae_pl.eval()
    vqvae = vqvae_pl.model.to(device)

    print(f"Loading CondTopPrior        : {top_prior_ckpt}")
    top_pl = CondTopPriorLightning.load_from_checkpoint(top_prior_ckpt, hp=hp, map_location=device)
    top_pl.eval()
    top_cond_mlp = top_pl.cond_mlp.to(device)
    top_prior    = top_pl.model.to(device)

    print(f"Loading CondBottomPrior     : {bot_prior_ckpt}")
    bot_pl = CondBottomPriorLightning.load_from_checkpoint(bot_prior_ckpt, hp=hp, map_location=device)
    bot_pl.eval()
    bot_cond_mlp = bot_pl.cond_mlp.to(device)
    bot_prior    = bot_pl.model.to(device)

    # ---- Prepare condition ----
    feat_tensor = torch.from_numpy(features_raw.astype(np.float32)).to(device)
    feat_batch  = feat_tensor.unsqueeze(0).expand(n_samples, -1)   # (B, 9)

    top_cond_token = top_cond_mlp(feat_batch)   # (B, 1, D)
    bot_cond_token = bot_cond_mlp(feat_batch)   # (B, 1, D)

    # ---- Sample codes ----
    print(f"Sampling {n_samples} top codes (seq_len={hp.top_seq_len})...")
    codes_top = top_prior.sample(top_cond_token, hp.top_seq_len, hp, device)   # (B, L_top)

    print(f"Sampling {n_samples} bottom codes (seq_len={hp.bot_seq_len})...")
    codes_bot = bot_prior.sample(codes_top, bot_cond_token, hp.bot_seq_len, hp, device)  # (B, L_bot)

    # ---- Decode ----
    print("Decoding to ECG signals...")
    ecgs = vqvae.decode_codes(codes_bot, codes_top, feat_batch).cpu().numpy()  # (B, 12, 5000)

    np.save(out_path, ecgs)
    print(f"Saved {n_samples} ECGs → {out_path}  shape: {ecgs.shape}")

    if plot:
        _plot_generated(ecgs, out_path, features_raw)


def _plot_generated(ecgs: np.ndarray, out_path: str, features_raw: np.ndarray):
    FEATURE_NAMES = [
        "RR Interval", "P Onset",  "P End",
        "QRS Onset",   "QRS End",  "T End",
        "P Axis",      "QRS Axis", "T Axis",
    ]
    n_show  = min(4, ecgs.shape[0])
    n_leads = min(6, ecgs.shape[1])

    fig, axs = plt.subplots(n_show, n_leads, figsize=(3 * n_leads, 2 * n_show),
                            sharex=True)
    if n_show == 1:
        axs = axs.reshape(1, -1)

    colors = plt.cm.tab10.colors
    for si in range(n_show):
        for li in range(n_leads):
            ax = axs[si, li]
            ax.plot(ecgs[si, li], linewidth=0.6, color=colors[li % len(colors)])
            if si == 0:
                ax.set_title(f"Lead {li}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        axs[si, 0].set_ylabel(f"Sample {si+1}", fontsize=8, rotation=0, labelpad=40)

    cond_lines = "  |  ".join(f"{n}: {v:+.4f}" for n, v in zip(FEATURE_NAMES[:5], features_raw[:5]))
    cond_lines += "\n" + "  |  ".join(f"{n}: {v:+.4f}" for n, v in zip(FEATURE_NAMES[5:], features_raw[5:]))
    plt.suptitle(f"Conditional ECG Generation\n{cond_lines}", fontsize=9)
    plt.tight_layout()

    png_path = Path(out_path).with_suffix(".png")
    plt.savefig(png_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization → {png_path}")


@torch.no_grad()
def _generate_k_ecgs(
    vqvae,
    top_prior,
    bot_prior,
    top_cond_mlp,
    bot_cond_mlp,
    features_raw: np.ndarray,
    k: int,
    hp: CondPriorHParams,
    device: torch.device,
) -> np.ndarray:
    """Generate K ECGs conditioned on features_raw. Returns (K, 12, 5000)."""
    feat_tensor = torch.from_numpy(features_raw.astype(np.float32)).to(device)
    feat_batch  = feat_tensor.unsqueeze(0).expand(k, -1)
    top_cond_token = top_cond_mlp(feat_batch)
    bot_cond_token = bot_cond_mlp(feat_batch)
    codes_top = top_prior.sample(top_cond_token, hp.top_seq_len, hp, device)
    codes_bot = bot_prior.sample(codes_top, bot_cond_token, hp.bot_seq_len, hp, device)
    ecgs = vqvae.decode_codes(codes_bot, codes_top, feat_batch).cpu().numpy()
    return ecgs


def _plot_comparison(
    ecg_real: np.ndarray,
    ecgs_gen: np.ndarray,
    features_raw: np.ndarray,
    out_path: str,
):
    """Plot real vs generated: 1 real row + K generated rows x 6 leads."""
    FEATURE_NAMES = [
        "RR Interval", "P Onset",  "P End",
        "QRS Onset",   "QRS End",  "T End",
        "P Axis",      "QRS Axis", "T Axis",
    ]
    n_leads = min(6, ecg_real.shape[0])
    n_rows  = 1 + ecgs_gen.shape[0]

    fig, axs = plt.subplots(n_rows, n_leads, figsize=(3 * n_leads, 2 * n_rows), sharex=True)
    if n_rows == 1:
        axs = axs.reshape(1, -1)

    colors = plt.cm.tab10.colors
    # Row 0: Real
    for li in range(n_leads):
        ax = axs[0, li]
        ax.plot(ecg_real[li], linewidth=0.6, color="C0")
        if li == 0:
            ax.set_ylabel("Real", fontsize=8, rotation=0, labelpad=30)
        if li == 0:
            ax.set_title(f"Lead {li}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    # Rows 1..K: Generated
    for si in range(ecgs_gen.shape[0]):
        for li in range(n_leads):
            ax = axs[si + 1, li]
            ax.plot(ecgs_gen[si, li], linewidth=0.6, color=colors[(si + 1) % len(colors)])
            if li == 0:
                ax.set_ylabel(f"Gen {si+1}", fontsize=8, rotation=0, labelpad=30)
            ax.set_xticks([])
            ax.set_yticks([])

    cond_lines = "  |  ".join(f"{n}: {v:+.4f}" for n, v in zip(FEATURE_NAMES[:5], features_raw[:5]))
    cond_lines += "\n" + "  |  ".join(f"{n}: {v:+.4f}" for n, v in zip(FEATURE_NAMES[5:], features_raw[5:]))
    plt.suptitle(f"Real vs Generated ECG\n{cond_lines}", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison → {out_path}")


@torch.no_grad()
def test_sample_from_data(
    data_dir: str,
    vqvae_ckpt: str,
    top_prior_ckpt: str,
    bot_prior_ckpt: str,
    n_test: int = 4,
    k_per_feature: int = 4,
    out_dir: str = "test_comparisons",
    hp: CondPriorHParams = PHP,
    seed: int = 42,
    val_split: float = 0.1,
    test_split: float = 0.1,
    skip_missing_check: bool = False,
):
    """
    Load N test ECGs, extract features, generate K samples per feature set,
    and save comparison plots (real vs generated).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ---- Load models ----
    print(f"Loading Cond-VQVAE2         : {vqvae_ckpt}")
    vqvae_pl = CondVQVAE2Lightning.load_from_checkpoint(vqvae_ckpt, map_location=device)
    vqvae_pl.eval()
    vqvae = vqvae_pl.model.to(device)

    print(f"Loading CondTopPrior        : {top_prior_ckpt}")
    top_pl = CondTopPriorLightning.load_from_checkpoint(top_prior_ckpt, hp=hp, map_location=device)
    top_pl.eval()
    top_cond_mlp = top_pl.cond_mlp.to(device)
    top_prior    = top_pl.model.to(device)

    print(f"Loading CondBottomPrior     : {bot_prior_ckpt}")
    bot_pl = CondBottomPriorLightning.load_from_checkpoint(bot_prior_ckpt, hp=hp, map_location=device)
    bot_pl.eval()
    bot_cond_mlp = bot_pl.cond_mlp.to(device)
    bot_prior    = bot_pl.model.to(device)

    # ---- Load test dataset ----
    dataset = MIMICIVECGDataset(
        mimic_path=data_dir,
        split="test",
        val_split=val_split,
        test_split=test_split,
        max_samples=None,
        seed=seed,
        skip_missing_check=skip_missing_check,
    )
    n_test = min(n_test, len(dataset))
    print(f"Loaded {len(dataset)} test samples, using first {n_test}")

    all_real = []
    all_gen  = []

    for i in range(n_test):
        ecg_real, features = dataset[i]
        ecg_real_np = ecg_real.numpy()
        features_np = features.numpy().astype(np.float32)

        print(f"  Test {i+1}/{n_test}: generating {k_per_feature} samples...")
        ecgs_gen = _generate_k_ecgs(
            vqvae, top_prior, bot_prior,
            top_cond_mlp, bot_cond_mlp,
            features_np, k_per_feature, hp, device,
        )

        all_real.append(ecg_real_np)
        all_gen.append(ecgs_gen)

        png_path = out_path / f"test_comparison_{i:03d}.png"
        _plot_comparison(ecg_real_np, ecgs_gen, features_np, str(png_path))

    all_real_np = np.stack(all_real, axis=0)
    all_gen_np  = np.stack(all_gen,  axis=0)
    np.save(out_path / "real_ecgs.npy",      all_real_np)
    np.save(out_path / "generated_ecgs.npy", all_gen_np)
    print(f"Saved real_ecgs.npy {all_real_np.shape}, generated_ecgs.npy {all_gen_np.shape} → {out_path}")


# ============================================================================
# Trainer factory
# ============================================================================

def build_trainer(args, name: str) -> pl.Trainer:
    callbacks = [
        ModelCheckpoint(
            monitor="val/loss", mode="min", save_top_k=3,
            filename=f"{name}-{{epoch:03d}}-{{val/loss:.4f}}",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    loggers = [CSVLogger("logs", name=name)]
    if hasattr(args, 'wandb') and args.wandb and WANDB_AVAILABLE:
        loggers.append(WandbLogger(
            project=getattr(args, 'wandb_project', 'cond-vqvae2-prior'),
            entity=getattr(args, 'wandb_entity', None),
            name=f"{name}_{args.wandb_run_name}" if getattr(args, 'wandb_run_name', None) else name,
            save_dir="logs",
        ))
        print("✓ W&B logging enabled")
    return pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=args.gpus if args.gpus else "auto",
        callbacks=callbacks,
        logger=loggers if len(loggers) > 1 else loggers[0],
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Conditional Transformer Prior for Cond-VQVAE2 ECG Generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # ------------------------------------------------------------------ extract
    ep = sub.add_parser("extract", help="Extract codes + features from Cond-VQVAE2")
    ep.add_argument("--vqvae-ckpt",   required=True, help="Path to Cond-VQVAE2 checkpoint")
    ep.add_argument("--data-dir",     required=True, help="MIMIC-IV-ECG data directory")
    ep.add_argument("--out-dir",      required=True, help="Output directory for .npy files")
    ep.add_argument("--batch-size",   type=int, default=32)
    ep.add_argument("--max-samples",  type=int, default=None)

    # ------------------------------------------------------------------ fit_top
    tp = sub.add_parser("fit_top", help="Train the Conditional Top Prior")
    tp.add_argument("--codes-dir",    required=True)
    tp.add_argument("--max-epochs",   type=int,   default=PHP.max_epochs)
    tp.add_argument("--batch-size",   type=int,   default=PHP.batch_size)
    tp.add_argument("--lr",           type=float, default=PHP.learning_rate)
    tp.add_argument("--gpus",         type=int,   default=None)
    tp.add_argument("--d-model",      type=int,   default=PHP.top_d_model)
    tp.add_argument("--n-layers",     type=int,   default=PHP.top_n_layers)
    tp.add_argument("--n-heads",      type=int,   default=PHP.top_n_heads)
    tp.add_argument("--cond-dim",     type=int,   default=PHP.cond_hidden)
    tp.add_argument("--wandb",            action="store_true")
    tp.add_argument("--wandb-project",    type=str, default="cond-vqvae2-prior")
    tp.add_argument("--wandb-entity",     type=str, default=None)
    tp.add_argument("--wandb-run-name",   type=str, default=None)

    # ------------------------------------------------------------------ fit_bot
    bp = sub.add_parser("fit_bot", help="Train the Conditional Bottom Prior")
    bp.add_argument("--codes-dir",    required=True)
    bp.add_argument("--max-epochs",   type=int,   default=PHP.max_epochs)
    bp.add_argument("--batch-size",   type=int,   default=PHP.batch_size)
    bp.add_argument("--lr",           type=float, default=PHP.learning_rate)
    bp.add_argument("--gpus",         type=int,   default=None)
    bp.add_argument("--d-model",      type=int,   default=PHP.bot_d_model)
    bp.add_argument("--n-layers",     type=int,   default=PHP.bot_n_layers)
    bp.add_argument("--n-heads",      type=int,   default=PHP.bot_n_heads)
    bp.add_argument("--cond-dim",     type=int,   default=PHP.cond_hidden)
    bp.add_argument("--wandb",            action="store_true")
    bp.add_argument("--wandb-project",    type=str, default="cond-vqvae2-prior")
    bp.add_argument("--wandb-entity",     type=str, default=None)
    bp.add_argument("--wandb-run-name",   type=str, default=None)
    bp.add_argument("--resume",           type=str, default=None,
                    help="Path to checkpoint to resume from (e.g. logs/cond_bot_prior/version_0/checkpoints/last.ckpt)")

    # ------------------------------------------------------------------ sample
    sp = sub.add_parser("sample", help="Generate ECGs conditioned on clinical features")
    sp.add_argument("--vqvae-ckpt",     required=True)
    sp.add_argument("--top-prior-ckpt", required=True)
    sp.add_argument("--bot-prior-ckpt", required=True)
    sp.add_argument("--n-samples",   type=int,   default=8)
    sp.add_argument("--out",         type=str,   default="cond_generated_ecgs.npy")
    sp.add_argument("--top-temp",    type=float, default=PHP.top_temperature)
    sp.add_argument("--bot-temp",    type=float, default=PHP.bot_temperature)
    sp.add_argument("--top-p",       type=float, default=PHP.top_top_p)
    sp.add_argument("--cond-dim",    type=int,   default=128,
                    help="Condition MLP output dim (must match the prior checkpoint; default 128)")
    sp.add_argument("--plot",        action="store_true",
                    help="Save a PNG grid of generated ECGs")

    # Clinical feature inputs (pre-normalized values)
    sp.add_argument("--rr-interval", type=float, required=True, metavar="NORM_VAL")
    sp.add_argument("--p-onset",     type=float, required=True, metavar="NORM_VAL")
    sp.add_argument("--p-end",       type=float, required=True, metavar="NORM_VAL")
    sp.add_argument("--qrs-onset",   type=float, required=True, metavar="NORM_VAL")
    sp.add_argument("--qrs-end",     type=float, required=True, metavar="NORM_VAL")
    sp.add_argument("--t-end",       type=float, required=True, metavar="NORM_VAL")
    sp.add_argument("--p-axis",      type=float, required=True, metavar="NORM_VAL")
    sp.add_argument("--qrs-axis",    type=float, required=True, metavar="NORM_VAL")
    sp.add_argument("--t-axis",      type=float, required=True, metavar="NORM_VAL")

    # ------------------------------------------------------------------ test_sample
    tsp = sub.add_parser("test_sample", help="Test: load N test ECGs, generate K per feature set, compare visually")
    tsp.add_argument("--data-dir",        required=True)
    tsp.add_argument("--vqvae-ckpt",      required=True)
    tsp.add_argument("--top-prior-ckpt",  required=True)
    tsp.add_argument("--bot-prior-ckpt",  required=True)
    tsp.add_argument("--n-test",       type=int, default=4, help="Number of test samples")
    tsp.add_argument("--k-per-feature", type=int, default=4, help="Generated samples per feature set")
    tsp.add_argument("--out-dir",      type=str, default="test_comparisons")
    tsp.add_argument("--cond-dim",     type=int, default=128)
    tsp.add_argument("--top-temp",     type=float, default=PHP.top_temperature)
    tsp.add_argument("--bot-temp",     type=float, default=PHP.bot_temperature)
    tsp.add_argument("--top-p",        type=float, default=PHP.top_top_p)
    tsp.add_argument("--seed",         type=int, default=42)
    tsp.add_argument("--val-split",    type=float, default=0.1)
    tsp.add_argument("--test-split",   type=float, default=0.1)
    tsp.add_argument("--skip-missing-check", action="store_true")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    # ------------------------------------------------------------------ dispatch
    if args.command == "extract":
        extract_codes(args.vqvae_ckpt, args.data_dir, args.out_dir,
                      args.batch_size, args.max_samples)

    elif args.command == "fit_top":
        hp = CondPriorHParams()
        hp.learning_rate = args.lr
        hp.batch_size    = args.batch_size
        hp.max_epochs    = args.max_epochs
        hp.top_d_model   = args.d_model
        hp.top_n_layers  = args.n_layers
        hp.top_n_heads   = args.n_heads
        hp.top_d_ff      = args.d_model * 4
        hp.cond_dim      = args.cond_dim
        model = CondTopPriorLightning(hp)
        dm    = CondCodeDataModule(args.codes_dir, hp)
        build_trainer(args, "cond_top_prior").fit(model, dm)

    elif args.command == "fit_bot":
        hp = CondPriorHParams()
        hp.learning_rate = args.lr
        hp.batch_size    = args.batch_size
        hp.max_epochs    = args.max_epochs
        hp.bot_d_model   = args.d_model
        hp.bot_n_layers  = args.n_layers
        hp.bot_n_heads   = args.n_heads
        hp.bot_d_ff      = args.d_model * 4
        hp.cond_dim      = args.cond_dim
        model = CondBottomPriorLightning(hp)
        dm    = CondCodeDataModule(args.codes_dir, hp)
        ckpt_path = getattr(args, 'resume', None)
        build_trainer(args, "cond_bot_prior").fit(model, dm, ckpt_path=ckpt_path)

    elif args.command == "sample":
        hp = CondPriorHParams()
        hp.cond_dim        = args.cond_dim
        hp.top_temperature = args.top_temp
        hp.bot_temperature = args.bot_temp
        hp.top_top_p       = args.top_p
        hp.bot_top_p       = args.top_p

        features_raw = np.array([
            args.rr_interval, args.p_onset, args.p_end,
            args.qrs_onset,   args.qrs_end, args.t_end,
            args.p_axis,      args.qrs_axis, args.t_axis,
        ], dtype=np.float32)

        FEATURE_NAMES = [
            "RR Interval", "P Onset",  "P End",
            "QRS Onset",   "QRS End",  "T End",
            "P Axis",      "QRS Axis", "T Axis",
        ]
        print("\nCondition (normalized features):")
        for name, val in zip(FEATURE_NAMES, features_raw):
            print(f"  {name:<15}: {val:>+8.4f}")
        print()

        generate_ecgs(
            vqvae_ckpt=args.vqvae_ckpt,
            top_prior_ckpt=args.top_prior_ckpt,
            bot_prior_ckpt=args.bot_prior_ckpt,
            features_raw=features_raw,
            n_samples=args.n_samples,
            out_path=args.out,
            hp=hp,
            plot=args.plot,
        )

    elif args.command == "test_sample":
        hp = CondPriorHParams()
        hp.cond_dim        = args.cond_dim
        hp.top_temperature = args.top_temp
        hp.bot_temperature = args.bot_temp
        hp.top_top_p       = args.top_p
        hp.bot_top_p       = args.top_p

        test_sample_from_data(
            data_dir=args.data_dir,
            vqvae_ckpt=args.vqvae_ckpt,
            top_prior_ckpt=args.top_prior_ckpt,
            bot_prior_ckpt=args.bot_prior_ckpt,
            n_test=args.n_test,
            k_per_feature=args.k_per_feature,
            out_dir=args.out_dir,
            hp=hp,
            seed=args.seed,
            val_split=args.val_split,
            test_split=args.test_split,
            skip_missing_check=args.skip_missing_check,
        )


if __name__ == "__main__":
    main()
