#!/usr/bin/env python3
"""
Transformer Prior for VQ-VAE-2 ECG Generation
===============================================
Stage 2 of the VQ-VAE-2 pipeline (Razavi et al., NeurIPS 2019).

Two auto-regressive Transformers are trained on the discrete code sequences
produced by a pre-trained VQ-VAE-2 (vqvae2.py):

  ┌─────────────────────────────────────────────────────────────┐
  │  TopPrior                                                   │
  │  Input : <BOS> + top_codes[:-1]   shape (B, 78)            │
  │  Output: logits over top codebook shape (B, 78, K_top)     │
  │  Plain causal Transformer (GPT-style)                       │
  └─────────────────────────────────────────────────────────────┘
              ↓  sampled top_codes
  ┌─────────────────────────────────────────────────────────────┐
  │  BottomPrior                                                │
  │  Input : <BOS> + bot_codes[:-1]   shape (B, 625)           │
  │  Cond  : top_codes upsampled to length 625                  │
  │  Output: logits over bot codebook shape (B, 625, K_bot)    │
  │  Conditioned causal Transformer – cross-attention on top    │
  └─────────────────────────────────────────────────────────────┘

Pipeline
--------
  1.  Train VQ-VAE-2         →  vqvae2.py
  2.  Extract codes           →  python transformer_prior.py extract ...
  3.  Train TopPrior          →  python transformer_prior.py fit_top ...
  4.  Train BottomPrior       →  python transformer_prior.py fit_bot ...
  5.  Sample new ECGs         →  python transformer_prior.py sample  ...

All five steps are in this single file. Steps 3 & 4 can run independently.

Sequence lengths (default VQ-VAE-2 strides ×8 per level)
---------------------------------------------------------
  top codes : 5000 // 64  =  78  (exact: floor(5000/64))
  bot codes : 5000 //  8  = 625
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

# Optional wandb import
try:
    from pytorch_lightning.loggers import WandbLogger
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    WandbLogger = None
    wandb = None

# ---------------------------------------------------------------------------
# Import VQ-VAE-2 from companion file
# ---------------------------------------------------------------------------
try:
    from vqvae2 import VQVAE2Lightning, VQVAE2Config, MIMICIVECGDataset
except ImportError:
    raise ImportError(
        "Could not import vqvae2.py. "
        "Place transformer_prior.py in the same directory as vqvae2.py."
    )


# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

class PriorHParams:
    # Must match the trained VQ-VAE-2
    n_embeddings_top: int = 512
    n_embeddings_bot: int = 512
    top_seq_len: int = 78        # 5000 // 64  (corrected from 79)
    bot_seq_len: int = 625       # 5000 //  8

    # Transformer – Top Prior
    top_d_model: int = 256
    top_n_heads: int = 8
    top_n_layers: int = 8
    top_d_ff: int = 1024
    top_dropout: float = 0.1

    # Transformer – Bottom Prior
    bot_d_model: int = 512
    bot_n_heads: int = 8
    bot_n_layers: int = 12
    bot_d_ff: int = 2048
    bot_dropout: float = 0.1

    # Training
    learning_rate: float = 3e-4
    warmup_steps: int = 2000
    batch_size: int = 16
    max_epochs: int = 100
    num_workers: int = 4
    val_fraction: float = 0.1
    label_smoothing: float = 0.1

    # Sampling
    top_temperature: float = 1.0
    bot_temperature: float = 1.0
    top_top_k: int = 0           # 0 = disabled
    top_top_p: float = 0.95      # nucleus sampling
    bot_top_k: int = 0
    bot_top_p: float = 0.95


PHP = PriorHParams()

BOS = 0   # Beginning-of-sequence token (shared; shifts codebook indices by 1)


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# ---------------------------------------------------------------------------
# Causal (decoder-only) Transformer block
# ---------------------------------------------------------------------------

class CausalTransformerBlock(nn.Module):
    """Standard GPT-style block with pre-norm and optional cross-attention."""

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
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.cross_attend = cross_attend
        if cross_attend:
            kv_dim = cross_d_model or d_model
            self.cross_attn = nn.MultiheadAttention(
                d_model, n_heads, kdim=kv_dim, vdim=kv_dim,
                dropout=dropout, batch_first=True)
            self.norm_cross = nn.LayerNorm(d_model)

    def _causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        L = x.size(1)
        mask = self._causal_mask(L, x.device)

        # Self-attention (causal)
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h, attn_mask=mask, is_causal=True)
        x = x + h

        # Cross-attention (optional, not causal – attends to all context)
        if self.cross_attend and context is not None:
            h = self.norm_cross(x)
            h, _ = self.cross_attn(h, context, context)
            x = x + h

        # Feed-forward
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Top Prior  –  unconditional autoregressive model over top codes
# ---------------------------------------------------------------------------

class TopPrior(nn.Module):
    """
    GPT-style model that learns p(z_top).
    Vocabulary size = n_embeddings_top + 1  (+ BOS token).
    """

    def __init__(self, hp: PriorHParams = PHP):
        super().__init__()
        vocab = hp.n_embeddings_top + 1   # +1 for BOS
        self.embed = nn.Embedding(vocab, hp.top_d_model)
        self.pe = SinusoidalPE(hp.top_d_model, max_len=hp.top_seq_len + 1)
        self.drop = nn.Dropout(hp.top_dropout)
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(hp.top_d_model, hp.top_n_heads,
                                   hp.top_d_ff, hp.top_dropout)
            for _ in range(hp.top_n_layers)
        ])
        self.norm = nn.LayerNorm(hp.top_d_model)
        self.head = nn.Linear(hp.top_d_model, hp.n_embeddings_top, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        codes: (B, L)  integer top codes  (0-indexed, BOS NOT prepended)
        returns logits: (B, L, K_top)
        """
        B, L = codes.shape
        bos = torch.full((B, 1), BOS, dtype=torch.long, device=codes.device)
        # Shift right: input = [BOS, c0, c1, ..., c_{L-1}]
        # Target = [c0, c1, ..., c_{L-1}] → handled in loss
        x = torch.cat([bos, codes[:, :-1] + 1], dim=1)  # offset by 1 (BOS=0)
        x = self.drop(self.pe(self.embed(x)))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))   # (B, L, K_top)

    @torch.no_grad()
    def sample(self, n: int, seq_len: int, hp: PriorHParams, device: torch.device) -> torch.Tensor:
        """Autoregressively sample top codes. Returns (n, seq_len) int64 tensor."""
        tokens = torch.full((n, 1), BOS, dtype=torch.long, device=device)
        for _ in range(seq_len):
            # Feed the full growing sequence
            x_in = tokens
            x = self.drop(self.pe(self.embed(x_in)))
            for block in self.blocks:
                x = block(x)
            logit = self.head(self.norm(x))[:, -1, :]  # (B, K)
            logit = logit / hp.top_temperature
            logit = _top_p_filter(logit, hp.top_top_p)
            if hp.top_top_k > 0:
                logit = _top_k_filter(logit, hp.top_top_k)
            next_tok = torch.multinomial(F.softmax(logit, dim=-1), 1)  # (B,1)
            tokens = torch.cat([tokens, next_tok + 1], dim=1)          # +1 for BOS offset
        return tokens[:, 1:] - 1   # strip BOS, un-shift


# ---------------------------------------------------------------------------
# Bottom Prior  –  p(z_bot | z_top)
# ---------------------------------------------------------------------------

class BottomPrior(nn.Module):
    """
    Conditioned autoregressive model for bottom codes given top codes.
    Cross-attention is used to condition each bottom token on the full
    top code sequence (already available at inference time).
    """

    def __init__(self, hp: PriorHParams = PHP):
        super().__init__()
        vocab_bot = hp.n_embeddings_bot + 1
        vocab_top = hp.n_embeddings_top + 1

        self.embed_bot = nn.Embedding(vocab_bot, hp.bot_d_model)
        self.embed_top = nn.Embedding(vocab_top, hp.bot_d_model)
        self.pe_bot = SinusoidalPE(hp.bot_d_model, max_len=hp.bot_seq_len + 1)
        self.pe_top = SinusoidalPE(hp.bot_d_model, max_len=hp.top_seq_len + 1)
        self.drop = nn.Dropout(hp.bot_dropout)

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
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _encode_top(self, top_codes: torch.Tensor) -> torch.Tensor:
        """top_codes: (B, L_top) → context (B, L_top, d_model)"""
        return self.drop(self.pe_top(self.embed_top(top_codes + 1)))

    def forward(self, bot_codes: torch.Tensor, top_codes: torch.Tensor) -> torch.Tensor:
        """
        bot_codes: (B, L_bot)  integer bottom codes
        top_codes: (B, L_top)  integer top codes (condition)
        returns logits: (B, L_bot, K_bot)
        """
        B, L = bot_codes.shape
        bos = torch.full((B, 1), BOS, dtype=torch.long, device=bot_codes.device)
        x_in = torch.cat([bos, bot_codes[:, :-1] + 1], dim=1)
        x = self.drop(self.pe_bot(self.embed_bot(x_in)))
        ctx = self._encode_top(top_codes)
        for block in self.blocks:
            x = block(x, context=ctx)
        return self.head(self.norm(x))   # (B, L_bot, K_bot)

    @torch.no_grad()
    def sample(
        self,
        top_codes: torch.Tensor,
        seq_len: int,
        hp: PriorHParams,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample bottom codes conditioned on top_codes. Returns (B, seq_len)."""
        B = top_codes.size(0)
        ctx = self._encode_top(top_codes.to(device))
        tokens = torch.full((B, 1), BOS, dtype=torch.long, device=device)
        for _ in range(seq_len):
            x = self.drop(self.pe_bot(self.embed_bot(tokens)))
            for block in self.blocks:
                x = block(x, context=ctx)
            logit = self.head(self.norm(x))[:, -1, :]
            logit = logit / hp.bot_temperature
            logit = _top_p_filter(logit, hp.bot_top_p)
            if hp.bot_top_k > 0:
                logit = _top_k_filter(logit, hp.bot_top_k)
            next_tok = torch.multinomial(F.softmax(logit, dim=-1), 1)
            tokens = torch.cat([tokens, next_tok + 1], dim=1)
        return tokens[:, 1:] - 1   # strip BOS, un-shift


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    vals, _ = torch.topk(logits, k)
    threshold = vals[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < threshold, float('-inf'))


def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative prob > p (shift right to keep first above threshold)
    remove = cumprobs - F.softmax(sorted_logits, dim=-1) > p
    sorted_logits[remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_idx, sorted_logits)


# ---------------------------------------------------------------------------
# Lightning modules
# ---------------------------------------------------------------------------

class TopPriorLightning(pl.LightningModule):

    def __init__(self, hp: PriorHParams = PHP):
        super().__init__()
        self.hp = hp
        self.save_hyperparameters(vars(hp))
        self.model = TopPrior(hp)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=hp.label_smoothing)

    def _step(self, batch, stage: str):
        top_codes, _ = batch                          # (B, L_top), (B, L_bot)
        logits = self.model(top_codes)                # (B, L, K)
        B, L, K = logits.shape
        loss = self.criterion(logits.reshape(B * L, K), top_codes.reshape(B * L))
        acc = (logits.argmax(-1) == top_codes).float().mean()
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}/acc",  acc,  prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, _):   return self._step(batch, "train")
    def validation_step(self, batch, _): self._step(batch, "val")
    def test_step(self, batch, _):       self._step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hp.learning_rate,
                                betas=(0.9, 0.98), weight_decay=1e-2)
        sched = _cosine_with_warmup(opt, self.hp.warmup_steps,
                                    self.hp.max_epochs * 1000)  # approx total steps
        return [opt], [{"scheduler": sched, "interval": "step"}]


class BottomPriorLightning(pl.LightningModule):

    def __init__(self, hp: PriorHParams = PHP):
        super().__init__()
        self.hp = hp
        self.save_hyperparameters(vars(hp))
        self.model = BottomPrior(hp)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=hp.label_smoothing)

    def _step(self, batch, stage: str):
        top_codes, bot_codes = batch                  # (B, L_top), (B, L_bot)
        logits = self.model(bot_codes, top_codes)     # (B, L_bot, K)
        B, L, K = logits.shape
        loss = self.criterion(logits.reshape(B * L, K), bot_codes.reshape(B * L))
        acc = (logits.argmax(-1) == bot_codes).float().mean()
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}/acc",  acc,  prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, _):   return self._step(batch, "train")
    def validation_step(self, batch, _): self._step(batch, "val")
    def test_step(self, batch, _):       self._step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hp.learning_rate,
                                betas=(0.9, 0.98), weight_decay=1e-2)
        sched = _cosine_with_warmup(opt, self.hp.warmup_steps,
                                    self.hp.max_epochs * 1000)
        return [opt], [{"scheduler": sched, "interval": "step"}]


def _cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Code Dataset
# ---------------------------------------------------------------------------

class CodeDataset(Dataset):
    """
    Loads pre-extracted VQ-VAE-2 codes saved by the `extract` command.
    Expects two files:
      codes_top.npy  –  shape (N, L_top)  int16 or int32
      codes_bot.npy  –  shape (N, L_bot)  int16 or int32
    """

    def __init__(self, codes_dir: str):
        codes_dir = Path(codes_dir)
        top = np.load(codes_dir / "codes_top.npy").astype(np.int64)
        bot = np.load(codes_dir / "codes_bot.npy").astype(np.int64)
        assert len(top) == len(bot), "Mismatch between top and bottom code counts"
        self.top = torch.from_numpy(top)
        self.bot = torch.from_numpy(bot)

    def __len__(self):
        return len(self.top)

    def __getitem__(self, idx):
        return self.top[idx], self.bot[idx]


class CodeDataModule(pl.LightningDataModule):

    def __init__(self, codes_dir: str, hp: PriorHParams = PHP):
        super().__init__()
        self.codes_dir = codes_dir
        self.hp = hp

    def setup(self, stage: Optional[str] = None):
        full = CodeDataset(self.codes_dir)
        n_val = max(1, int(len(full) * self.hp.val_fraction))
        self.train_ds, self.val_ds = random_split(
            full, [len(full) - n_val, n_val],
            generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hp.batch_size,
                          shuffle=True, num_workers=self.hp.num_workers,
                          pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hp.batch_size,
                          shuffle=False, num_workers=self.hp.num_workers,
                          pin_memory=True)

    def test_dataloader(self):
        return self.val_dataloader()


# ---------------------------------------------------------------------------
# Code extraction (runs VQ-VAE-2 over data, saves codes)
# ---------------------------------------------------------------------------

def extract_codes(vqvae_ckpt: str, data_dir: str, out_dir: str, batch_size: int = 32, max_samples: Optional[int] = None):
    """
    Encode all ECG samples and save the discrete code indices.
    Produces:
      <out_dir>/codes_top.npy  (N, L_top)
      <out_dir>/codes_bot.npy  (N, L_bot)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading VQ-VAE-2 from: {vqvae_ckpt}")
    vqvae_pl = VQVAE2Lightning.load_from_checkpoint(vqvae_ckpt, map_location=device)
    vqvae_pl.eval()
    vqvae = vqvae_pl.model.to(device)

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

    all_top, all_bot = [], []
    print(f"Extracting codes from {len(dataset)} samples...")
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            x = x.to(device)
            _, _, _, _, codes_bot, codes_top = vqvae.encode(x)
            all_top.append(codes_top.cpu().short())
            all_bot.append(codes_bot.cpu().short())
            if (i + 1) % 20 == 0:
                print(f"  batch {i+1}/{len(loader)}")

    codes_top_np = torch.cat(all_top, 0).numpy()
    codes_bot_np = torch.cat(all_bot, 0).numpy()
    np.save(out_dir / "codes_top.npy", codes_top_np)
    np.save(out_dir / "codes_bot.npy", codes_bot_np)
    print(f"Saved codes_top {codes_top_np.shape} and codes_bot {codes_bot_np.shape} to {out_dir}")


# ---------------------------------------------------------------------------
# Full end-to-end sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_ecgs(
    vqvae_ckpt: str,
    top_prior_ckpt: str,
    bot_prior_ckpt: str,
    n_samples: int,
    out_path: str,
    hp: PriorHParams = PHP,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    print(f"Loading VQ-VAE-2 from: {vqvae_ckpt}")
    vqvae_pl = VQVAE2Lightning.load_from_checkpoint(vqvae_ckpt, map_location=device)
    vqvae_pl.eval()
    vqvae = vqvae_pl.model.to(device)

    print(f"Loading top prior from: {top_prior_ckpt}")
    top_pl = TopPriorLightning.load_from_checkpoint(top_prior_ckpt, hp=hp, map_location=device)
    top_pl.eval()
    top_prior = top_pl.model.to(device)

    print(f"Loading bottom prior from: {bot_prior_ckpt}")
    bot_pl = BottomPriorLightning.load_from_checkpoint(bot_prior_ckpt, hp=hp, map_location=device)
    bot_pl.eval()
    bot_prior = bot_pl.model.to(device)

    print(f"Sampling {n_samples} top codes (seq_len={hp.top_seq_len})...")
    codes_top = top_prior.sample(n_samples, hp.top_seq_len, hp, device)   # (B, L_top)

    print(f"Sampling {n_samples} bottom codes (seq_len={hp.bot_seq_len})...")
    codes_bot = bot_prior.sample(codes_top, hp.bot_seq_len, hp, device)   # (B, L_bot)

    print("Decoding to ECG signals...")
    ecgs = vqvae.decode_codes(codes_bot, codes_top).cpu().numpy()          # (B, 12, 5000)

    np.save(out_path, ecgs)
    print(f"Saved {n_samples} ECGs → {out_path}  shape: {ecgs.shape}")


# ---------------------------------------------------------------------------
# Trainer factory
# ---------------------------------------------------------------------------

def build_trainer(args, name: str) -> pl.Trainer:
    callbacks = [
        ModelCheckpoint(
            monitor="val/loss", mode="min", save_top_k=3,
            filename=f"{name}-{{epoch:03d}}-{{val/loss:.4f}}",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    
    # Setup loggers
    loggers = []
    
    # Always use CSV logger as backup
    csv_logger = CSVLogger("logs", name=name)
    loggers.append(csv_logger)
    
    # Add wandb logger if available and requested
    if hasattr(args, 'wandb') and args.wandb and WANDB_AVAILABLE:
        wandb_logger = WandbLogger(
            project=args.wandb_project if hasattr(args, 'wandb_project') else "vqvae2-prior",
            entity=args.wandb_entity if hasattr(args, 'wandb_entity') else None,
            name=f"{name}_{args.wandb_run_name}" if hasattr(args, 'wandb_run_name') and args.wandb_run_name else name,
            save_dir="logs",
        )
        loggers.append(wandb_logger)
        print(f"✓ Weights & Biases logging enabled")
    elif hasattr(args, 'wandb') and args.wandb and not WANDB_AVAILABLE:
        print("⚠ Warning: wandb requested but not installed. Install with: pip install wandb")
    
    return pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=args.gpus if args.gpus else "auto",
        callbacks=callbacks,
        logger=loggers if len(loggers) > 1 else loggers[0],
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Transformer Prior for VQ-VAE-2 ECG Generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # ---- extract ----
    ep = sub.add_parser("extract", help="Extract VQ codes from a trained VQ-VAE-2")
    ep.add_argument("--vqvae-ckpt", required=True, help="Path to VQ-VAE-2 checkpoint")
    ep.add_argument("--data-dir",   required=True, help="Directory with MIMIC-IV-ECG dataset")
    ep.add_argument("--out-dir",    required=True, help="Directory to save code .npy files")
    ep.add_argument("--batch-size", type=int, default=32)
    ep.add_argument("--max-samples", type=int, default=None, help="Max samples to extract (for debugging)")

    # ---- fit_top ----
    tp = sub.add_parser("fit_top", help="Train the Top Prior Transformer")
    tp.add_argument("--codes-dir",  required=True, help="Directory with code .npy files")
    tp.add_argument("--max-epochs", type=int, default=PHP.max_epochs)
    tp.add_argument("--batch-size", type=int, default=PHP.batch_size)
    tp.add_argument("--lr",         type=float, default=PHP.learning_rate)
    tp.add_argument("--gpus",       type=int, default=None)
    tp.add_argument("--d-model",    type=int, default=PHP.top_d_model)
    tp.add_argument("--n-layers",   type=int, default=PHP.top_n_layers)
    tp.add_argument("--n-heads",    type=int, default=PHP.top_n_heads)
    tp.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    tp.add_argument("--wandb-project", type=str, default="vqvae2-prior", help="W&B project name")
    tp.add_argument("--wandb-entity", type=str, default=None, help="W&B entity (username/team)")
    tp.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")

    # ---- fit_bot ----
    bp = sub.add_parser("fit_bot", help="Train the Bottom Prior Transformer")
    bp.add_argument("--codes-dir",  required=True, help="Directory with code .npy files")
    bp.add_argument("--max-epochs", type=int, default=PHP.max_epochs)
    bp.add_argument("--batch-size", type=int, default=PHP.batch_size)
    bp.add_argument("--lr",         type=float, default=PHP.learning_rate)
    bp.add_argument("--gpus",       type=int, default=None)
    bp.add_argument("--d-model",    type=int, default=PHP.bot_d_model)
    bp.add_argument("--n-layers",   type=int, default=PHP.bot_n_layers)
    bp.add_argument("--n-heads",    type=int, default=PHP.bot_n_heads)
    bp.add_argument("--ckpt-path",  type=str, default=None, help="Resume from checkpoint (path to last.ckpt)")
    bp.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    bp.add_argument("--wandb-project", type=str, default="vqvae2-prior", help="W&B project name")
    bp.add_argument("--wandb-entity", type=str, default=None, help="W&B entity (username/team)")
    bp.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")

    # ---- sample ----
    sp = sub.add_parser("sample", help="Generate ECGs using trained priors")
    sp.add_argument("--vqvae-ckpt",     required=True)
    sp.add_argument("--top-prior-ckpt", required=True)
    sp.add_argument("--bot-prior-ckpt", required=True)
    sp.add_argument("--n-samples",  type=int, default=16)
    sp.add_argument("--out",        type=str, default="generated_ecgs.npy")
    sp.add_argument("--top-temp",   type=float, default=PHP.top_temperature)
    sp.add_argument("--bot-temp",   type=float, default=PHP.bot_temperature)
    sp.add_argument("--top-p",      type=float, default=PHP.top_top_p)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    # ------------------------------------------------------------------
    if args.command == "extract":
        extract_codes(args.vqvae_ckpt, args.data_dir, args.out_dir, args.batch_size, args.max_samples)

    # ------------------------------------------------------------------
    elif args.command == "fit_top":
        hp = PriorHParams()
        hp.learning_rate = args.lr
        hp.batch_size = args.batch_size
        hp.max_epochs = args.max_epochs
        hp.top_d_model = args.d_model
        hp.top_n_layers = args.n_layers
        hp.top_n_heads = args.n_heads
        hp.top_d_ff = args.d_model * 4
        model = TopPriorLightning(hp)
        dm = CodeDataModule(args.codes_dir, hp)
        trainer = build_trainer(args, "top_prior")
        trainer.fit(model, dm)

    # ------------------------------------------------------------------
    elif args.command == "fit_bot":
        hp = PriorHParams()
        hp.learning_rate = args.lr
        hp.batch_size = args.batch_size
        hp.max_epochs = args.max_epochs
        hp.bot_d_model = args.d_model
        hp.bot_n_layers = args.n_layers
        hp.bot_n_heads = args.n_heads
        hp.bot_d_ff = args.d_model * 4
        model = BottomPriorLightning(hp)
        dm = CodeDataModule(args.codes_dir, hp)
        trainer = build_trainer(args, "bot_prior")
        ckpt_path = getattr(args, "ckpt_path", None) or None
        trainer.fit(model, dm, ckpt_path=ckpt_path)

    # ------------------------------------------------------------------
    elif args.command == "sample":
        hp = PriorHParams()
        hp.top_temperature = args.top_temp
        hp.bot_temperature = args.bot_temp
        hp.top_top_p = args.top_p
        hp.bot_top_p = args.top_p
        generate_ecgs(
            args.vqvae_ckpt, args.top_prior_ckpt, args.bot_prior_ckpt,
            args.n_samples, args.out, hp,
        )


if __name__ == "__main__":
    main()
