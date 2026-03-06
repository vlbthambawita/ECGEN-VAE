#!/usr/bin/env python3
"""
PixelCNN Prior Quality Diagnostic Script

Checks:
  1. NLL (Negative Log-Likelihood) on validation data
  2. Token sequence diversity (entropy, repetition)
  3. Token transition patterns (does it learn structure?)
  4. Generated vs real token distribution comparison
  5. Reconstruction of generated tokens via VQ-VAE decoder
  6. Temperature sweep — how generation quality changes with temperature

Usage:
    python check_prior_quality.py \
        --prior-checkpoint runs/prior_mimic_standalone/seed_42/checkpoints/best.ckpt \
        --data-dir /path/to/mimic-iv-ecg \
        --n-batches 20 \
        --output-dir prior_diagnostics

    # Quick mode
    python check_prior_quality.py \
        --prior-checkpoint runs/prior_mimic_standalone/seed_42/checkpoints/best.ckpt \
        --data-dir /path/to/mimic-iv-ecg \
        --n-batches 5 --quick
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── Import your training module ──────────────────────────────────────────────
try:
    from train_vqvae_standalone import PriorLightning, MIMICIVECGDataset
except ImportError:
    print("[ERROR] Could not import from train_vqvae_standalone.py")
    print("        Make sure this script is in the same directory.")
    sys.exit(1)


LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]


class ECGDataset(MIMICIVECGDataset):
    """
    Thin wrapper around MIMICIVECGDataset to match the older ECGDataset API
    used by this diagnostics script.
    """

    def __init__(
        self,
        data_dir,
        seq_length=5000,
        split="val",
        val_split=0.1,
        test_split=0.1,
        max_samples=None,
        seed=42,
        skip_missing_check=False,
        num_leads=12,
    ):
        super().__init__(
            mimic_path=data_dir,
            split=split,
            val_split=val_split,
            test_split=test_split,
            max_samples=max_samples,
            seed=seed,
            skip_missing_check=skip_missing_check,
            num_leads=num_leads,
            seq_length=seq_length,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Token extraction helper
# ─────────────────────────────────────────────────────────────────────────────

def encode_to_tokens(model: PriorLightning, x: torch.Tensor) -> torch.Tensor:
    """
    Encode real ECG batch to discrete token indices using the VQ-VAE encoder.

    Args:
        model: PriorLightning (contains model.vqvae, a VQVAELightning)
        x:     [B, 12, T] real ECG

    Returns:
        indices: [B, T_enc] integer token indices
    """
    if model.vqvae is None:
        raise RuntimeError("VQ-VAE not loaded inside PriorLightning")

    # Use the helper on VQVAELightning, which handles encoder + quantizer.
    indices = model.vqvae.encode_to_indices(x)  # [B, T_enc]
    return indices


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_nll_histogram(nlls: list, save_path: Path) -> None:
    """Histogram of per-sample NLL values."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(nlls, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(nlls), color="red", linewidth=2, linestyle="--",
               label=f"Mean = {np.mean(nlls):.3f}")
    ax.axvline(np.median(nlls), color="orange", linewidth=2, linestyle="--",
               label=f"Median = {np.median(nlls):.3f}")
    ax.set_xlabel("NLL per token", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Prior NLL Distribution over Validation Samples", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_token_distribution_comparison(
    real_counts: Counter,
    gen_counts: Counter,
    num_embeddings: int,
    save_path: Path,
) -> None:
    """Compare token frequency distribution: real vs generated."""
    real_arr = np.zeros(num_embeddings)
    gen_arr  = np.zeros(num_embeddings)

    total_real = max(sum(real_counts.values()), 1)
    total_gen  = max(sum(gen_counts.values()), 1)

    for k, v in real_counts.items():
        if k < num_embeddings:
            real_arr[k] = v / total_real
    for k, v in gen_counts.items():
        if k < num_embeddings:
            gen_arr[k] = v / total_gen

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle("Token Distribution: Real vs Generated", fontsize=13, fontweight="bold")

    # Real distribution
    ax1 = axes[0]
    ax1.bar(range(num_embeddings), real_arr, width=1.0, color="steelblue", alpha=0.85)
    ax1.set_title("Real ECG Token Frequencies", fontsize=11)
    ax1.set_xlabel("Token Index")
    ax1.set_ylabel("Frequency")
    ax1.grid(True, alpha=0.3, axis="y")

    # Generated distribution
    ax2 = axes[1]
    ax2.bar(range(num_embeddings), gen_arr, width=1.0, color="darkorange", alpha=0.85)
    ax2.set_title("Generated Token Frequencies", fontsize=11)
    ax2.set_xlabel("Token Index")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, alpha=0.3, axis="y")

    # Difference
    ax3 = axes[2]
    diff = gen_arr - real_arr
    colors = ["green" if d >= 0 else "red" for d in diff]
    ax3.bar(range(num_embeddings), diff, width=1.0, color=colors, alpha=0.75)
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_title("Difference (Generated − Real)  |  green=overused, red=underused by prior", fontsize=11)
    ax3.set_xlabel("Token Index")
    ax3.set_ylabel("Frequency Difference")
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_token_transition_matrix(
    indices: np.ndarray,
    num_embeddings: int,
    save_path: Path,
    max_codes: int = 64,
) -> None:
    """
    Bigram transition matrix of token sequences.
    A structured diagonal = prior learned local structure.
    A uniform matrix = prior learned nothing.
    """
    # Use only top-N most frequent codes for readability
    all_codes = indices.flatten()
    top_codes = [c for c, _ in Counter(all_codes.tolist()).most_common(max_codes)]
    top_codes = sorted(top_codes)
    n = len(top_codes)
    code_to_idx = {c: i for i, c in enumerate(top_codes)}

    matrix = np.zeros((n, n), dtype=np.float32)
    for seq in indices:
        for t in range(len(seq) - 1):
            a, b = int(seq[t]), int(seq[t + 1])
            if a in code_to_idx and b in code_to_idx:
                matrix[code_to_idx[a], code_to_idx[b]] += 1

    # Row-normalize
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix /= row_sums

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="hot", aspect="auto", vmin=0, vmax=matrix.max())
    plt.colorbar(im, ax=ax, label="Transition Probability")
    ax.set_title(
        f"Token Transition Matrix (top-{n} codes from real data)\n"
        f"Diagonal structure = prior learned local patterns",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("Next Token", fontsize=10)
    ax.set_ylabel("Current Token", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_ecg_grid(
    samples: np.ndarray,
    save_path: Path,
    title: str = "Generated ECG Samples",
    n_cols: int = 4,
) -> None:
    """Plot multiple ECGs (all 12 leads stacked) in a grid."""
    n_samples = min(samples.shape[0], 16)
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx in range(n_samples):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        ecg = samples[idx]   # [12, T]
        offset = 0
        for lead_idx in range(12):
            ax.plot(ecg[lead_idx] + offset, linewidth=0.5, alpha=0.8,
                    label=LEAD_NAMES[lead_idx])
            offset -= 3
        ax.set_title(f"Sample {idx + 1}", fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.axis("off")

    for idx in range(n_samples, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_single_ecg(ecg: np.ndarray, save_path: Path, title: str = "") -> None:
    """Plot a single ECG all 12 leads stacked."""
    fig, axes = plt.subplots(12, 1, figsize=(16, 13))
    fig.suptitle(title, fontsize=12, fontweight="bold")
    for i in range(12):
        axes[i].plot(ecg[i], linewidth=0.8, color="black")
        axes[i].set_ylabel(LEAD_NAMES[i], fontsize=9, fontweight="bold")
        axes[i].grid(True, alpha=0.25)
        axes[i].set_xlim(0, ecg.shape[1])
        if i < 11:
            axes[i].set_xticks([])
        else:
            axes[i].set_xlabel("Time (samples)", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_temperature_comparison(
    samples_by_temp: dict,
    save_path: Path,
) -> None:
    """
    For each temperature, show one ECG side by side.
    Reveals whether low T is too repetitive and high T is too random.
    """
    temps = sorted(samples_by_temp.keys())
    n = len(temps)

    fig, axes = plt.subplots(12, n, figsize=(5 * n, 14))
    fig.suptitle("Temperature Sweep — One ECG per Temperature", fontsize=13, fontweight="bold")

    if n == 1:
        axes = axes.reshape(-1, 1)

    for col, temp in enumerate(temps):
        ecg = samples_by_temp[temp][0]   # first sample, [12, T]
        for lead_idx in range(12):
            ax = axes[lead_idx, col]
            ax.plot(ecg[lead_idx], linewidth=0.7, color="steelblue")
            ax.set_xlim(0, ecg.shape[1])
            ax.grid(True, alpha=0.2)
            if lead_idx == 0:
                ax.set_title(f"T={temp}", fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(LEAD_NAMES[lead_idx], fontsize=8, fontweight="bold")
            if lead_idx < 11:
                ax.set_xticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_repetition_score(
    rep_real: list,
    rep_gen_by_temp: dict,
    save_path: Path,
) -> None:
    """
    Bar chart of token repetition rate for real vs generated at each temperature.
    High repetition = prior is mode-collapsing / generating repetitive codes.
    """
    labels = ["Real"] + [f"T={t}" for t in sorted(rep_gen_by_temp.keys())]
    values = [np.mean(rep_real)] + [np.mean(v) for _, v in sorted(rep_gen_by_temp.items())]
    colors = ["steelblue"] + ["darkorange"] * len(rep_gen_by_temp)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="white")
    ax.set_ylabel("Token Repetition Rate", fontsize=11)
    ax.set_title(
        "Token Repetition Rate: Real vs Generated\n"
        "(fraction of consecutive identical tokens — ideal ≈ real)",
        fontsize=12, fontweight="bold"
    )
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def compute_repetition_rate(indices: np.ndarray) -> list:
    """
    Fraction of consecutive identical tokens per sequence.
    High value → prior is stuck in repetitive loops.
    """
    rates = []
    for seq in indices:
        reps = np.sum(seq[1:] == seq[:-1])
        rates.append(reps / max(len(seq) - 1, 1))
    return rates


def compute_entropy(counts: Counter, total: int) -> float:
    """Shannon entropy of a token distribution."""
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy


# ─────────────────────────────────────────────────────────────────────────────
# Core diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def run_prior_diagnostics(
    prior_checkpoint: str,
    data_dir: str,
    n_batches: int = 20,
    batch_size: int = 16,
    seq_length: int = 5000,
    output_dir: str = "prior_diagnostics",
    device: str = "cuda",
    quick: bool = False,
    val_split: float = 0.1,
    test_split: float = 0.1,
    num_workers: int = 4,
    temperatures: list = None,
) -> dict:

    if temperatures is None:
        temperatures = [0.5, 0.8, 1.0, 1.2] if not quick else [0.8, 1.0]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Loading Prior model from: {prior_checkpoint}")
    print(f"{'='*70}")

    model = PriorLightning.load_from_checkpoint(prior_checkpoint)
    model.eval()
    model.to(device)

    if model.vqvae is None:
        print("[ERROR] VQ-VAE not loaded inside PriorLightning — cannot run diagnostics")
        sys.exit(1)

    try:
        # model.vqvae is VQVAELightning; its inner VQ-VAE lives at .vqvae
        num_embeddings = model.vqvae.vqvae.vq.num_embeddings
    except AttributeError:
        num_embeddings = getattr(model.hparams, "num_embeddings", 512)
        if num_embeddings == 512:
            print(f"  [WARN] Could not read num_embeddings, using {num_embeddings}")

    print(f"  Codebook size: {num_embeddings}")
    print(f"  Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n  Loading dataset from: {data_dir}")
    dataset = ECGDataset(
        data_dir=data_dir,
        seq_length=seq_length,
        split="val",
        val_split=val_split,
        test_split=test_split,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device == "cuda"),
    )
    print(f"  Validation set size: {len(dataset)}")
    print(f"  Evaluating {n_batches} batches × {batch_size} = up to {n_batches * batch_size} samples")

    # ─────────────────────────────────────────────────────────────────────────
    # PART 1: NLL on validation data + real token stats
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  PART 1: NLL evaluation on real validation data")
    print(f"{'='*70}")

    all_nll:       list = []
    real_token_counter = Counter()
    real_token_indices_list = []
    real_rep_rates: list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= n_batches:
                break

            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device).float()

            # Get discrete token indices from VQ-VAE encoder
            try:
                indices = encode_to_tokens(model, x)   # [B, T_enc]
            except Exception as e:
                if batch_idx == 0:
                    print(f"  [WARN] Token extraction failed: {e}")
                    print(f"         Skipping token-level analysis.")
                indices = None

            # Compute prior NLL
            # Adjust forward call to match your PriorLightning signature
            try:
                prior_out = model.prior(indices) if indices is not None else None
                if prior_out is not None:
                    # prior_out: [B, T_enc, num_embeddings] logits
                    B, T, V = prior_out.shape
                    targets = indices[:, 1:].contiguous()          # [B, T-1]
                    logits  = prior_out[:, :-1, :].contiguous()    # [B, T-1, V]
                    nll = F.cross_entropy(
                        logits.view(B * (T - 1), V),
                        targets.view(B * (T - 1)),
                        reduction="none",
                    ).view(B, T - 1).mean(dim=1)                   # [B]
                    all_nll.extend(nll.cpu().numpy().tolist())
            except Exception as e:
                if batch_idx == 0:
                    print(f"  [WARN] Prior NLL computation failed: {e}")
                    print(f"         Your PriorLightning.prior forward may have a different signature.")

            if indices is not None:
                idx_np = indices.cpu().numpy()   # [B, T_enc]
                real_token_counter.update(idx_np.flatten().tolist())
                real_token_indices_list.append(idx_np)
                real_rep_rates.extend(compute_repetition_rate(idx_np))

            if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
                nll_str = f"{np.mean(all_nll):.4f}" if all_nll else "N/A"
                print(f"  Batch {batch_idx + 1:3d}/{n_batches}  |  Mean NLL so far: {nll_str}")

    mean_nll    = float(np.mean(all_nll))    if all_nll    else float("nan")
    median_nll  = float(np.median(all_nll))  if all_nll    else float("nan")
    mean_real_rep = float(np.mean(real_rep_rates)) if real_rep_rates else float("nan")

    real_total = sum(real_token_counter.values())
    real_entropy = compute_entropy(real_token_counter, real_total)
    real_used_codes = len(real_token_counter)

    print(f"\n  NLL Results:")
    print(f"    Mean NLL:   {mean_nll:.4f}")
    print(f"    Median NLL: {median_nll:.4f}")
    print(f"\n  Real Token Stats:")
    print(f"    Unique codes used:  {real_used_codes} / {num_embeddings}")
    print(f"    Token entropy:      {real_entropy:.3f} bits  (max={np.log2(num_embeddings):.2f})")
    print(f"    Repetition rate:    {mean_real_rep:.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # PART 2: Generate samples at multiple temperatures
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  PART 2: Generating samples at temperatures: {temperatures}")
    print(f"{'='*70}")

    n_gen = min(8, batch_size)
    samples_by_temp:    dict = {}
    gen_token_by_temp:  dict = {}
    gen_rep_by_temp:    dict = {}
    gen_entropy_by_temp: dict = {}

    with torch.no_grad():
        for temp in temperatures:
            print(f"\n  Generating {n_gen} samples at T={temp}...")
            try:
                samples = model.sample(
                    n_samples=n_gen,
                    seq_length=seq_length,
                    temperature=temp,
                )
                samples_np = samples.cpu().numpy()   # [N, 12, T]
                samples_by_temp[temp] = samples_np

                # Also generate tokens directly for analysis
                gen_indices = encode_to_tokens(model, samples.to(device))
                gen_idx_np = gen_indices.cpu().numpy()
                gen_counter = Counter(gen_idx_np.flatten().tolist())
                gen_entropy = compute_entropy(gen_counter, sum(gen_counter.values()))
                gen_rep = compute_repetition_rate(gen_idx_np)

                gen_token_by_temp[temp]   = gen_counter
                gen_rep_by_temp[temp]     = gen_rep
                gen_entropy_by_temp[temp] = gen_entropy

                print(f"    Generated shape: {samples_np.shape}")
                print(f"    Unique tokens:   {len(gen_counter)} / {num_embeddings}")
                print(f"    Token entropy:   {gen_entropy:.3f} bits  (real={real_entropy:.3f})")
                print(f"    Repetition rate: {np.mean(gen_rep):.4f}  (real={mean_real_rep:.4f})")

            except Exception as e:
                print(f"    [WARN] Generation at T={temp} failed: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # PART 3: Diagnosis
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  DIAGNOSIS")
    print(f"{'='*70}")

    issues_found = 0

    # NLL check
    if not np.isnan(mean_nll):
        max_nll = np.log2(num_embeddings)   # worst case (uniform)
        nll_ratio = mean_nll / max_nll
        if nll_ratio > 0.85:
            print(f"\n  🔴 CRITICAL — NLL very high ({mean_nll:.3f})")
            print(f"     Prior is barely better than random uniform ({max_nll:.2f})")
            print(f"     → Prior has not learned the token distribution")
            print(f"     FIXES: increase hidden_dim (≥512), num_layers (≥8),")
            print(f"            train longer, reduce LR to 1e-4")
            issues_found += 1
        elif nll_ratio > 0.6:
            print(f"\n  🟡 WARNING — NLL moderately high ({mean_nll:.3f}  /  max={max_nll:.2f})")
            print(f"     Prior has learned something but not enough")
            print(f"     FIXES: increase model capacity and/or training epochs")
            issues_found += 1
        else:
            print(f"\n  ✅ NLL looks reasonable ({mean_nll:.3f})")

    # Token diversity check
    if gen_entropy_by_temp:
        for temp, gen_ent in sorted(gen_entropy_by_temp.items()):
            entropy_ratio = gen_ent / max(real_entropy, 1e-6)
            if entropy_ratio < 0.3:
                print(f"\n  🔴 CRITICAL — T={temp}: Generated token entropy={gen_ent:.2f} "
                      f"vs real={real_entropy:.2f} ({100*entropy_ratio:.0f}%)")
                print(f"     → Prior is collapsing to very few tokens (mode collapse)")
                issues_found += 1
            elif entropy_ratio < 0.7:
                print(f"\n  🟡 WARNING — T={temp}: Low token entropy ratio ({100*entropy_ratio:.0f}%)")
            else:
                print(f"\n  ✅ T={temp}: Token entropy ratio OK ({100*entropy_ratio:.0f}%)")

    # Repetition check
    if gen_rep_by_temp:
        for temp, gen_rep in sorted(gen_rep_by_temp.items()):
            mean_gen_rep = float(np.mean(gen_rep))
            if mean_gen_rep > 3 * mean_real_rep and mean_real_rep > 0:
                print(f"\n  🔴 CRITICAL — T={temp}: Repetition rate={mean_gen_rep:.3f} "
                      f"vs real={mean_real_rep:.3f}")
                print(f"     → Prior generates repetitive token sequences")
                print(f"     → This directly causes non-continuous ECG patterns")
                print(f"     FIXES: increase model depth, train longer")
                issues_found += 1
            elif mean_gen_rep > 1.5 * mean_real_rep and mean_real_rep > 0:
                print(f"\n  🟡 WARNING — T={temp}: Slightly high repetition ({mean_gen_rep:.3f} vs {mean_real_rep:.3f})")

    if issues_found == 0:
        print(f"\n  ✅ Prior looks healthy — check reconstruction plots for visual quality")
    else:
        print(f"\n  ⚠️  {issues_found} issue(s) found in the prior")

    # Best temperature recommendation
    if gen_entropy_by_temp:
        best_temp = min(
            gen_entropy_by_temp.keys(),
            key=lambda t: abs(gen_entropy_by_temp[t] - real_entropy)
        )
        print(f"\n  💡 Temperature closest to real distribution: T={best_temp}")
        print(f"     (entropy={gen_entropy_by_temp.get(best_temp, '?'):.3f} vs real={real_entropy:.3f})")

    # ─────────────────────────────────────────────────────────────────────────
    # PART 4: Save plots
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Saving diagnostic plots to: {output_path}")
    print(f"{'='*70}\n")

    # NLL histogram
    if all_nll:
        plot_nll_histogram(all_nll, output_path / "nll_histogram.png")

    # Token distribution comparison (T=1.0 or closest)
    if gen_token_by_temp:
        ref_temp = min(gen_token_by_temp.keys(), key=lambda t: abs(t - 1.0))
        plot_token_distribution_comparison(
            real_token_counter,
            gen_token_by_temp[ref_temp],
            num_embeddings,
            output_path / f"token_distribution_T{ref_temp}.png",
        )

    # Token transition matrix (real data)
    if real_token_indices_list:
        all_real_idx = np.concatenate(real_token_indices_list, axis=0)
        plot_token_transition_matrix(
            all_real_idx, num_embeddings,
            output_path / "token_transition_real.png",
        )

    # Generated ECG grids per temperature
    for temp, samples_np in samples_by_temp.items():
        plot_ecg_grid(
            samples_np,
            output_path / f"generated_grid_T{temp}.png",
            title=f"Generated ECGs — Temperature={temp}",
        )
        plot_single_ecg(
            samples_np[0],
            output_path / f"generated_single_T{temp}.png",
            title=f"Generated ECG (Sample 1) — Temperature={temp}",
        )

    # Temperature comparison side-by-side
    if len(samples_by_temp) > 1:
        plot_temperature_comparison(samples_by_temp, output_path / "temperature_comparison.png")

    # Repetition rate bar chart
    if gen_rep_by_temp and real_rep_rates:
        plot_repetition_score(real_rep_rates, gen_rep_by_temp, output_path / "repetition_rate.png")

    # ── Save JSON summary ─────────────────────────────────────────────────────
    summary = {
        "checkpoint": prior_checkpoint,
        "num_embeddings": num_embeddings,
        "mean_nll": mean_nll,
        "median_nll": median_nll,
        "real_token_entropy": real_entropy,
        "real_used_codes": real_used_codes,
        "real_repetition_rate": mean_real_rep,
        "generated": {
            str(temp): {
                "entropy": float(gen_entropy_by_temp.get(temp, float("nan"))),
                "repetition_rate": float(np.mean(gen_rep_by_temp.get(temp, [float("nan")]))),
                "unique_tokens": len(gen_token_by_temp.get(temp, {})),
            }
            for temp in temperatures
        },
        "issues_found": issues_found,
        "n_batches_evaluated": n_batches,
    }

    summary_path = output_path / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved JSON summary: {summary_path}")

    print(f"\n{'='*70}")
    print(f"  ✓ Prior diagnostics complete — results in: {output_path}")
    print(f"{'='*70}\n")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PixelCNN Prior Quality Diagnostics"
    )
    parser.add_argument("--prior-checkpoint", type=str, required=True,
                        help="Path to trained Prior checkpoint (.ckpt)")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to MIMIC-IV-ECG dataset directory")
    parser.add_argument("--n-batches", type=int, default=20,
                        help="Validation batches to evaluate (default: 20)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--seq-length", type=int, default=5000,
                        help="ECG sequence length (default: 5000)")
    parser.add_argument("--output-dir", type=str, default="prior_diagnostics",
                        help="Output directory (default: prior_diagnostics)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--temperatures", type=float, nargs="+",
                        default=[0.5, 0.8, 1.0, 1.2],
                        help="Temperatures to evaluate (default: 0.5 0.8 1.0 1.2)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer temperatures, skip slower plots")

    args = parser.parse_args()

    if not os.path.exists(args.prior_checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.prior_checkpoint}")
        sys.exit(1)
    if not os.path.isdir(args.data_dir):
        print(f"[ERROR] Data directory not found: {args.data_dir}")
        sys.exit(1)

    run_prior_diagnostics(
        prior_checkpoint=args.prior_checkpoint,
        data_dir=args.data_dir,
        n_batches=args.n_batches,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        output_dir=args.output_dir,
        device=args.device,
        quick=args.quick,
        val_split=args.val_split,
        test_split=args.test_split,
        num_workers=args.num_workers,
        temperatures=args.temperatures,
    )


if __name__ == "__main__":
    main()