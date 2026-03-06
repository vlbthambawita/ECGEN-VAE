#!/usr/bin/env python3
"""
VQ-VAE Quality Diagnostic Script

Checks:
  1. Codebook perplexity (codebook utilization)
  2. Reconstruction quality (MSE, MAE per lead)
  3. Codebook usage distribution (which codes are actually used)
  4. Visual reconstruction comparison plots

Usage:
    python check_vqvae_quality.py \
        --vqvae-checkpoint runs/vqvae_mimic_standalone/seed_42/checkpoints/best.ckpt \
        --data-dir /path/to/mimic-iv-ecg \
        --n-batches 20 \
        --output-dir diagnostics

    # Quick mode (fewer batches, faster)
    python check_vqvae_quality.py \
        --vqvae-checkpoint runs/vqvae_mimic_standalone/seed_42/checkpoints/best.ckpt \
        --data-dir /path/to/mimic-iv-ecg \
        --n-batches 5 \
        --quick
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── Import your training module ──────────────────────────────────────────────
try:
    from train_vqvae_standalone import VQVAELightning, MIMICIVECGDataset
except ImportError:
    print("[ERROR] Could not import from train_vqvae_standalone.py")
    print("        Make sure this script is in the same directory as train_vqvae_standalone.py")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

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


def plot_reconstruction_comparison(
    original: np.ndarray,
    reconstructed: np.ndarray,
    save_path: Path,
    sample_idx: int = 0,
    title_suffix: str = "",
) -> None:
    """
    Plot original vs reconstructed ECG side by side, all 12 leads.

    Args:
        original:      [B, 12, T]
        reconstructed: [B, 12, T]
        save_path:     where to save the figure
        sample_idx:    which sample in the batch to plot
    """
    orig = original[sample_idx]   # [12, T]
    recon = reconstructed[sample_idx]  # [12, T]

    fig, axes = plt.subplots(12, 2, figsize=(20, 18))
    fig.suptitle(f"VQ-VAE Reconstruction Quality {title_suffix}", fontsize=14, fontweight="bold")

    for lead_idx in range(12):
        # Original
        ax_orig = axes[lead_idx, 0]
        ax_orig.plot(orig[lead_idx], linewidth=0.7, color="black")
        ax_orig.set_ylabel(LEAD_NAMES[lead_idx], fontsize=9, fontweight="bold")
        ax_orig.set_xlim(0, orig.shape[1])
        ax_orig.grid(True, alpha=0.3)
        if lead_idx == 0:
            ax_orig.set_title("Original", fontsize=11, fontweight="bold")
        if lead_idx < 11:
            ax_orig.set_xticks([])

        # Reconstructed
        ax_recon = axes[lead_idx, 1]
        ax_recon.plot(recon[lead_idx], linewidth=0.7, color="steelblue")
        ax_recon.set_xlim(0, recon.shape[1])
        ax_recon.grid(True, alpha=0.3)
        if lead_idx == 0:
            ax_recon.set_title("Reconstructed", fontsize=11, fontweight="bold")
        if lead_idx < 11:
            ax_recon.set_xticks([])
        else:
            ax_orig.set_xlabel("Time (samples)", fontsize=9)
            ax_recon.set_xlabel("Time (samples)", fontsize=9)

        # Per-lead MSE in title
        mse = float(np.mean((orig[lead_idx] - recon[lead_idx]) ** 2))
        axes[lead_idx, 1].set_ylabel(f"MSE={mse:.4f}", fontsize=7, color="gray")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_reconstruction_overlay(
    original: np.ndarray,
    reconstructed: np.ndarray,
    save_path: Path,
    sample_idx: int = 0,
) -> None:
    """
    Plot original (black) and reconstructed (red) overlaid for quick visual check.
    """
    orig = original[sample_idx]   # [12, T]
    recon = reconstructed[sample_idx]

    fig, axes = plt.subplots(12, 1, figsize=(18, 14))
    fig.suptitle("Original (black) vs Reconstructed (red) — Overlay", fontsize=13, fontweight="bold")

    for lead_idx in range(12):
        ax = axes[lead_idx]
        ax.plot(orig[lead_idx], linewidth=0.8, color="black", label="Original", alpha=0.9)
        ax.plot(recon[lead_idx], linewidth=0.8, color="red", label="Reconstructed", alpha=0.7)
        ax.set_ylabel(LEAD_NAMES[lead_idx], fontsize=9, fontweight="bold")
        ax.set_xlim(0, orig.shape[1])
        ax.grid(True, alpha=0.2)
        if lead_idx == 0:
            ax.legend(fontsize=8, loc="upper right")
        if lead_idx < 11:
            ax.set_xticks([])
        else:
            ax.set_xlabel("Time (samples)", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_codebook_usage(
    code_counts: Counter,
    num_embeddings: int,
    save_path: Path,
) -> None:
    """
    Bar chart of codebook usage — reveals dead codes.
    """
    counts = np.zeros(num_embeddings, dtype=np.int64)
    for code_idx, count in code_counts.items():
        if code_idx < num_embeddings:
            counts[code_idx] = count

    used = int(np.sum(counts > 0))
    dead = num_embeddings - used
    usage_pct = 100.0 * used / num_embeddings

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(
        f"Codebook Utilization  |  Used: {used}/{num_embeddings} ({usage_pct:.1f}%)  |  Dead codes: {dead}",
        fontsize=13, fontweight="bold"
    )

    # Full bar chart
    ax1 = axes[0]
    ax1.bar(range(num_embeddings), counts, width=1.0, color="steelblue", alpha=0.8)
    ax1.set_xlabel("Codebook Index", fontsize=10)
    ax1.set_ylabel("Usage Count", fontsize=10)
    ax1.set_title("All Codes", fontsize=10)
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.grid(True, alpha=0.3, axis="y")

    # Sorted usage (descending) — shows how skewed the distribution is
    ax2 = axes[1]
    sorted_counts = np.sort(counts)[::-1]
    ax2.bar(range(num_embeddings), sorted_counts, width=1.0, color="darkorange", alpha=0.8)
    ax2.set_xlabel("Rank (most used → least used)", fontsize=10)
    ax2.set_ylabel("Usage Count", fontsize=10)
    ax2.set_title("Sorted by Usage (ideal = flat)", fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_per_lead_metrics(
    per_lead_mse: np.ndarray,
    per_lead_mae: np.ndarray,
    save_path: Path,
) -> None:
    """Bar chart of MSE and MAE per ECG lead."""
    x = np.arange(12)
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Reconstruction Error per Lead", fontsize=13, fontweight="bold")

    ax1 = axes[0]
    bars = ax1.bar(x, per_lead_mse, width=0.6, color="steelblue", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(LEAD_NAMES, fontsize=10)
    ax1.set_ylabel("MSE", fontsize=10)
    ax1.set_title("Mean Squared Error per Lead", fontsize=11)
    ax1.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, per_lead_mse):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=7)

    ax2 = axes[1]
    bars2 = ax2.bar(x, per_lead_mae, width=0.6, color="coral", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(LEAD_NAMES, fontsize=10)
    ax2.set_ylabel("MAE", fontsize=10)
    ax2.set_title("Mean Absolute Error per Lead", fontsize=11)
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars2, per_lead_mae):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_error_over_time(
    original: np.ndarray,
    reconstructed: np.ndarray,
    save_path: Path,
    n_samples: int = 50,
) -> None:
    """
    Plot mean absolute error as a function of time position.
    Reveals if errors are localised or spread uniformly.
    """
    n = min(n_samples, original.shape[0])
    orig = original[:n]    # [n, 12, T]
    recon = reconstructed[:n]

    mae_time = np.mean(np.abs(orig - recon), axis=(0, 1))  # [T]

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(mae_time, linewidth=0.8, color="steelblue")
    ax.set_xlabel("Time (samples)", fontsize=11)
    ax.set_ylabel("Mean Absolute Error", fontsize=11)
    ax.set_title(f"Reconstruction Error over Time (avg over {n} samples, all 12 leads)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.fill_between(range(len(mae_time)), mae_time, alpha=0.2, color="steelblue")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Core diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def run_diagnostics(
    vqvae_checkpoint: str,
    data_dir: str,
    n_batches: int = 20,
    batch_size: int = 16,
    seq_length: int = 5000,
    output_dir: str = "diagnostics",
    device: str = "cuda",
    quick: bool = False,
    val_split: float = 0.1,
    test_split: float = 0.1,
    num_workers: int = 4,
) -> dict:
    """
    Run full VQ-VAE quality diagnostics.

    Returns dict with summary metrics.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Loading VQ-VAE from: {vqvae_checkpoint}")
    print(f"{'='*70}")

    model = VQVAELightning.load_from_checkpoint(vqvae_checkpoint)
    model.eval()
    model.to(device)

    # Try to get codebook size from model
    try:
        num_embeddings = model.vqvae.vq.num_embeddings
    except AttributeError:
        num_embeddings = getattr(model.hparams, "num_embeddings", 512)
        if num_embeddings == 512:
            print(f"  [WARN] Could not read num_embeddings from model, using {num_embeddings}")

    print(f"  Codebook size (num_embeddings): {num_embeddings}")
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
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    print(f"  Validation set size: {len(dataset)}")
    print(f"  Will process {n_batches} batches × {batch_size} = up to {n_batches * batch_size} samples")

    # ── Run inference ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  Running inference...")
    print(f"{'='*70}")

    all_mse:      list[float] = []
    all_mae:      list[float] = []
    all_perplexities: list[float] = []
    per_lead_mse  = np.zeros(12)
    per_lead_mae  = np.zeros(12)
    code_counter  = Counter()

    # Collect originals & reconstructions for plotting
    plot_originals:     list[np.ndarray] = []
    plot_reconstructed: list[np.ndarray] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= n_batches:
                break

            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            x = x.to(device).float()   # [B, 12, T]

            # Forward pass (returns recon, vq_loss, indices)
            output = model(x)
            x_recon = output[0] if isinstance(output, tuple) else output
            vq_loss = output[1] if isinstance(output, tuple) and len(output) >= 2 else None
            indices = output[2] if isinstance(output, tuple) and len(output) >= 3 else None

            # Metrics
            mse = F.mse_loss(x_recon, x, reduction="none")  # [B, 12, T]
            mae = torch.abs(x_recon - x)                    # [B, 12, T]

            batch_mse = float(mse.mean().cpu())
            batch_mae = float(mae.mean().cpu())
            all_mse.append(batch_mse)
            all_mae.append(batch_mae)

            # Per-lead
            per_lead_mse += mse.mean(dim=(0, 2)).cpu().numpy()  # [12]
            per_lead_mae += mae.mean(dim=(0, 2)).cpu().numpy()

            # Perplexity: compute from indices (exp of entropy of code distribution)
            batch_perplexity = float("nan")
            if indices is not None:
                flat = indices.cpu().numpy().flatten()
                counts = np.bincount(flat, minlength=num_embeddings)
                probs = counts / (counts.sum() + 1e-10)
                probs = probs[probs > 0]
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                batch_perplexity = float(np.exp(entropy))
                all_perplexities.append(batch_perplexity)
                code_counter.update(flat.tolist())

            # Collect samples for plotting (first 8 per batch)
            if len(plot_originals) < 8:
                plot_originals.append(x.cpu().numpy())
                plot_reconstructed.append(x_recon.cpu().numpy())

            if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
                perp_str = f"{batch_perplexity:.1f}" if not np.isnan(batch_perplexity) else "N/A"
                print(f"  Batch {batch_idx + 1:3d}/{n_batches}  |  "
                      f"MSE={batch_mse:.5f}  MAE={batch_mae:.5f}  Perplexity={perp_str}")

    # ── Aggregate ──────────────────────────────────────────────────────────────
    per_lead_mse /= n_batches
    per_lead_mae /= n_batches

    mean_mse = float(np.mean(all_mse))
    mean_mae = float(np.mean(all_mae))
    mean_perplexity = float(np.mean(all_perplexities)) if all_perplexities else float("nan")

    # Codebook utilization
    if len(code_counter) > 0:
        used_codes = len(code_counter)
        dead_codes = num_embeddings - used_codes
        utilization_pct = 100.0 * used_codes / num_embeddings
    else:
        used_codes = -1
        dead_codes = -1
        utilization_pct = float("nan")

    # ── Print Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  DIAGNOSTIC SUMMARY")
    print(f"{'='*70}")
    print(f"\n  📊 Reconstruction Quality")
    print(f"     Mean MSE:  {mean_mse:.6f}")
    print(f"     Mean MAE:  {mean_mae:.6f}")
    print(f"\n  🗂️  Codebook Utilization")
    print(f"     Perplexity:        {mean_perplexity:.1f}  (ideal ≈ {num_embeddings * 0.6:.0f}–{num_embeddings:.0f})")

    if used_codes > 0:
        print(f"     Used codes:        {used_codes} / {num_embeddings}  ({utilization_pct:.1f}%)")
        print(f"     Dead codes:        {dead_codes}")

    print(f"\n  📈 Per-Lead MSE")
    for lead_name, mse_val in zip(LEAD_NAMES, per_lead_mse):
        bar = "█" * int(mse_val * 2000)
        print(f"     {lead_name:>4s}:  {mse_val:.6f}  {bar}")

    # ── Diagnosis ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  DIAGNOSIS")
    print(f"{'='*70}")

    issues_found = 0

    if not np.isnan(mean_perplexity):
        if mean_perplexity < num_embeddings * 0.05:
            print(f"\n  🔴 CRITICAL — Codebook collapsed!")
            print(f"     Perplexity={mean_perplexity:.1f} << {num_embeddings * 0.05:.0f} (5% of codebook)")
            print(f"     → Only a tiny fraction of codes are being used")
            print(f"     → Reconstructions and generations will look patchy/discontinuous")
            print(f"     FIXES: increase commitment_cost (try 0.5–1.0), increase num_embeddings,")
            print(f"            try EMA codebook updates, reduce learning rate")
            issues_found += 1
        elif mean_perplexity < num_embeddings * 0.2:
            print(f"\n  🟡 WARNING — Low codebook utilization")
            print(f"     Perplexity={mean_perplexity:.1f} (only {100*mean_perplexity/num_embeddings:.0f}% of codebook used)")
            print(f"     → Some codebook collapse, may limit generation diversity")
            print(f"     FIXES: increase commitment_cost slightly, train longer")
            issues_found += 1
        else:
            print(f"\n  ✅ Codebook utilization OK (perplexity={mean_perplexity:.1f})")

    if used_codes > 0:
        if utilization_pct < 20:
            print(f"\n  🔴 CRITICAL — Only {utilization_pct:.1f}% of codes used ({used_codes}/{num_embeddings})")
            issues_found += 1
        elif utilization_pct < 50:
            print(f"\n  🟡 WARNING — {utilization_pct:.1f}% of codes used — consider larger codebook")
            issues_found += 1
        else:
            print(f"\n  ✅ {utilization_pct:.1f}% of codes in use")

    if mean_mse > 0.05:
        print(f"\n  🔴 CRITICAL — Very high reconstruction MSE ({mean_mse:.4f})")
        print(f"     → VQ-VAE is the bottleneck, not just the prior")
        print(f"     FIXES: increase base_channels (try 128), num_res_blocks (try 4),")
        print(f"            latent_channels (try 128), train longer")
        issues_found += 1
    elif mean_mse > 0.01:
        print(f"\n  🟡 WARNING — Moderate reconstruction MSE ({mean_mse:.4f})")
        print(f"     → Reconstructions may miss fine ECG morphology")
        issues_found += 1
    else:
        print(f"\n  ✅ Reconstruction MSE looks good ({mean_mse:.6f})")

    worst_lead_idx = int(np.argmax(per_lead_mse))
    best_lead_idx  = int(np.argmin(per_lead_mse))
    print(f"\n  📋 Worst reconstructed lead: {LEAD_NAMES[worst_lead_idx]} (MSE={per_lead_mse[worst_lead_idx]:.6f})")
    print(f"     Best reconstructed lead:  {LEAD_NAMES[best_lead_idx]}  (MSE={per_lead_mse[best_lead_idx]:.6f})")

    if issues_found == 0:
        print(f"\n  ✅ VQ-VAE looks healthy — if generations are still poor,")
        print(f"     the problem is in the Prior (too small, not trained enough)")
        print(f"     → Try: hidden_dim=512, num_layers=10, more epochs, lower LR")
    else:
        print(f"\n  ⚠️  {issues_found} issue(s) found — fix VQ-VAE before retraining Prior")

    # ── Save plots ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Saving diagnostic plots to: {output_path}")
    print(f"{'='*70}\n")

    all_orig_np  = np.concatenate(plot_originals,     axis=0)
    all_recon_np = np.concatenate(plot_reconstructed, axis=0)

    # 1. Side-by-side comparison (first 4 samples)
    for i in range(min(4, all_orig_np.shape[0])):
        plot_reconstruction_comparison(
            all_orig_np, all_recon_np,
            save_path=output_path / f"recon_sidebyside_sample{i:02d}.png",
            sample_idx=i,
            title_suffix=f"(Sample {i})",
        )

    # 2. Overlay comparison (first 4 samples)
    for i in range(min(4, all_orig_np.shape[0])):
        plot_reconstruction_overlay(
            all_orig_np, all_recon_np,
            save_path=output_path / f"recon_overlay_sample{i:02d}.png",
            sample_idx=i,
        )

    # 3. Codebook usage
    if len(code_counter) > 0:
        plot_codebook_usage(
            code_counter, num_embeddings,
            save_path=output_path / "codebook_usage.png",
        )

    # 4. Per-lead metrics
    plot_per_lead_metrics(
        per_lead_mse, per_lead_mae,
        save_path=output_path / "per_lead_metrics.png",
    )

    # 5. Error over time
    if not quick:
        plot_error_over_time(
            all_orig_np, all_recon_np,
            save_path=output_path / "error_over_time.png",
            n_samples=min(50, all_orig_np.shape[0]),
        )

    # ── Save text summary ─────────────────────────────────────────────────────
    summary = {
        "checkpoint": vqvae_checkpoint,
        "num_embeddings": num_embeddings,
        "mean_mse": mean_mse,
        "mean_mae": mean_mae,
        "mean_perplexity": mean_perplexity,
        "used_codes": used_codes,
        "dead_codes": dead_codes,
        "utilization_pct": utilization_pct,
        "per_lead_mse": {LEAD_NAMES[i]: float(per_lead_mse[i]) for i in range(12)},
        "per_lead_mae": {LEAD_NAMES[i]: float(per_lead_mae[i]) for i in range(12)},
        "n_batches_evaluated": n_batches,
        "n_samples_evaluated": n_batches * batch_size,
        "issues_found": issues_found,
    }

    import json
    summary_path = output_path / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved JSON summary: {summary_path}")

    print(f"\n{'='*70}")
    print(f"  ✓ Diagnostics complete — results in: {output_path}")
    print(f"{'='*70}\n")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VQ-VAE Quality Diagnostics — perplexity, reconstruction, codebook usage"
    )

    parser.add_argument(
        "--vqvae-checkpoint", type=str, required=True,
        help="Path to trained VQ-VAE checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Path to MIMIC-IV-ECG dataset directory"
    )
    parser.add_argument(
        "--n-batches", type=int, default=20,
        help="Number of batches to evaluate (default: 20)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for evaluation (default: 16)"
    )
    parser.add_argument(
        "--seq-length", type=int, default=5000,
        help="ECG sequence length (default: 5000)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="diagnostics",
        help="Directory to save diagnostic outputs (default: diagnostics)"
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: cuda or cpu"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader workers (default: 4)"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1,
        help="Validation split fraction (default: 0.1)"
    )
    parser.add_argument(
        "--test-split", type=float, default=0.1,
        help="Test split fraction (default: 0.1)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: skip slower plots (error-over-time)"
    )

    args = parser.parse_args()

    # Validate checkpoint
    if not os.path.exists(args.vqvae_checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.vqvae_checkpoint}")
        sys.exit(1)

    # Validate data dir
    if not os.path.isdir(args.data_dir):
        print(f"[ERROR] Data directory not found: {args.data_dir}")
        sys.exit(1)

    run_diagnostics(
        vqvae_checkpoint=args.vqvae_checkpoint,
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
    )


if __name__ == "__main__":
    main()