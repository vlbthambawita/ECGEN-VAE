#!/usr/bin/env python3
"""
Generate ECG samples from trained VQ-VAE + Prior models.

Usage:
    # Generate samples with default settings
    python generate_samples.py --prior-checkpoint runs/prior_mimic_standalone/seed_42/checkpoints/best.ckpt
    
    # Generate more samples with different temperature
    python generate_samples.py \
        --prior-checkpoint runs/prior_mimic_standalone/seed_42/checkpoints/best.ckpt \
        --n-samples 32 \
        --temperature 0.8 \
        --output-dir generated_samples
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from train_vqvae_standalone import PriorLightning


def plot_ecg_samples(
    samples: np.ndarray,
    save_path: Path,
    n_cols: int = 4,
    lead_names: Optional[list] = None,
) -> None:
    """Plot ECG samples in a grid layout.
    
    Args:
        samples: ECG samples [N, 12, 5000]
        save_path: Path to save the figure
        n_cols: Number of columns in the grid
        lead_names: Names of ECG leads
    """
    if lead_names is None:
        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    
    n_samples = samples.shape[0]
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx in range(n_samples):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        ecg = samples[idx]  # [12, 5000]
        
        # Plot all 12 leads stacked vertically with offset
        offset = 0
        for lead_idx in range(12):
            signal = ecg[lead_idx] + offset
            ax.plot(signal, linewidth=0.5, alpha=0.8, label=lead_names[lead_idx])
            offset -= 3  # Offset for next lead
        
        ax.set_title(f"Sample {idx + 1}")
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("Amplitude (normalized)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='right', fontsize=6, ncol=2)
    
    # Hide empty subplots
    for idx in range(n_samples, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {save_path}")


def plot_single_ecg(
    ecg: np.ndarray,
    save_path: Path,
    lead_names: Optional[list] = None,
    title: str = "Generated ECG",
) -> None:
    """Plot a single ECG with all 12 leads.
    
    Args:
        ecg: Single ECG [12, 5000]
        save_path: Path to save the figure
        lead_names: Names of ECG leads
        title: Title for the plot
    """
    if lead_names is None:
        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    
    fig, axes = plt.subplots(12, 1, figsize=(15, 12))
    
    for lead_idx in range(12):
        ax = axes[lead_idx]
        signal = ecg[lead_idx]
        ax.plot(signal, linewidth=0.8, color='black')
        ax.set_ylabel(lead_names[lead_idx], fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, len(signal))
        
        if lead_idx < 11:
            ax.set_xticks([])
        else:
            ax.set_xlabel("Time (samples)", fontsize=10)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved detailed plot: {save_path}")


def save_samples_npy(samples: np.ndarray, save_path: Path) -> None:
    """Save samples as numpy array."""
    np.save(save_path, samples)
    print(f"Saved numpy array: {save_path}")


def generate_samples(
    prior_checkpoint: str,
    n_samples: int = 16,
    temperature: float = 1.0,
    seq_length: int = 5000,
    output_dir: str = "generated_samples",
    device: str = "cuda",
    save_npy: bool = True,
    plot_grid: bool = True,
    plot_individual: bool = True,
) -> np.ndarray:
    """Generate ECG samples from trained Prior model.
    
    Args:
        prior_checkpoint: Path to trained Prior checkpoint
        n_samples: Number of samples to generate
        temperature: Sampling temperature (higher = more diverse)
        seq_length: Length of ECG sequence
        output_dir: Directory to save outputs
        device: Device to use for generation
        save_npy: Save samples as numpy array
        plot_grid: Plot samples in a grid
        plot_individual: Plot each sample individually
        
    Returns:
        Generated samples as numpy array [N, 12, seq_length]
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading Prior model from: {prior_checkpoint}")
    model = PriorLightning.load_from_checkpoint(prior_checkpoint)
    model.eval()
    model.to(device)
    
    if model.vqvae is None:
        raise RuntimeError("VQ-VAE not loaded in Prior model")
    
    print(f"\nGenerating {n_samples} samples...")
    print(f"  Temperature: {temperature}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Device: {device}")
    
    # Generate samples
    with torch.no_grad():
        samples = model.sample(
            n_samples=n_samples,
            seq_length=seq_length,
            temperature=temperature,
        )
    
    # Convert to numpy
    samples_np = samples.cpu().numpy()
    print(f"Generated samples shape: {samples_np.shape}")
    
    # Save as numpy array
    if save_npy:
        npy_path = output_path / f"samples_n{n_samples}_t{temperature:.2f}.npy"
        save_samples_npy(samples_np, npy_path)
    
    # Plot grid of samples
    if plot_grid:
        grid_path = output_path / f"samples_grid_n{n_samples}_t{temperature:.2f}.png"
        plot_ecg_samples(samples_np, grid_path, n_cols=4)
    
    # Plot individual samples
    if plot_individual:
        individual_dir = output_path / "individual"
        individual_dir.mkdir(exist_ok=True)
        
        for idx in range(min(n_samples, 10)):  # Save first 10 individual plots
            sample_path = individual_dir / f"sample_{idx:03d}_t{temperature:.2f}.png"
            plot_single_ecg(
                samples_np[idx],
                sample_path,
                title=f"Generated ECG Sample {idx + 1} (T={temperature:.2f})"
            )
    
    print(f"\n✓ Generation complete!")
    print(f"  Output directory: {output_path}")
    print(f"  Total samples: {n_samples}")
    
    return samples_np


def compute_statistics(samples: np.ndarray) -> dict:
    """Compute basic statistics of generated samples."""
    stats = {
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples)),
        "min": float(np.min(samples)),
        "max": float(np.max(samples)),
        "median": float(np.median(samples)),
    }
    
    # Per-lead statistics
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    lead_stats = {}
    for lead_idx, lead_name in enumerate(lead_names):
        lead_data = samples[:, lead_idx, :]
        lead_stats[lead_name] = {
            "mean": float(np.mean(lead_data)),
            "std": float(np.std(lead_data)),
            "min": float(np.min(lead_data)),
            "max": float(np.max(lead_data)),
        }
    
    stats["per_lead"] = lead_stats
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate ECG samples from trained VQ-VAE + Prior models"
    )
    
    # Required arguments
    parser.add_argument(
        "--prior-checkpoint",
        type=str,
        required=True,
        help="Path to trained Prior checkpoint (.ckpt file)",
    )
    
    # Generation settings
    parser.add_argument(
        "--n-samples",
        type=int,
        default=16,
        help="Number of samples to generate (default: 16)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature - higher = more diverse (default: 1.0)",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=5000,
        help="ECG sequence length (default: 5000)",
    )
    
    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_samples",
        help="Directory to save generated samples (default: generated_samples)",
    )
    parser.add_argument(
        "--no-npy",
        action="store_true",
        help="Don't save samples as numpy array",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Don't plot samples in a grid",
    )
    parser.add_argument(
        "--no-individual",
        action="store_true",
        help="Don't plot individual samples",
    )
    
    # Device settings
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    
    # Multiple temperatures
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        help="Generate samples with multiple temperatures (e.g., --temperatures 0.5 0.8 1.0 1.2)",
    )
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.prior_checkpoint):
        print(f"Error: Prior checkpoint not found: {args.prior_checkpoint}")
        print("\nAvailable checkpoints:")
        import subprocess
        result = subprocess.run(
            ["find", "runs", "-path", "*/prior*/seed_*/checkpoints/*.ckpt", "-type", "f"],
            capture_output=True,
            text=True,
        )
        if result.stdout:
            for line in result.stdout.strip().split("\n")[:5]:
                print(f"  - {line}")
        return
    
    # Generate with multiple temperatures if specified
    if args.temperatures:
        print(f"Generating samples with temperatures: {args.temperatures}")
        for temp in args.temperatures:
            print(f"\n{'='*80}")
            print(f"Temperature: {temp}")
            print('='*80)
            generate_samples(
                prior_checkpoint=args.prior_checkpoint,
                n_samples=args.n_samples,
                temperature=temp,
                seq_length=args.seq_length,
                output_dir=args.output_dir,
                device=args.device,
                save_npy=not args.no_npy,
                plot_grid=not args.no_grid,
                plot_individual=not args.no_individual,
            )
    else:
        # Generate with single temperature
        samples = generate_samples(
            prior_checkpoint=args.prior_checkpoint,
            n_samples=args.n_samples,
            temperature=args.temperature,
            seq_length=args.seq_length,
            output_dir=args.output_dir,
            device=args.device,
            save_npy=not args.no_npy,
            plot_grid=not args.no_grid,
            plot_individual=not args.no_individual,
        )
        
        # Print statistics
        print("\nSample Statistics:")
        stats = compute_statistics(samples)
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std:  {stats['std']:.4f}")
        print(f"  Min:  {stats['min']:.4f}")
        print(f"  Max:  {stats['max']:.4f}")
        print(f"  Median: {stats['median']:.4f}")


if __name__ == "__main__":
    main()
