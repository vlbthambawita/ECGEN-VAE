#!/usr/bin/env python3
"""
ECG Plotting Script for VQ-VAE-2 Generated Samples
===================================================
Visualize generated ECG samples using the ecg-plot library.

Usage:
    python plot_ecgs.py --input generated_ecgs.npy --output-dir plots --n-samples 8
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt

try:
    import ecg_plot
except ImportError:
    print("ERROR: ecg-plot library not found. Install with: pip install ecg-plot")
    sys.exit(1)


def validate_ecg_data(ecg_data: np.ndarray) -> None:
    """
    Validate ECG data shape and values.
    
    Args:
        ecg_data: ECG array to validate
        
    Raises:
        ValueError: If data shape is invalid
    """
    if ecg_data.ndim != 3:
        raise ValueError(
            f"Expected 3D array (N, 12, L), got {ecg_data.ndim}D array with shape {ecg_data.shape}"
        )
    
    n_samples, n_leads, seq_len = ecg_data.shape
    
    if n_leads != 12:
        raise ValueError(
            f"Expected 12 leads, got {n_leads} leads. "
            f"ECG data should have shape (N, 12, L)"
        )
    
    if seq_len < 1000:
        raise ValueError(
            f"Sequence length too short: {seq_len}. "
            f"Expected at least 1000 samples for meaningful ECG visualization"
        )
    
    print(f"✓ Validated ECG data: {n_samples} samples, {n_leads} leads, {seq_len} samples per lead")


def plot_single_ecg(
    ecg_data: np.ndarray,
    idx: int,
    output_path: str,
    sample_rate: int = 500,
    style: Optional[str] = None,
    columns: int = 2,
    title_prefix: str = "Generated ECG",
) -> None:
    """
    Plot a single 12-lead ECG and save to file.
    
    Args:
        ecg_data: ECG data of shape (12, L)
        idx: Sample index for title
        output_path: Path to save the plot
        sample_rate: Sampling rate in Hz
        style: Plot style ('bw' for black/white, None for standard)
        columns: Number of columns in plot layout
        title_prefix: Prefix for plot title
    """
    # Ensure data is in correct shape (12, L)
    if ecg_data.shape[0] != 12:
        if ecg_data.shape[1] == 12:
            ecg_data = ecg_data.T
        else:
            raise ValueError(f"Invalid ECG shape: {ecg_data.shape}")
    
    # Create the plot
    title = f"{title_prefix} #{idx:04d}"
    
    ecg_plot.plot(
        ecg=ecg_data,
        sample_rate=sample_rate,
        title=title,
        lead_index=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
        lead_order=None,
        style=style,
        columns=columns,
        row_height=6,
        show_lead_name=True,
        show_grid=True,
        show_separate_line=True,
    )
    
    # Save the plot
    ecg_plot.save_as_png(output_path)
    plt.close('all')  # Close all figures to free memory


def plot_ecgs(
    input_file: str,
    output_dir: str,
    n_samples: Optional[int] = None,
    sample_rate: int = 500,
    prefix: str = "ecg_",
    style: Optional[str] = None,
    columns: int = 2,
) -> None:
    """
    Plot multiple ECG samples from a numpy file.
    
    Args:
        input_file: Path to .npy file containing ECG data
        output_dir: Directory to save plots
        n_samples: Number of samples to plot (None for all)
        sample_rate: Sampling rate in Hz
        prefix: Filename prefix for saved plots
        style: Plot style ('bw' for black/white, None for standard)
        columns: Number of columns in plot layout
    """
    # Load ECG data
    print(f"Loading ECG data from: {input_file}")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    ecg_data = np.load(input_file)
    print(f"Loaded ECG data with shape: {ecg_data.shape}")
    
    # Validate data
    validate_ecg_data(ecg_data)
    
    # Determine number of samples to plot
    total_samples = ecg_data.shape[0]
    if n_samples is None:
        n_samples = total_samples
    else:
        n_samples = min(n_samples, total_samples)
    
    print(f"Plotting {n_samples} out of {total_samples} ECG samples")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    
    # Plot each ECG
    print("\nGenerating plots...")
    for i in range(n_samples):
        # Note: ecg_plot.save_as_png() adds .png extension automatically
        output_path = output_dir / f"{prefix}{i:04d}"
        
        try:
            plot_single_ecg(
                ecg_data=ecg_data[i],
                idx=i,
                output_path=str(output_path),
                sample_rate=sample_rate,
                style=style,
                columns=columns,
            )
            
            # Progress indicator
            if (i + 1) % 10 == 0 or i == n_samples - 1:
                print(f"  Progress: {i + 1}/{n_samples} plots generated")
                
        except Exception as e:
            print(f"  WARNING: Failed to plot sample {i}: {e}")
            continue
    
    print(f"\n✓ Successfully generated {n_samples} ECG plots in: {output_dir.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot generated ECG samples using ecg-plot library",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default="generated_ecgs.npy",
        help="Path to .npy file containing ECG data (shape: N x 12 x L)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory to save generated plots",
    )
    
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of samples to plot (default: all)",
    )
    
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=500,
        help="ECG sampling rate in Hz",
    )
    
    parser.add_argument(
        "--prefix",
        type=str,
        default="ecg_",
        help="Filename prefix for saved plots",
    )
    
    parser.add_argument(
        "--style",
        type=str,
        choices=[None, "bw"],
        default=None,
        help="Plot style (None for standard colors, 'bw' for black/white)",
    )
    
    parser.add_argument(
        "--columns",
        type=int,
        default=2,
        help="Number of columns in plot layout",
    )
    
    args = parser.parse_args()
    
    try:
        plot_ecgs(
            input_file=args.input,
            output_dir=args.output_dir,
            n_samples=args.n_samples,
            sample_rate=args.sample_rate,
            prefix=args.prefix,
            style=args.style,
            columns=args.columns,
        )
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
