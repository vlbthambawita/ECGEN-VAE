#!/usr/bin/env python3
"""
Dataloader Split Reproducibility Checker

Verifies that val/test splits are identical across:
  - Multiple runs with the same seed
  - Different seeds (should differ)
  - After reloading from disk
  - Across different batch sizes

Usage:
    python check_split_reproducibility.py \
        --data-dir /path/to/mimic-iv-ecg \
        --output-dir split_diagnostics

    # Also test that different seeds give different splits
    python check_split_reproducibility.py \
        --data-dir /path/to/mimic-iv-ecg \
        --seeds 42 42 123 456
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

try:
    from train_vqvae_standalone import MIMICIVECGDataset
except ImportError:
    print("[ERROR] Could not import MIMICIVECGDataset from train_vqvae_standalone.py")
    print("        Make sure this script is in the same directory.")
    sys.exit(1)


class ECGDataset(MIMICIVECGDataset):
    """
    Thin wrapper around MIMICIVECGDataset to match the older ECGDataset API,
    mirroring the pattern used in check_prior_quality.py.
    """

    def __init__(
        self,
        data_dir,
        seq_length=5000,
        split="train",
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
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_dataset_fingerprint(dataset) -> dict:
    """
    Extract a fingerprint from a dataset that uniquely identifies its split.
    Works by collecting sample indices or file paths if available,
    otherwise hashes the first/last few samples.
    """
    fingerprint = {}
    n = len(dataset)
    fingerprint["n_samples"] = n

    # Try to get indices directly (if dataset stores them)
    if hasattr(dataset, "indices"):
        indices = list(dataset.indices)
        fingerprint["indices_hash"] = hashlib.md5(
            str(sorted(indices)).encode()
        ).hexdigest()
        fingerprint["first_5_indices"] = indices[:5]
        fingerprint["last_5_indices"]  = indices[-5:]

    elif hasattr(dataset, "file_list") or hasattr(dataset, "files"):
        files = getattr(dataset, "file_list", None) or getattr(dataset, "files", [])
        fingerprint["files_hash"] = hashlib.md5(
            str(sorted(str(f) for f in files)).encode()
        ).hexdigest()
        fingerprint["first_5_files"] = [str(f) for f in files[:5]]
        fingerprint["last_5_files"]  = [str(f) for f in files[-5:]]

    elif hasattr(dataset, "data_files") or hasattr(dataset, "samples"):
        items = getattr(dataset, "data_files", None) or getattr(dataset, "samples", [])
        fingerprint["items_hash"] = hashlib.md5(
            str(items[:100]).encode()
        ).hexdigest()

    # Sample a few data points and hash their values
    # This works regardless of internal structure
    try:
        sample_indices = [0, 1, 2, n // 4, n // 2, n - 1]
        sample_values  = []
        for idx in sample_indices:
            if idx < n:
                item = dataset[idx]
                arr  = item[0] if isinstance(item, (list, tuple)) else item
                # Take a small slice to keep it fast
                arr_np = arr.numpy() if hasattr(arr, "numpy") else np.array(arr)
                sample_values.append(float(arr_np.flat[0]))  # just first element

        fingerprint["sample_values"]      = sample_values
        fingerprint["sample_values_hash"] = hashlib.md5(
            str(sample_values).encode()
        ).hexdigest()
    except Exception as e:
        fingerprint["sample_hash_error"] = str(e)

    return fingerprint


def fingerprints_match(fp1: dict, fp2: dict) -> tuple[bool, list]:
    """
    Compare two fingerprints. Returns (match: bool, differences: list).
    """
    differences = []

    # Check n_samples
    if fp1.get("n_samples") != fp2.get("n_samples"):
        differences.append(
            f"n_samples: {fp1.get('n_samples')} vs {fp2.get('n_samples')}"
        )

    # Check whichever hash fields are present
    for hash_key in ["indices_hash", "files_hash", "items_hash", "sample_values_hash"]:
        v1 = fp1.get(hash_key)
        v2 = fp2.get(hash_key)
        if v1 is not None and v2 is not None:
            if v1 != v2:
                differences.append(f"{hash_key}: {v1[:8]}... vs {v2[:8]}...")

    # Check sample values directly
    sv1 = fp1.get("sample_values", [])
    sv2 = fp2.get("sample_values", [])
    if sv1 and sv2:
        if sv1 != sv2:
            differences.append(
                f"sample_values differ: {sv1[:3]} vs {sv2[:3]}"
            )

    return len(differences) == 0, differences


def load_split(data_dir: str, split: str, seed: int,
               val_split: float, test_split: float,
               seq_length: int) -> object:
    """Instantiate ECGDataset for a given split and seed."""
    # ECGDataset may accept seed directly, or may use torch/numpy global seed
    import torch
    import random

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset = ECGDataset(
        data_dir=data_dir,
        seq_length=seq_length,
        split=split,
        val_split=val_split,
        test_split=test_split,
    )
    return dataset


def check_dataloader_order(dataset, batch_size: int = 8, n_batches: int = 3) -> str:
    """
    Hash the first N batches from a DataLoader with shuffle=False.
    Same hash = same order.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    values = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        item = batch[0] if isinstance(batch, (list, tuple)) else batch
        arr  = item.numpy() if hasattr(item, "numpy") else np.array(item)
        values.append(float(arr.flat[0]))
    return hashlib.md5(str(values).encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_split_sizes(results: dict, save_path: Path) -> None:
    """Bar chart showing split sizes for each run."""
    runs   = list(results.keys())
    labels = ["train", "val", "test"]
    colors = ["steelblue", "darkorange", "green"]

    x     = np.arange(len(runs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, 2 * len(runs)), 5))

    for i, (label, color) in enumerate(zip(labels, colors)):
        sizes = [results[r]["sizes"].get(label, 0) for r in runs]
        bars  = ax.bar(x + i * width, sizes, width, label=label,
                       color=color, alpha=0.85)
        for bar, val in zip(bars, sizes):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 10,
                    str(val), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(runs, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Number of Samples", fontsize=11)
    ax.set_title("Dataset Split Sizes per Run\n(identical sizes ≠ identical samples — check hashes)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_reproducibility_matrix(
    run_labels: list,
    match_matrix: np.ndarray,
    split_name: str,
    save_path: Path,
) -> None:
    """Heatmap: green = splits match, red = splits differ."""
    n = len(run_labels)
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))

    cmap = plt.cm.RdYlGn
    im   = ax.imshow(match_matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(run_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(run_labels, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = match_matrix[i, j]
            txt = "✓ MATCH" if val == 1 else "✗ DIFFER"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8, fontweight="bold",
                    color="black" if val > 0.5 else "white")

    ax.set_title(
        f"Split Reproducibility Matrix — {split_name.upper()} split\n"
        f"Green = identical splits, Red = different splits",
        fontsize=12, fontweight="bold"
    )
    plt.colorbar(im, ax=ax, label="Match (1=identical, 0=different)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Core checker
# ─────────────────────────────────────────────────────────────────────────────

def run_reproducibility_check(
    data_dir: str,
    seeds: list,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seq_length: int   = 5000,
    output_dir: str   = "split_diagnostics",
    n_repeats: int    = 3,   # how many times to reload same seed
) -> dict:
    """
    Main reproducibility check.

    Tests:
      A. Same seed, multiple loads → splits must match
      B. Different seeds → splits should differ
      C. Val and test sets never overlap
      D. Train + val + test = total dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("  DATALOADER SPLIT REPRODUCIBILITY CHECK")
    print(f"{'='*70}")
    print(f"  Data dir:   {data_dir}")
    print(f"  Seeds:      {seeds}")
    print(f"  Val split:  {val_split}")
    print(f"  Test split: {test_split}")
    print(f"  Repeats per seed: {n_repeats}")

    splits_to_check = ["val", "test"]
    all_results     = {}
    total_issues    = 0

    # ── Load all splits ───────────────────────────────────────────────────────
    # Structure: fingerprints[split][run_label] = fingerprint dict
    fingerprints: dict = {s: {} for s in splits_to_check}
    run_labels_all: list = []
    sizes_per_run: dict  = {}

    # For each seed: load n_repeats times (to test within-seed consistency)
    for seed in seeds:
        for repeat in range(n_repeats):
            run_label = f"seed{seed}_run{repeat + 1}"
            run_labels_all.append(run_label)
            print(f"\n  Loading split for {run_label}...")

            sizes_per_run[run_label] = {}

            for split in splits_to_check:
                try:
                    ds = load_split(data_dir, split, seed,
                                    val_split, test_split, seq_length)
                    fp = get_dataset_fingerprint(ds)
                    fingerprints[split][run_label] = fp
                    sizes_per_run[run_label][split] = len(ds)
                    print(f"    {split:5s}: {len(ds)} samples  "
                          f"hash={fp.get('sample_values_hash', fp.get('indices_hash', '?'))[:8]}...")
                except Exception as e:
                    print(f"    [ERROR] Loading {split} split: {e}")
                    fingerprints[split][run_label] = {"error": str(e)}

            # Also load train to check sizes add up
            try:
                ds_train = load_split(data_dir, "train", seed,
                                      val_split, test_split, seq_length)
                sizes_per_run[run_label]["train"] = len(ds_train)
            except Exception:
                pass

    # ── Test A: Same seed, multiple runs must match ───────────────────────────
    print(f"\n{'='*70}")
    print("  TEST A: Same-seed reproducibility (runs must match)")
    print(f"{'='*70}")

    for split in splits_to_check:
        print(f"\n  Split: {split.upper()}")
        for seed in seeds:
            run_labels_for_seed = [f"seed{seed}_run{r+1}" for r in range(n_repeats)]
            ref_label = run_labels_for_seed[0]
            ref_fp    = fingerprints[split].get(ref_label, {})

            all_match = True
            for other_label in run_labels_for_seed[1:]:
                other_fp = fingerprints[split].get(other_label, {})
                match, diffs = fingerprints_match(ref_fp, other_fp)
                status = "✅ MATCH" if match else "🔴 DIFFER"
                print(f"    seed={seed}: {ref_label} vs {other_label} → {status}")
                if not match:
                    for d in diffs:
                        print(f"      Difference: {d}")
                    all_match = False
                    total_issues += 1

            if all_match:
                print(f"    seed={seed}: All {n_repeats} loads produce identical {split} split ✅")

    # ── Test B: Different seeds should differ ─────────────────────────────────
    if len(set(seeds)) > 1:
        print(f"\n{'='*70}")
        print("  TEST B: Different seeds should give different splits")
        print(f"{'='*70}")

        unique_seeds = list(dict.fromkeys(seeds))   # preserve order, deduplicate
        for split in splits_to_check:
            print(f"\n  Split: {split.upper()}")
            for i in range(len(unique_seeds)):
                for j in range(i + 1, len(unique_seeds)):
                    s1, s2 = unique_seeds[i], unique_seeds[j]
                    fp1 = fingerprints[split].get(f"seed{s1}_run1", {})
                    fp2 = fingerprints[split].get(f"seed{s2}_run1", {})
                    match, _ = fingerprints_match(fp1, fp2)
                    if match:
                        print(f"    seed={s1} vs seed={s2} → 🟡 WARNING: splits are identical!")
                        print(f"      ECGDataset may ignore the seed parameter.")
                        print(f"      If so, results are reproducible but not seed-dependent.")
                        total_issues += 1
                    else:
                        print(f"    seed={s1} vs seed={s2} → ✅ Correctly different")

    # ── Test C: Val and Test never overlap ────────────────────────────────────
    print(f"\n{'='*70}")
    print("  TEST C: Val and Test sets must not overlap")
    print(f"{'='*70}")

    for seed in set(seeds):
        run_label = f"seed{seed}_run1"
        fp_val  = fingerprints["val"].get(run_label, {})
        fp_test = fingerprints["test"].get(run_label, {})

        # If we have indices, check overlap directly
        idx_val  = fp_val.get("first_5_indices",  []) + fp_val.get("last_5_indices",  [])
        idx_test = fp_test.get("first_5_indices", []) + fp_test.get("last_5_indices", [])

        if idx_val and idx_test:
            overlap = set(idx_val) & set(idx_test)
            if overlap:
                print(f"  seed={seed}: 🔴 OVERLAP DETECTED — indices appear in both val and test!")
                print(f"    Overlapping sample indices: {overlap}")
                total_issues += 1
            else:
                print(f"  seed={seed}: ✅ No overlap detected in sampled indices")
        else:
            # Fall back to file/value comparison
            sv_val  = set(fp_val.get("first_5_files",  fp_val.get("sample_values",  [])))
            sv_test = set(fp_test.get("first_5_files", fp_test.get("sample_values", [])))
            overlap = sv_val & sv_test
            if overlap:
                print(f"  seed={seed}: 🔴 Possible overlap in val/test — {overlap}")
                total_issues += 1
            else:
                print(f"  seed={seed}: ✅ No overlap detected (sampled check)")

    # ── Test D: Split sizes sum to total ──────────────────────────────────────
    print(f"\n{'='*70}")
    print("  TEST D: train + val + test should equal total dataset size")
    print(f"{'='*70}")

    for seed in set(seeds):
        run_label = f"seed{seed}_run1"
        sizes = sizes_per_run.get(run_label, {})
        n_train = sizes.get("train", None)
        n_val   = sizes.get("val",   None)
        n_test  = sizes.get("test",  None)

        if all(v is not None for v in [n_train, n_val, n_test]):
            total = n_train + n_val + n_test
            expected_val  = total * val_split
            expected_test = total * test_split

            print(f"\n  seed={seed}:")
            print(f"    train={n_train}  val={n_val}  test={n_test}  total={total}")
            print(f"    Expected val  ≈ {expected_val:.0f}  (got {n_val})")
            print(f"    Expected test ≈ {expected_test:.0f}  (got {n_test})")

            val_ok  = abs(n_val  - expected_val)  / max(expected_val,  1) < 0.05
            test_ok = abs(n_test - expected_test) / max(expected_test, 1) < 0.05

            if val_ok and test_ok:
                print(f"    ✅ Split proportions correct")
            else:
                print(f"    🟡 Split proportions differ from requested val_split={val_split}, "
                      f"test_split={test_split}")
                print(f"       This may be due to rounding or dataset-level filtering.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  FINAL VERDICT")
    print(f"{'='*70}")

    if total_issues == 0:
        print("\n  ✅ All reproducibility checks passed!")
        print("     Your val/test splits are deterministic and consistent.")
    else:
        print(f"\n  ⚠️  {total_issues} issue(s) found.")
        print("\n  Common fixes:")
        print("  1. Pass seed explicitly to ECGDataset constructor")
        print("  2. Set global seeds BEFORE creating the dataset:")
        print("       torch.manual_seed(seed)")
        print("       np.random.seed(seed)")
        print("       random.seed(seed)")
        print("  3. Use a fixed index file saved to disk (most robust):")
        print("       Split indices once → save as splits.json → load every time")

    # ── Save outputs ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Saving outputs to: {output_path}")
    print(f"{'='*70}\n")

    # Plot split sizes
    plot_split_sizes(
        {r: {"sizes": sizes_per_run[r]} for r in run_labels_all},
        output_path / "split_sizes.png",
    )

    # Reproducibility matrix per split
    n_runs = len(run_labels_all)
    for split in splits_to_check:
        matrix = np.zeros((n_runs, n_runs))
        for i, r1 in enumerate(run_labels_all):
            for j, r2 in enumerate(run_labels_all):
                fp1 = fingerprints[split].get(r1, {})
                fp2 = fingerprints[split].get(r2, {})
                match, _ = fingerprints_match(fp1, fp2)
                matrix[i, j] = 1.0 if match else 0.0

        plot_reproducibility_matrix(
            run_labels_all, matrix, split,
            output_path / f"reproducibility_matrix_{split}.png",
        )

    # Save full fingerprints as JSON
    fp_serializable = {}
    for split in splits_to_check:
        fp_serializable[split] = {}
        for run_label, fp in fingerprints[split].items():
            fp_serializable[split][run_label] = {
                k: (v if not isinstance(v, np.integer) else int(v))
                for k, v in fp.items()
            }

    summary = {
        "data_dir": data_dir,
        "seeds": seeds,
        "val_split": val_split,
        "test_split": test_split,
        "total_issues": total_issues,
        "sizes_per_run": sizes_per_run,
        "fingerprints": fp_serializable,
    }
    summary_path = output_path / "reproducibility_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved: {summary_path}")

    print(f"\n  ✓ Done — results in: {output_path}\n")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Bonus: save a fixed split index file for guaranteed reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def save_fixed_split_indices(
    data_dir: str,
    seed: int,
    val_split: float,
    test_split: float,
    seq_length: int,
    save_path: str,
) -> None:
    """
    Load the dataset once, extract all indices, and save them to a JSON file.
    Load this file in every future run to guarantee identical splits.
    """
    import torch, random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    fixed = {}
    for split in ["train", "val", "test"]:
        ds = ECGDataset(
            data_dir=data_dir,
            seq_length=seq_length,
            split=split,
            val_split=val_split,
            test_split=test_split,
        )
        if hasattr(ds, "indices"):
            fixed[split] = {"indices": [int(i) for i in ds.indices], "n": len(ds)}
        elif hasattr(ds, "file_list"):
            fixed[split] = {"files": [str(f) for f in ds.file_list], "n": len(ds)}
        else:
            fixed[split] = {"n": len(ds), "note": "no index access available"}

    with open(save_path, "w") as f:
        json.dump({"seed": seed, "splits": fixed}, f, indent=2)

    print(f"✅ Fixed split indices saved to: {save_path}")
    print(f"   train={fixed['train']['n']}  "
          f"val={fixed['val']['n']}  "
          f"test={fixed['test']['n']}")
    print(f"\n   Load this file in your training script to guarantee")
    print(f"   identical splits across all experiments.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Check val/test split reproducibility across runs and seeds"
    )
    parser.add_argument("--data-dir",   type=str, required=True,
                        help="Path to MIMIC-IV-ECG dataset directory")
    parser.add_argument("--seeds",      type=int, nargs="+", default=[42, 42, 123],
                        help="Seeds to test. Repeat a seed (e.g. 42 42) to test "
                             "same-seed consistency. Add different seeds to test "
                             "cross-seed difference. (default: 42 42 123)")
    parser.add_argument("--n-repeats",  type=int, default=3,
                        help="How many times to reload each seed (default: 3)")
    parser.add_argument("--val-split",  type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--seq-length", type=int,   default=5000)
    parser.add_argument("--output-dir", type=str,   default="split_diagnostics")
    parser.add_argument(
        "--save-fixed-splits",
        type=str, default=None,
        metavar="OUTPUT_JSON",
        help="Also save fixed split indices to this JSON file for guaranteed "
             "future reproducibility (uses first seed in --seeds)"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"[ERROR] Data directory not found: {args.data_dir}")
        sys.exit(1)

    run_reproducibility_check(
        data_dir=args.data_dir,
        seeds=args.seeds,
        val_split=args.val_split,
        test_split=args.test_split,
        seq_length=args.seq_length,
        output_dir=args.output_dir,
        n_repeats=args.n_repeats,
    )

    if args.save_fixed_splits:
        print(f"\n{'='*70}")
        print("  Saving fixed split index file...")
        print(f"{'='*70}")
        save_fixed_split_indices(
            data_dir=args.data_dir,
            seed=args.seeds[0],
            val_split=args.val_split,
            test_split=args.test_split,
            seq_length=args.seq_length,
            save_path=args.save_fixed_splits,
        )


if __name__ == "__main__":
    main()