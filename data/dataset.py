"""
MIMIC-IV-ECG, PTB-XL, and Deepfake ECG dataset implementations.

Provides:
- MIMICTestDataset: Uses record_list.csv, row-level 80/10/10 splits
- MIMICIVECGDataset: Uses machine_measurements.csv, subject-level splits
- PTBXLDataset: Uses ptbxl_database.csv, strat_fold splits, optional MUSE reports
- DeepfakeECGDataset: .asc files (8 leads), derives 12 leads, no conditioning
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import wfdb
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import Dataset

# Shared constants
FEATURE_NAMES = [
    "rr_interval", "p_onset", "p_end", "qrs_onset", "qrs_end", "t_end",
    "p_axis", "qrs_axis", "t_axis",
]

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


class MIMICTestDataset(Dataset):
    """Loads the test split from MIMIC-IV-ECG.
    Uses record_list.csv (with optional merge from machine_measurements.csv).
    Row-level 80/10/10 train/val/test split.
    """

    def __init__(
        self,
        data_dir: str,
        max_samples: Optional[int],
        seq_length: int,
        seed: int,
        val_split: float = 0.1,
        test_split: float = 0.1,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.seq_length = seq_length

        records_csv = self.data_dir / "record_list.csv"
        if not records_csv.exists():
            raise FileNotFoundError(f"record_list.csv not found in {data_dir}")

        df = pd.read_csv(records_csv)

        # Try to load machine measurements
        machine_csv = self.data_dir / "machine_measurements.csv"
        if machine_csv.exists():
            meas = pd.read_csv(machine_csv)
            df = df.merge(meas, on="study_id", how="left") if "study_id" in df.columns else df.merge(
                meas, left_index=True, right_index=True, how="left"
            )
        for fn in FEATURE_NAMES:
            if fn not in df.columns:
                df[fn] = 0.0

        # 80 / 10 / 10 split — reproduce training split exactly
        train_df, temp_df = train_test_split(
            df, test_size=val_split + test_split, random_state=seed
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_split / (val_split + test_split),
            random_state=seed,
        )

        if max_samples:
            test_df = test_df.head(max_samples)

        self.df = test_df.reset_index(drop=True)

        # Feature stats from train set
        feat = train_df[FEATURE_NAMES].values.astype(np.float32)
        self.feature_mean = np.nanmean(feat, axis=0)
        self.feature_std = np.nanstd(feat, axis=0) + 1e-6

        print(f"[Dataset] Test samples: {len(self.df)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        row = self.df.iloc[idx]
        path = str(self.data_dir / row["path"])
        try:
            record = wfdb.rdrecord(path)
            ecg = record.p_signal.T.astype(np.float32)  # [12, T]
        except Exception:
            ecg = np.zeros((12, self.seq_length), dtype=np.float32)

        # Pad / crop
        T = ecg.shape[1]
        if T >= self.seq_length:
            ecg = ecg[:, : self.seq_length]
        else:
            ecg = np.pad(ecg, ((0, 0), (0, self.seq_length - T)))

        # Normalise per sample
        std = ecg.std() + 1e-6
        ecg = (ecg - ecg.mean()) / std

        features = row[FEATURE_NAMES].values.astype(np.float32)
        features = (features - self.feature_mean) / self.feature_std

        return torch.from_numpy(ecg), torch.from_numpy(features)


class MIMICIVECGDataset(Dataset):
    """MIMIC-IV-ECG dataset.
    Uses machine_measurements.csv with subject-level splits.
    """

    FEATURE_NAMES = FEATURE_NAMES

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

        # Final check for empty dataset
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
                UserWarning,
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
        return os.path.join(
            self.mimic_path, "files", p_dir, p_subdir, s_dir, str(study_id)
        )

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


# SCP codes by superclass (form column in scp_statements.csv)
SCP_SUPERCLASS_CODES: dict[str, frozenset[str]] = {
    "HYP": frozenset({"LVH", "LAO/LAE", "RVH", "RAO/RAE", "SEHYP", "VCLVH"}),
}

# MUSE feature names mapped to FEATURE_NAMES (p_offset->p_end, qrs_offset->qrs_end, t_offset->t_end)
_MUSE_TO_FEATURE = {
    "rr_interval": "rr_interval",
    "p_onset": "p_onset",
    "p_offset": "p_end",
    "qrs_onset": "qrs_onset",
    "qrs_offset": "qrs_end",
    "t_offset": "t_end",
    "p_axis": "p_axis",
    "qrs_axis": "qrs_axis",
    "t_axis": "t_axis",
}


class PTBXLDataset(Dataset):
    """PTB-XL ECG dataset with optional MUSE report conditioning.
    Returns (ecg [12, seq_length], features [9]) compatible with FEATURE_NAMES.
    Uses strat_fold: 1–8 train, 9 val, 10 test.
    """

    FEATURE_NAMES = FEATURE_NAMES

    def __init__(
        self,
        ptbxl_path: str,
        split: str = "train",
        muse_path: Optional[str] = None,
        scp_superclass: Optional[str] = None,
        seq_length: int = 5000,
        num_leads: int = 12,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.ptbxl_path = Path(ptbxl_path)
        self.split = split
        self.muse_path = Path(muse_path) if muse_path else None
        self.scp_superclass = scp_superclass
        self.seq_length = seq_length
        self.num_leads = num_leads
        self.seed = seed

        self._load_metadata()
        self._load_muse_reports()
        self._compute_feature_stats()

        if max_samples is not None:
            self.metadata = self.metadata.head(max_samples).reset_index(drop=True)

        scp_info = f" scp={self.scp_superclass}" if self.scp_superclass else ""
        print(f"[PTBXLDataset] {split}: {len(self.metadata)} samples{scp_info}")

    def _load_metadata(self) -> None:
        metadata_path = self.ptbxl_path / "ptbxl_database.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"ptbxl_database.csv not found at {self.ptbxl_path}")
        self.metadata = pd.read_csv(metadata_path)

        if self.split == "train":
            self.metadata = self.metadata[self.metadata.strat_fold < 9]
        elif self.split == "val":
            self.metadata = self.metadata[self.metadata.strat_fold == 9]
        elif self.split == "test":
            self.metadata = self.metadata[self.metadata.strat_fold == 10]
        else:
            raise ValueError(f"Unknown split: {self.split}")

        # Filter by SCP superclass (e.g. HYP for Hypertrophy)
        if self.scp_superclass is not None:
            codes = SCP_SUPERCLASS_CODES.get(self.scp_superclass)
            if codes is None:
                raise ValueError(
                    f"Unknown scp_superclass: {self.scp_superclass}. "
                    f"Supported: {list(SCP_SUPERCLASS_CODES.keys())}"
                )
            if "scp_codes" not in self.metadata.columns:
                raise ValueError("ptbxl_database.csv has no scp_codes column")
            mask = []
            for raw in self.metadata["scp_codes"]:
                try:
                    d = ast.literal_eval(str(raw)) if isinstance(raw, str) else {}
                except (ValueError, SyntaxError):
                    d = {}
                has_match = bool(d and (set(d.keys()) & codes))
                mask.append(has_match)
            self.metadata = self.metadata[pd.Series(mask, index=self.metadata.index)]

        self.metadata = self.metadata.reset_index(drop=True)

    def _load_muse_reports(self) -> None:
        self.muse_data: dict = {}
        self.use_muse = False

        if self.muse_path is None:
            return

        muse_dir = self.muse_path / "muse_reports"
        if not muse_dir.exists():
            return

        try:
            import xmltodict
        except ImportError:
            return

        for _, row in self.metadata.iterrows():
            ecg_id = row["ecg_id"]
            patient_id = row["patient_id"]
            muse_file = muse_dir / f"{patient_id:05d}" / f"{ecg_id:05d}.xml"
            if muse_file.exists():
                try:
                    with open(muse_file, "r", encoding="utf-8") as f:
                        xml_data = xmltodict.parse(f.read())
                    self.muse_data[ecg_id] = self._parse_muse_xml(xml_data)
                except Exception:
                    pass

        if len(self.muse_data) > 0:
            self.use_muse = True

    def _parse_muse_xml(self, xml_data: dict) -> dict:
        """Extract features from MUSE XML and map to FEATURE_NAMES."""
        out: dict = {fn: 0.0 for fn in FEATURE_NAMES}
        try:
            root = (
                xml_data.get("RestingECG")
                or xml_data.get("RestingECGMeasurements")
                or xml_data
            )
            m = root.get("Measurements") or {}

            raw = {
                "rr_interval": m.get("RRInterval"),
                "p_onset": m.get("POnset"),
                "p_offset": m.get("POffset"),
                "qrs_onset": m.get("QRSOnset"),
                "qrs_offset": m.get("QRSOffset"),
                "t_offset": m.get("TOffset"),
                "p_axis": m.get("PAxis"),
                "qrs_axis": m.get("QRSAxis"),
                "t_axis": m.get("TAxis"),
            }
            for muse_key, feat_key in _MUSE_TO_FEATURE.items():
                v = raw.get(muse_key)
                if v is not None:
                    try:
                        out[feat_key] = float(v)
                    except (ValueError, TypeError):
                        pass
        except Exception:
            pass
        return out

    def _compute_feature_stats(self) -> None:
        if not self.use_muse or len(self.muse_data) == 0:
            self.feature_mean = np.zeros(9, dtype=np.float32)
            self.feature_std = np.ones(9, dtype=np.float32)
            return

        values = {fn: [] for fn in FEATURE_NAMES}
        for feat_dict in self.muse_data.values():
            for fn in FEATURE_NAMES:
                v = feat_dict.get(fn, 0.0)
                if v != 0.0:
                    values[fn].append(v)

        mean = np.zeros(9, dtype=np.float32)
        std = np.ones(9, dtype=np.float32)
        for i, fn in enumerate(FEATURE_NAMES):
            arr = np.array(values[fn], dtype=np.float32)
            if len(arr) > 0:
                mean[i] = np.nanmean(arr)
                std[i] = np.nanstd(arr) + 1e-6
        self.feature_mean = mean
        self.feature_std = std

    def _get_features(self, ecg_id: int) -> np.ndarray:
        feat = np.zeros(9, dtype=np.float32)
        if self.use_muse and ecg_id in self.muse_data:
            d = self.muse_data[ecg_id]
            for i, fn in enumerate(FEATURE_NAMES):
                feat[i] = d.get(fn, 0.0)
        feat = (feat - self.feature_mean) / self.feature_std
        return feat

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        row = self.metadata.iloc[idx]
        filename = row["filename_hr"].replace(".hea", "")
        record_path = str(self.ptbxl_path / filename)

        try:
            record = wfdb.rdrecord(record_path)
            ecg = record.p_signal.T.astype(np.float32)
        except Exception:
            ecg = np.zeros((self.num_leads, self.seq_length), dtype=np.float32)

        if ecg.shape[0] != self.num_leads:
            ecg = ecg[: self.num_leads] if ecg.shape[0] >= self.num_leads else np.pad(
                ecg, ((0, self.num_leads - ecg.shape[0]), (0, 0))
            )
        T = ecg.shape[1]
        if T >= self.seq_length:
            ecg = ecg[:, : self.seq_length]
        else:
            ecg = np.pad(ecg, ((0, 0), (0, self.seq_length - T)))

        ecg = (ecg - ecg.mean()) / (ecg.std() + 1e-6)

        features = self._get_features(int(row["ecg_id"]))

        return torch.from_numpy(ecg), torch.from_numpy(features)


class DeepfakeECGDataset(Dataset):
    """Deepfake ECG dataset: .asc files with 8 leads (I, II, V1–V6).
    Derives III, aVR, aVL, aVF from I and II. No conditioning; features are zeros.
    """

    FEATURE_NAMES = FEATURE_NAMES

    def __init__(
        self,
        data_dir: str,
        max_samples: Optional[int] = None,
        seq_length: int = 5000,
        num_leads: int = 12,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.seq_length = seq_length
        self.num_leads = num_leads

        self.files = sorted(
            self.data_dir.glob("*.asc"),
            key=lambda p: int(p.stem) if p.stem.isdigit() else float("inf"),
        )
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .asc files found in {data_dir}")

        if max_samples is not None:
            self.files = self.files[:max_samples]

        self.feature_mean = np.zeros(9, dtype=np.float32)
        self.feature_std = np.ones(9, dtype=np.float32)
        print(f"[DeepfakeECGDataset] {len(self.files)} samples")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        path = self.files[idx]
        try:
            ecg = pd.read_csv(path, header=None, sep=r"\s+").values.astype(np.float32)
        except Exception:
            ecg = np.zeros((5000, 8), dtype=np.float32)

        # Raw shape (5000, 8): columns 0=I, 1=II, 2-7=V1-V6
        I, II = ecg[:, 0], ecg[:, 1]
        V1, V2, V3, V4, V5, V6 = ecg[:, 2], ecg[:, 3], ecg[:, 4], ecg[:, 5], ecg[:, 6], ecg[:, 7]
        III = II - I
        aVR = -0.5 * (I + II)
        aVL = I - 0.5 * II
        aVF = II - 0.5 * I
        ecg_12 = np.stack(
            [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6], axis=0
        ).astype(np.float32)

        # Crop / pad to seq_length
        T = ecg_12.shape[1]
        if T >= self.seq_length:
            ecg_12 = ecg_12[:, : self.seq_length]
        else:
            ecg_12 = np.pad(ecg_12, ((0, 0), (0, self.seq_length - T)))

        ecg_12 = (ecg_12 - ecg_12.mean()) / (ecg_12.std() + 1e-6)
        features = np.zeros(9, dtype=np.float32)

        return torch.from_numpy(ecg_12), torch.from_numpy(features)
