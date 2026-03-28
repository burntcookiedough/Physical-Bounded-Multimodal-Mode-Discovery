"""
Dataset adapters for CWRU Bearing and NASA CMAPSS datasets.

Each adapter loads raw data, applies windowing/segmentation, and returns
a standardized dict of per-modality arrays + metadata for downstream
feature extraction.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat


# ────────────────────────────────────────────────────────────────────
#  CWRU Bearing Dataset Adapter
# ────────────────────────────────────────────────────────────────────

class CWRUAdapter:
    """
    Loads CWRU Bearing .mat files and segments into windows.

    Expected directory layout:
        data/cwru/
          ├── Normal/
          │     └── *.mat
          ├── IR007/ IR014/ IR021/
          ├── OR007/ OR014/ OR021/
          └── Ball007/ Ball014/ Ball021/

    Each .mat file contains keys like 'X097_DE_time', 'X097_FE_time', etc.
    """

    # Map folder names to fault metadata
    FAULT_MAP = {
        'Normal':  {'fault_type': 'normal', 'fault_diameter_mils': 0},
        'IR007':   {'fault_type': 'inner_race', 'fault_diameter_mils': 7},
        'IR014':   {'fault_type': 'inner_race', 'fault_diameter_mils': 14},
        'IR021':   {'fault_type': 'inner_race', 'fault_diameter_mils': 21},
        'OR007':   {'fault_type': 'outer_race', 'fault_diameter_mils': 7},
        'OR014':   {'fault_type': 'outer_race', 'fault_diameter_mils': 14},
        'OR021':   {'fault_type': 'outer_race', 'fault_diameter_mils': 21},
        'Ball007': {'fault_type': 'ball', 'fault_diameter_mils': 7},
        'Ball014': {'fault_type': 'ball', 'fault_diameter_mils': 14},
        'Ball021': {'fault_type': 'ball', 'fault_diameter_mils': 21},
    }

    def __init__(
        self,
        data_dir: str,
        config_path: str = 'configs/cwru_params.json',
        window_size: int = 500,
        overlap: float = 0.5,
        sampling_rate: int = 12000,
    ):
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.step_size = int(window_size * (1 - overlap))
        self.sampling_rate = sampling_rate

        with open(config_path, 'r') as f:
            self.params = json.load(f)

    def _extract_signal(self, mat_data: dict, suffix: str) -> Optional[np.ndarray]:
        """Extract a signal array matching the given suffix ('DE_time', 'FE_time')."""
        for key in mat_data:
            if key.endswith(suffix):
                arr = mat_data[key].flatten()
                return arr
        return None

    def _infer_load_hp(self, filename: str) -> int:
        """Infer load condition (0–3 HP) from the file number in the CWRU naming."""
        # CWRU files are numbered: 97–100 → 0HP, 105–108 → 1HP, etc.
        # Simplified: parse from filename or default to -1 (unknown)
        name = Path(filename).stem
        digits = ''.join(c for c in name if c.isdigit())
        if not digits:
            return -1
        file_num = int(digits)
        # 12k DE Mapping
        if file_num in [97, 105, 118, 130, 169, 209]:
            return 0
        elif file_num in [98, 106, 119, 131, 170, 210]:
            return 1
        elif file_num in [99, 107, 120, 132, 171, 211]:
            return 2
        elif file_num in [100, 108, 121, 133, 172, 212]:
            return 3
        return -1

    def _window_signal(self, signal: np.ndarray) -> np.ndarray:
        """Segment a 1-D signal into overlapping windows. Returns (n_windows, window_size)."""
        n_windows = max(0, (len(signal) - self.window_size) // self.step_size + 1)
        if n_windows == 0:
            return np.empty((0, self.window_size))
        windows = np.zeros((n_windows, self.window_size))
        for i in range(n_windows):
            start = i * self.step_size
            windows[i] = signal[start : start + self.window_size]
        return windows

    def load(self) -> Dict:
        """
        Load all CWRU .mat files.

        Returns:
            {
                'vibration_de':  np.ndarray (N, window_size),  # drive-end accel
                'vibration_fe':  np.ndarray (N, window_size),  # fan-end accel
                'labels': pd.DataFrame with columns
                    ['fault_type', 'fault_diameter', 'load_hp', 'source_file'],
                'params': dict (from cwru_params.json),
                'modality_names': ['vibration_de', 'vibration_fe'],
            }
        """
        all_de, all_fe = [], []
        all_labels = []

        for fault_dir_name, fault_meta in self.FAULT_MAP.items():
            fault_dir = self.data_dir / fault_dir_name
            if not fault_dir.exists():
                continue

            mat_files = sorted(fault_dir.glob('*.mat'))
            for mat_file in mat_files:
                try:
                    mat = loadmat(str(mat_file))
                except Exception:
                    continue

                de = self._extract_signal(mat, 'DE_time')
                fe = self._extract_signal(mat, 'FE_time')

                if de is None:
                    continue

                de_windows = self._window_signal(de)
                n_windows = de_windows.shape[0]

                if fe is not None:
                    fe_windows = self._window_signal(fe)
                    # Align to shorter
                    n_windows = min(n_windows, fe_windows.shape[0])
                    de_windows = de_windows[:n_windows]
                    fe_windows = fe_windows[:n_windows]
                else:
                    fe_windows = np.zeros_like(de_windows)

                all_de.append(de_windows)
                all_fe.append(fe_windows)

                load_hp = self._infer_load_hp(mat_file.name)
                for _ in range(n_windows):
                    all_labels.append({
                        'fault_type': fault_meta['fault_type'],
                        'fault_diameter': fault_meta['fault_diameter_mils'],
                        'load_hp': load_hp,
                        'source_file': mat_file.name,
                    })

        if not all_de:
            raise FileNotFoundError(
                f"No valid .mat files found in {self.data_dir}. "
                "Expected subfolders: Normal/, IR007/, OR007/, Ball007/, etc."
            )

        return {
            'vibration_de': np.vstack(all_de),
            'vibration_fe': np.vstack(all_fe),
            'labels': pd.DataFrame(all_labels),
            'params': self.params,
            'modality_names': ['vibration_de', 'vibration_fe'],
            'sampling_rate': self.sampling_rate,
            'window_size': self.window_size,
        }


# ────────────────────────────────────────────────────────────────────
#  NASA CMAPSS Dataset Adapter
# ────────────────────────────────────────────────────────────────────

class CMAPSSAdapter:
    """
    Loads NASA CMAPSS turbofan engine degradation simulation data.

    Expected directory layout:
        data/cmapss/
          ├── train_FD001.txt
          ├── test_FD001.txt
          ├── RUL_FD001.txt
          └── (optionally FD002–FD004)

    Columns per official documentation:
        unit_id, cycle, op_setting1, op_setting2, op_setting3, s1..s21
    """

    COLUMN_NAMES = (
        ['unit_id', 'cycle', 'op1', 'op2', 'op3']
        + [f's{i}' for i in range(1, 22)]
    )

    def __init__(
        self,
        data_dir: str,
        config_path: str = 'configs/cmapss_params.json',
        subset: str = 'FD001',
    ):
        self.data_dir = Path(data_dir)
        self.subset = subset
        self.train_file = self.data_dir / f'train_{subset}.txt'
        self.test_file = self.data_dir / f'test_{subset}.txt'
        self.rul_file = self.data_dir / f'RUL_{subset}.txt'

        with open(config_path, 'r') as f:
            self.params = json.load(f)

        # Modality group indices (0-based within s1..s21 columns)
        self.thermal_idx = self.params.get('thermal_indices', [0, 1, 2, 3])
        self.pressure_idx = self.params.get('pressure_indices', [4, 5, 6])
        self.mechanical_idx = self.params.get('mechanical_indices', [7, 8, 14])

    def _read_cmapss_file(self, filepath: Path) -> pd.DataFrame:
        """Read a whitespace-delimited CMAPSS text file."""
        df = pd.read_csv(
            filepath,
            sep=r'\s+',
            header=None,
            names=self.COLUMN_NAMES,
            engine='python',
        )
        return df

    def _get_sensor_columns(self) -> List[str]:
        """Return the 21 sensor column names."""
        return [f's{i}' for i in range(1, 22)]

    def load(self, split: str = 'train') -> Dict:
        """
        Load CMAPSS data and split into modality groups.

        Args:
            split: 'train' or 'test'

        Returns:
            {
                'thermal':    np.ndarray (N, 4),   # T2, T24, T30, T50
                'pressure':   np.ndarray (N, 3),   # P2, P15, P30
                'mechanical': np.ndarray (N, 3),   # Nf, Nc, BPR
                'all_sensors': np.ndarray (N, 21),  # full 21-channel matrix
                'op_settings': np.ndarray (N, 3),
                'unit_ids':    np.ndarray (N,),
                'cycles':      np.ndarray (N,),
                'labels': pd.DataFrame with unit_id, cycle, op_settings
                'params': dict,
                'modality_names': ['thermal', 'pressure', 'mechanical'],
                'sensor_columns': list of 21 sensor names,
            }
        """
        filepath = self.train_file if split == 'train' else self.test_file
        if not filepath.exists():
            raise FileNotFoundError(
                f"CMAPSS file not found: {filepath}. "
                f"Download from NASA Prognostics Data Repository."
            )

        df = self._read_cmapss_file(filepath)
        sensor_cols = self._get_sensor_columns()
        all_sensors = df[sensor_cols].values  # (N, 21)

        # Split into modality groups
        thermal = all_sensors[:, self.thermal_idx]
        pressure = all_sensors[:, self.pressure_idx]
        mechanical = all_sensors[:, self.mechanical_idx]

        # Operational settings as additional context
        op_settings = df[['op1', 'op2', 'op3']].values

        # Build labels DataFrame
        labels = df[['unit_id', 'cycle', 'op1', 'op2', 'op3']].copy()
        # Discretize op settings → regime label (for ground truth comparison)
        labels['regime'] = self._assign_regime_labels(op_settings)

        return {
            'thermal': thermal,
            'pressure': pressure,
            'mechanical': mechanical,
            'all_sensors': all_sensors,
            'op_settings': op_settings,
            'unit_ids': df['unit_id'].values,
            'cycles': df['cycle'].values,
            'labels': labels,
            'params': self.params,
            'modality_names': ['thermal', 'pressure', 'mechanical'],
            'sensor_columns': sensor_cols,
        }

    def _assign_regime_labels(self, op_settings: np.ndarray) -> np.ndarray:
        """
        Cluster operational settings to identify distinct regimes.
        CMAPSS FD001 has 1 operating condition, FD002/FD004 have 6.
        Uses simple rounding to discretize continuous settings.
        """
        from sklearn.cluster import KMeans

        # Try k=1..6, pick best silhouette
        if op_settings.shape[0] < 10:
            return np.zeros(op_settings.shape[0], dtype=int)

        # Round to reduce noise — settings are nearly discrete in CMAPSS
        rounded = np.round(op_settings, decimals=2)
        unique_combos = np.unique(rounded, axis=0)
        n_regimes = min(len(unique_combos), 6)

        if n_regimes <= 1:
            return np.zeros(op_settings.shape[0], dtype=int)

        km = KMeans(n_clusters=n_regimes, n_init=10, random_state=42)
        regime_labels = km.fit_predict(rounded)
        return regime_labels

    def load_rul(self) -> Optional[np.ndarray]:
        """Load remaining useful life labels for test set."""
        if not self.rul_file.exists():
            return None
        return np.loadtxt(self.rul_file, dtype=int)


# ────────────────────────────────────────────────────────────────────
#  Utility: quick summary of loaded data
# ────────────────────────────────────────────────────────────────────

def summarize_dataset(data: Dict, name: str = '') -> str:
    """Print a quick summary of a loaded dataset dict."""
    lines = [f"=== Dataset: {name} ==="]
    for key, val in data.items():
        if isinstance(val, np.ndarray):
            lines.append(f"  {key}: shape={val.shape}, dtype={val.dtype}")
        elif isinstance(val, pd.DataFrame):
            lines.append(f"  {key}: DataFrame shape={val.shape}")
            lines.append(f"    columns: {list(val.columns)}")
        elif isinstance(val, dict):
            lines.append(f"  {key}: dict with {len(val)} keys")
        elif isinstance(val, list):
            lines.append(f"  {key}: list with {len(val)} items")
        else:
            lines.append(f"  {key}: {val}")
    summary = '\n'.join(lines)
    print(summary)
    return summary
