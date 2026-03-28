"""
Feature extraction for multimodal sensor data.

Extracts time-domain, frequency-domain, statistical, and cross-modal
features from windowed sensor signals. Includes preprocessing pipeline
(imputation → Z-score normalization → IQR/3σ clipping).
"""

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.stats import skew, kurtosis
from typing import Dict, List, Optional, Tuple


# ────────────────────────────────────────────────────────────────────
#  Per-Window Feature Extraction
# ────────────────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Extract features from 1-D signal windows.
    Designed for vibration signals (CWRU) and sensor readings (CMAPSS).
    """

    def __init__(self, sampling_rate: float = 12000.0):
        self.sampling_rate = sampling_rate

    # ── Time-Domain Features ────────────────────────────────────────

    def _time_domain(self, window: np.ndarray) -> Dict[str, float]:
        """RMS, peak, crest factor, skewness, kurtosis."""
        rms = np.sqrt(np.mean(window ** 2))
        peak = np.max(np.abs(window))
        crest = peak / rms if rms > 1e-12 else 0.0
        return {
            'rms': rms,
            'peak': peak,
            'crest_factor': crest,
            'skewness': float(skew(window)),
            'kurtosis': float(kurtosis(window)),
        }

    # ── Frequency-Domain Features ───────────────────────────────────

    def _frequency_domain(self, window: np.ndarray) -> Dict[str, float]:
        """FFT top-5 harmonics, spectral entropy, dominant freq, spectral centroid."""
        n = len(window)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sampling_rate)
        fft_mag = np.abs(np.fft.rfft(window))

        # Skip DC component
        freqs = freqs[1:]
        fft_mag = fft_mag[1:]

        if fft_mag.sum() < 1e-12:
            return {
                'dominant_freq': 0.0,
                'spectral_centroid': 0.0,
                'spectral_entropy': 0.0,
                **{f'harmonic_{i}_freq': 0.0 for i in range(5)},
                **{f'harmonic_{i}_mag': 0.0 for i in range(5)},
            }

        # Power spectral density (normalized)
        psd = fft_mag ** 2
        psd_norm = psd / psd.sum()

        # Spectral entropy
        psd_nz = psd_norm[psd_norm > 0]
        spectral_entropy = -np.sum(psd_nz * np.log2(psd_nz))

        # Spectral centroid
        spectral_centroid = np.sum(freqs * psd_norm)

        # Dominant frequency
        dominant_idx = np.argmax(fft_mag)
        dominant_freq = freqs[dominant_idx]

        # Top-5 harmonics
        top5_idx = np.argsort(fft_mag)[-5:][::-1]
        features = {
            'dominant_freq': dominant_freq,
            'spectral_centroid': spectral_centroid,
            'spectral_entropy': spectral_entropy,
        }
        for i, idx in enumerate(top5_idx):
            features[f'harmonic_{i}_freq'] = freqs[idx]
            features[f'harmonic_{i}_mag'] = fft_mag[idx]

        return features

    # ── Statistical Features ────────────────────────────────────────

    def _statistical(self, window: np.ndarray) -> Dict[str, float]:
        """Mean, variance, range, IQR, zero-crossing rate, autocorrelation lag-1."""
        q75, q25 = np.percentile(window, [75, 25])
        zcr = np.sum(np.diff(np.sign(window - np.mean(window))) != 0) / len(window)

        # Autocorrelation at lag 1
        if len(window) > 1 and np.std(window) > 1e-12:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                corr = np.corrcoef(window[:-1], window[1:])
            autocorr = corr[0, 1]
            if np.isnan(autocorr):
                autocorr = 0.0
        else:
            autocorr = 0.0

        return {
            'mean': np.mean(window),
            'variance': np.var(window),
            'range': np.ptp(window),
            'iqr': q75 - q25,
            'zero_crossing_rate': zcr,
            'autocorrelation_lag1': float(autocorr),
        }

    def extract_window_features(self, window: np.ndarray) -> Dict[str, float]:
        """Extract all feature categories from a single window."""
        features = {}
        features.update(self._time_domain(window))
        features.update(self._frequency_domain(window))
        features.update(self._statistical(window))
        return features

    def extract_batch(self, windows: np.ndarray, prefix: str = '') -> pd.DataFrame:
        """
        Extract features from all windows.

        Args:
            windows: (n_windows, window_size) array
            prefix: column name prefix, e.g. 'de_' for drive-end

        Returns:
            DataFrame (n_windows, n_features)
        """
        all_features = []
        for i in range(windows.shape[0]):
            feat = self.extract_window_features(windows[i])
            if prefix:
                feat = {f'{prefix}{k}': v for k, v in feat.items()}
            all_features.append(feat)
        return pd.DataFrame(all_features)


# ────────────────────────────────────────────────────────────────────
#  CMAPSS Multi-Sensor Feature Extraction
# ────────────────────────────────────────────────────────────────────

class CMAPSSFeatureExtractor:
    """
    Extract features from CMAPSS 21-channel sensor data.
    Since CMAPSS data is per-cycle (not windowed vibration), features
    are computed from rolling windows over cycle sequences per engine unit.
    """

    def __init__(self, window_size: int = 30, stride: int = 1):
        self.window_size = window_size
        self.stride = stride

    def _rolling_stats(self, series: np.ndarray) -> Dict[str, float]:
        """Statistical features from a 1-D series segment."""
        if len(series) == 0 or np.all(np.isnan(series)):
            return {'mean': 0, 'std': 0, 'slope': 0, 'range': 0, 'iqr': 0}

        q75, q25 = np.percentile(series, [75, 25])
        # Linear slope (trend)
        x = np.arange(len(series))
        if np.std(series) > 1e-12:
            slope = np.polyfit(x, series, 1)[0]
        else:
            slope = 0.0

        return {
            'mean': np.mean(series),
            'std': np.std(series),
            'slope': slope,
            'range': np.ptp(series),
            'iqr': q75 - q25,
        }

    def extract_modality_features(
        self,
        sensor_data: np.ndarray,
        unit_ids: np.ndarray,
        cycles: np.ndarray,
        sensor_names: List[str],
        prefix: str = '',
    ) -> pd.DataFrame:
        """
        Extract rolling features per modality group per engine unit.

        Args:
            sensor_data: (N, n_sensors) array for one modality group
            unit_ids: (N,) engine unit IDs
            cycles: (N,) cycle numbers
            sensor_names: list of sensor names for column naming
            prefix: modality prefix like 'thermal_'

        Returns:
            DataFrame with features + unit_id + cycle columns
        """
        all_features = []
        unique_units = np.unique(unit_ids)

        for uid in unique_units:
            mask = unit_ids == uid
            unit_data = sensor_data[mask]
            unit_cycles = cycles[mask]

            # Sort by cycle
            sort_idx = np.argsort(unit_cycles)
            unit_data = unit_data[sort_idx]
            unit_cycles = unit_cycles[sort_idx]

            n_samples = len(unit_data)
            for start in range(0, n_samples - self.window_size + 1, self.stride):
                end = start + self.window_size
                window_data = unit_data[start:end]
                center_cycle = unit_cycles[start + self.window_size // 2]

                row = {'unit_id': uid, 'cycle': center_cycle}
                for s_idx, s_name in enumerate(sensor_names):
                    stats = self._rolling_stats(window_data[:, s_idx])
                    for stat_name, val in stats.items():
                        row[f'{prefix}{s_name}_{stat_name}'] = val

                # Cross-sensor correlation within modality
                if sensor_data.shape[1] >= 2:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        corr_matrix = np.corrcoef(window_data.T)
                    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
                    # Average off-diagonal correlation
                    n_s = corr_matrix.shape[0]
                    mask_offdiag = ~np.eye(n_s, dtype=bool)
                    avg_corr = np.mean(corr_matrix[mask_offdiag])
                    row[f'{prefix}avg_intra_correlation'] = avg_corr

                all_features.append(row)

        return pd.DataFrame(all_features)


# ────────────────────────────────────────────────────────────────────
#  Cross-Modal Features
# ────────────────────────────────────────────────────────────────────

def compute_cross_modal_features(
    modality_features: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Compute cross-modal correlation features between modalities.

    Args:
        modality_features: {'thermal': df, 'pressure': df, 'mechanical': df}

    Returns:
        DataFrame with cross-modal correlation columns, indexed by (unit_id, cycle)
    """
    mod_names = list(modality_features.keys())
    if len(mod_names) < 2:
        return pd.DataFrame()

    # Align all modalities on unit_id + cycle
    merged = None
    for name in mod_names:
        df = modality_features[name]
        if 'unit_id' not in df.columns or 'cycle' not in df.columns:
            continue
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on=['unit_id', 'cycle'], how='inner')

    if merged is None or merged.empty:
        return pd.DataFrame()

    cross_features = merged[['unit_id', 'cycle']].copy()

    # Pairwise mean-feature correlations (per-row approximation via product)
    for i in range(len(mod_names)):
        for j in range(i + 1, len(mod_names)):
            name_i, name_j = mod_names[i], mod_names[j]
            # Get mean columns for each modality
            cols_i = [c for c in merged.columns if c.startswith(f'{name_i}_') and '_mean' in c]
            cols_j = [c for c in merged.columns if c.startswith(f'{name_j}_') and '_mean' in c]

            if cols_i and cols_j:
                # Average correlation between mean features
                mean_i = merged[cols_i].mean(axis=1)
                mean_j = merged[cols_j].mean(axis=1)
                cross_features[f'cross_{name_i}_{name_j}_product'] = mean_i * mean_j

                # Time-lag correlation via diff
                diff_i = mean_i.diff().fillna(0)
                diff_j = mean_j.diff().fillna(0)
                cross_features[f'cross_{name_i}_{name_j}_diff_product'] = diff_i * diff_j

    return cross_features


# ────────────────────────────────────────────────────────────────────
#  Preprocessing Pipeline
# ────────────────────────────────────────────────────────────────────

def preprocess_features(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Preprocessing: imputation → Z-score → IQR clipping + 3σ cap.

    Args:
        df: raw feature DataFrame
        exclude_cols: columns to skip (e.g., 'unit_id', 'cycle', 'fault_type')

    Returns:
        (preprocessed_df, scaler_stats_dict)
    """
    if exclude_cols is None:
        exclude_cols = []

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    result = df.copy()

    # Step 1: Imputation (forward-fill → interpolation → mean)
    result[feature_cols] = result[feature_cols].ffill()
    result[feature_cols] = result[feature_cols].interpolate(method='linear', limit_direction='both')
    col_means = result[feature_cols].mean().fillna(0.0)
    result[feature_cols] = result[feature_cols].fillna(col_means)
    result[feature_cols] = result[feature_cols].fillna(0.0) # Fallback for all-NaN columns

    # Step 2: Z-score normalization
    stats = {}
    for col in feature_cols:
        mean_val = result[col].mean()
        std_val = result[col].std()
        if std_val < 1e-12:
            std_val = 1.0  # constant column, avoid div-by-zero
        result[col] = (result[col] - mean_val) / std_val
        stats[col] = {'mean': mean_val, 'std': std_val}

    # Step 3: IQR clipping + 3σ cap
    for col in feature_cols:
        q75, q25 = result[col].quantile(0.75), result[col].quantile(0.25)
        iqr = q75 - q25
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr
        # Also apply 3σ cap
        lower = max(lower, -3.0)
        upper = min(upper, 3.0)
        result[col] = result[col].clip(lower=lower, upper=upper)

    return result, stats
