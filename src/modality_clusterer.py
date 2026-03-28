"""
Per-modality HDBSCAN clustering.

Direction 1 component: runs HDBSCAN independently on each modality's
feature matrix to discover modality-specific micro-clusters before
consensus arbitration.
"""

import numpy as np
import pandas as pd
import hdbscan
from sklearn.metrics import silhouette_score
from typing import Dict, List, Optional, Tuple


class ModalityClusterer:
    """
    HDBSCAN per-modality micro-clustering.

    For each modality (e.g., vibration_de, thermal, pressure, mechanical),
    runs HDBSCAN and returns labels, probabilities, and confidence metrics.
    """

    def __init__(
        self,
        min_cluster_size: int = 15,
        min_samples: int = 5,
        cluster_selection_method: str = 'eom',
        metric: str = 'euclidean',
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_method = cluster_selection_method
        self.metric = metric
        self.models: Dict[str, hdbscan.HDBSCAN] = {}

    def fit_modality(
        self,
        X: np.ndarray,
        modality_name: str,
    ) -> Dict:
        """
        Cluster a single modality's feature matrix.

        Args:
            X: (n_samples, n_features) for this modality
            modality_name: identifier like 'thermal', 'vibration_de'

        Returns:
            {
                'labels': np.ndarray (n_samples,), -1 = noise
                'probabilities': np.ndarray (n_samples,), membership strength
                'outlier_scores': np.ndarray (n_samples,),
                'n_clusters': int,
                'noise_ratio': float,
                'silhouette': float or None,
            }
        """
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_method=self.cluster_selection_method,
            metric=self.metric,
        )
        labels = clusterer.fit_predict(X)
        self.models[modality_name] = clusterer

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = np.sum(labels == -1) / len(labels)

        # Silhouette only if >=2 clusters and some non-noise points
        sil = None
        non_noise = labels >= 0
        if n_clusters >= 2 and np.sum(non_noise) > n_clusters:
            try:
                sil = silhouette_score(X[non_noise], labels[non_noise])
            except ValueError:
                sil = None

        return {
            'labels': labels,
            'probabilities': clusterer.probabilities_,
            'outlier_scores': clusterer.outlier_scores_,
            'n_clusters': n_clusters,
            'noise_ratio': noise_ratio,
            'silhouette': sil,
        }

    def fit_all(
        self,
        modality_features: Dict[str, np.ndarray],
    ) -> Dict[str, Dict]:
        """
        Cluster all modalities independently.

        Args:
            modality_features: {'modality_name': feature_matrix}

        Returns:
            {'modality_name': cluster_result_dict}
        """
        results = {}
        for name, X in modality_features.items():
            if X.shape[0] == 0:
                continue
            results[name] = self.fit_modality(X, name)
            n = results[name]['n_clusters']
            noise = results[name]['noise_ratio']
            sil = results[name].get('silhouette', 'N/A')
            print(f"  [{name}] → {n} clusters, {noise:.1%} noise, silhouette={sil}")
        return results
