"""
Baseline pipeline: K-means on concatenated joint features.

This establishes the comparison benchmark — the "obvious approach"
that our patent-novel architecture must beat.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from typing import Dict, Tuple


class BaselinePipeline:
    """
    K-means baseline on joint (all-modalities-concatenated) feature matrix.
    Scans k=2..k_max, picks best silhouette, computes all internal metrics.
    """

    def __init__(self, k_max: int = 10, random_state: int = 42):
        self.k_max = k_max
        self.random_state = random_state
        self.best_k = None
        self.best_model = None
        self.best_labels = None
        self.scan_results = []

    def fit(self, X: np.ndarray) -> Dict:
        """
        Run k-means silhouette scan k=2..k_max.

        Args:
            X: (n_samples, n_features) preprocessed feature matrix

        Returns:
            Dict with best_k, labels, metrics, scan_results
        """
        best_silhouette = -1
        self.scan_results = []

        for k in range(2, self.k_max + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
            labels = km.fit_predict(X)

            # Skip if degenerate (all same label)
            n_unique = len(np.unique(labels))
            if n_unique < 2:
                continue

            sil = silhouette_score(X, labels)
            db = davies_bouldin_score(X, labels)
            ch = calinski_harabasz_score(X, labels)

            result = {
                'k': k,
                'silhouette': sil,
                'davies_bouldin': db,
                'calinski_harabasz': ch,
                'inertia': km.inertia_,
            }
            self.scan_results.append(result)

            if sil > best_silhouette:
                best_silhouette = sil
                self.best_k = k
                self.best_model = km
                self.best_labels = labels

        metrics = self._compute_final_metrics(X, self.best_labels)
        return {
            'best_k': self.best_k,
            'labels': self.best_labels,
            'centroids': self.best_model.cluster_centers_ if self.best_model else None,
            'metrics': metrics,
            'scan_results': pd.DataFrame(self.scan_results),
        }

    def _compute_final_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict:
        """Compute internal clustering metrics for the best model."""
        if labels is None or len(np.unique(labels)) < 2:
            return {'silhouette': 0, 'davies_bouldin': 0, 'calinski_harabasz': 0}

        return {
            'silhouette': silhouette_score(X, labels),
            'davies_bouldin': davies_bouldin_score(X, labels),
            'calinski_harabasz': calinski_harabasz_score(X, labels),
        }

    def transition_matrix(self, labels: np.ndarray) -> np.ndarray:
        """
        Build mode transition probability matrix.

        Args:
            labels: (n_samples,) cluster assignments in temporal order

        Returns:
            (k, k) transition probability matrix
        """
        k = len(np.unique(labels[labels >= 0]))
        T = np.zeros((k, k))
        for i in range(len(labels) - 1):
            if labels[i] >= 0 and labels[i + 1] >= 0:
                T[labels[i], labels[i + 1]] += 1

        # Normalize rows
        row_sums = T.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid div-by-zero
        T = T / row_sums
        return T
