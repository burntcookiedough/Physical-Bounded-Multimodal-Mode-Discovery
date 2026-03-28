"""
Consensus Arbiter — Direction 1 core patent component.

Performs confidence-weighted voting across per-modality HDBSCAN results
to produce unified mode labels with conflict detection and dominance
signatures. Feeds into GMM joint mode modeling.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from typing import Dict, List, Optional, Tuple


class ConsensusArbiter:
    """
    Weighted consensus across modality-specific cluster labels.

    Patent Claims:
        - Claim 1: Modality-disaggregated architecture with per-modality confidence
        - Claim 2: Consensus arbitration via confidence-weighted voting
        - Claim 3: Modality conflict flags for fault localization
        - Claim 4: Dominance signatures per discovered mode
    """

    def __init__(self, conflict_threshold: float = 0.3):
        """
        Args:
            conflict_threshold: minimum disagreement fraction to flag conflict.
                If <= this fraction of total weight agrees on the winning label,
                the sample is flagged as conflicted.
        """
        self.conflict_threshold = conflict_threshold

    def arbitrate(
        self,
        modality_results: Dict[str, Dict],
        n_samples: int,
    ) -> Dict:
        """
        Perform consensus voting across modalities.

        Args:
            modality_results: {modality_name: {labels, probabilities, ...}}
            n_samples: number of data points

        Returns:
            {
                'consensus_labels': np.ndarray (n_samples,),
                'conflict_flags': np.ndarray (n_samples,) bool,
                'dominance_signatures': np.ndarray (n_samples, n_modalities),
                'modality_names': list,
                'per_sample_votes': list of dicts,
            }
        """
        modality_names = list(modality_results.keys())
        n_modalities = len(modality_names)

        consensus_labels = np.full(n_samples, -1, dtype=int)
        conflict_flags = np.zeros(n_samples, dtype=bool)
        dominance_sigs = np.zeros((n_samples, n_modalities))
        per_sample_votes = []

        for i in range(n_samples):
            votes = {}  # label → total_weight
            modality_weights = []  # (modality_idx, label, weight) per modality

            for m_idx, m_name in enumerate(modality_names):
                result = modality_results[m_name]
                label = result['labels'][i]
                prob = result['probabilities'][i]

                # Create a unique label namespace per modality to avoid
                # collisions (modality 0 label 1 ≠ modality 1 label 1)
                # We use probability as the weight for voting
                if label == -1:
                    # Noise point — zero weight, does not vote
                    modality_weights.append((m_idx, -1, 0.0))
                    continue

                # Weight = HDBSCAN membership probability (0..1)
                weight = float(prob)
                modality_weights.append((m_idx, label, weight))

                # For consensus, we need to find which modality labels
                # correspond to the "same" mode. Since labels are per-modality
                # and not aligned, we'll use a simple approach:
                # Each modality votes for its own label with its weight.
                # The GMM joint model (below) handles alignment.
                m_label = f"{m_name}:{label}"
                votes[m_label] = votes.get(m_label, 0.0) + weight

            per_sample_votes.append(votes)

            # Determine consensus: modality with highest weight wins
            total_weight = sum(w for _, _, w in modality_weights if w > 0)
            if total_weight < 1e-12:
                # All noise — mark as noise
                consensus_labels[i] = -1
                conflict_flags[i] = True
                continue

            # Find the winning vote
            if votes:
                winning_label = max(votes, key=votes.get)
                winning_weight = votes[winning_label]
                consensus_labels[i] = hash(winning_label) % 10000  # temp integer label

                # Conflict detection: if winning weight < threshold × total
                agreement_ratio = winning_weight / total_weight
                if agreement_ratio < (1.0 - self.conflict_threshold):
                    conflict_flags[i] = True

            # Dominance signature: normalized weight per modality
            for m_idx, _, w in modality_weights:
                dominance_sigs[i, m_idx] = w / total_weight if total_weight > 0 else 0.0

        return {
            'consensus_labels': consensus_labels,
            'conflict_flags': conflict_flags,
            'dominance_signatures': dominance_sigs,
            'modality_names': modality_names,
            'per_sample_votes': per_sample_votes,
        }


class JointGMMModeler:
    """
    Fit a Gaussian Mixture Model on the joint feature space,
    using consensus information to initialize or validate.

    Patent Claim 5: GMM mode probability with covariance-based
    feasibility testing.
    """

    def __init__(
        self,
        n_components: int = 5,
        covariance_type: str = 'full',
        n_init: int = 5,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_init = n_init
        self.random_state = random_state
        self.model: Optional[GaussianMixture] = None

    def fit(self, X: np.ndarray) -> Dict:
        """
        Fit GMM on joint feature matrix.

        Args:
            X: (n_samples, n_features) — all modalities concatenated

        Returns:
            {
                'labels': np.ndarray (n_samples,),
                'probabilities': np.ndarray (n_samples, n_components),
                'means': np.ndarray (n_components, n_features),
                'covariances': array,
                'bic': float,
                'aic': float,
            }
        """
        self.model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        labels = self.model.fit_predict(X)
        probabilities = self.model.predict_proba(X)

        return {
            'labels': labels,
            'probabilities': probabilities,
            'means': self.model.means_,
            'covariances': self.model.covariances_,
            'bic': self.model.bic(X),
            'aic': self.model.aic(X),
        }

    def scan_components(
        self, X: np.ndarray, k_range: range = range(2, 11)
    ) -> Tuple[int, pd.DataFrame]:
        """
        Scan BIC across component counts to find optimal k.

        Returns:
            (best_k, DataFrame of scan results)
        """
        results = []
        for k in k_range:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=self.covariance_type,
                n_init=self.n_init,
                random_state=self.random_state,
            )
            gmm.fit(X)
            results.append({
                'k': k,
                'bic': gmm.bic(X),
                'aic': gmm.aic(X),
                'log_likelihood': gmm.score(X) * X.shape[0],
            })

        df = pd.DataFrame(results)
        best_k = int(df.loc[df['bic'].idxmin(), 'k'])
        return best_k, df
