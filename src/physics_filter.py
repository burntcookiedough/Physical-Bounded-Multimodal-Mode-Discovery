"""
Physics Feasibility Filter — Direction 2 core patent component.

Validates discovered operational modes against parametric physical
constraints. Domain-configurable: electromechanical (CWRU) and
turbofan (CMAPSS) constraints loaded from config files.

Patent Claims:
    - Claim 6: Physics-constrained filtering with boundary refinement
    - Claim 7: Vibration violations retained as FAULT STATES
    - Claim 8: Domain-agnostic configurable constraint store
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, FrozenSet

# Fault taxonomy: maps violation pattern → semantic label + recommended action
FAULT_TAXONOMY = {
    frozenset(): {
        'label': 'Normal Operation',
        'severity': 'NOMINAL',
        'recommended_action': 'No action required',
        'physical_interpretation': 'All physical constraints satisfied; system operating within design limits'
    },
    frozenset(['joule_heating']): {
        'label': 'Electrothermal Stress',
        'severity': 'WARNING',
        'recommended_action': 'Check cooling adequacy and current draw under load',
        'physical_interpretation': 'Temperature exceeds Joule heating prediction → excessive resistive loss or cooling degradation'
    },
    frozenset(['vibration_bounds']): {
        'label': 'Mechanical Imbalance / Bearing Wear',
        'severity': 'WARNING',
        'recommended_action': 'Schedule bearing inspection; check shaft alignment',
        'physical_interpretation': 'Vibration exceeds ISO 10816 limit → mechanical fault independent of electrical load'
    },
    frozenset(['cross_modal_coherence']): {
        'label': 'Sensor Disagreement / Mixed-Regime Cluster',
        'severity': 'DATA_QUALITY',
        'recommended_action': 'Verify sensor calibration; consider re-clustering with finer granularity',
        'physical_interpretation': 'Current and temperature dynamics are uncorrelated → cluster spans two distinct physical regimes'
    },
    frozenset(['joule_heating', 'vibration_bounds']): {
        'label': 'Coupled Electromechanical Fault',
        'severity': 'CRITICAL',
        'recommended_action': 'Immediate inspection: simultaneous electrical overload and mechanical stress detected',
        'physical_interpretation': 'Both thermal excess and vibration excess present → fault propagated across subsystems'
    },
    frozenset(['joule_heating', 'cross_modal_coherence']): {
        'label': 'Thermal Runaway Onset',
        'severity': 'CRITICAL',
        'recommended_action': 'Reduce load immediately; check for bearing seizure or winding insulation breakdown',
        'physical_interpretation': 'Joule limit violated AND thermal-electrical coupling lost → runaway thermal accumulation'
    },
    frozenset(['vibration_bounds', 'cross_modal_coherence']): {
        'label': 'Mechanical Fault with Sensor Artifact',
        'severity': 'WARNING',
        'recommended_action': 'Validate vibration sensor placement; inspect mechanical components',
        'physical_interpretation': 'Vibration anomaly without thermal-electrical coupling → isolated mechanical fault or sensor issue'
    },
    frozenset(['joule_heating', 'vibration_bounds', 'cross_modal_coherence']): {
        'label': 'Complete System Degradation',
        'severity': 'CRITICAL',
        'recommended_action': 'EMERGENCY SHUTDOWN recommended. All subsystems showing constraint violations.',
        'physical_interpretation': 'All three constraint domains violated simultaneously → multi-failure state'
    }
}


class PhysicsFeasibilityFilter:
    """
    Tests cluster centroids (and optionally covariance spreads)
    against parametric physical constraint inequalities.

    Supports two domains:
      - 'electromechanical' (CWRU): Joule heating, ISO 10816, thermal rate, coherence
      - 'turbofan' (CMAPSS): speed bounds, EGT limit, fuel-thermal coherence
    """

    def __init__(self, constraint_params: Dict):
        """
        Args:
            constraint_params: loaded from cwru_params.json or cmapss_params.json
        """
        self.params = constraint_params
        self.domain = constraint_params.get('domain', 'electromechanical')
        self.violation_log: List[Dict] = []

    # ────────────────────────────────────────────────────────────────
    #  Domain-Specific Constraint Tests
    # ────────────────────────────────────────────────────────────────

    def _test_joule_heating(self, centroid: Dict[str, float]) -> Tuple[bool, str]:
        """
        Joule heating: T ≤ T_ambient + (I²·R)/(h·A) + ε

        Only for electromechanical domain.
        """
        if self.domain != 'electromechanical':
            return True, 'N/A (turbofan domain)'

        p = self.params
        T = centroid.get('temperature', p['T_ambient_celsius'])
        I = centroid.get('current', p['rated_current_A'])
        R = p['R_stator_ohm']
        h = p['h_conv_W_m2K']
        A = p['A_surface_m2']
        T_amb = p['T_ambient_celsius']
        epsilon = p['epsilon_joule']

        T_max_joule = T_amb + (I**2 * R) / (h * A) + epsilon

        if T <= T_max_joule:
            return True, f'T={T:.1f}°C ≤ T_max_joule={T_max_joule:.1f}°C → PASS'
        return False, f'T={T:.1f}°C > T_max_joule={T_max_joule:.1f}°C → FAIL (Joule violation)'

    def _test_vibration_bounds(self, centroid: Dict[str, float]) -> Tuple[bool, str]:
        """
        ISO 10816 vibration bound: a ≤ a_max.
        VIOLATION → FAULT STATE (Claim 7), NOT rejected.
        """
        if self.domain != 'electromechanical':
            return True, 'N/A (turbofan domain)'

        a = centroid.get('vibration_rms', 0.0)
        a_max = self.params['a_max_g_ISO10816']

        if a <= a_max:
            return True, f'a_rms={a:.3f}g ≤ ISO_limit={a_max:.3f}g → PASS'
        return False, f'a_rms={a:.3f}g > ISO_limit={a_max:.3f}g → FAULT STATE'

    def _test_thermal_rate(self, centroid: Dict[str, float]) -> Tuple[bool, str]:
        """Thermal rate limit: |ΔT/Δt| ≤ max_dT_per_sec."""
        if self.domain != 'electromechanical':
            return True, 'N/A'

        dT_dt = abs(centroid.get('thermal_rate', 0.0))
        max_rate = self.params['max_dT_per_sec']

        if dT_dt <= max_rate:
            return True, f'|ΔT/Δt|={dT_dt:.2f}°C/s ≤ {max_rate}°C/s → PASS'
        return False, f'|ΔT/Δt|={dT_dt:.2f}°C/s > {max_rate}°C/s → FAIL'

    def _test_efficiency(self, centroid: Dict[str, float]) -> Tuple[bool, str]:
        """Power efficiency bound: η ≤ η_nominal + ε."""
        if self.domain != 'electromechanical':
            return True, 'N/A'

        eta = centroid.get('efficiency', self.params['eta_nominal'])
        eta_max = self.params['eta_nominal'] + self.params['epsilon_efficiency']

        if eta <= eta_max:
            return True, f'η={eta:.3f} ≤ {eta_max:.3f} → PASS'
        return False, f'η={eta:.3f} > {eta_max:.3f} → FAIL (efficiency violation)'

    def _test_cross_modal_coherence(self, centroid: Dict[str, float]) -> Tuple[bool, str]:
        """
        Cross-modal coherence.
        - Electromechanical: corr(ΔI/Δt, ΔT/Δt) ≥ threshold
        - Turbofan: corr(Wf, ΔT50/Δt) ≥ threshold
        """
        if self.domain == 'electromechanical':
            corr = centroid.get('correlation_I_T', 0.0)
            threshold = self.params['min_correlation_I_T']
            if corr >= threshold:
                return True, f'corr(ΔI,ΔT)={corr:.3f} ≥ {threshold} → PASS'
            return False, f'corr(ΔI,ΔT)={corr:.3f} < {threshold} → FAIL (incoherent)'

        elif self.domain == 'turbofan':
            corr = centroid.get('correlation_fuel_T50', 0.0)
            threshold = self.params.get('min_correlation_fuel_T50', 0.5)
            if corr >= threshold:
                return True, f'corr(Wf,ΔT50)={corr:.3f} ≥ {threshold} → PASS'
            return False, f'corr(Wf,ΔT50)={corr:.3f} < {threshold} → FAIL'

        return True, 'N/A'

    def _test_speed_bounds(self, centroid: Dict[str, float]) -> Tuple[bool, str]:
        """Turbofan speed bounds: Nf ≤ Nf_max, Nc ≤ Nc_max."""
        if self.domain != 'turbofan':
            return True, 'N/A'

        bounds = self.params.get('speed_bounds', {})
        Nf = centroid.get('Nf', 0)
        Nc = centroid.get('Nc', 0)
        Nf_max = bounds.get('Nf_max_rpm', 99999)
        Nc_max = bounds.get('Nc_max_rpm', 99999)

        violations = []
        if Nf > Nf_max:
            violations.append(f'Nf={Nf:.0f} > {Nf_max}')
        if Nc > Nc_max:
            violations.append(f'Nc={Nc:.0f} > {Nc_max}')

        if not violations:
            return True, f'Nf={Nf:.0f}, Nc={Nc:.0f} within bounds → PASS'
        return False, 'Speed violation: ' + ', '.join(violations)

    def _test_egt_limit(self, centroid: Dict[str, float]) -> Tuple[bool, str]:
        """Turbofan EGT limit: T50 ≤ EGT_limit."""
        if self.domain != 'turbofan':
            return True, 'N/A'

        T50 = centroid.get('T50', 0)
        egt_limit = self.params.get('EGT_limit_R', 99999)

        if T50 <= egt_limit:
            return True, f'T50={T50:.0f}R ≤ EGT_limit={egt_limit}R → PASS'
        return False, f'T50={T50:.0f}R > EGT_limit={egt_limit}R → FAIL'

    # ────────────────────────────────────────────────────────────────
    #  Main Filter Logic
    # ────────────────────────────────────────────────────────────────

    def test_mode(self, mode_id: int, centroid: Dict[str, float]) -> Dict:
        """
        Test a single discovered mode against all relevant constraints.

        Args:
            mode_id: cluster/mode identifier
            centroid: dict of denormalized physical quantities

        Returns:
            {
                'mode_id': int,
                'feasible': bool,
                'is_fault_state': bool,
                'violations': list of str,
                'passes': list of str,
                'constraint_details': list of (name, passed, explanation),
            }
        """
        tests = [
            ('joule_heating', self._test_joule_heating),
            ('vibration_bounds', self._test_vibration_bounds),
            ('thermal_rate', self._test_thermal_rate),
            ('efficiency', self._test_efficiency),
            ('cross_modal_coherence', self._test_cross_modal_coherence),
            ('speed_bounds', self._test_speed_bounds),
            ('egt_limit', self._test_egt_limit),
        ]

        violations = []
        passes = []
        details = []
        violated_names = set()
        is_fault_state = False

        for name, test_fn in tests:
            passed, explanation = test_fn(centroid)
            details.append((name, passed, explanation))

            if 'N/A' in explanation:
                continue
            if passed:
                passes.append(f'{name}: {explanation}')
            else:
                violations.append(f'{name}: {explanation}')
                violated_names.add(name)
                # Vibration violation → FAULT STATE, not rejection
                if name == 'vibration_bounds':
                    is_fault_state = True

        feasible = len([v for v in violations if 'FAULT STATE' not in v]) == 0

        result = {
            'mode_id': mode_id,
            'feasible': feasible,
            'is_fault_state': is_fault_state,
            'violations': violations,
            'violated_names': violated_names,
            'passes': passes,
            'constraint_details': details,
        }

        self.violation_log.append(result)
        return result

    def compute_mode_risk_score(self, centroid: Dict[str, float]) -> Dict:
        """Computes a continuous risk score [0, 1] per mode."""
        risk_components = []
        p = self.params

        if self.domain == 'electromechanical':
            # Joule heating severity
            T_measured = centroid.get('temperature', p['T_ambient_celsius'])
            I = centroid.get('current', p['rated_current_A'])
            R, h, A = p['R_stator_ohm'], p['h_conv_W_m2K'], p['A_surface_m2']
            T_max = p['T_ambient_celsius'] + (I**2 * R) / (h * A)
            joule_severity = max(0, (T_measured - T_max) / T_max) if T_max > 0 else 0
            risk_components.append(('joule_heating', joule_severity, 0.35))

            # Vibration severity
            a_measured = centroid.get('vibration_rms', 0.0)
            a_max = p['a_max_g_ISO10816']
            vibration_severity = max(0, (a_measured - a_max) / a_max) if a_max > 0 else 0
            risk_components.append(('vibration_bounds', vibration_severity, 0.30))

            # Cross-modal decoupling severity
            corr = centroid.get('correlation_I_T', 0.0)
            corr_min = p['min_correlation_I_T']
            decoupling_severity = max(0, (corr_min - corr) / corr_min) if corr < corr_min else 0
            risk_components.append(('cross_modal_coherence', decoupling_severity, 0.35))
        elif self.domain == 'turbofan':
            # Speed bounds severity
            bounds = p.get('speed_bounds', {})
            Nf_measured = centroid.get('Nf', 0)
            Nf_max = bounds.get('Nf_max_rpm', 99999)
            speed_severity = max(0, (Nf_measured - Nf_max) / Nf_max) if Nf_max > 0 else 0
            risk_components.append(('speed_bounds', speed_severity, 0.4))
            
            # EGT limit severity
            T50_measured = centroid.get('T50', 0)
            T50_max = p.get('EGT_limit_R', 99999)
            egt_severity = max(0, (T50_measured - T50_max) / T50_max) if T50_max > 0 else 0
            risk_components.append(('egt_limit', egt_severity, 0.4))
            
            # Coherence severity
            corr = centroid.get('correlation_fuel_T50', 0.0)
            corr_min = p.get('min_correlation_fuel_T50', 0.5)
            decoupling_severity = max(0, (corr_min - corr) / corr_min) if corr < corr_min else 0
            risk_components.append(('cross_modal_coherence', decoupling_severity, 0.2))

        if not risk_components:
            return {'risk_score': 0.0, 'dominant_risk_factor': None, 'components': {}}

        weighted_sum = sum(sev * w for _, sev, w in risk_components)
        weight_total = sum(w for _, _, w in risk_components)
        risk_score = min(1.0, weighted_sum / weight_total) if weight_total > 0 else 0.0

        dominant = max(risk_components, key=lambda x: x[1])[0]

        return {
            'risk_score': risk_score,
            'dominant_risk_factor': dominant,
            'components': {name: sev for name, sev, _ in risk_components}
        }

    def auto_label_mode(self, violated_names: set, risk_score: float, dominance_signature: Dict[str, float] = None) -> Dict:
        """Assigns semantic label from violation pattern."""
        key = frozenset(violated_names)
        base = FAULT_TAXONOMY.get(
            key,
            {'label': 'Unknown State', 'severity': 'UNCLASSIFIED',
             'recommended_action': 'Manual inspection required',
             'physical_interpretation': 'Violation pattern not in taxonomy'}
        )
        
        label_with_context = base['label']
        if dominance_signature:
            dominant_modality = max(dominance_signature, key=dominance_signature.get)
            label_with_context = f"{base['label']} [{dominant_modality.upper()}-dominant]"

        return {
            'semantic_label': label_with_context,
            'severity_class': base['severity'],
            'recommended_action': base['recommended_action'],
            'physical_interpretation': base['physical_interpretation'],
            'risk_score': risk_score,
            'auto_labeled': True,
            'label_source': 'physics_constraint_taxonomy'
        }

    def filter_modes(
        self,
        mode_centroids: Dict[int, Dict[str, float]],
        dominance_signatures: Dict[int, Dict[str, float]] = None
    ) -> Dict:
        """
        Test and rank all discovered modes against physics constraints.
        """
        feasible = []
        infeasible = []
        fault_states = []
        results = []

        all_scores = {}

        for mode_id, centroid in mode_centroids.items():
            result = self.test_mode(mode_id, centroid)
            
            # Patent Addition 1: Risk Scorer
            risk_data = self.compute_mode_risk_score(centroid)
            result.update(risk_data)
            all_scores[mode_id] = risk_data['risk_score']

            # Patent Addition 2: Auto-Labeler
            dom_sig = dominance_signatures.get(mode_id) if dominance_signatures else None
            # Filter the violated_names to match taxonomy keys
            taxonomy_keys = {'joule_heating', 'vibration_bounds', 'cross_modal_coherence', 'speed_bounds', 'egt_limit'}
            valid_violations = result['violated_names'].intersection(taxonomy_keys)
            
            label_data = self.auto_label_mode(valid_violations, risk_data['risk_score'], dom_sig)
            result.update(label_data)
            
            results.append(result)

            if result['is_fault_state']:
                fault_states.append(mode_id)
                feasible.append(mode_id)
            elif result['feasible']:
                feasible.append(mode_id)
            else:
                infeasible.append(mode_id)

        # Rank modes by risk (Addition 1)
        sorted_results = sorted(results, key=lambda x: x['risk_score'], reverse=True)
        for rank, res in enumerate(sorted_results, 1):
            res['risk_rank'] = rank

        total = len(mode_centroids)
        coherence_rate = len(feasible) / total if total > 0 else 0.0

        return {
            'feasible_modes': feasible,
            'infeasible_modes': infeasible,
            'fault_states': fault_states,
            'coherence_rate': coherence_rate,
            'results': sorted_results,
        }


# ────────────────────────────────────────────────────────────────────
#  Boundary Refinement (Claim 6)
# ────────────────────────────────────────────────────────────────────

class BoundaryRefiner:
    """
    Handles incoherent clusters discovered by the physics filter.

    Four options (from patent spec):
      A: Merge with nearest valid cluster
      B: Split via sub-clustering within valid region
      C: Constraint gradient adjustment
      D: Flag as FAULT STATE
    """

    def __init__(self, strategy: str = 'merge'):
        """
        Args:
            strategy: one of 'merge', 'split', 'gradient', 'fault'
        """
        if strategy not in ('merge', 'split', 'gradient', 'fault'):
            raise ValueError(f"Unknown strategy: {strategy}. "
                             f"Use 'merge', 'split', 'gradient', or 'fault'.")
        self.strategy = strategy

    def refine(
        self,
        labels: np.ndarray,
        infeasible_modes: List[int],
        feasible_modes: List[int],
        X: np.ndarray,
        centroids: np.ndarray,
    ) -> np.ndarray:
        """
        Refine labels for infeasible clusters.

        Args:
            labels: (n_samples,) current cluster assignments
            infeasible_modes: list of mode IDs that failed physics
            feasible_modes: list of mode IDs that passed
            X: (n_samples, n_features) feature matrix
            centroids: (n_clusters, n_features) cluster centers

        Returns:
            refined_labels: (n_samples,) updated assignments
        """
        refined = labels.copy()

        if self.strategy == 'merge':
            refined = self._merge_nearest(
                refined, infeasible_modes, feasible_modes, centroids
            )
        elif self.strategy == 'split':
            refined = self._split_subclustering(
                refined, infeasible_modes, X
            )
        elif self.strategy == 'gradient':
            refined = self._gradient_adjust(
                refined, infeasible_modes, feasible_modes, X, centroids
            )
        elif self.strategy == 'fault':
            refined = self._flag_as_fault(refined, infeasible_modes)

        return refined

    def _merge_nearest(
        self,
        labels: np.ndarray,
        infeasible: List[int],
        feasible: List[int],
        centroids: np.ndarray,
    ) -> np.ndarray:
        """Strategy A: Merge infeasible modes into nearest feasible cluster."""
        if not feasible:
            return labels

        for bad_mode in infeasible:
            # Find nearest feasible centroid
            bad_centroid = centroids[bad_mode]
            distances = [
                np.linalg.norm(bad_centroid - centroids[f]) for f in feasible
            ]
            nearest_feasible = feasible[np.argmin(distances)]
            labels[labels == bad_mode] = nearest_feasible

        return labels

    def _split_subclustering(
        self,
        labels: np.ndarray,
        infeasible: List[int],
        X: np.ndarray,
    ) -> np.ndarray:
        """Strategy B: Sub-cluster infeasible points and retest."""
        from sklearn.cluster import KMeans

        max_label = labels.max()
        for bad_mode in infeasible:
            mask = labels == bad_mode
            if mask.sum() < 4:
                continue  # too few points to sub-cluster

            sub_X = X[mask]
            km = KMeans(n_clusters=2, n_init=5, random_state=42)
            sub_labels = km.fit_predict(sub_X)

            # Assign new unique labels
            new_labels = sub_labels + max_label + 1
            labels[mask] = new_labels
            max_label = labels.max()

        return labels

    def _gradient_adjust(
        self,
        labels: np.ndarray,
        infeasible: List[int],
        feasible: List[int],
        X: np.ndarray,
        centroids: np.ndarray,
    ) -> np.ndarray:
        """
        Strategy C: Move boundary points toward constraint-satisfying region.
        Points nearest to a feasible centroid get reassigned.
        """
        if not feasible:
            return labels

        feasible_centroids = centroids[feasible]

        for bad_mode in infeasible:
            mask = labels == bad_mode
            if mask.sum() == 0:
                continue

            bad_points = X[mask]
            # For each bad point, find nearest feasible centroid
            for i, pt in enumerate(bad_points):
                dists = np.linalg.norm(feasible_centroids - pt, axis=1)
                nearest = feasible[np.argmin(dists)]
                indices = np.where(mask)[0]
                labels[indices[i]] = nearest

        return labels

    def _flag_as_fault(self, labels: np.ndarray, infeasible: List[int]) -> np.ndarray:
        """Strategy D: Mark infeasible modes with special FAULT label (-2)."""
        for bad_mode in infeasible:
            labels[labels == bad_mode] = -2  # fault-state sentinel
        return labels
