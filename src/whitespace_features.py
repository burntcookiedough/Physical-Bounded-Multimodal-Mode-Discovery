import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from scipy.spatial.distance import cosine

# --- Addition 5: Physics-Feasibility-Bounded Synthetic Fault Augmentation (PFBSFA) ---

def passes_all_constraints(candidate_physical, bounds):
    """Tests a denormalized candidate point against all expected physics constraints."""
    # This is an example constraint check; should be adapted dynamically to the domain
    if bounds.get('domain') == 'turbofan':
        # CMAPSS constraints
        Nf_ok = candidate_physical.get('Nf', 0) <= bounds['speed_bounds']['Nf_max_rpm']
        Nc_ok = candidate_physical.get('Nc', 0) <= bounds['speed_bounds']['Nc_max_rpm']
        EGT_ok = candidate_physical.get('T50', 0) <= bounds['EGT_limit_R']
        return Nf_ok and Nc_ok and EGT_ok
    else:
        # CWRU constraints
        try:
            T = candidate_physical.get('temperature', 0)
            I = candidate_physical.get('current', 0)
            a = candidate_physical.get('vibration', 0)
            
            joule_ok  = T <= bounds.get('T_ambient_celsius', 25) + (I**2 * bounds.get('R_stator_ohm', 2.1)) / (bounds.get('h_conv_W_m2K', 15) * bounds.get('A_surface_m2', 0.18)) + bounds.get('epsilon_joule', 5)
            vibr_ok   = a <= bounds.get('a_max_g_ISO10816', 0.46)
            power_ok  = True # Simplified for this check
            
            return joule_ok and vibr_ok and power_ok
        except KeyError:
            return True

def denormalize_point(point, scaler=None, feature_cols=None):
    """Denormalize a candidate point back to physical units."""
    if scaler is None or feature_cols is None:
        # Return a mock dict if no scaler provided
        return {f'f_{i}': val for i, val in enumerate(point)}
    
    # If standard scaler is used:
    try:
        denormed = point * scaler.scale_ + scaler.mean_
        return {col: val for col, val in zip(feature_cols, denormed)}
    except Exception:
        return {f'f_{i}': val for i, val in enumerate(point)}

def project_to_feasible_boundary(candidate, bounds):
    """Mock projection for candidate rejected points."""
    return candidate * 0.99  # Example naive projection

def physics_bounded_synthetic_sampling(
        fault_cluster_features,
        mode_constraint_bounds,
        scaler=None,
        feature_cols=None,
        n_synthetic=100,
        max_attempts_per_sample=50
    ):
    """
    Generate synthetic samples within the physics-feasible subspace of a fault cluster.
    """
    if len(fault_cluster_features) < 2:
        return np.array([])
        
    n_neighbors = min(5, len(fault_cluster_features))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(fault_cluster_features)
    synthetic = []
    
    for _ in range(n_synthetic):
        attempts = 0
        accepted = False
        
        while not accepted and attempts < max_attempts_per_sample:
            idx = np.random.randint(len(fault_cluster_features))
            seed = fault_cluster_features[idx]
            
            _, neighbor_indices = nbrs.kneighbors([seed])
            neighbor = fault_cluster_features[
                neighbor_indices[0][np.random.randint(1, n_neighbors)]
            ]
            
            alpha = np.random.uniform(0, 1)
            candidate = seed + alpha * (neighbor - seed)
            
            candidate_physical = denormalize_point(candidate, scaler, feature_cols)
            if passes_all_constraints(candidate_physical, mode_constraint_bounds):
                synthetic.append(candidate)
                accepted = True
            
            attempts += 1
        
        if not accepted:
            projected = project_to_feasible_boundary(candidate, mode_constraint_bounds)
            synthetic.append(projected)
    
    return np.array(synthetic)


# --- Addition 6: Constraint-Violation Temporal Causality Chain (CVTCC) ---

class ConstraintViolationCausalityChain:
    """
    Builds a directed temporal causality graph from sequences of
    constraint violations observed across engine/machine unit lifecycles.
    """
    
    def __init__(self):
        self.causal_edges = defaultdict(list)
        self.violation_sequences = []
    
    def ingest_unit_lifecycle(self, unit_id, time_ordered_violations):
        first_appearances = {}
        for cycle, violations in time_ordered_violations:
            for constraint in violations:
                if constraint not in first_appearances:
                    first_appearances[constraint] = cycle
        
        constraints = sorted(first_appearances.keys(), key=lambda c: first_appearances[c])
        for i, early_constraint in enumerate(constraints):
            for late_constraint in constraints[i+1:]:
                lead_time = first_appearances[late_constraint] - first_appearances[early_constraint]
                self.causal_edges[(early_constraint, late_constraint)].append(lead_time)
        
        self.violation_sequences.append((unit_id, first_appearances))
    
    def build_causal_graph(self, min_support=0.6):
        n_units = len(self.violation_sequences)
        causal_graph = {}
        if n_units == 0:
            return causal_graph
            
        for (A, B), lead_times in self.causal_edges.items():
            support = len(lead_times) / n_units
            if support >= min_support:
                causal_graph[(A, B)] = {
                    'mean_lead_time_cycles': np.mean(lead_times),
                    'std_lead_time_cycles': np.std(lead_times),
                    'support': support,
                    'earliest_warning_cycles': np.percentile(lead_times, 10) if len(lead_times) > 0 else 0,
                    'interpretation': f"'{A}' precedes '{B}' by avg {np.mean(lead_times):.1f} cycles in {support*100:.0f}% of units"
                }
        
        return causal_graph


# --- Addition 7: Unsupervised Cross-Dataset Mode Alignment via Dominance-Constraint Fingerprinting (DCFP) ---

class CrossDatasetModeAligner:
    """
    Aligns discovered operational modes across datasets from different
    physical domains using dominance signatures and violation fingerprints.
    """
    
    def __init__(self):
        self.mode_registry = {}
    
    def register_dataset_modes(self, dataset_id, validated_modes):
        self.mode_registry[dataset_id] = {}
        for mode_id, mode_data in validated_modes.items():
            fingerprint = self._build_fingerprint(mode_data)
            self.mode_registry[dataset_id][mode_id] = {'fingerprint': fingerprint, 'label': mode_data.get('semantic_label', '')}
    
    def _build_fingerprint(self, mode_data):
        constraint_types = ['joule_heating', 'power_balance', 'vibration_bounds',
                           'thermal_inertia', 'cross_modal_decoupling']
        
        vp = mode_data.get('violation_pattern', set())
        violation_vec = np.array([
            1.0 if c in vp else 0.0
            for c in constraint_types
        ])
        
        dom = mode_data.get('dominance_signature', {})
        modality_roles = ['thermal', 'electrical_pressure', 'mechanical']
        
        role_mapping = {
            'temperature': 'thermal', 'T_group': 'thermal', 'thermal': 'thermal',
            'current': 'electrical_pressure', 'P_group': 'electrical_pressure', 'pressure': 'electrical_pressure',
            'vibration': 'mechanical', 'vibration_de': 'mechanical',
            'vibration_fe': 'mechanical', 'M_group': 'mechanical', 'mechanical': 'mechanical'
        }
        
        role_dominance = {}
        for modality, score in dom.items():
            role = role_mapping.get(modality, 'mechanical')
            role_dominance[role] = role_dominance.get(role, 0) + score
            
        role_vec = np.array([role_dominance.get(r, 0) for r in modality_roles])
        if role_vec.sum() > 0:
            role_vec /= role_vec.sum()
        
        risk_vec = np.array([mode_data.get('risk_score', 0)])
        
        return np.concatenate([violation_vec, role_vec, risk_vec])
    
    def align_across_datasets(self, dataset_a, dataset_b, similarity_threshold=0.75):
        modes_a = self.mode_registry.get(dataset_a, {})
        modes_b = self.mode_registry.get(dataset_b, {})
        
        alignments = []
        
        for mode_id_a, data_a in modes_a.items():
            fp_a = data_a['fingerprint']
            best_match = None
            best_similarity = 0
            
            for mode_id_b, data_b in modes_b.items():
                fp_b = data_b['fingerprint']
                if np.linalg.norm(fp_a) == 0 or np.linalg.norm(fp_b) == 0:
                    continue
                similarity = 1 - cosine(fp_a, fp_b)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = mode_id_b
            
            if best_match is not None and best_similarity >= similarity_threshold:
                alignments.append({
                    f'{dataset_a}_mode': mode_id_a,
                    f'{dataset_b}_mode': best_match,
                    'alignment_similarity': best_similarity,
                    'interpretation': (
                        f"Mode '{mode_id_a}' ({dataset_a}) is structurally analogous "
                        f"to Mode '{best_match}' ({dataset_b}) (sim: {best_similarity:.2f})"
                    )
                })
        
        return sorted(alignments, key=lambda x: x['alignment_similarity'], reverse=True)
