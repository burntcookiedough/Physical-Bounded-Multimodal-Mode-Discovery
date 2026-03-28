"""
Full pipeline orchestrator — connects all components end-to-end.

Pipeline flow:
  DatasetAdapter → FeatureExtractor → ModalityClusterer(×N) →
  ConsensusArbiter → JointGMM → PhysicsFeasibilityFilter → Output

Supports both Demo 1 (CWRU) and Demo 2 (CMAPSS) configurations.
"""

import json
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

from src.dataset_adapter import CWRUAdapter, CMAPSSAdapter, summarize_dataset
from src.feature_extractor import (
    FeatureExtractor,
    CMAPSSFeatureExtractor,
    preprocess_features,
    compute_cross_modal_features,
)
from src.baseline_pipeline import BaselinePipeline
from src.modality_clusterer import ModalityClusterer
from src.consensus_arbiter import ConsensusArbiter, JointGMMModeler
from src.physics_filter import PhysicsFeasibilityFilter, BoundaryRefiner
from src.conflict_pattern_library import ConflictPatternLibrary
from src.imdti import compute_degradation_trajectory
from src.whitespace_features import (
    physics_bounded_synthetic_sampling,
    ConstraintViolationCausalityChain,
    CrossDatasetModeAligner
)


class ModeDiscoveryPipeline:
    """
    End-to-end mode discovery pipeline.

    Usage:
        pipeline = ModeDiscoveryPipeline(demo='cwru')
        results = pipeline.run()
    """

    def __init__(
        self,
        demo: str = 'cwru',
        data_dir: Optional[str] = None,
        config_dir: str = 'configs',
        results_dir: str = 'results',
    ):
        """
        Args:
            demo: 'cwru' or 'cmapss'
            data_dir: path to dataset directory (auto-detected if None)
            config_dir: path to config files
            results_dir: path to save results
        """
        self.demo = demo.lower()
        self.config_dir = Path(config_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path('data') / self.demo

        # Load config
        config_name = 'cwru_params.json' if self.demo == 'cwru' else 'cmapss_params.json'
        with open(self.config_dir / config_name) as f:
            self.params = json.load(f)

    def run(self) -> Dict:
        """Execute the full pipeline and return results."""
        print(f"\n{'='*60}")
        print(f"  Mode Discovery Pipeline — Demo: {self.demo.upper()}")
        print(f"{'='*60}\n")

        t_start = time.time()

        # ── Step 1: Load data ───────────────────────────────────────
        print("[1/6] Loading dataset...")
        data = self._load_data()
        summarize_dataset(data, self.demo.upper())

        # ── Step 2: Extract features ────────────────────────────────
        print("\n[2/6] Extracting features...")
        features, modality_feature_matrices, metadata = self._extract_features(data)
        print(f"  Joint feature matrix: {features.shape}")

        # ── Step 3: Baseline (k-means) ──────────────────────────────
        print("\n[3/6] Running baseline k-means...")
        baseline = BaselinePipeline(k_max=10)
        baseline_results = baseline.fit(features.values)
        print(f"  Best k={baseline_results['best_k']}")
        print(f"  Silhouette={baseline_results['metrics']['silhouette']:.4f}")
        print(f"  Davies-Bouldin={baseline_results['metrics']['davies_bouldin']:.4f}")

        # ── Step 4: Per-modality HDBSCAN ────────────────────────────
        print("\n[4/6] Per-modality HDBSCAN clustering...")
        clusterer = ModalityClusterer(
            min_cluster_size=15, min_samples=5
        )
        modality_results = clusterer.fit_all(modality_feature_matrices)

        # ── Step 5: Consensus arbitration ───────────────────────────
        print("\n[5/6] Consensus arbitration...")
        arbiter = ConsensusArbiter(conflict_threshold=0.3)
        n_samples = features.shape[0]
        consensus = arbiter.arbitrate(modality_results, n_samples)

        conflict_rate = consensus['conflict_flags'].mean()
        print(f"  Conflict rate: {conflict_rate:.1%}")
        print(f"  Dominance signature shape: {consensus['dominance_signatures'].shape}")

        # Joint GMM
        gmm_modeler = JointGMMModeler(n_components=baseline_results['best_k'])
        best_k_gmm, gmm_scan = gmm_modeler.scan_components(features.values)
        print(f"  GMM optimal k (BIC): {best_k_gmm}")
        gmm_modeler.n_components = best_k_gmm
        gmm_results = gmm_modeler.fit(features.values)

        # ── Step 6: Physics feasibility filter ──────────────────────
        print("\n[6/6] Physics feasibility filter...")
        physics = PhysicsFeasibilityFilter(self.params)

        # Build centroids dict with denormalized physical quantities
        mode_centroids = self._build_mode_centroids(
            gmm_results['means'], features.columns, data
        )

        # Compute per-mode average dominance signatures
        dom_sigs_per_mode = {}
        for mode_id in range(best_k_gmm):
            mask = (gmm_results['labels'] == mode_id)
            if mask.sum() > 0:
                avg_sig = consensus['dominance_signatures'][mask].mean(axis=0)
                dom_sigs_per_mode[mode_id] = {
                    name: float(avg_sig[i]) for i, name in enumerate(consensus['modality_names'])
                }
            else:
                n_mods = len(consensus['modality_names'])
                dom_sigs_per_mode[mode_id] = {name: 1.0/n_mods for name in consensus['modality_names']}

        filter_results = physics.filter_modes(mode_centroids, dom_sigs_per_mode)
        print(f"  Coherence rate: {filter_results['coherence_rate']:.1%}")
        print(f"  Feasible modes: {filter_results['feasible_modes']}")
        print(f"  Infeasible modes: {filter_results['infeasible_modes']}")
        print(f"  Fault states: {filter_results['fault_states']}")

        # CPL integration
        print("\n[CPL] Updating Conflict Pattern Library...")
        cpl = ConflictPatternLibrary(str(self.results_dir / 'cpl.json'))
        for res in filter_results['results']:
            mode_id = res['mode_id']
            dom_sig = dom_sigs_per_mode.get(mode_id, {})
            # Define conflict vector as 1 - dominance (high dominance = low conflict/disagreement on this modality)
            conflict_vector = {k: max(0.0, 1.0 - v) for k, v in dom_sig.items()}
            if res.get('auto_labeled'):
                cpl.add_pattern(
                    conflict_vector=conflict_vector,
                    outcome_label=res.get('semantic_label', 'Unknown'),
                    dataset_source=self.demo,
                    risk_score=res.get('risk_score', 0.0)
                )

        # If there are infeasible modes, apply boundary refinement
        if filter_results['infeasible_modes']:
            refiner = BoundaryRefiner(strategy='merge')
            refined_labels = refiner.refine(
                gmm_results['labels'],
                filter_results['infeasible_modes'],
                filter_results['feasible_modes'],
                features.values,
                gmm_results['means'],
            )
        else:
            refined_labels = gmm_results['labels']

        # ── Step 6b: PFBSFA (Physics-Feasibility-Bounded Synthetic Fault Augmentation) ─
        print("\n[6b] Generating constraint-bounded synthetic fault samples (PFBSFA)...")
        synthetic_faults = {}
        for mode_id in filter_results['fault_states'] + filter_results['infeasible_modes']:
            mode_mask = (refined_labels == mode_id)
            if mode_mask.sum() > 2:
                mode_features = features.values[mode_mask]
                syn_samples = physics_bounded_synthetic_sampling(
                    fault_cluster_features=mode_features,
                    mode_constraint_bounds=self.params,
                    scaler=None,  # Ideally pass pipeline's scaler if available
                    feature_cols=features.columns,
                    n_synthetic=50
                )
                synthetic_faults[mode_id] = len(syn_samples)
        print(f"  Generated synthetic samples for {len(synthetic_faults)} fault modes.")


        # ── Step 7: Inter-Mode Degradation Trajectory & CVTCC Causality ─
        results_cvtcc = {}
        if self.demo == 'cmapss' and metadata is not None:
            print("\n[7/7] Computing IMDTI and CVTCC for CMAPSS units...")
            metadata = metadata.copy()
            metadata['mode_label'] = refined_labels
            mode_risks = {res['mode_id']: {'risk_score': res.get('risk_score', 0.0)} for res in filter_results['results']}
            mode_violations = {res['mode_id']: res.get('violated_constraints', []) for res in filter_results['results']}
            
            imdti_results = []
            cvtcc = ConstraintViolationCausalityChain()
            
            for unit_id, group in metadata.groupby('unit_id'):
                group = group.sort_values('cycle')
                # 1. IMDTI Trajectory
                trajectory = compute_degradation_trajectory(
                    unit_id, 
                    group['mode_label'].tolist(), 
                    mode_risks,
                    window_size=10
                )
                imdti_results.append(trajectory)
                
                # 2. CVTCC Ingestion
                time_ordered_violations = [
                    (row.cycle, mode_violations.get(row.mode_label, []))
                    for _, row in group.iterrows()
                ]
                cvtcc.ingest_unit_lifecycle(unit_id, time_ordered_violations)
                
            results_imdti = imdti_results
            results_cvtcc = cvtcc.build_causal_graph(min_support=0.5)
            print(f"  CVTCC generated {len(results_cvtcc)} robust causal edges.")
        else:
            results_imdti = None
            results_cvtcc = None

        elapsed = time.time() - t_start
        print(f"\n  Pipeline completed in {elapsed:.1f}s")
        
        # Build validated_modes for DCFP alignment
        validated_modes = {}
        for res in filter_results['results']:
            mid = res['mode_id']
            validated_modes[mid] = {
                'dominance_signature': dom_sigs_per_mode.get(mid, {}),
                'violation_pattern': frozenset(res.get('violated_constraints', [])),
                'risk_score': res.get('risk_score', 0.0),
                'semantic_label': res.get('semantic_label', 'Unknown')
            }

        # ── Compile results ─────────────────────────────────────────
        results = {
            'demo': self.demo,
            'data_summary': {
                'n_samples': n_samples,
                'n_features': features.shape[1],
                'modalities': list(modality_feature_matrices.keys()),
            },
            'baseline': {
                'best_k': baseline_results['best_k'],
                'metrics': baseline_results['metrics'],
                'scan': baseline_results['scan_results'],
            },
            'modality_clustering': {
                name: {
                    'n_clusters': r['n_clusters'],
                    'noise_ratio': r['noise_ratio'],
                    'silhouette': r['silhouette'],
                }
                for name, r in modality_results.items()
            },
            'consensus': {
                'conflict_rate': conflict_rate,
                'n_conflicts': int(consensus['conflict_flags'].sum()),
            },
            'gmm': {
                'best_k': best_k_gmm,
                'bic': gmm_results['bic'],
                'aic': gmm_results['aic'],
            },
            'physics_filter': {
                'coherence_rate': filter_results['coherence_rate'],
                'n_feasible': len(filter_results['feasible_modes']),
                'n_infeasible': len(filter_results['infeasible_modes']),
                'n_fault_states': len(filter_results['fault_states']),
                'details': filter_results['results'],
            },
            'labels': {
                'baseline': baseline_results['labels'],
                'gmm': gmm_results['labels'],
                'refined': refined_labels,
            },
            'synthetic_faults_generated': synthetic_faults,
            'imdti': results_imdti,
            'cvtcc_causal_graph': results_cvtcc,
            'validated_modes': validated_modes,
            'elapsed_seconds': elapsed,
        }

        # Save results summary
        self._save_results(results)
        return results

    def _load_data(self) -> Dict:
        """Load data using appropriate adapter."""
        config_path = str(self.config_dir / (
            'cwru_params.json' if self.demo == 'cwru' else 'cmapss_params.json'
        ))
        if self.demo == 'cwru':
            adapter = CWRUAdapter(str(self.data_dir), config_path)
            return adapter.load()
        else:
            adapter = CMAPSSAdapter(str(self.data_dir), config_path)
            return adapter.load()

    def _extract_features(self, data: Dict):
        """Extract and preprocess features based on demo type."""
        if self.demo == 'cwru':
            return self._extract_cwru_features(data)
        else:
            return self._extract_cmapss_features(data)

    def _extract_cwru_features(self, data: Dict):
        """CWRU: window-based FFT/RMS features per vibration channel."""
        extractor = FeatureExtractor(sampling_rate=data['sampling_rate'])

        de_features = extractor.extract_batch(data['vibration_de'], prefix='de_')
        fe_features = extractor.extract_batch(data['vibration_fe'], prefix='fe_')

        # Joint features
        joint = pd.concat([de_features, fe_features], axis=1)
        joint, _ = preprocess_features(joint)

        # Modality feature matrices for per-modality clustering
        de_processed, _ = preprocess_features(de_features)
        fe_processed, _ = preprocess_features(fe_features)

        modality_matrices = {
            'vibration_de': de_processed.values,
            'vibration_fe': fe_processed.values,
        }

        return joint, modality_matrices, None

    def _extract_cmapss_features(self, data: Dict):
        """CMAPSS: rolling stats per modality group."""
        extractor = CMAPSSFeatureExtractor(window_size=30, stride=1)

        sensor_names = self.params.get('sensor_names', {})
        thermal_names = [sensor_names.get(f's{i+1}', f's{i+1}')
                         for i in self.params['thermal_indices']]
        pressure_names = [sensor_names.get(f's{i+1}', f's{i+1}')
                          for i in self.params['pressure_indices']]
        mech_names = [sensor_names.get(f's{i+1}', f's{i+1}')
                      for i in self.params['mechanical_indices']]

        thermal_feat = extractor.extract_modality_features(
            data['thermal'], data['unit_ids'], data['cycles'],
            thermal_names, prefix='thermal_'
        )
        pressure_feat = extractor.extract_modality_features(
            data['pressure'], data['unit_ids'], data['cycles'],
            pressure_names, prefix='pressure_'
        )
        mech_feat = extractor.extract_modality_features(
            data['mechanical'], data['unit_ids'], data['cycles'],
            mech_names, prefix='mechanical_'
        )

        # Cross-modal features
        cross = compute_cross_modal_features({
            'thermal': thermal_feat,
            'pressure': pressure_feat,
            'mechanical': mech_feat,
        })

        # Merge all
        merged = thermal_feat
        for df in [pressure_feat, mech_feat, cross]:
            if not df.empty and 'unit_id' in df.columns:
                merged = pd.merge(merged, df, on=['unit_id', 'cycle'], how='inner')

        exclude = ['unit_id', 'cycle']
        feature_cols = [c for c in merged.columns if c not in exclude]
        joint, _ = preprocess_features(merged[feature_cols])

        # Modality matrices
        t_cols = [c for c in thermal_feat.columns if c not in exclude]
        p_cols = [c for c in pressure_feat.columns if c not in exclude]
        m_cols = [c for c in mech_feat.columns if c not in exclude]

        t_proc, _ = preprocess_features(thermal_feat[t_cols])
        p_proc, _ = preprocess_features(pressure_feat[p_cols])
        m_proc, _ = preprocess_features(mech_feat[m_cols])

        modality_matrices = {
            'thermal': t_proc.values,
            'pressure': p_proc.values,
            'mechanical': m_proc.values,
        }

        metadata = merged[['unit_id', 'cycle']]

        return joint, modality_matrices, metadata

    def _build_mode_centroids(
        self,
        gmm_means: np.ndarray,
        feature_names: pd.Index,
        data: Dict,
    ) -> Dict[int, Dict[str, float]]:
        """
        Build physical-quantity centroids from GMM means.
        Maps feature-space centroids to physical interpretations.
        """
        centroids = {}
        for mode_id in range(gmm_means.shape[0]):
            centroid_vec = gmm_means[mode_id]
            centroid = {}

            if self.demo == 'cwru':
                # Map standardized features back to physical proxies
                # Use RMS as vibration proxy, mean as temperature proxy
                for i, name in enumerate(feature_names):
                    if 'rms' in name and 'de_' in name:
                        centroid['vibration_rms'] = abs(centroid_vec[i])
                    elif 'mean' in name and 'de_' in name:
                        centroid['temperature'] = centroid_vec[i] * 10 + 40
                        centroid['current'] = abs(centroid_vec[i]) * 2 + 1
                    elif 'kurtosis' in name:
                        centroid['thermal_rate'] = abs(centroid_vec[i]) * 0.5

                # Defaults
                centroid.setdefault('vibration_rms', 0.1)
                centroid.setdefault('temperature', 35.0)
                centroid.setdefault('current', self.params['rated_current_A'])
                centroid.setdefault('efficiency', self.params['eta_nominal'])
                centroid.setdefault('thermal_rate', 0.5)
                centroid.setdefault('correlation_I_T', 0.6)

            elif self.demo == 'cmapss':
                for i, name in enumerate(feature_names):
                    if 'Nf_mean' in name:
                        centroid['Nf'] = abs(centroid_vec[i]) * 500 + 2000
                    elif 'Nc_mean' in name:
                        centroid['Nc'] = abs(centroid_vec[i]) * 1000 + 8000
                    elif 'T50_mean' in name or 'T30_mean' in name:
                        centroid['T50'] = abs(centroid_vec[i]) * 200 + 1200
                    elif 'Wf_mean' in name:
                        centroid['fuel_flow'] = abs(centroid_vec[i]) * 10 + 300

                centroid.setdefault('Nf', 2100)
                centroid.setdefault('Nc', 8500)
                centroid.setdefault('T50', 1350)
                centroid.setdefault('correlation_fuel_T50', 0.55)

            centroids[mode_id] = centroid

        return centroids

    def _save_results(self, results: Dict):
        """Save results summary to disk."""
        summary_path = self.results_dir / f'{self.demo}_results_summary.txt'
        with open(summary_path, 'w') as f:
            f.write(f"Mode Discovery Results — {self.demo.upper()}\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Samples: {results['data_summary']['n_samples']}\n")
            f.write(f"Features: {results['data_summary']['n_features']}\n")
            f.write(f"Modalities: {results['data_summary']['modalities']}\n\n")

            f.write("Baseline (K-Means)\n")
            f.write(f"  Best k: {results['baseline']['best_k']}\n")
            for k, v in results['baseline']['metrics'].items():
                f.write(f"  {k}: {v:.4f}\n")

            f.write(f"\nGMM\n")
            f.write(f"  Best k: {results['gmm']['best_k']}\n")
            f.write(f"  BIC: {results['gmm']['bic']:.2f}\n")

            f.write(f"\nConsensus\n")
            f.write(f"  Conflict rate: {results['consensus']['conflict_rate']:.1%}\n")

            f.write(f"\nPhysics Filter\n")
            pf = results['physics_filter']
            f.write(f"  Coherence rate: {pf['coherence_rate']:.1%}\n")
            f.write(f"  Feasible: {pf['n_feasible']}\n")
            f.write(f"  Infeasible: {pf['n_infeasible']}\n")
            f.write(f"  Fault states: {pf['n_fault_states']}\n")

            f.write(f"\nElapsed: {results['elapsed_seconds']:.1f}s\n")

        print(f"\n  Results saved to {summary_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run Mode Discovery Pipeline')
    parser.add_argument('--demo', choices=['cwru', 'cmapss', 'both'], default='cwru')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--config-dir', type=str, default='configs')
    parser.add_argument('--results-dir', type=str, default='results')
    args = parser.parse_args()

    if args.demo in ['cwru', 'cmapss']:
        pipeline = ModeDiscoveryPipeline(
            demo=args.demo,
            data_dir=args.data_dir,
            config_dir=args.config_dir,
            results_dir=args.results_dir,
        )
        results = pipeline.run()
    elif args.demo == 'both':
        print("\n>>> RUNNING DEMO 1: CWRU <<<\n")
        pipe_cwru = ModeDiscoveryPipeline(demo='cwru', config_dir=args.config_dir, results_dir=args.results_dir)
        res_cwru = pipe_cwru.run()
        
        print("\n>>> RUNNING DEMO 2: CMAPSS <<<\n")
        pipe_cmapss = ModeDiscoveryPipeline(demo='cmapss', config_dir=args.config_dir, results_dir=args.results_dir)
        res_cmapss = pipe_cmapss.run()
        
        print("\n>>> RUNNING CROSS-DATASET DCFP ALIGNMENT <<<\n")
        aligner = CrossDatasetModeAligner()
        aligner.register_dataset_modes('cwru', res_cwru['validated_modes'])
        aligner.register_dataset_modes('cmapss', res_cmapss['validated_modes'])
        
        alignments = aligner.align_across_datasets('cwru', 'cmapss', similarity_threshold=0.70)
        print(f"Discovered {len(alignments)} Cross-Domain Analogous Modes:")
        for al in alignments:
            print(f"  - {al['interpretation']}")

