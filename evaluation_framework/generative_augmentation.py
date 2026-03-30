import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp

def evaluate_synthetic_data(X_real, X_synth, real_labels=None, synth_labels=None):
    """
    Evaluates the physics-bounded synthetic data generated to augment minority modes.
    
    X_real: Original feature matrix.
    X_synth: Generative/Augmented feature matrix.
    real_labels: Original mode labels.
    synth_labels: Augmented mode labels.
    """
    metrics = {}
    
    # 1. Distribution Similarity (Wasserstein Distance per feature)
    w_distances = []
    for f in range(X_real.shape[1]):
        wd = wasserstein_distance(X_real[:, f], X_synth[:, f])
        w_distances.append(wd)
    metrics['Mean_Wasserstein_Distance'] = np.mean(w_distances)
    
    # Kolmogorov-Smirnov test per feature (measures if distributions differ significantly)
    ks_stats = []
    for f in range(X_real.shape[1]):
        stat, p_val = ks_2samp(X_real[:, f], X_synth[:, f])
        ks_stats.append(stat)
    metrics['Mean_KS_Statistic'] = np.mean(ks_stats)
    
    # 2. Feature Space Coverage Increase
    # Compares the bounding hyperbox (volume) of real vs combined real+synth
    vol_real = np.prod(np.max(X_real, axis=0) - np.min(X_real, axis=0))
    
    X_combined = np.vstack((X_real, X_synth))
    vol_combined = np.prod(np.max(X_combined, axis=0) - np.min(X_combined, axis=0))
    
    metrics['Coverage_Increase_Ratio'] = (vol_combined / max(1e-9, vol_real)) - 1.0
    
    # 3. Synthetic vs Real Cluster Overlap Score (Using MMD or proxy)
    # Fast proxy: distance between centroids of same mode.
    if real_labels is not None and synth_labels is not None:
        overlap_scores = {}
        unique_modes = set(np.unique(real_labels)).intersection(set(np.unique(synth_labels)))
        
        for mode in unique_modes:
            m_real = X_real[real_labels == mode]
            m_synth = X_synth[synth_labels == mode]
            
            if len(m_real) > 0 and len(m_synth) > 0:
                cent_real = np.mean(m_real, axis=0)
                cent_synth = np.mean(m_synth, axis=0)
                # Euclidean distance between centroids as proxy for overlap
                # Ideally, this should be small but non-zero (indicating bounded diversity)
                overlap_scores[mode] = np.linalg.norm(cent_real - cent_synth)
                
        if overlap_scores:
            metrics['Mean_Centroid_Shift_Per_Mode'] = np.mean(list(overlap_scores.values()))
        else:
            metrics['Mean_Centroid_Shift_Per_Mode'] = np.nan
    
    # 4. Mode Diversity Gain
    # Variance change within specific modes before and after augmentation
    if real_labels is not None and synth_labels is not None:
        diversity_gain = {}
        for mode in unique_modes:
            var_real = np.mean(np.var(X_real[real_labels == mode], axis=0))
            var_synth = np.mean(np.var(X_synth[synth_labels == mode], axis=0))
            # Ratio of synthetic variance to real variance
            diversity_gain[mode] = var_synth / max(1e-9, var_real)
            
        if diversity_gain:
            metrics['Mean_Diversity_Gain_Ratio'] = np.mean(list(diversity_gain.values()))
        else:
            metrics['Mean_Diversity_Gain_Ratio'] = np.nan

    return metrics

if __name__ == "__main__":
    from dataset_simulator import IndustrialMultimodalSimulator
    
    sim_real = IndustrialMultimodalSimulator(n_samples=500, random_state=42)
    df_real = sim_real.generate_domain(domain_shift=0.0)
    
    # Synthetic generator creates slightly noisier data but bounded
    sim_synth = IndustrialMultimodalSimulator(n_samples=200, random_state=99)
    df_synth = sim_synth.generate_domain(domain_shift=0.1) # Small shift
    
    X_r = df_real[['Vibration', 'Current', 'Temperature']].values
    L_r = df_real['True_Mode'].values
    
    X_s = df_synth[['Vibration', 'Current', 'Temperature']].values
    L_s = df_synth['True_Mode'].values
    
    metrics = evaluate_synthetic_data(X_real=X_r, X_synth=X_s, real_labels=L_r, synth_labels=L_s)
    
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
