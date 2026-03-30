import numpy as np
from sklearn.metrics import adjusted_rand_score, v_measure_score
from scipy.stats import wasserstein_distance

def evaluate_cross_domain_robustness(X_source, X_target, labels_source, labels_target_mapped):
    """
    Evaluates cross-domain generalization (e.g., CWRU vs CMAPSS, or Asset A vs Asset B).
    
    X_source: Features from Domain A.
    X_target: Features from Domain B.
    labels_source: Ground truth or discovered modes for source.
    labels_target_mapped: Predicted modes in Domain B using Domain A's aligned model.
    """
    metrics = {}
    
    # 1. Feature Distribution Shift (Wasserstein)
    # Measures how far the raw physics manifolds have drifted between domains
    w_distances = []
    for f in range(X_source.shape[1]):
        w_distances.append(wasserstein_distance(X_source[:, f], X_target[:, f]))
    metrics['Domain_Shift_Intensity'] = np.mean(w_distances)
    
    # 2. Domain Transfer Consistency Score 
    # How well does the model preserve the topological relationships?
    # Proxy: Are the transition matrices similar?
    # Here, we measure label consistency if we have ground truth for target.
    # Assuming labels_target_mapped is evaluated against theoretical ideal mappings
    
    # 3. Cluster Alignment Score
    # We compare the variance and topology of modes between source and target
    # If Domain B is just Domain A shifted, the intra-cluster variance should remain identical
    alignment_scores = []
    common_modes = set(np.unique(labels_source)).intersection(set(np.unique(labels_target_mapped)))
    
    for mode in common_modes:
        src_mode_data = X_source[labels_source == mode]
        tgt_mode_data = X_target[labels_target_mapped == mode]
        
        if len(src_mode_data) > 0 and len(tgt_mode_data) > 0:
            var_src = np.var(src_mode_data, axis=0)
            var_tgt = np.var(tgt_mode_data, axis=0)
            
            # Ratio of variances (Target / Source)
            # A score near 1.0 means perfect structural alignment
            var_ratio = np.mean(var_tgt / max(1e-9, np.mean(var_src)))
            alignment_scores.append(abs(1.0 - var_ratio))
            
    if alignment_scores:
        metrics['Mean_Topological_Deformation'] = np.mean(alignment_scores)
    else:
        metrics['Mean_Topological_Deformation'] = np.nan
        
    # 4. Mode Mapping Stability (Proxy)
    # Are rare modes preserved, or did they collapse?
    src_dist = np.array([np.sum(labels_source == m) for m in common_modes]) / len(labels_source)
    tgt_dist = np.array([np.sum(labels_target_mapped == m) for m in common_modes]) / len(labels_target_mapped)
    
    if len(src_dist) > 0 and len(tgt_dist) > 0:
        kl_div = np.sum(np.where(src_dist != 0, src_dist * np.log((src_dist + 1e-9) / (tgt_dist + 1e-9)), 0))
        metrics['Mode_Mapping_KL_Divergence'] = kl_div
    else:
        metrics['Mode_Mapping_KL_Divergence'] = np.nan
        
    return metrics

if __name__ == "__main__":
    from dataset_simulator import IndustrialMultimodalSimulator
    
    # Source Domain (e.g., Asset A)
    sim_a = IndustrialMultimodalSimulator(n_samples=1000, random_state=42)
    df_src = sim_a.generate_domain(domain_shift=0.0)
    
    # Target Domain (e.g., Asset B, structurally similar but baseline temps/vib differ)
    sim_b = IndustrialMultimodalSimulator(n_samples=500, random_state=99)
    df_tgt = sim_b.generate_domain(domain_shift=5.0) # Domain gap
    
    X_src = df_src[['Vibration', 'Current', 'Temperature']].values
    L_src = df_src['True_Mode'].values
    
    X_tgt = df_tgt[['Vibration', 'Current', 'Temperature']].values
    L_tgt_pred = df_tgt['True_Mode'].values # Assuming perfect alignment for test
    
    # Introduce mapping error
    np.random.seed(1)
    mask = np.random.rand(500) < 0.1
    L_tgt_pred[mask] = 0 # Misclassify 10% as Off mode
    
    metrics = evaluate_cross_domain_robustness(X_src, X_tgt, L_src, L_tgt_pred)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
