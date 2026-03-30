import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings

def compute_clustering_metrics(X, labels):
    """
    Computes Core ML Clustering Metrics.
    X: Feature matrix (numpy array or pandas DataFrame)
    labels: Cluster labels (from HDBSCAN or KMeans). Noise should be -1.
    """
    metrics = {}
    
    # Filter out noise points for standard centroid-based metrics
    core_mask = labels != -1
    X_core = X[core_mask] if isinstance(X, np.ndarray) else X.iloc[core_mask].values
    labels_core = labels[core_mask]
    
    # 1. Noise/Outlier Ratio
    metrics['Noise_Ratio'] = np.mean(labels == -1)
    
    if len(np.unique(labels_core)) > 1:
        # 2. Silhouette Score (Intra vs Inter distance)
        metrics['Silhouette'] = silhouette_score(X_core, labels_core)
        
        # 3. Davies-Bouldin Index (Cluster similarity)
        metrics['Davies_Bouldin'] = davies_bouldin_score(X_core, labels_core)
        
        # 4. Calinski-Harabasz Score (Variance ratio)
        metrics['Calinski_Harabasz'] = calinski_harabasz_score(X_core, labels_core)
    else:
        metrics['Silhouette'] = np.nan
        metrics['Davies_Bouldin'] = np.nan
        metrics['Calinski_Harabasz'] = np.nan
        
    # 5. DBCV (Density-Based Clustering Validation)
    try:
        import hdbscan
        # DBCV expects all points including noise, HDBSCAN validity index handles -1
        score = hdbscan.validity.validity_index(X.astype(np.float64), labels)
        metrics['DBCV'] = score
    except ImportError:
        warnings.warn("hdbscan module not found. Skipping DBCV calculation.")
        metrics['DBCV'] = np.nan
    except ValueError:
        # Happens if hdbscan fails to compute validity (e.g. all points are noise)
        metrics['DBCV'] = np.nan
        
    return metrics

def compute_cluster_stability(X_list, labels_list):
    """
    Computes clustering stability across multiple runs (e.g., bootstrapping or cross-domain).
    For simplicity, measures the Adjusted Rand Index (ARI) if comparing labels of same points,
    or the variance of Silhouette scores.
    """
    from sklearn.metrics import adjusted_rand_score
    
    if len(labels_list) < 2:
        return np.nan
        
    aris = []
    for i in range(len(labels_list)-1):
        ari = adjusted_rand_score(labels_list[i], labels_list[i+1])
        aris.append(ari)
        
    return np.mean(aris)

if __name__ == "__main__":
    from dataset_simulator import IndustrialMultimodalSimulator
    
    sim = IndustrialMultimodalSimulator(n_samples=500)
    df = sim.generate_domain()
    
    # Synthetic clustering (pretending True_Mode are predicted labels, with some noise)
    labels = df['True_Mode'].values.copy()
    labels[df['Is_Violation']] = -1 # Let anomalies be noise
    
    X = df[['Vibration', 'Current', 'Temperature']].values
    
    metrics = compute_clustering_metrics(X, labels)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
