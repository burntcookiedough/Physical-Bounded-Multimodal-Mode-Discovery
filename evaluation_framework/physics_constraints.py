import numpy as np
import pandas as pd

def compute_physics_metrics(df, mode_labels, time_step=1.0, config=None):
    """
    Evaluates the physical feasibility of the discovered operational modes.
    
    df: DataFrame containing required columns based on config.
    mode_labels: The final consensus mode labels assigned to each sample.
    time_step: The sampling interval in seconds (e.g., 1.0s).
    config: parameter dictionary loaded from JSON.
    """
    metrics = {}
    
    act = {}
    if config and 'constraint_activation' in config:
        act = config['constraint_activation']
    else:
        # Default to old behavior for backward compat
        act = {
            'joule_heating': True,
            'cross_modal_coupling': {
                'active': True,
                'mode': 'current_temperature',
                'channels': ['Current', 'Temperature']
            }
        }
    
    N = len(df)
    total_violations = np.zeros(N, dtype=bool)
    V = df['Vibration'].values
    N = len(df)
    
    # 1. Thermal Rate & Joule Heating (Only if active)
    if act.get('joule_heating', True) and 'Current' in df.columns and 'Temperature' in df.columns:
        I = df['Current'].values
        T = df['Temperature'].values
        dT = np.diff(T) / time_step
        dT = np.insert(dT, 0, 0)
        
        max_physical_rate = 5.0
        thermal_violations = np.abs(dT) > max_physical_rate
        
        alpha = 0.05 
        predicted_dT = alpha * (I**2)
        cooling_factor = 0.05 * (T - 25.0)
        energy_balance_error = np.abs(dT - (predicted_dT - cooling_factor))
        
        metrics['Joule_Heating_Consistency_Error'] = np.mean(energy_balance_error)
        eb_violations = energy_balance_error > 10.0
        
        total_violations |= thermal_violations
        total_violations |= eb_violations
        
        metrics['Violation_Breakdown'] = {
            'Thermal_Rate_Violations': np.mean(thermal_violations),
            'Energy_Balance_Violations': np.mean(eb_violations)
        }
    else:
        metrics['Joule_Heating_Consistency_Error'] = np.nan
        metrics['Violation_Breakdown'] = {}

    # 2. Vibration Limit Violation
    if 'Vibration' in df.columns:
        V = df['Vibration'].values
        max_vibration = 1.5 
        vib_violations = V > max_vibration
        total_violations |= vib_violations
        metrics['Violation_Breakdown']['Vibration_Limit_Violations'] = np.mean(vib_violations)
    
    metrics['Overall_Constraint_Violation_Rate'] = np.mean(total_violations)
    
    # 3. Cross-Modal Correlation
    cross_conf = act.get('cross_modal_coupling', {})
    if cross_conf.get('active', False):
        channels = cross_conf.get('channels', [])
        if len(channels) == 2 and channels[0] in df.columns and channels[1] in df.columns:
            ch1 = df[channels[0]].values
            ch2 = df[channels[1]].values
            
            d1 = np.insert(np.diff(ch1), 0, 0)
            d2 = np.insert(np.diff(ch2), 0, 0)
            
            if len(d1) > 1:
                corr = np.corrcoef(d1, d2)[0, 1]
            else:
                corr = np.nan
                
            metrics['Cross_Modal_Correlation'] = corr
            metrics['Cross_Modal_Mode'] = cross_conf.get('mode', 'unknown')
        else:
            metrics['Cross_Modal_Correlation'] = np.nan
    else:
        metrics['Cross_Modal_Correlation'] = np.nan
    
    # 5. Physical Feasibility Score per cluster
    # What % of points in a discovered mode pass all physical filters?
    cluster_feasibility = {}
    unique_modes = np.unique(mode_labels)
    
    for mode in unique_modes:
        idx = mode_labels == mode
        if np.sum(idx) > 0:
            passes_physics = ~total_violations[idx]
            cluster_feasibility[mode] = np.mean(passes_physics)
            
    metrics['Physical_Feasibility_Per_Cluster'] = cluster_feasibility
    
    # 6. Percentage of Clusters Rejected
    # If a cluster is < 95% physically feasible, the whole mode might be an artifact (sensor error)
    feasibility_threshold = 0.95
    rejected_clusters = sum(1 for v in cluster_feasibility.values() if v < feasibility_threshold)
    metrics['Percentage_Clusters_Rejected'] = rejected_clusters / max(1, len(unique_modes))
    
    return metrics

if __name__ == "__main__":
    from dataset_simulator import IndustrialMultimodalSimulator
    
    sim = IndustrialMultimodalSimulator(n_samples=500)
    df = sim.generate_domain()
    mode_labels = df['True_Mode'].values
    
    metrics = compute_physics_metrics(df, mode_labels)
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"\n{k}:")
            for sub_k, sub_v in v.items():
                print(f"  {sub_k}: {sub_v:.4f}")
        else:
            print(f"{k}: {v:.4f}")
