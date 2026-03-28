import numpy as np
from collections import Counter

def compute_degradation_trajectory(unit_id, time_ordered_mode_labels, 
                                    mode_risk_scores, window_size=20):
    """
    For a single engine unit, compute its degradation trajectory
    through discovered mode space over time (cycles).
    
    Returns: trajectory dict with IMDTI scores over time + degradation rate
    """
    n = len(time_ordered_mode_labels)
    imdti_trajectory = []
    
    for t in range(window_size, n):
        window_modes = time_ordered_mode_labels[t - window_size : t]
        
        # safely handle missing modes in mode_risk_scores
        window_risks = [mode_risk_scores.get(m, {}).get('risk_score', 0.0) for m in window_modes]
        
        # Metric 1: Average risk in window (absolute health state)
        avg_risk = np.mean(window_risks) if window_risks else 0.0
        
        # Metric 2: Mode transition entropy (higher entropy = more unstable)
        mode_counts = Counter(window_modes)
        total = sum(mode_counts.values())
        if total > 0:
            probs = [c / total for c in mode_counts.values()]
            transition_entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        else:
            transition_entropy = 0.0
            
        # Normalize entropy
        max_modes = max(len(mode_risk_scores), 1)
        norm_entropy = transition_entropy / np.log(max_modes + 1e-10)
        
        # Metric 3: High-risk mode dwell time (fraction of window in FAULT/CRITICAL modes)
        high_risk_fraction = sum(1 for r in window_risks if r > 0.5) / window_size if window_size > 0 else 0.0
        
        # Composite IMDTI
        imdti = 0.50 * avg_risk + 0.25 * norm_entropy + 0.25 * high_risk_fraction
        
        dom_mode = mode_counts.most_common(1)[0][0] if mode_counts else None
        
        imdti_trajectory.append({
            'cycle': t,
            'imdti': imdti,
            'avg_risk': avg_risk,
            'transition_entropy': transition_entropy,
            'high_risk_dwell': high_risk_fraction,
            'dominant_mode': dom_mode
        })
    
    # Compute degradation rate (slope of IMDTI over last 50 cycles)
    if len(imdti_trajectory) >= 50:
        recent = [x['imdti'] for x in imdti_trajectory[-50:]]
        degradation_rate = np.polyfit(range(50), recent, 1)[0]
    else:
        degradation_rate = 0.0
    
    return {
        'unit_id': unit_id,
        'trajectory': imdti_trajectory,
        'degradation_rate': float(degradation_rate),
        'predicted_failure_cycles': estimate_rul_from_trajectory(imdti_trajectory)
    }

def estimate_rul_from_trajectory(trajectory, failure_threshold=0.85):
    """
    Fit a linear trend to IMDTI and extrapolate to failure threshold.
    Gives a RUL estimate in cycles purely from unsupervised mode structure.
    """
    imdti_values = [x['imdti'] for x in trajectory]
    cycles = list(range(len(imdti_values)))
    
    if len(imdti_values) < 10:
        return None
    
    slope, intercept = np.polyfit(cycles, imdti_values, 1)
    if slope <= 0:
        return float('inf')  # Not degrading
    
    cycles_to_failure = (failure_threshold - intercept) / slope
    current_cycle = cycles[-1]
    rul = max(0.0, cycles_to_failure - current_cycle)
    return float(rul)
