import numpy as np
import pandas as pd
from scipy.stats import entropy
from collections import Counter

def calculate_temporal_metrics(sequence, risk_scores=None, K=10, window_size=50, high_risk_threshold=0.6):
    """
    Evaluates the temporal evolution of discovered operational modes.
    
    sequence: 1D numpy array or pandas Series of mode labels over time.
    risk_scores: 1D numpy array of normalized risk scores corresponding to sequence.
    K: Total number of discovered modes.
    """
    metrics = {}
    
    # 1. Mode Persistence Duration (Run-length encoding)
    def rle(inarray):
        """ run length encoding. """
        ia = np.asarray(inarray)
        n = len(ia)
        if n == 0: 
            return (None, None, None)
        else:
            y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)   # must include last element pos
            z = np.diff(np.append(-1, i))       # run lengths
            p = ia[i]                           # values
            return(p, z)
            
    modes, durations = rle(sequence)
    avg_persistence = {mode: np.mean(durations[modes == mode]) for mode in set(modes)}
    
    metrics['Mean_Persistence_Duration'] = avg_persistence
    
    # 2. Transition Probability Matrix (TPM)
    unique_modes = sorted(list(set(sequence)))
    n_modes = len(unique_modes)
    mode_to_idx = {m: i for i, m in enumerate(unique_modes)}
    idx_to_mode = {i: m for i, m in enumerate(unique_modes)}
    
    tpm = np.zeros((n_modes, n_modes))
    
    for i in range(len(sequence) - 1):
        m1 = sequence[i]
        m2 = sequence[i+1]
        tpm[mode_to_idx[m1], mode_to_idx[m2]] += 1
        
    # Normalize rows
    row_sums = tpm.sum(axis=1)
    tpm_norm = np.divide(tpm, row_sums[:, np.newaxis], out=np.zeros_like(tpm), where=row_sums[:, np.newaxis]!=0)
    
    metrics['Transition_Probability_Matrix'] = pd.DataFrame(tpm_norm, index=unique_modes, columns=unique_modes)
    
    # 3. Transition Entropy 
    # H = - sum(p * log(p)) over the TPM. High entropy = chaotic state thrashing.
    # We calculate the conditional entropy H(X_{t+1} | X_{t})
    mode_probs = np.array([sum(sequence == m) for m in unique_modes]) / len(sequence)
    ent_per_state = [entropy(tpm_norm[i], base=2) for i in range(n_modes)]
    cond_entropy = np.sum(mode_probs * ent_per_state)
    
    metrics['System_Transition_Entropy'] = cond_entropy
    
    # 4. Rare Transition Frequency
    # Transitions that have < 5% probability in the overall history
    flat_transitions = []
    for i in range(len(sequence) - 1):
        m1 = sequence[i]
        m2 = sequence[i+1]
        if m1 != m2: # Only inter-mode jumps
            prob = tpm_norm[mode_to_idx[m1], mode_to_idx[m2]]
            if prob < 0.05 and prob > 0:
                flat_transitions.append((m1, m2))
                
    metrics['Rare_Transition_Frequency'] = len(flat_transitions) / max(1, len(durations) - 1)
    
    # 5. Stability Index
    # Ratio of time staying in same state vs changing. (Also derivable from TPM trace).
    same_state_transitions = np.trace(tpm)
    all_transitions = np.sum(tpm)
    metrics['Stability_Index'] = same_state_transitions / max(1, all_transitions)
    
    # 6. Sequence Pattern Mining (Top Topologies)
    # Extracts the most common sequence length of 3
    if len(modes) > 3:
        n_gram = 3
        seqs = [tuple(modes[i:i+n_gram]) for i in range(len(modes) - n_gram + 1)]
        top_seqs = Counter(seqs).most_common(5)
        metrics['Top_Behavioral_Sequences'] = top_seqs
    else:
        metrics['Top_Behavioral_Sequences'] = []
        
    # 7. Inter-Mode Degradation Trajectory Index (IMDTI)
    if risk_scores is not None and len(risk_scores) == len(sequence):
        # Implement IMDTI exactly per patent Claim 3
        n = len(sequence)
        imdti_series = []
        
        w1, w2, w3 = 0.50, 0.25, 0.25
        
        for t in range(window_size, n):
            window_labels = sequence[t - window_size : t]
            window_risks  = risk_scores[t - window_size : t]

            # Component 1: mean risk score — already [0,1]
            avg_risk = np.clip(np.mean(window_risks), 0, 1)

            # Component 2: transition entropy
            unique_labels, counts = np.unique(window_labels, return_counts=True)
            probs = counts / counts.sum()
            raw_entropy = -np.sum(probs * np.log(probs + 1e-10))
            norm_entropy = np.clip(raw_entropy / np.log(K + 1e-10), 0, 1)

            # Component 3: high-risk dwell fraction
            high_risk_frac = np.mean(window_risks >= high_risk_threshold)

            # Weighted combination
            imdti_val = w1 * avg_risk + w2 * norm_entropy + w3 * high_risk_frac
            imdti_series.append(np.clip(imdti_val, 0, 1))

        if len(imdti_series) > 0:
            metrics['IMDTI'] = float(np.mean(imdti_series))
        else:
            metrics['IMDTI'] = 0.0
    else:
        metrics['IMDTI'] = np.nan
        
    return metrics

if __name__ == "__main__":
    from dataset_simulator import IndustrialMultimodalSimulator
    
    sim = IndustrialMultimodalSimulator(n_samples=1000)
    df = sim.generate_domain()
    
    seq = df['True_Mode'].values
    # Fake centroids just for testing
    centroids = {
        0: [0.0, 0.0],
        1: [1.0, 1.0],
        2: [2.0, 2.0],
        3: [5.0, 5.0]
    }
    
    metrics = calculate_temporal_metrics(seq, centroids=centroids)
    for k, v in metrics.items():
        if isinstance(v, pd.DataFrame):
            print(f"\n{k}:")
            print(v)
        else:
            print(f"{k}: {v}")
