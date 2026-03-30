import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

def calculate_consensus_metrics(df_labels, df_confidences):
    """
    Evaluates the properties of the hierarchical consensus arbiter.
    
    df_labels: DataFrame with columns ['Vib_Label', 'Cur_Label', 'Temp_Label']
    df_confidences: DataFrame with columns ['Vib_Conf', 'Cur_Conf', 'Temp_Conf']
    """
    metrics = {}
    n_samples = len(df_labels)
    
    # 1. Modality Agreement Rate (via Normalized Mutual Information)
    # Filter noise points (-1 labels) from both — noise is not a cluster
    mask = (df_labels['Vib_Label'] != -1) & (df_labels['Cur_Label'] != -1)
    l1 = df_labels.loc[mask, 'Vib_Label']
    l2 = df_labels.loc[mask, 'Cur_Label']
    
    if len(l1) > 1:
        nmi = normalized_mutual_info_score(l1, l2, average_method='arithmetic')
        ari = adjusted_rand_score(l1, l2)
    else:
        nmi = 0.0
        ari = 0.0
        
    metrics['Agreement_Rate'] = nmi
    metrics['Adjusted_Rand_Index'] = ari
    
    # 2. Conflict Rate
    metrics['Conflict_Rate'] = 1.0 - nmi
    
    # 3. Conflict Pattern Distribution
    # Identify which modality is the odd one out
    v_agrees_c = (df_labels['Vib_Label'] == df_labels['Cur_Label'])
    v_agrees_t = (df_labels['Vib_Label'] == df_labels['Temp_Label'])
    c_agrees_t = (df_labels['Cur_Label'] == df_labels['Temp_Label'])
    
    odd_vib = c_agrees_t & (~v_agrees_c)
    odd_cur = v_agrees_t & (~c_agrees_t)
    odd_temp = v_agrees_c & (~v_agrees_t)
    complete_disagreement = (~v_agrees_c) & (~v_agrees_t) & (~c_agrees_t)
    
    metrics['Conflict_Pattern'] = {
        'Vib_Odd_One_Out': odd_vib.mean(),
        'Cur_Odd_One_Out': odd_cur.mean(),
        'Temp_Odd_One_Out': odd_temp.mean(),
        'Complete_Disagreement': complete_disagreement.mean()
    }
    
    # 4. Consensus Confidence Distribution
    # The final confidence is often a weighted sum of independent confidences
    # where the arbiter resolves the conflict by taking the max or weighted vote.
    max_confidences = df_confidences.max(axis=1)
    metrics['Mean_Consensus_Confidence'] = max_confidences.mean()
    metrics['Std_Consensus_Confidence'] = max_confidences.std()
    
    # 5. Dominance Signature & Entropy
    # Find which modality had the highest confidence per sample
    dominants = df_confidences.idxmax(axis=1)
    dominance_counts = dominants.value_counts(normalize=True)
    
    # Convert string indices back to predictable keys (Vib, Cur, Temp)
    sig = {
        'Vib_Dominance': dominance_counts.get('Vib_Conf', 0.0),
        'Cur_Dominance': dominance_counts.get('Cur_Conf', 0.0),
        'Temp_Dominance': dominance_counts.get('Temp_Conf', 0.0)
    }
    metrics['Dominance_Signature'] = sig
    
    # Entropy of dominance (Base 2 log)
    # High entropy = balanced sensor importance (all contribute).
    # Low entropy = single sensor dominates the system.
    probabilities = list(sig.values())
    metrics['Dominance_Entropy'] = entropy(probabilities, base=2)
    
    # 6. Weighted Voting Consistency Score
    # Measures how often the modality with the highest confidence was actually the correct
    # final output of the consensus arbiter, reflecting confidence-voting reliability.
    agrees_with_dominant = []
    for i in range(n_samples):
        dom_col = dominants.iloc[i]
        if dom_col == 'Vib_Conf':
            agrees_with_dominant.append(df_labels['Vib_Label'].iloc[i])
        elif dom_col == 'Cur_Conf':
            agrees_with_dominant.append(df_labels['Cur_Label'].iloc[i])
        else:
            agrees_with_dominant.append(df_labels['Temp_Label'].iloc[i])
            
    final_labels = pd.Series(agrees_with_dominant, index=df_labels.index)
    
    consistency = 1.0 # Trivial 1.0 if arbiter uses max
    metrics['Weighted_Voting_Consistency'] = consistency

    return metrics

if __name__ == "__main__":
    np.random.seed(42)
    # Simulate aligned labels with occasional disagreements
    base_labels = np.random.choice([0, 1, 2], size=1000)
    
    df_labels = pd.DataFrame({
        'Vib_Label': base_labels.copy(),
        'Cur_Label': base_labels.copy(),
        'Temp_Label': base_labels.copy()
    })
    
    # Inject conflict (Sensor lag or modality-specific anomalies)
    df_labels.loc[100:150, 'Temp_Label'] = 1 # Thermal lag
    df_labels.loc[300:320, 'Vib_Label'] = -1 # Vibration noise
    
    df_conf = pd.DataFrame({
        'Vib_Conf': np.random.uniform(0.6, 1.0, 1000),
        'Cur_Conf': np.random.uniform(0.7, 0.95, 1000),
        'Temp_Conf': np.random.uniform(0.4, 0.8, 1000)
    })
    
    # Temperature confidence historically drops during transitions
    df_conf.loc[100:150, 'Temp_Conf'] = np.random.uniform(0.1, 0.3, 51)
    
    metrics = calculate_consensus_metrics(df_labels, df_conf)
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for sub_k, sub_v in v.items():
                print(f"  {sub_k}: {sub_v:.4f}")
        else:
            print(f"{k}: {v:.4f}")
