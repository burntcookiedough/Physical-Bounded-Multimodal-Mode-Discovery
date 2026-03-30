import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class EvaluationVisualizer:
    def __init__(self, output_dir="evaluation_framework/output_plots"):
        self.out_dir = output_dir
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        # Apply research-style theme
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        
    def _save(self, filename):
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    # --- 1. CLUSTERING PLOTS ---
    def plot_latent_space(self, X, labels, method='pca', filename="clustering_latent.png"):
        """PCA or t-SNE plot highlighting clusters and noise (-1)."""
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:
            reducer = PCA(n_components=2)
            
        X_reduced = reducer.fit_transform(X)
        
        plt.figure(figsize=(8, 6))
        
        # Plot noise first (grey)
        noise_idx = labels == -1
        plt.scatter(X_reduced[noise_idx, 0], X_reduced[noise_idx, 1], 
                    c='grey', alpha=0.3, label='Noise/Anomalies', s=20)
        
        # Plot valid clusters
        valid_idx = labels != -1
        scatter = plt.scatter(X_reduced[valid_idx, 0], X_reduced[valid_idx, 1], 
                              c=labels[valid_idx], cmap='tab10', alpha=0.8, s=30)
        
        plt.title(f"{method.upper()} Projection of Multimodal Feature Space")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.colorbar(scatter, label="Mode Label")
        plt.legend()
        self._save(filename)

    # --- 2. CONSENSUS & CONFLICT PLOTS ---
    def plot_dominance_signature(self, dominance_dict, filename="consensus_dominance.png"):
        """Bar chart showing which sensor type dominated the consensus decision."""
        plt.figure(figsize=(6, 4))
        sensors = list(dominance_dict.keys())
        values = list(dominance_dict.values())
        
        sns.barplot(x=sensors, y=values, palette="viridis")
        plt.title("Sensor Dominance Signature across Modes")
        plt.ylabel("Dominance Probability")
        plt.ylim(0, 1)
        self._save(filename)
        
    def plot_conflict_heatmap(self, conflict_dict, filename="consensus_conflict_heat.png"):
        """Heatmap of conflict patterns."""
        plt.figure(figsize=(6, 4))
        df_conf = pd.DataFrame([conflict_dict])
        sns.heatmap(df_conf.T, annot=True, cmap="YlOrRd", fmt=".3f", cbar=False)
        plt.title("Conflict Pattern Distribution")
        plt.ylabel("Disagreement Type")
        self._save(filename)

    # --- 3. TRANSITION PLOTS ---
    def plot_transition_matrix(self, tpm_df, filename="transition_tpm.png"):
        """Heatmap of the mode Transition Probability Matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(tpm_df, annot=True, cmap="Blues", fmt=".2f", vmin=0, vmax=1)
        plt.title("Mode Transition Probability Matrix")
        plt.xlabel("Next Mode")
        plt.ylabel("Current Mode")
        self._save(filename)
        
    def plot_mode_sequence(self, sequence, time_axis=None, filename="transition_sequence.png"):
        """Time-series step plot of operational modes."""
        plt.figure(figsize=(12, 4))
        if time_axis is None:
            time_axis = np.arange(len(sequence))
            
        plt.step(time_axis, sequence, where='post', color="#2ca02c", linewidth=2)
        plt.title("Operational Mode Sequence over Time")
        plt.xlabel("Time (Windows)")
        plt.ylabel("Discovered Mode Label")
        plt.yticks(np.unique(sequence))
        self._save(filename)

    # --- 4. PHYSICS VALIDATION PLOTS ---
    def plot_physics_scatter(self, df, mode_labels, filename="physics_scatter.png"):
        """2D Phase plot. Current vs Temperature, or Vibration DE vs Vibration FE depending on context."""
        plt.figure(figsize=(8, 6))
        
        if 'Current' in df.columns and 'Temperature' in df.columns:
            scatter = plt.scatter(df['Current'], df['Temperature'], c=mode_labels, cmap='tab10', alpha=0.6, s=15)
            x_bounds = np.linspace(df['Current'].min(), df['Current'].max(), 100)
            y_bounds = 25.0 + 0.1 * (x_bounds**2)
            plt.plot(x_bounds, y_bounds, 'r--', label='Thermodynamic Bound (Proxy)', linewidth=2)
            plt.title("Cross-Modal Physics Constraint Mapping (Current vs Temp)")
            plt.xlabel("Current (A)")
            plt.ylabel("Temperature (°C)")
        elif 'vibration_de' in df.columns and 'vibration_fe' in df.columns:
            scatter = plt.scatter(df['vibration_de'], df['vibration_fe'], c=mode_labels, cmap='tab10', alpha=0.6, s=15)
            plt.axline((0, 0), slope=1, color='r', linestyle='--', label='1:1 Transfer Bound')
            plt.title("Cross-Modal Physics Constraint Mapping (Vib DE vs FE)")
            plt.xlabel("Vibration DE (RMS)")
            plt.ylabel("Vibration FE (RMS)")
        else:
            cols = df.columns[:2]
            scatter = plt.scatter(df[cols[0]], df[cols[1]], c=mode_labels, cmap='tab10', alpha=0.6, s=15)
            plt.title(f"Phase Portrait ({cols[0]} vs {cols[1]})")
            plt.xlabel(cols[0])
            plt.ylabel(cols[1])
            
        plt.colorbar(scatter, label="Mode")
        plt.legend()
        self._save(filename)

    # --- 5. GENERATIVE & SYSTEM PLOTS ---
    def plot_real_vs_synthetic(self, X_real, X_synth, feature_idx=1, filename="generative_dist.png"):
        """KDE overlay of real vs synthetic data to show distribution similarity & bounding."""
        plt.figure(figsize=(8, 5))
        sns.kdeplot(X_real[:, feature_idx], fill=True, label="Real Data", color="blue", alpha=0.5)
        sns.kdeplot(X_synth[:, feature_idx], fill=True, label="Physics-Bounded Synthetic", color="orange", alpha=0.5)
        plt.title(f"Real vs Generative Distribution Overlay (Feature {feature_idx})")
        plt.xlabel("Feature Magnitude")
        plt.ylabel("Density")
        plt.legend()
        self._save(filename)

    def plot_throughput_scaling(self, scalability_dict, filename="system_scaling.png"):
        """Line plot of system latency vs batch size."""
        plt.figure(figsize=(8, 5))
        sizes = list(scalability_dict.keys())
        times = list(scalability_dict.values())
        
        plt.plot(sizes, times, marker='o', linestyle='-', color="#d62728", linewidth=2)
        plt.title("Inference Latency Scalability")
        plt.xlabel("Batch Size (Windows)")
        plt.ylabel("Processing Time (Seconds)")
        plt.grid(True, which="both", ls="--")
        self._save(filename)

# Example usage tester
if __name__ == "__main__":
    from dataset_simulator import IndustrialMultimodalSimulator
    
    vis = EvaluationVisualizer()
    sim = IndustrialMultimodalSimulator(n_samples=500)
    df = sim.generate_domain()
    
    # Fake tests
    vis.plot_latent_space(df[['Vibration', 'Current', 'Temperature']].values, df['True_Mode'].values, method='pca')
    vis.plot_physics_scatter(df, df['True_Mode'].values)
    vis.plot_mode_sequence(df['True_Mode'].values[:100])
    
    print(f"Generated test plots in {vis.out_dir}")
