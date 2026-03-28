"""
Visualization for mode discovery results.

Generates comparison charts, silhouette plots, transition matrices,
dominance heatmaps, and physics filter summaries.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path

# Use a clean, publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')


def plot_silhouette_scan(scan_results: pd.DataFrame, save_path: Optional[str] = None):
    """Plot silhouette score vs. k from baseline scan."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(scan_results['k'], scan_results['silhouette'], 'o-', linewidth=2)
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Baseline K-Means: Silhouette Scan', fontsize=14)
    ax.grid(True, alpha=0.3)

    best_idx = scan_results['silhouette'].idxmax()
    best_k = scan_results.loc[best_idx, 'k']
    best_sil = scan_results.loc[best_idx, 'silhouette']
    ax.annotate(f'Best k={best_k}\nSil={best_sil:.3f}',
                xy=(best_k, best_sil),
                xytext=(best_k + 0.5, best_sil - 0.02),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_gmm_bic_scan(scan_df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot BIC/AIC vs. k for GMM component scan."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(scan_df['k'], scan_df['bic'], 'o-', label='BIC', linewidth=2)
    ax.plot(scan_df['k'], scan_df['aic'], 's--', label='AIC', linewidth=2, alpha=0.7)
    ax.set_xlabel('Number of Components (k)', fontsize=12)
    ax.set_ylabel('Information Criterion', fontsize=12)
    ax.set_title('GMM Component Selection', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_transition_matrix(T: np.ndarray, save_path: Optional[str] = None):
    """Plot mode transition probability matrix as heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(T, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                xticklabels=[f'Mode {i}' for i in range(T.shape[1])],
                yticklabels=[f'Mode {i}' for i in range(T.shape[0])])
    ax.set_xlabel('To Mode', fontsize=12)
    ax.set_ylabel('From Mode', fontsize=12)
    ax.set_title('Mode Transition Probability Matrix', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_modality_comparison(modality_results: Dict, save_path: Optional[str] = None):
    """Bar chart comparing per-modality clustering metrics."""
    names = list(modality_results.keys())
    n_clusters = [modality_results[n]['n_clusters'] for n in names]
    noise_ratios = [modality_results[n]['noise_ratio'] for n in names]
    silhouettes = [modality_results[n]['silhouette'] or 0 for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].bar(names, n_clusters, color=sns.color_palette('Set2'))
    axes[0].set_title('Clusters per Modality', fontsize=13)
    axes[0].set_ylabel('Count')

    axes[1].bar(names, noise_ratios, color=sns.color_palette('Set2'))
    axes[1].set_title('Noise Ratio per Modality', fontsize=13)
    axes[1].set_ylabel('Ratio')

    axes[2].bar(names, silhouettes, color=sns.color_palette('Set2'))
    axes[2].set_title('Silhouette per Modality', fontsize=13)
    axes[2].set_ylabel('Score')

    for ax in axes:
        ax.tick_params(axis='x', rotation=15)

    plt.suptitle('Per-Modality HDBSCAN Results', fontsize=15, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_dominance_heatmap(
    dominance_sigs: np.ndarray,
    modality_names: List[str],
    labels: np.ndarray,
    save_path: Optional[str] = None,
):
    """Heatmap of average dominance signatures per mode."""
    unique_labels = np.unique(labels[labels >= 0])
    avg_dom = np.zeros((len(unique_labels), len(modality_names)))

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        avg_dom[i] = dominance_sigs[mask].mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(avg_dom, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                xticklabels=modality_names,
                yticklabels=[f'Mode {l}' for l in unique_labels])
    ax.set_xlabel('Modality', fontsize=12)
    ax.set_ylabel('Discovered Mode', fontsize=12)
    ax.set_title('Dominance Signatures: Which Modality Drives Each Mode', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_physics_filter_summary(filter_results: Dict, save_path: Optional[str] = None):
    """Stacked bar showing feasible/infeasible/fault modes."""
    n_f = filter_results['n_feasible'] - filter_results['n_fault_states']
    n_i = filter_results['n_infeasible']
    n_fault = filter_results['n_fault_states']

    fig, ax = plt.subplots(figsize=(6, 5))
    categories = ['Feasible', 'Infeasible\n(phantom)', 'Fault State\n(Claim 7)']
    counts = [n_f, n_i, n_fault]
    colors = ['#2ecc71', '#e74c3c', '#f39c12']

    bars = ax.bar(categories, counts, color=colors, edgecolor='white', linewidth=1.5)
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    str(count), ha='center', fontsize=13, fontweight='bold')

    ax.set_ylabel('Number of Modes', fontsize=12)
    ax.set_title(f'Physics Filter Results (Coherence: {filter_results["coherence_rate"]:.0%})',
                 fontsize=14)
    ax.set_ylim(0, max(counts) + 1.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_baseline_vs_novel(
    baseline_metrics: Dict,
    novel_metrics: Dict,
    save_path: Optional[str] = None,
):
    """Side-by-side comparison of baseline vs. novel pipeline metrics."""
    metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
    baseline_vals = [baseline_metrics.get(m, 0) for m in metrics]
    novel_vals = [novel_metrics.get(m, 0) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline (K-Means)',
                   color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, novel_vals, width, label='Novel (HDBSCAN+GMM+Physics)',
                   color='#e74c3c', alpha=0.8)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Baseline vs. Novel Pipeline Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(['Silhouette ↑', 'Davies-Bouldin ↓', 'Calinski-Harabasz ↑'],
                       fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def generate_all_plots(results: Dict, save_dir: str = 'results'):
    """Generate all visualization plots from pipeline results."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    demo = results.get('demo', 'unknown')

    print(f"\nGenerating plots for {demo.upper()}...")

    # 1. Silhouette scan
    if 'scan' in results.get('baseline', {}):
        plot_silhouette_scan(
            results['baseline']['scan'],
            save_path=str(save_dir / f'{demo}_silhouette_scan.png')
        )

    # 2. Transition matrix
    if 'labels' in results and results['labels']['baseline'] is not None:
        from src.baseline_pipeline import BaselinePipeline
        bp = BaselinePipeline()
        T = bp.transition_matrix(results['labels']['baseline'])
        plot_transition_matrix(T, save_path=str(save_dir / f'{demo}_transition_matrix.png'))

    # 3. Per-modality comparison
    if 'modality_clustering' in results:
        plot_modality_comparison(
            results['modality_clustering'],
            save_path=str(save_dir / f'{demo}_modality_comparison.png')
        )

    # 4. Physics filter summary
    if 'physics_filter' in results:
        plot_physics_filter_summary(
            results['physics_filter'],
            save_path=str(save_dir / f'{demo}_physics_filter.png')
        )

    print(f"  All plots saved to {save_dir}/")
