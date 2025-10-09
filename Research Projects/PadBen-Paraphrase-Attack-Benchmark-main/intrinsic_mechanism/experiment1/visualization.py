#!/usr/bin/env python3
"""
Visualization script for semantic vs paraphrase experiment results.

This script creates comprehensive visualizations for the experiment results,
including distance comparisons, semantic space plots, and clustering analysis.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ExperimentVisualizer:
    """
    Visualization class for semantic vs paraphrase experiment results.
    """
    
    def __init__(self, results_dir: str = "results", use_full_dataset: bool = False):
        """
        Initialize the visualizer.
        
        Args:
            results_dir: Directory containing experiment results
            use_full_dataset: If True, use full_results directory
        """
        if use_full_dataset:
            results_dir = "full_results"
        self.results_dir = Path(results_dir)
        self.distance_results = None
        self.exploration_results = None
        self.use_full_dataset = use_full_dataset
        
    def load_results(self) -> Tuple[Dict, Dict]:
        """
        Load experiment results from files.
        
        Returns:
            Tuple of (distance_results, exploration_results)
        """
        # Load distance results
        with open(self.results_dir / "distance_results.json", 'r') as f:
            self.distance_results = json.load(f)
        
        # Load exploration summary
        with open(self.results_dir / "exploration_summary.json", 'r') as f:
            exploration_summary = json.load(f)
        
        # Load arrays
        pca_2d = np.load(self.results_dir / "pca_2d.npy")
        kmeans_labels = np.load(self.results_dir / "kmeans_labels.npy")
        labels = np.load(self.results_dir / "labels.npy")
        
        self.exploration_results = {
            **exploration_summary,
            'pca_2d': pca_2d,
            'kmeans_labels': kmeans_labels,
            'labels': labels
        }
        
        return self.distance_results, self.exploration_results
    
    def plot_distance_comparison(self, save_path: str = None) -> None:
        """
        Create distance comparison plots.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.distance_results:
            raise ValueError("Distance results not loaded. Call load_results() first.")
        
        # Prepare data for plotting
        comparisons = []
        metrics = []
        values = []
        
        for comparison, distances in self.distance_results.items():
            for metric, value in distances.items():
                comparisons.append(comparison.replace('_', ' vs '))
                metrics.append(metric)
                values.append(value)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Comparison': comparisons,
            'Metric': metrics,
            'Distance': values
        })
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, metric in enumerate(['cosine_similarity', 'euclidean', 'manhattan']):
            metric_data = df[df['Metric'] == metric]
            
            sns.barplot(
                data=metric_data,
                x='Comparison',
                y='Distance',
                hue='Comparison',
                ax=axes[i],
                palette='viridis',
                legend=False
            )
            
            axes[i].set_title(f'{metric.replace("_", " ").title()} Distance')
            axes[i].set_ylabel('Distance')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for j, v in enumerate(metric_data['Distance']):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved distance comparison plot to {save_path}")
        plt.show()
    
    def plot_semantic_space(self, method: str = 'pca', save_path: str = None) -> None:
        """
        Plot semantic space visualization.
        
        Args:
            method: 'pca' (only PCA supported)
            save_path: Path to save the plot
        """
        if not self.exploration_results:
            raise ValueError("Exploration results not loaded. Call load_results() first.")
        
        # Get 2D coordinates (only PCA supported)
        if method == 'pca':
            coords = self.exploration_results['pca_2d']
            title = 'PCA Semantic Space'
        else:
            raise ValueError("Only PCA method is supported. Use method='pca'")
        
        labels = self.exploration_results['labels']
        type_mapping = self.exploration_results['type_mapping']
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot by type
        colors = ['red', 'blue', 'green']
        type_names = ['Type1 (Human Original)', 'Type2 (LLM Generated)', 'Type4 (LLM Paraphrased)']
        
        # Get unique type IDs and map them properly
        unique_types = np.unique(labels)
        # Map 0,1,2 to type1,type2,type4
        type_name_mapping = {0: 'Type1 (Human Original)', 1: 'Type2 (LLM Generated)', 2: 'Type4 (LLM Paraphrased)'}
        
        for i, type_id in enumerate(unique_types):
            mask = labels == type_id
            if np.any(mask):  # Only plot if there are points for this type
                color_idx = i % len(colors)
                type_name = type_name_mapping.get(type_id, f'Type{type_id}')
                ax1.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=colors[color_idx], label=type_name, alpha=0.6, s=20
                )
        
        ax1.set_title(f'{title} - By Text Type')
        ax1.set_xlabel(f'{method.upper()} Component 1')
        ax1.set_ylabel(f'{method.upper()} Component 2')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot by cluster
        kmeans_labels = self.exploration_results['kmeans_labels']
        scatter = ax2.scatter(
            coords[:, 0], coords[:, 1],
            c=kmeans_labels, cmap='tab10', alpha=0.6, s=20
        )
        
        # Get unique clusters for subtitle
        unique_clusters = np.unique(kmeans_labels)
        cluster_count = len(unique_clusters)
        
        ax2.set_title(f'{title} - By Cluster (n={cluster_count} clusters)')
        ax2.set_xlabel(f'{method.upper()} Component 1')
        ax2.set_ylabel(f'{method.upper()} Component 2')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar for clusters
        plt.colorbar(scatter, ax=ax2, label='Cluster')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved {method.upper()} semantic space plot to {save_path}")
        plt.show()
    
    def plot_cluster_analysis(self, save_path: str = None) -> None:
        """
        Create cluster analysis plots.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.exploration_results:
            raise ValueError("Exploration results not loaded. Call load_results() first.")
        
        labels = self.exploration_results['labels']
        kmeans_labels = self.exploration_results['kmeans_labels']
        type_mapping = self.exploration_results['type_mapping']
        
        # Debug information
        print(f"Debug: labels shape: {labels.shape}, unique labels: {np.unique(labels)}")
        print(f"Debug: kmeans_labels shape: {kmeans_labels.shape}, unique clusters: {np.unique(kmeans_labels)}")
        print(f"Debug: type_mapping: {type_mapping}")
        
        # Create cluster vs type analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Cluster distribution
        unique_clusters, cluster_counts = np.unique(kmeans_labels, return_counts=True)
        axes[0, 0].bar(unique_clusters, cluster_counts, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Cluster Size Distribution')
        axes[0, 0].set_xlabel('Cluster ID')
        axes[0, 0].set_ylabel('Number of Samples')
        
        # Type distribution
        unique_types, type_counts = np.unique(labels, return_counts=True)
        type_name_mapping = {0: 'Type1 (Human Original)', 1: 'Type2 (LLM Generated)', 2: 'Type4 (LLM Paraphrased)'}
        type_names = [type_name_mapping.get(t, f'Type{t}') for t in unique_types]
        axes[0, 1].bar(type_names, type_counts, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Text Type Distribution')
        axes[0, 1].set_ylabel('Number of Samples')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Cluster vs Type heatmap
        cluster_type_matrix = np.zeros((len(unique_clusters), len(unique_types)))
        for i, cluster in enumerate(unique_clusters):
            for j, type_id in enumerate(unique_types):
                cluster_type_matrix[i, j] = np.sum((kmeans_labels == cluster) & (labels == type_id))
        
        # Ensure matrix has integer values for proper formatting
        cluster_type_matrix = cluster_type_matrix.astype(int)
        
        # Create proper type names mapping
        type_name_mapping = {0: 'Type1 (Human Original)', 1: 'Type2 (LLM Generated)', 2: 'Type4 (LLM Paraphrased)'}
        type_names = [type_name_mapping.get(t, f'Type{t}') for t in unique_types]
        
        # Debug matrix information
        print(f"Debug: cluster_type_matrix shape: {cluster_type_matrix.shape}")
        print(f"Debug: cluster_type_matrix values: {cluster_type_matrix}")
        print(f"Debug: type_names: {type_names}")
        print(f"Debug: unique_clusters: {unique_clusters}")
        
        try:
            sns.heatmap(
                cluster_type_matrix,
                xticklabels=type_names,
                yticklabels=unique_clusters,
                annot=True,
                fmt='d',
                cmap='Blues',
                ax=axes[1, 0]
            )
        except Exception as e:
            print(f"Error creating heatmap: {e}")
            # Fallback: create heatmap without annotations
            sns.heatmap(
                cluster_type_matrix,
                xticklabels=type_names,
                yticklabels=unique_clusters,
                annot=False,
                cmap='Blues',
                ax=axes[1, 0]
            )
        axes[1, 0].set_title('Cluster vs Text Type Matrix')
        axes[1, 0].set_xlabel('Text Type')
        axes[1, 0].set_ylabel('Cluster ID')
        
        # Cluster purity analysis
        cluster_purity = []
        for cluster in unique_clusters:
            cluster_mask = kmeans_labels == cluster
            cluster_types = labels[cluster_mask]
            if len(cluster_types) > 0:
                most_common_type = np.bincount(cluster_types).argmax()
                purity = np.sum(cluster_types == most_common_type) / len(cluster_types)
                cluster_purity.append(purity)
            else:
                cluster_purity.append(0)
        
        axes[1, 1].bar(unique_clusters, cluster_purity, color='lightgreen', alpha=0.7)
        axes[1, 1].set_title('Cluster Purity (Dominant Type Ratio)')
        axes[1, 1].set_xlabel('Cluster ID')
        axes[1, 1].set_ylabel('Purity Score')
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved cluster analysis plot to {save_path}")
        plt.show()
    
    def plot_cluster_type_matrix(self, save_path: str = None) -> None:
        """
        Create a dedicated cluster vs text type matrix heatmap.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.exploration_results:
            raise ValueError("Exploration results not loaded. Call load_results() first.")
        
        labels = self.exploration_results['labels']
        kmeans_labels = self.exploration_results['kmeans_labels']
        
        # Get unique clusters and types
        unique_clusters = np.unique(kmeans_labels)
        unique_types = np.unique(labels)
        
        # Create cluster vs type matrix
        cluster_type_matrix = np.zeros((len(unique_clusters), len(unique_types)))
        for i, cluster in enumerate(unique_clusters):
            for j, type_id in enumerate(unique_types):
                cluster_type_matrix[i, j] = np.sum((kmeans_labels == cluster) & (labels == type_id))
        
        # Ensure matrix has integer values for proper formatting
        cluster_type_matrix = cluster_type_matrix.astype(int)
        
        # Create proper type names mapping
        type_name_mapping = {0: 'Type1 (Human Original)', 1: 'Type2 (LLM Generated)', 2: 'Type4 (LLM Paraphrased)'}
        type_names = [type_name_mapping.get(t, f'Type{t}') for t in unique_types]
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            cluster_type_matrix,
            xticklabels=type_names,
            yticklabels=[f'Cluster {c}' for c in unique_clusters],
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar_kws={'label': 'Number of Samples'}
        )
        
        plt.title('Cluster vs Text Type Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Text Type', fontsize=12)
        plt.ylabel('Cluster ID', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved cluster vs text type matrix to {save_path}")
        plt.show()
    
    def create_summary_report(self, save_path: str = None) -> None:
        """
        Create a comprehensive summary report.
        
        Args:
            save_path: Path to save the report
        """
        if not self.distance_results or not self.exploration_results:
            raise ValueError("Results not loaded. Call load_results() first.")
        
        # Create summary figure
        fig = plt.figure(figsize=(20, 16))
        
        # Distance comparison
        ax1 = plt.subplot(3, 3, 1)
        comparisons = list(self.distance_results.keys())
        cosine_distances = [self.distance_results[comp]['cosine_similarity'] for comp in comparisons]
        bars = ax1.bar(range(len(comparisons)), cosine_distances, color='skyblue', alpha=0.7)
        ax1.set_title('Cosine Distance Comparison')
        ax1.set_xticks(range(len(comparisons)))
        ax1.set_xticklabels([comp.replace('_', ' vs ') for comp in comparisons], rotation=45)
        ax1.set_ylabel('Cosine Distance')
        
        # Add value labels
        for i, v in enumerate(cosine_distances):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # PCA plot
        ax2 = plt.subplot(3, 3, 2)
        coords = self.exploration_results['pca_2d']
        labels = self.exploration_results['labels']
        type_mapping = self.exploration_results['type_mapping']
        
        colors = ['red', 'blue', 'green']
        unique_types = np.unique(labels)
        type_name_mapping = {0: 'Type1 (Human Original)', 1: 'Type2 (LLM Generated)', 2: 'Type4 (LLM Paraphrased)'}
        
        for i, type_id in enumerate(unique_types):
            mask = labels == type_id
            if np.any(mask):
                color_idx = i % len(colors)
                type_name = type_name_mapping.get(type_id, f'Type{type_id}')
                ax2.scatter(coords[mask, 0], coords[mask, 1], c=colors[color_idx], label=type_name, alpha=0.6, s=10)
        
        ax2.set_title('PCA Semantic Space')
        ax2.set_xlabel('PCA 1')
        ax2.set_ylabel('PCA 2')
        ax2.legend()
        
        
        # Cluster analysis
        ax4 = plt.subplot(3, 3, 4)
        kmeans_labels = self.exploration_results['kmeans_labels']
        unique_clusters, cluster_counts = np.unique(kmeans_labels, return_counts=True)
        ax4.bar(unique_clusters, cluster_counts, color='lightcoral', alpha=0.7)
        ax4.set_title('Cluster Distribution')
        ax4.set_xlabel('Cluster ID')
        ax4.set_ylabel('Count')
        
        # Type distribution
        ax5 = plt.subplot(3, 3, 5)
        unique_types, type_counts = np.unique(labels, return_counts=True)
        type_names = [type_name_mapping.get(t, f'Type{t}') for t in unique_types]
        ax5.bar(type_names, type_counts, color='lightgreen', alpha=0.7)
        ax5.set_title('Text Type Distribution')
        ax5.set_ylabel('Count')
        ax5.tick_params(axis='x', rotation=45)
        
        # Key insights text
        ax6 = plt.subplot(3, 3, (6, 9))
        ax6.axis('off')
        
        # Calculate key insights
        type2_vs_type4_cosine = self.distance_results.get('type2_vs_type4', {}).get('cosine_similarity', 'N/A')
        type1_vs_type2_cosine = self.distance_results.get('type1_vs_type2', {}).get('cosine_similarity', 'N/A')
        type1_vs_type4_cosine = self.distance_results.get('type1_vs_type4', {}).get('cosine_similarity', 'N/A')
        
        insights_text = f"""
        KEY INSIGHTS:
        
        1. Distance Analysis:
           • Type2 vs Type4 Cosine Distance: {type2_vs_type4_cosine:.3f}
           • Type1 vs Type2 Cosine Distance: {type1_vs_type2_cosine:.3f}
           • Type1 vs Type4 Cosine Distance: {type1_vs_type4_cosine:.3f}
        
        2. Semantic Space Analysis:
           • Total samples: {len(labels)}
           • Number of clusters: {len(unique_clusters)}
           • Type distribution: {dict(zip(type_names, type_counts))}
        
        3. Key Questions:
           • Do Type2 and Type4 form distinct clusters?
           • Are Type4 texts closer to Type1 or Type2?
           • What does this tell us about paraphrase attacks?
        
        4. Interpretation:
           • Lower distance = higher semantic similarity
           • Cluster separation indicates distinct semantic spaces
           • Type4 positioning reveals paraphrase behavior
        """
        
        ax6.text(0.05, 0.95, insights_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved summary report to {save_path}")
        plt.show()


def main():
    """Main execution function for visualization."""
    import sys
    
    # Check if we should use full dataset results
    use_full_dataset = "--full" in sys.argv or "full" in sys.argv
    
    if use_full_dataset:
        print("Using FULL dataset results from full_results/ directory")
        visualizer = ExperimentVisualizer(use_full_dataset=True)
        output_dir = "full_results"
    else:
        print("Using sample results from results/ directory")
        visualizer = ExperimentVisualizer(use_full_dataset=False)
        output_dir = "results"
    
    try:
        # Load results
        distance_results, exploration_results = visualizer.load_results()
        print("Results loaded successfully!")
        
        # Create visualizations
        print("Creating distance comparison plot...")
        visualizer.plot_distance_comparison(f"{output_dir}/distance_comparison.png")
        
        print("Creating PCA semantic space plot...")
        visualizer.plot_semantic_space('pca', f"{output_dir}/pca_semantic_space.png")
        
        
        print("Creating cluster analysis plot...")
        visualizer.plot_cluster_analysis(f"{output_dir}/cluster_analysis.png")
        
        print("Creating cluster vs text type matrix...")
        visualizer.plot_cluster_type_matrix(f"{output_dir}/cluster_type_matrix.png")
        
        print("Creating comprehensive summary report...")
        visualizer.create_summary_report(f"{output_dir}/summary_report.png")
        
        print(f"All figures saved to {output_dir}/ directory:")
        print("  - distance_comparison.png")
        print("  - pca_semantic_space.png")
        print("  - cluster_analysis.png")
        print("  - cluster_type_matrix.png")
        print("  - summary_report.png")
        
        print("All visualizations completed!")
        
    except Exception as e:
        print(f"Visualization failed: {e}")
        raise


if __name__ == "__main__":
    main()
