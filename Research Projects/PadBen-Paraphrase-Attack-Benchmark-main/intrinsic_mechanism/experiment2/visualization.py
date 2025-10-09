"""
Visualization module for semantic space experiment results.

This module contains functions for creating various visualizations of the
experimental results, including PCA plots, trajectory analysis, and
statistical summaries.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import pandas as pd
from scipy.spatial.distance import cosine, euclidean, cityblock

logger = logging.getLogger(__name__)


class SemanticSpaceVisualizer:
    """Class for creating visualizations of semantic space experiment results."""
    
    def __init__(self, output_dir: str, style: str = 'default'):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            style: Matplotlib style to use
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use(style)
        sns.set_palette("husl")
        
        # Color schemes for iterations
        self.iteration_colors = plt.cm.viridis(np.linspace(0, 1, 6))  # Up to 6 iterations
        self.type_colors = {'type1': '#FF6B6B', 'type2': '#4ECDC4'}
    
    def create_pca_visualization(
        self, 
        pca_results: Dict[str, Any], 
        max_iterations: int = 5
    ) -> None:
        """
        Create comprehensive PCA visualization.
        
        Args:
            pca_results: Results from PCA analysis
            max_iterations: Maximum number of iterations
        """
        logger.info("Creating PCA visualization")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Semantic Space Analysis: Iterative Paraphrasing Effects', 
                    fontsize=16, fontweight='bold')
        
        # Plot hidden states PCA
        for i, text_type in enumerate(['type1', 'type2']):
            ax = axes[0, i]
            self._plot_pca_scatter(
                ax, 
                pca_results['hidden_states'][text_type], 
                max_iterations,
                f'Hidden States PCA - {text_type.upper()}'
            )
        
        # Plot embeddings PCA
        for i, text_type in enumerate(['type1', 'type2']):
            ax = axes[1, i]
            self._plot_pca_scatter(
                ax, 
                pca_results['embeddings'][text_type], 
                max_iterations,
                f'Embeddings PCA - {text_type.upper()}'
            )
        
        plt.tight_layout()
        
        # Save the visualization
        viz_path = self.output_dir / 'pca_comprehensive.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"PCA visualization saved to: {viz_path}")
        
        plt.show()
    
    def _plot_pca_scatter(
        self, 
        ax: plt.Axes, 
        data: Dict[str, Any], 
        max_iterations: int,
        title: str
    ) -> None:
        """
        Plot PCA scatter plot on given axes.
        
        Args:
            ax: Matplotlib axes
            data: PCA data for one feature type and text type
            max_iterations: Maximum number of iterations
            title: Plot title
        """
        for iteration in range(1, max_iterations + 1):
            mask = np.array(data['iteration_labels']) == iteration
            points = data['pca_components'][mask]
            
            if len(points) > 0:
                ax.scatter(
                    points[:, 0], points[:, 1],
                    c=[self.iteration_colors[iteration-1]], 
                    label=f'Iteration {iteration}',
                    alpha=0.7,
                    s=60,
                    edgecolors='white',
                    linewidth=0.5
                )
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(f'PC1 ({data["explained_variance_ratio"][0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({data["explained_variance_ratio"][1]:.1%} variance)')
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        # Add convex hull for each iteration
        self._add_convex_hulls(ax, data, max_iterations)
    
    def _add_convex_hulls(
        self, 
        ax: plt.Axes, 
        data: Dict[str, Any], 
        max_iterations: int
    ) -> None:
        """
        Add convex hulls around each iteration's points.
        
        Args:
            ax: Matplotlib axes
            data: PCA data
            max_iterations: Maximum number of iterations
        """
        try:
            from scipy.spatial import ConvexHull
            
            for iteration in range(1, max_iterations + 1):
                mask = np.array(data['iteration_labels']) == iteration
                points = data['pca_components'][mask]
                
                if len(points) > 2:  # Need at least 3 points for convex hull
                    try:
                        hull = ConvexHull(points)
                        for simplex in hull.simplices:
                            ax.plot(
                                points[simplex, 0], points[simplex, 1], 
                                color=self.iteration_colors[iteration-1],
                                alpha=0.3,
                                linewidth=1
                            )
                    except Exception:
                        continue  # Skip if convex hull fails
        except ImportError:
            logger.warning("scipy not available, skipping convex hulls")
    
    def create_trajectory_analysis(
        self, 
        pca_results: Dict[str, Any], 
        max_iterations: int = 5
    ) -> None:
        """
        Create trajectory analysis showing how samples move through iterations.
        
        Args:
            pca_results: Results from PCA analysis
            max_iterations: Maximum number of iterations
        """
        logger.info("Creating trajectory analysis")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Semantic Trajectory Analysis: Movement Across Iterations', 
                    fontsize=16, fontweight='bold')
        
        for row, feature_type in enumerate(['hidden_states', 'embeddings']):
            for col, text_type in enumerate(['type1', 'type2']):
                ax = axes[row, col]
                self._plot_trajectory(
                    ax, 
                    pca_results[feature_type][text_type], 
                    max_iterations,
                    f'{feature_type.replace("_", " ").title()} - {text_type.upper()}'
                )
        
        plt.tight_layout()
        
        # Save the visualization
        viz_path = self.output_dir / 'trajectory_analysis.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"Trajectory analysis saved to: {viz_path}")
        
        plt.show()
    
    def _plot_trajectory(
        self, 
        ax: plt.Axes, 
        data: Dict[str, Any], 
        max_iterations: int,
        title: str
    ) -> None:
        """
        Plot trajectory of samples across iterations.
        
        Args:
            ax: Matplotlib axes
            data: PCA data
            max_iterations: Maximum number of iterations
            title: Plot title
        """
        components = data['pca_components']
        labels = np.array(data['iteration_labels'])
        
        # Calculate number of samples per iteration
        samples_per_iter = len(labels) // max_iterations
        
        # Plot trajectories for a subset of samples
        sample_indices = np.arange(0, min(20, samples_per_iter))  # Show max 20 trajectories
        
        for sample_idx in sample_indices:
            trajectory_x = []
            trajectory_y = []
            
            for iteration in range(1, max_iterations + 1):
                point_idx = (iteration - 1) * samples_per_iter + sample_idx
                if point_idx < len(components):
                    trajectory_x.append(components[point_idx, 0])
                    trajectory_y.append(components[point_idx, 1])
            
            if len(trajectory_x) > 1:
                ax.plot(
                    trajectory_x, trajectory_y, 
                    alpha=0.6, 
                    linewidth=1,
                    marker='o',
                    markersize=3
                )
        
        # Add iteration centers
        for iteration in range(1, max_iterations + 1):
            mask = labels == iteration
            if np.any(mask):
                center_x = np.mean(components[mask, 0])
                center_y = np.mean(components[mask, 1])
                
                ax.scatter(
                    center_x, center_y,
                    c=[self.iteration_colors[iteration-1]],
                    s=200,
                    marker='*',
                    label=f'Iter {iteration} center',
                    edgecolors='black',
                    linewidth=1
                )
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(f'PC1 ({data["explained_variance_ratio"][0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({data["explained_variance_ratio"][1]:.1%} variance)')
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
    
    def create_variance_analysis(self, pca_results: Dict[str, Any]) -> None:
        """
        Create analysis of explained variance across components.
        
        Args:
            pca_results: Results from PCA analysis
        """
        logger.info("Creating variance analysis")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('PCA Explained Variance Analysis', fontsize=16, fontweight='bold')
        
        for row, feature_type in enumerate(['hidden_states', 'embeddings']):
            for col, text_type in enumerate(['type1', 'type2']):
                ax = axes[row, col]
                
                variance_ratio = pca_results[feature_type][text_type]['explained_variance_ratio']
                
                # Bar plot of explained variance
                ax.bar(
                    ['PC1', 'PC2'], 
                    variance_ratio,
                    color=[self.type_colors[text_type], self.type_colors[text_type]],
                    alpha=0.7,
                    edgecolor='black'
                )
                
                ax.set_title(f'{feature_type.replace("_", " ").title()} - {text_type.upper()}')
                ax.set_ylabel('Explained Variance Ratio')
                ax.set_ylim(0, 1)
                
                # Add percentage labels on bars
                for i, v in enumerate(variance_ratio):
                    ax.text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save the visualization
        viz_path = self.output_dir / 'variance_analysis.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"Variance analysis saved to: {viz_path}")
        
        plt.show()
    
    def create_distance_analysis(
        self, 
        pca_results: Dict[str, Any], 
        max_iterations: int = 5
    ) -> None:
        """
        Create comprehensive distance analysis between iterations.
        
        Args:
            pca_results: Results from PCA analysis
            max_iterations: Maximum number of iterations
        """
        logger.info("Creating enhanced distance analysis")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Enhanced Distance Analysis: Semantic Drift Across Iterations', 
                    fontsize=16, fontweight='bold')
        
        for row, feature_type in enumerate(['hidden_states', 'embeddings']):
            for col, text_type in enumerate(['type1', 'type2']):
                # Sequential distance plot (existing)
                ax = axes[row, col]
                
                distances = self._calculate_iteration_distances(
                    pca_results[feature_type][text_type], 
                    max_iterations
                )
                
                iterations = list(range(2, max_iterations + 1))
                ax.plot(
                    iterations, distances, 
                    marker='o', 
                    linewidth=2,
                    markersize=8,
                    color=self.type_colors[text_type],
                    label='Sequential distances'
                )
                
                ax.set_title(f'{feature_type.replace("_", " ").title()} - {text_type.upper()}\nSequential Distances')
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Distance from Previous Iteration')
                ax.grid(True, alpha=0.3)
                
                # Add trend line
                if len(distances) > 1:
                    z = np.polyfit(iterations, distances, 1)
                    p = np.poly1d(z)
                    ax.plot(iterations, p(iterations), "--", alpha=0.8, color='red', label='Trend')
                    
                    # Add trend statistics
                    slope = z[0]
                    trend_text = f"Trend: {'Increasing' if slope > 0 else 'Decreasing'}\nSlope: {slope:.4f}"
                    ax.text(0.05, 0.95, trend_text, transform=ax.transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                ax.legend()
            
            # Distance matrix heatmap (new)
            ax_heatmap = axes[row, 2]
            self._create_distance_matrix_heatmap(
                ax_heatmap, 
                pca_results[feature_type], 
                max_iterations,
                f'{feature_type.replace("_", " ").title()}\nAll Pairwise Distances'
            )
        
        plt.tight_layout()
        
        # Save the visualization
        viz_path = self.output_dir / 'distance_analysis.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"Enhanced distance analysis saved to: {viz_path}")
        
        plt.show()
    
    def _calculate_iteration_distances(
        self, 
        data: Dict[str, Any], 
        max_iterations: int
    ) -> List[float]:
        """
        Calculate distances between consecutive iterations.
        
        Args:
            data: PCA data
            max_iterations: Maximum number of iterations
            
        Returns:
            List of distances between consecutive iterations
        """
        components = data['pca_components']
        labels = np.array(data['iteration_labels'])
        
        distances = []
        samples_per_iter = len(labels) // max_iterations
        
        for iteration in range(2, max_iterations + 1):
            # Get centroids of consecutive iterations
            prev_mask = labels == (iteration - 1)
            curr_mask = labels == iteration
            
            if np.any(prev_mask) and np.any(curr_mask):
                prev_centroid = np.mean(components[prev_mask], axis=0)
                curr_centroid = np.mean(components[curr_mask], axis=0)
                
                distance = np.linalg.norm(curr_centroid - prev_centroid)
                distances.append(distance)
        
        return distances
    
    def _create_distance_matrix_heatmap(
        self, 
        ax: plt.Axes, 
        data_dict: Dict[str, Dict[str, Any]], 
        max_iterations: int,
        title: str
    ) -> None:
        """
        Create a heatmap showing distances between all pairs of iterations.
        
        Args:
            ax: Matplotlib axes
            data_dict: Dictionary containing data for both text types
            max_iterations: Maximum number of iterations
            title: Plot title
        """
        # Calculate distance matrix for both text types combined
        distance_matrix = np.zeros((max_iterations, max_iterations))
        
        for text_type in ['type1', 'type2']:
            data = data_dict[text_type]
            components = data['pca_components']
            labels = np.array(data['iteration_labels'])
            
            # Calculate centroids for each iteration
            centroids = {}
            for iteration in range(1, max_iterations + 1):
                mask = labels == iteration
                if np.any(mask):
                    centroids[iteration] = np.mean(components[mask], axis=0)
            
            # Calculate pairwise distances
            for i in range(1, max_iterations + 1):
                for j in range(1, max_iterations + 1):
                    if i in centroids and j in centroids:
                        distance = np.linalg.norm(centroids[i] - centroids[j])
                        distance_matrix[i-1, j-1] += distance
        
        # Average across text types
        distance_matrix /= 2
        
        # Create heatmap
        im = ax.imshow(distance_matrix, cmap='viridis', aspect='auto')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Set labels
        ax.set_xticks(range(max_iterations))
        ax.set_yticks(range(max_iterations))
        ax.set_xticklabels([f'Iter {i+1}' for i in range(max_iterations)])
        ax.set_yticklabels([f'Iter {i+1}' for i in range(max_iterations)])
        ax.set_title(title)
        
        # Add text annotations
        for i in range(max_iterations):
            for j in range(max_iterations):
                text = ax.text(j, i, f'{distance_matrix[i, j]:.3f}',
                             ha="center", va="center", color="white" if distance_matrix[i, j] > distance_matrix.max()/2 else "black")
    
    def create_center_movement_analysis(
        self, 
        pca_results: Dict[str, Any], 
        max_iterations: int = 5
    ) -> None:
        """
        Create analysis of how iteration centers move through semantic space.
        
        Args:
            pca_results: Results from PCA analysis
            max_iterations: Maximum number of iterations
        """
        logger.info("Creating center movement analysis")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Center Movement Analysis: Semantic Drift of Iteration Centroids', 
                    fontsize=16, fontweight='bold')
        
        for row, feature_type in enumerate(['hidden_states', 'embeddings']):
            for col, text_type in enumerate(['type1', 'type2']):
                ax = axes[row, col]
                
                # Calculate centroids for each iteration
                centroids, movement_stats = self._calculate_center_movements(
                    pca_results[feature_type][text_type], 
                    max_iterations
                )
                
                # Plot centroid trajectory
                if len(centroids) > 1:
                    centroid_array = np.array(list(centroids.values()))
                    
                    # Plot trajectory line
                    ax.plot(
                        centroid_array[:, 0], centroid_array[:, 1], 
                        'k--', alpha=0.7, linewidth=2, label='Centroid trajectory'
                    )
                    
                    # Plot individual centroids
                    for i, (iteration, centroid) in enumerate(centroids.items()):
                        ax.scatter(
                            centroid[0], centroid[1],
                            c=[self.iteration_colors[iteration-1]],
                            s=200,
                            marker='*',
                            label=f'Iter {iteration} center',
                            edgecolors='black',
                            linewidth=1,
                            zorder=5
                        )
                        
                        # Add iteration labels
                        ax.annotate(
                            f'{iteration}', 
                            (centroid[0], centroid[1]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=10, fontweight='bold'
                        )
                    
                    # Add movement vectors
                    for i in range(len(centroid_array) - 1):
                        start = centroid_array[i]
                        end = centroid_array[i + 1]
                        ax.annotate('', xy=end, xytext=start,
                                  arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.7))
                
                ax.set_title(f'{feature_type.replace("_", " ").title()} - {text_type.upper()}')
                ax.set_xlabel(f'PC1 ({pca_results[feature_type][text_type]["explained_variance_ratio"][0]:.1%} variance)')
                ax.set_ylabel(f'PC2 ({pca_results[feature_type][text_type]["explained_variance_ratio"][1]:.1%} variance)')
                ax.grid(True, alpha=0.3)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
                # Add movement statistics
                if movement_stats:
                    stats_text = f"Total movement: {movement_stats['total_distance']:.4f}\n"
                    stats_text += f"Avg step size: {movement_stats['avg_step_size']:.4f}\n"
                    stats_text += f"Max step size: {movement_stats['max_step_size']:.4f}"
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the visualization
        viz_path = self.output_dir / 'center_movement_analysis.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"Center movement analysis saved to: {viz_path}")
        
        plt.show()
    
    def _calculate_center_movements(
        self, 
        data: Dict[str, Any], 
        max_iterations: int
    ) -> Tuple[Dict[int, np.ndarray], Dict[str, float]]:
        """
        Calculate centroid positions and movement statistics.
        
        Args:
            data: PCA data
            max_iterations: Maximum number of iterations
            
        Returns:
            Tuple of (centroids dict, movement statistics dict)
        """
        components = data['pca_components']
        labels = np.array(data['iteration_labels'])
        
        centroids = {}
        
        # Calculate centroids for each iteration
        for iteration in range(1, max_iterations + 1):
            mask = labels == iteration
            if np.any(mask):
                centroids[iteration] = np.mean(components[mask], axis=0)
        
        # Calculate movement statistics
        movement_stats = {}
        if len(centroids) > 1:
            centroid_list = [centroids[i] for i in sorted(centroids.keys())]
            
            # Calculate step sizes
            step_sizes = []
            total_distance = 0
            
            for i in range(len(centroid_list) - 1):
                step_size = np.linalg.norm(centroid_list[i + 1] - centroid_list[i])
                step_sizes.append(step_size)
                total_distance += step_size
            
            movement_stats = {
                'total_distance': total_distance,
                'avg_step_size': np.mean(step_sizes) if step_sizes else 0,
                'max_step_size': np.max(step_sizes) if step_sizes else 0,
                'step_sizes': step_sizes
            }
        
        return centroids, movement_stats
    
    def create_comprehensive_analysis(
        self, 
        pca_results: Dict[str, Any], 
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        """
        Create comprehensive analysis including all visualizations and statistics.
        
        Args:
            pca_results: Results from PCA analysis
            max_iterations: Maximum number of iterations
            
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Creating comprehensive analysis")
        
        analysis_results = {
            'center_movements': {},
            'distance_matrices': {},
            'trajectory_stats': {}
        }
        
        # Calculate detailed statistics for each feature type and text type
        for feature_type in ['hidden_states', 'embeddings']:
            analysis_results['center_movements'][feature_type] = {}
            analysis_results['distance_matrices'][feature_type] = {}
            
            for text_type in ['type1', 'type2']:
                # Center movement analysis
                centroids, movement_stats = self._calculate_center_movements(
                    pca_results[feature_type][text_type], 
                    max_iterations
                )
                analysis_results['center_movements'][feature_type][text_type] = {
                    'centroids': {k: v.tolist() for k, v in centroids.items()},
                    'movement_stats': movement_stats
                }
                
                # Distance matrix
                distance_matrix = self._calculate_full_distance_matrix(
                    pca_results[feature_type][text_type], 
                    max_iterations
                )
                analysis_results['distance_matrices'][feature_type][text_type] = distance_matrix.tolist()
        
        # Save detailed analysis results
        analysis_path = self.output_dir / 'comprehensive_analysis.json'
        import json
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        logger.info(f"Comprehensive analysis saved to: {analysis_path}")
        
        return analysis_results
    
    def _calculate_full_distance_matrix(
        self, 
        data: Dict[str, Any], 
        max_iterations: int
    ) -> np.ndarray:
        """
        Calculate full distance matrix between all iterations.
        
        Args:
            data: PCA data
            max_iterations: Maximum number of iterations
            
        Returns:
            Distance matrix as numpy array
        """
        components = data['pca_components']
        labels = np.array(data['iteration_labels'])
        
        # Calculate centroids
        centroids = {}
        for iteration in range(1, max_iterations + 1):
            mask = labels == iteration
            if np.any(mask):
                centroids[iteration] = np.mean(components[mask], axis=0)
        
        # Calculate distance matrix
        distance_matrix = np.zeros((max_iterations, max_iterations))
        for i in range(1, max_iterations + 1):
            for j in range(1, max_iterations + 1):
                if i in centroids and j in centroids:
                    distance = np.linalg.norm(centroids[i] - centroids[j])
                    distance_matrix[i-1, j-1] = distance
        
        return distance_matrix
    
    def create_pairwise_distance_analysis(
        self, 
        output_dir: Path, 
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        """
        Create comprehensive pairwise distance analysis between corresponding samples.
        
        Args:
            output_dir: Directory containing experiment results
            max_iterations: Maximum number of iterations
            
        Returns:
            Dictionary containing distance analysis results
        """
        logger.info("Creating pairwise distance analysis")
        
        results = {
            'hidden_states': {'type1': [], 'type2': []},
            'embeddings': {'type1': [], 'type2': []}
        }
        
        for data_type in ['hidden_states', 'embeddings']:
            for text_type in ['type1', 'type2']:
                logger.info(f"Processing {data_type} {text_type}")
                
                # Load all iterations
                all_data = {}
                for iteration in range(1, max_iterations + 1):
                    try:
                        file_path = output_dir / str(iteration) / text_type / f"{data_type}.npy"
                        data = np.load(file_path)
                        all_data[iteration] = data
                    except FileNotFoundError:
                        logger.warning(f"Skipping {data_type} {text_type} iteration {iteration}")
                        continue
                
                # Compute distances from iteration 1 to all others
                if 1 in all_data:
                    data_1 = all_data[1]
                    
                    for iteration in range(2, max_iterations + 1):
                        if iteration in all_data:
                            data_i = all_data[iteration]
                            
                            # Ensure both iterations have same number of samples
                            min_samples = min(len(data_1), len(data_i))
                            
                            # Compute distances for each corresponding pair
                            cosine_distances = []
                            euclidean_distances = []
                            manhattan_distances = []
                            
                            for sample_idx in range(min_samples):
                                sample_1 = data_1[sample_idx]
                                sample_i = data_i[sample_idx]
                                
                                # Cosine distance (1 - cosine similarity)
                                cosine_sim = np.dot(sample_1, sample_i) / (np.linalg.norm(sample_1) * np.linalg.norm(sample_i))
                                cosine_dist = 1 - cosine_sim
                                cosine_distances.append(cosine_dist)
                                
                                # Euclidean distance
                                euclidean_dist = euclidean(sample_1, sample_i)
                                euclidean_distances.append(euclidean_dist)
                                
                                # Manhattan distance
                                manhattan_dist = cityblock(sample_1, sample_i)
                                manhattan_distances.append(manhattan_dist)
                            
                            # Compute averages
                            avg_cosine = np.mean(cosine_distances)
                            avg_euclidean = np.mean(euclidean_distances)
                            avg_manhattan = np.mean(manhattan_distances)
                            
                            results[data_type][text_type].append({
                                'comparison': f'1-{iteration}',
                                'cosine_distance': float(avg_cosine),
                                'euclidean_distance': float(avg_euclidean),
                                'manhattan_distance': float(avg_manhattan),
                                'num_samples': min_samples
                            })
        
        return results
    
    def create_distance_trend_plots(
        self, 
        distance_results: Dict[str, Any],
        save_path: Path
    ) -> None:
        """
        Create line charts showing distance trends across iterations.
        
        Args:
            distance_results: Results from pairwise distance analysis
            save_path: Path to save the plot
        """
        logger.info("Creating distance trend plots")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Distance metrics to plot
        metrics = ['cosine_distance', 'euclidean_distance', 'manhattan_distance']
        metric_labels = ['Cosine Distance', 'Euclidean Distance', 'Manhattan Distance']
        
        # Colors for type1 and type2
        colors = {'type1': '#2E86AB', 'type2': '#A23B72'}
        
        # Plot hidden states (top row)
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[0, i]
            
            for text_type in ['type1', 'type2']:
                data = distance_results['hidden_states'][text_type]
                if data:
                    iterations = [int(row['comparison'].split('-')[1]) for row in data]
                    values = [row[metric] for row in data]
                    
                    ax.plot(iterations, values, 
                           marker='o', linewidth=2.5, markersize=8,
                           color=colors[text_type], label=text_type.upper(),
                           alpha=0.8)
            
            ax.set_title(f'Hidden States - {label}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Iteration')
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xticks([2, 3, 4, 5])
            
            # Format y-axis based on metric scale
            if metric == 'cosine_distance':
                ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Plot embeddings (bottom row)
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[1, i]
            
            for text_type in ['type1', 'type2']:
                data = distance_results['embeddings'][text_type]
                if data:
                    iterations = [int(row['comparison'].split('-')[1]) for row in data]
                    values = [row[metric] for row in data]
                    
                    ax.plot(iterations, values, 
                           marker='s', linewidth=2.5, markersize=8,
                           color=colors[text_type], label=text_type.upper(),
                           alpha=0.8)
            
            ax.set_title(f'Embeddings - {label}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Iteration')
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xticks([2, 3, 4, 5])
            
            # Format y-axis based on metric scale
            if metric == 'cosine_distance':
                ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Distance trends plot saved to: {save_path}")
        
        plt.show()
    
    def create_centroid_trajectory_plots(
        self, 
        output_dir: Path, 
        save_path: Path,
        max_iterations: int = 5
    ) -> None:
        """
        Create focused centroid trajectory visualization.
        
        Args:
            output_dir: Directory containing experiment results
            save_path: Path to save the plot
            max_iterations: Maximum number of iterations
        """
        logger.info("Creating centroid trajectory plots")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Data types to analyze
        data_types = ['hidden_states', 'embeddings']
        data_labels = ['Hidden States', 'Embeddings']
        
        # Colors for iterations
        iteration_colors = {
            1: '#000000',  # Black
            2: '#8E44AD',  # Purple
            3: '#16A085',  # Teal
            4: '#27AE60',  # Green
            5: '#F39C12'   # Orange
        }
        
        # Markers for iterations
        iteration_markers = {1: 'o', 2: 's', 3: '^', 4: 'D', 5: 'v'}
        
        for data_idx, (data_type, data_label) in enumerate(zip(data_types, data_labels)):
            try:
                # Compute centroids
                centroids = self._compute_centroids_from_files(output_dir, data_type, max_iterations)
                
                # Apply PCA
                pca_centroids, variance_ratio = self._apply_pca_to_centroids(centroids)
                
                # Plot for each text type
                for type_idx, text_type in enumerate(['type1', 'type2']):
                    ax = axes[data_idx, type_idx]
                    
                    if text_type not in pca_centroids or not pca_centroids[text_type]:
                        ax.text(0.5, 0.5, f'No data for {text_type}', 
                               ha='center', va='center', transform=ax.transAxes)
                        continue
                    
                    # Extract trajectory points
                    iterations = sorted(pca_centroids[text_type].keys())
                    trajectory_x = [pca_centroids[text_type][it][0] for it in iterations]
                    trajectory_y = [pca_centroids[text_type][it][1] for it in iterations]
                    
                    # Plot trajectory line
                    ax.plot(trajectory_x, trajectory_y, 
                           color='gray', linewidth=3, alpha=0.7, zorder=1,
                           label='Centroid Trajectory')
                    
                    # Plot centroid points for each iteration
                    for i, iteration in enumerate(iterations):
                        x, y = pca_centroids[text_type][iteration]
                        ax.scatter(x, y, 
                                 c=iteration_colors[iteration], 
                                 marker=iteration_markers[iteration],
                                 s=200, 
                                 edgecolors='white', 
                                 linewidth=2,
                                 zorder=3,
                                 label=f'Iteration {iteration}')
                        
                        # Add iteration number as text
                        ax.annotate(str(iteration), 
                                  (x, y), 
                                  xytext=(5, 5), 
                                  textcoords='offset points',
                                  fontsize=12, 
                                  fontweight='bold',
                                  color=iteration_colors[iteration])
                    
                    # Set title and labels
                    ax.set_title(f'{data_label} - {text_type.upper()}', 
                               fontsize=14, fontweight='bold')
                    ax.set_xlabel(f'PC1 ({variance_ratio[0]:.1%} variance)')
                    ax.set_ylabel(f'PC2 ({variance_ratio[1]:.1%} variance)')
                    ax.grid(True, alpha=0.3)
                    
                    # Add legend only for the first subplot
                    if data_idx == 0 and type_idx == 0:
                        legend_elements = []
                        for iteration in iterations:
                            legend_elements.append(
                                plt.scatter([], [], 
                                          c=iteration_colors[iteration],
                                          marker=iteration_markers[iteration],
                                          s=100,
                                          edgecolors='white',
                                          linewidth=1,
                                          label=f'Iteration {iteration}')
                            )
                        ax.legend(handles=legend_elements, 
                                loc='upper right', 
                                fontsize=10,
                                title='Iterations')
                    
                    # Auto-adjust axis limits with padding
                    if trajectory_x and trajectory_y:
                        x_range = max(trajectory_x) - min(trajectory_x)
                        y_range = max(trajectory_y) - min(trajectory_y)
                        
                        x_padding = max(x_range * 0.2, 0.1)
                        y_padding = max(y_range * 0.2, 0.1)
                        
                        ax.set_xlim(min(trajectory_x) - x_padding, 
                                   max(trajectory_x) + x_padding)
                        ax.set_ylim(min(trajectory_y) - y_padding, 
                                   max(trajectory_y) + y_padding)
                
            except Exception as e:
                logger.error(f"Error processing {data_type}: {e}")
                for type_idx in range(2):
                    ax = axes[data_idx, type_idx]
                    ax.text(0.5, 0.5, f'Error: {str(e)}', 
                           ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Centroid trajectory plot saved to: {save_path}")
        
        plt.show()
    
    def _compute_centroids_from_files(
        self, 
        output_dir: Path, 
        data_type: str, 
        max_iterations: int
    ) -> Dict[str, Dict[int, np.ndarray]]:
        """Compute centroids from saved files."""
        centroids = {'type1': {}, 'type2': {}}
        
        for text_type in ['type1', 'type2']:
            for iteration in range(1, max_iterations + 1):
                try:
                    file_path = output_dir / str(iteration) / text_type / f"{data_type}.npy"
                    data = np.load(file_path)
                    centroid = np.mean(data, axis=0)
                    centroids[text_type][iteration] = centroid
                except FileNotFoundError:
                    continue
        
        return centroids
    
    def _apply_pca_to_centroids(self, centroids: Dict[str, Dict[int, np.ndarray]]) -> Tuple[Dict[str, Dict[int, np.ndarray]], np.ndarray]:
        """Apply PCA to centroids for visualization."""
        # Collect all centroids for PCA fitting
        all_centroids = []
        centroid_labels = []
        
        for text_type in ['type1', 'type2']:
            for iteration in sorted(centroids[text_type].keys()):
                all_centroids.append(centroids[text_type][iteration])
                centroid_labels.append((text_type, iteration))
        
        if not all_centroids:
            raise ValueError("No centroids found for PCA")
        
        # Convert to numpy array and apply PCA
        all_centroids = np.array(all_centroids)
        pca = PCA(n_components=2)
        pca_centroids_array = pca.fit_transform(all_centroids)
        
        # Organize PCA results back into dictionary structure
        pca_results = {'type1': {}, 'type2': {}}
        
        for i, (text_type, iteration) in enumerate(centroid_labels):
            pca_results[text_type][iteration] = pca_centroids_array[i]
        
        return pca_results, pca.explained_variance_ratio_
    
    def save_distance_tables(
        self, 
        distance_results: Dict[str, Any], 
        output_dir: Path
    ) -> None:
        """
        Save distance analysis results as CSV and JSON files.
        
        Args:
            distance_results: Results from pairwise distance analysis
            output_dir: Directory to save files
        """
        logger.info("Saving distance analysis tables")
        
        # Save hidden states table
        if distance_results['hidden_states']['type1'] or distance_results['hidden_states']['type2']:
            combined_hidden = []
            
            for text_type in ['type1', 'type2']:
                for row in distance_results['hidden_states'][text_type]:
                    combined_row = row.copy()
                    combined_row['text_type'] = text_type
                    combined_hidden.append(combined_row)
            
            if combined_hidden:
                df_hidden = pd.DataFrame(combined_hidden)
                df_hidden = df_hidden[['text_type', 'comparison', 'cosine_distance', 'euclidean_distance', 'manhattan_distance', 'num_samples']]
                df_hidden.to_csv(output_dir / 'hidden_states_distance_table.csv', index=False)
        
        # Save embeddings table
        if distance_results['embeddings']['type1'] or distance_results['embeddings']['type2']:
            combined_embeddings = []
            
            for text_type in ['type1', 'type2']:
                for row in distance_results['embeddings'][text_type]:
                    combined_row = row.copy()
                    combined_row['text_type'] = text_type
                    combined_embeddings.append(combined_row)
            
            if combined_embeddings:
                df_embeddings = pd.DataFrame(combined_embeddings)
                df_embeddings = df_embeddings[['text_type', 'comparison', 'cosine_distance', 'euclidean_distance', 'manhattan_distance', 'num_samples']]
                df_embeddings.to_csv(output_dir / 'embeddings_distance_table.csv', index=False)
        
        # Save as JSON
        import json
        with open(output_dir / 'distance_tables.json', 'w') as f:
            json.dump(distance_results, f, indent=2)
        
        logger.info(f"Distance tables saved to: {output_dir}")
    
    def create_summary_report(
        self, 
        pca_results: Dict[str, Any], 
        experiment_config: Dict[str, Any],
        max_iterations: int = 5
    ) -> None:
        """
        Create an enhanced summary report with key statistics including movement and distance analysis.
        
        Args:
            pca_results: Results from PCA analysis
            experiment_config: Experiment configuration
            max_iterations: Maximum number of iterations
        """
        logger.info("Creating enhanced summary report")
        
        # Create text report
        report = []
        report.append("=" * 80)
        report.append("ENHANCED SEMANTIC SPACE EXPERIMENT SUMMARY REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Experiment configuration
        report.append("EXPERIMENT CONFIGURATION:")
        report.append(f"  - Number of samples per type: {experiment_config.get('num_samples', 'N/A')}")
        report.append(f"  - Maximum iterations: {experiment_config.get('max_iterations', 'N/A')}")
        report.append(f"  - Paraphrase model: {experiment_config.get('paraphrase_model', 'N/A')}")
        report.append(f"  - Embedding model: {experiment_config.get('embedding_model', 'N/A')}")
        report.append("")
        
        # PCA analysis results
        report.append("PCA ANALYSIS RESULTS:")
        report.append("")
        
        for feature_type in ['hidden_states', 'embeddings']:
            report.append(f"{feature_type.replace('_', ' ').upper()}:")
            for text_type in ['type1', 'type2']:
                variance_ratio = pca_results[feature_type][text_type]['explained_variance_ratio']
                total_variance = sum(variance_ratio)
                
                report.append(f"  {text_type.upper()}:")
                report.append(f"    - PC1 explains {variance_ratio[0]:.1%} of variance")
                report.append(f"    - PC2 explains {variance_ratio[1]:.1%} of variance")
                report.append(f"    - Total variance explained: {total_variance:.1%}")
            report.append("")
        
        # Center movement analysis
        report.append("CENTER MOVEMENT ANALYSIS:")
        report.append("")
        
        for feature_type in ['hidden_states', 'embeddings']:
            report.append(f"{feature_type.replace('_', ' ').upper()}:")
            for text_type in ['type1', 'type2']:
                centroids, movement_stats = self._calculate_center_movements(
                    pca_results[feature_type][text_type], 
                    max_iterations
                )
                
                report.append(f"  {text_type.upper()}:")
                if movement_stats:
                    report.append(f"    - Total centroid movement: {movement_stats['total_distance']:.4f}")
                    report.append(f"    - Average step size: {movement_stats['avg_step_size']:.4f}")
                    report.append(f"    - Maximum step size: {movement_stats['max_step_size']:.4f}")
                    
                    # Analyze movement pattern
                    if len(movement_stats['step_sizes']) > 1:
                        step_trend = np.polyfit(range(len(movement_stats['step_sizes'])), movement_stats['step_sizes'], 1)[0]
                        trend_desc = "increasing" if step_trend > 0 else "decreasing" if step_trend < 0 else "stable"
                        report.append(f"    - Movement trend: {trend_desc} (slope: {step_trend:.4f})")
                else:
                    report.append("    - No movement data available")
            report.append("")
        
        # Distance analysis
        report.append("ITERATION DISTANCE ANALYSIS:")
        report.append("")
        
        for feature_type in ['hidden_states', 'embeddings']:
            report.append(f"{feature_type.replace('_', ' ').upper()}:")
            for text_type in ['type1', 'type2']:
                distances = self._calculate_iteration_distances(
                    pca_results[feature_type][text_type], 
                    max_iterations
                )
                
                report.append(f"  {text_type.upper()}:")
                if distances:
                    report.append(f"    - Sequential distances: {[f'{d:.4f}' for d in distances]}")
                    report.append(f"    - Average sequential distance: {np.mean(distances):.4f}")
                    report.append(f"    - Maximum sequential distance: {np.max(distances):.4f}")
                    report.append(f"    - Minimum sequential distance: {np.min(distances):.4f}")
                    
                    # Distance trend analysis
                    if len(distances) > 1:
                        distance_trend = np.polyfit(range(len(distances)), distances, 1)[0]
                        trend_desc = "increasing" if distance_trend > 0 else "decreasing" if distance_trend < 0 else "stable"
                        report.append(f"    - Distance trend: {trend_desc} (slope: {distance_trend:.4f})")
                else:
                    report.append("    - No distance data available")
            report.append("")
        
        # Trajectory analysis interpretation
        report.append("TRAJECTORY ANALYSIS INTERPRETATION:")
        report.append("")
        report.append("The trajectory analysis shows how text samples move through semantic space")
        report.append("across paraphrasing iterations. Key insights:")
        report.append("")
        report.append("1. CENTROID MOVEMENT:")
        report.append("   - Tracks how the 'center of mass' of each iteration shifts")
        report.append("   - Large movements indicate significant semantic drift")
        report.append("   - Consistent direction suggests systematic bias in paraphrasing")
        report.append("")
        report.append("2. DISTANCE ANALYSIS:")
        report.append("   - Sequential distances show step-by-step semantic changes")
        report.append("   - Increasing trend suggests accelerating semantic drift")
        report.append("   - Decreasing trend suggests convergence to a stable state")
        report.append("")
        report.append("3. VARIANCE EXPLAINED:")
        report.append("   - Higher PC1/PC2 variance indicates more structured semantic space")
        report.append("   - Low variance suggests high-dimensional, complex relationships")
        report.append("")
        
        # Save report
        report_path = self.output_dir / 'enhanced_summary_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Enhanced summary report saved to: {report_path}")
        
        # Print to console
        print('\n'.join(report))
