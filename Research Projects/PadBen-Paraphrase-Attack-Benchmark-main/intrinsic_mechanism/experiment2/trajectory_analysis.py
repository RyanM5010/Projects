#!/usr/bin/env python3
"""
Trajectory Analysis for Semantic Space Experiment

This script performs PCA-based trajectory analysis to visualize how centroids
move through semantic space across iterations.

For both hidden states and embeddings:
1. Performs PCA with n_components=2
2. Computes centroid for each iteration
3. Visualizes centroid movement trajectory
4. Saves results and visualizations
"""

import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrajectoryAnalyzer:
    """Analyzer for computing trajectory analysis using PCA."""
    
    def __init__(self, output_dir: str, max_iterations: int = 5):
        """
        Initialize the trajectory analyzer.
        
        Args:
            output_dir: Directory containing experiment results
            max_iterations: Maximum number of iterations
        """
        self.output_dir = Path(output_dir)
        self.max_iterations = max_iterations
        
        # Set up plotting style
        plt.style.use('default')
        self.colors = plt.cm.viridis(np.linspace(0, 1, max_iterations))
        
    def load_data(self, iteration: int, text_type: str, data_type: str) -> np.ndarray:
        """
        Load data for a specific iteration, text type, and data type.
        
        Args:
            iteration: Iteration number (1-based)
            text_type: 'type1' or 'type2'
            data_type: 'hidden_states' or 'embeddings'
            
        Returns:
            Numpy array containing the data
        """
        file_path = self.output_dir / str(iteration) / text_type / f"{data_type}.npy"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        data = np.load(file_path)
        logger.info(f"Loaded {data_type} for iteration {iteration}, {text_type}: shape {data.shape}")
        return data
    
    def perform_pca_analysis(self, data_type: str) -> Dict[str, Any]:
        """
        Perform PCA analysis on the specified data type.
        
        Args:
            data_type: 'hidden_states' or 'embeddings'
            
        Returns:
            Dictionary containing PCA analysis results
        """
        logger.info(f"Performing PCA analysis for {data_type}")
        
        results = {
            'data_type': data_type,
            'text_types': {}
        }
        
        for text_type in ['type1', 'type2']:
            logger.info(f"Processing {text_type}")
            
            # Collect all data across iterations
            all_data = []
            iteration_labels = []
            centroids = {}
            
            for iteration in range(1, self.max_iterations + 1):
                try:
                    data = self.load_data(iteration, text_type, data_type)
                    all_data.append(data)
                    iteration_labels.extend([iteration] * len(data))
                    
                    # Compute centroid for this iteration
                    centroid = np.mean(data, axis=0)
                    centroids[iteration] = centroid
                    
                except FileNotFoundError as e:
                    logger.warning(f"Skipping iteration {iteration}: {e}")
                    continue
            
            if not all_data:
                logger.error(f"No data found for {text_type}")
                continue
            
            # Concatenate all data
            all_data_combined = np.vstack(all_data)
            logger.info(f"Combined data shape for {text_type}: {all_data_combined.shape}")
            
            # Standardize the data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(all_data_combined)
            
            # Perform PCA
            pca = PCA(n_components=2)
            data_pca = pca.fit_transform(data_scaled)
            
            # Transform centroids to PCA space
            centroids_scaled = {}
            centroids_pca = {}
            for iteration, centroid in centroids.items():
                centroid_scaled = scaler.transform([centroid])[0]
                centroid_pca = pca.transform([centroid_scaled])[0]
                centroids_scaled[iteration] = centroid_scaled
                centroids_pca[iteration] = centroid_pca
            
            # Store results
            results['text_types'][text_type] = {
                'pca_components': data_pca.tolist(),
                'iteration_labels': iteration_labels,
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'centroids_original': {str(k): v.tolist() for k, v in centroids.items()},
                'centroids_pca': {str(k): v.tolist() for k, v in centroids_pca.items()},
                'num_samples': len(all_data_combined),
                'num_iterations': len(centroids)
            }
            
            logger.info(f"PCA completed for {text_type}: explained variance = {pca.explained_variance_ratio_}")
        
        return results
    
    def create_trajectory_visualization(self, results: Dict[str, Any]) -> None:
        """
        Create trajectory visualization showing centroid movement.
        
        Args:
            results: PCA analysis results
        """
        data_type = results['data_type']
        logger.info(f"Creating trajectory visualization for {data_type}")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'Trajectory Analysis: {data_type.replace("_", " ").title()} Centroid Movement', 
                    fontsize=16, fontweight='bold')
        
        for i, text_type in enumerate(['type1', 'type2']):
            if text_type not in results['text_types']:
                continue
                
            ax = axes[i]
            data = results['text_types'][text_type]
            
            # Get PCA components and labels
            pca_components = np.array(data['pca_components'])
            iteration_labels = np.array(data['iteration_labels'])
            centroids_pca = data['centroids_pca']
            
            # Plot all data points with different colors for each iteration
            for iteration in range(1, self.max_iterations + 1):
                if str(iteration) in centroids_pca:
                    mask = iteration_labels == iteration
                    if np.any(mask):
                        points = pca_components[mask]
                        ax.scatter(
                            points[:, 0], points[:, 1],
                            c=[self.colors[iteration-1]], 
                            alpha=0.3,
                            s=20,
                            label=f'Iteration {iteration}'
                        )
            
            # Plot centroids and trajectory
            centroid_iterations = sorted([int(k) for k in centroids_pca.keys()])
            centroid_points = np.array([centroids_pca[str(it)] for it in centroid_iterations])
            
            if len(centroid_points) > 1:
                # Plot trajectory line
                ax.plot(
                    centroid_points[:, 0], centroid_points[:, 1], 
                    'k--', alpha=0.7, linewidth=2, label='Centroid trajectory'
                )
                
                # Plot individual centroids
                for j, iteration in enumerate(centroid_iterations):
                    centroid = centroid_points[j]
                    ax.scatter(
                        centroid[0], centroid[1],
                        c=[self.colors[iteration-1]],
                        s=200,
                        marker='*',
                        edgecolors='black',
                        linewidth=2,
                        zorder=5
                    )
                    
                    # Add iteration labels
                    ax.annotate(
                        f'{iteration}', 
                        (centroid[0], centroid[1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=12, fontweight='bold'
                    )
                
                # Add movement vectors
                for j in range(len(centroid_points) - 1):
                    start = centroid_points[j]
                    end = centroid_points[j + 1]
                    ax.annotate('', xy=end, xytext=start,
                              arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.7))
            
            # Formatting
            variance_ratio = data['explained_variance_ratio']
            ax.set_title(f'{text_type.upper()}', fontweight='bold')
            ax.set_xlabel(f'PC1 ({variance_ratio[0]:.1%} variance)')
            ax.set_ylabel(f'PC2 ({variance_ratio[1]:.1%} variance)')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Add movement statistics
            if len(centroid_points) > 1:
                total_distance = self._calculate_trajectory_distance(centroid_points)
                max_step = self._calculate_max_step_size(centroid_points)
                
                stats_text = f"Total movement: {total_distance:.4f}\nMax step size: {max_step:.4f}"
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.output_dir / 'trajectory_analysis' / f'{data_type}_trajectory.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"Trajectory visualization saved to: {viz_path}")
        
        plt.show()
    
    def _calculate_trajectory_distance(self, points: np.ndarray) -> float:
        """Calculate total distance traveled along trajectory."""
        total_distance = 0.0
        for i in range(len(points) - 1):
            distance = np.linalg.norm(points[i + 1] - points[i])
            total_distance += distance
        return total_distance
    
    def _calculate_max_step_size(self, points: np.ndarray) -> float:
        """Calculate maximum step size in trajectory."""
        max_step = 0.0
        for i in range(len(points) - 1):
            distance = np.linalg.norm(points[i + 1] - points[i])
            max_step = max(max_step, distance)
        return max_step
    
    def compute_trajectory_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute detailed trajectory statistics.
        
        Args:
            results: PCA analysis results
            
        Returns:
            Dictionary containing trajectory statistics
        """
        data_type = results['data_type']
        logger.info(f"Computing trajectory statistics for {data_type}")
        
        stats = {
            'data_type': data_type,
            'text_types': {}
        }
        
        for text_type in ['type1', 'type2']:
            if text_type not in results['text_types']:
                continue
                
            data = results['text_types'][text_type]
            centroids_pca = data['centroids_pca']
            
            # Get centroid trajectory
            centroid_iterations = sorted([int(k) for k in centroids_pca.keys()])
            centroid_points = np.array([centroids_pca[str(it)] for it in centroid_iterations])
            
            if len(centroid_points) > 1:
                # Calculate step sizes
                step_sizes = []
                for i in range(len(centroid_points) - 1):
                    step_size = np.linalg.norm(centroid_points[i + 1] - centroid_points[i])
                    step_sizes.append(step_size)
                
                # Calculate trajectory statistics
                total_distance = sum(step_sizes)
                avg_step_size = np.mean(step_sizes)
                max_step_size = np.max(step_sizes)
                min_step_size = np.min(step_sizes)
                
                # Calculate trajectory direction consistency
                directions = []
                for i in range(len(centroid_points) - 1):
                    direction = centroid_points[i + 1] - centroid_points[i]
                    directions.append(direction / np.linalg.norm(direction))  # Normalize
                
                # Calculate average direction consistency (dot product of consecutive directions)
                direction_consistency = []
                for i in range(len(directions) - 1):
                    consistency = np.dot(directions[i], directions[i + 1])
                    direction_consistency.append(consistency)
                
                avg_direction_consistency = np.mean(direction_consistency) if direction_consistency else 0.0
                
                stats['text_types'][text_type] = {
                    'total_distance': total_distance,
                    'average_step_size': avg_step_size,
                    'max_step_size': max_step_size,
                    'min_step_size': min_step_size,
                    'step_sizes': step_sizes,
                    'num_steps': len(step_sizes),
                    'direction_consistency': avg_direction_consistency,
                    'centroid_coordinates': {str(k): v for k, v in zip(centroid_iterations, centroid_points.tolist())},
                    'explained_variance': data['explained_variance_ratio']
                }
            else:
                stats['text_types'][text_type] = {
                    'error': 'Insufficient data for trajectory analysis'
                }
        
        return stats
    
    def run_analysis(self) -> None:
        """Run complete trajectory analysis and save results."""
        logger.info("Starting trajectory analysis")
        
        # Create output directory
        output_dir = self.output_dir / 'trajectory_analysis'
        output_dir.mkdir(exist_ok=True)
        
        # Analyze hidden states
        logger.info("Analyzing hidden states...")
        hidden_states_results = self.perform_pca_analysis('hidden_states')
        hidden_states_stats = self.compute_trajectory_statistics(hidden_states_results)
        
        # Create visualization for hidden states
        self.create_trajectory_visualization(hidden_states_results)
        
        # Save hidden states results
        with open(output_dir / 'hidden_states_trajectory.json', 'w') as f:
            json.dump({
                'pca_analysis': hidden_states_results,
                'trajectory_statistics': hidden_states_stats
            }, f, indent=2)
        
        logger.info("Hidden states trajectory analysis saved")
        
        # Analyze embeddings
        logger.info("Analyzing embeddings...")
        embeddings_results = self.perform_pca_analysis('embeddings')
        embeddings_stats = self.compute_trajectory_statistics(embeddings_results)
        
        # Create visualization for embeddings
        self.create_trajectory_visualization(embeddings_results)
        
        # Save embeddings results
        with open(output_dir / 'embeddings_trajectory.json', 'w') as f:
            json.dump({
                'pca_analysis': embeddings_results,
                'trajectory_statistics': embeddings_stats
            }, f, indent=2)
        
        logger.info("Embeddings trajectory analysis saved")
        
        # Create summary report
        self._create_summary_report(output_dir, hidden_states_stats, embeddings_stats)
        
        print("\n" + "="*60)
        print("TRAJECTORY ANALYSIS COMPLETED!")
        print("="*60)
        print(f"📁 Results saved to: {output_dir}")
        print("📊 Files created:")
        print("  - hidden_states_trajectory.json")
        print("  - embeddings_trajectory.json")
        print("  - hidden_states_trajectory.png")
        print("  - embeddings_trajectory.png")
        print("  - trajectory_analysis_summary.txt")
        print("="*60)
    
    def _create_summary_report(self, output_dir: Path, hidden_stats: Dict, embedding_stats: Dict) -> None:
        """Create a summary report of the trajectory analysis."""
        report = []
        report.append("=" * 60)
        report.append("TRAJECTORY ANALYSIS SUMMARY REPORT")
        report.append("=" * 60)
        report.append("")
        
        report.append("This analysis shows how centroids move through 2D PCA space across iterations.")
        report.append("Key metrics:")
        report.append("- Total distance: Total movement of centroid across all iterations")
        report.append("- Average step size: Mean distance between consecutive iterations")
        report.append("- Direction consistency: How consistent the movement direction is (-1 to 1)")
        report.append("")
        
        # Hidden states summary
        report.append("HIDDEN STATES TRAJECTORY:")
        report.append("")
        for text_type in ['type1', 'type2']:
            if text_type in hidden_stats['text_types'] and 'total_distance' in hidden_stats['text_types'][text_type]:
                data = hidden_stats['text_types'][text_type]
                
                report.append(f"  {text_type.upper()}:")
                report.append(f"    - Total distance: {data['total_distance']:.4f}")
                report.append(f"    - Average step size: {data['average_step_size']:.4f}")
                report.append(f"    - Max step size: {data['max_step_size']:.4f}")
                report.append(f"    - Direction consistency: {data['direction_consistency']:.4f}")
                report.append(f"    - Explained variance: PC1={data['explained_variance'][0]:.1%}, PC2={data['explained_variance'][1]:.1%}")
                report.append("")
        
        # Embeddings summary
        report.append("EMBEDDINGS TRAJECTORY:")
        report.append("")
        for text_type in ['type1', 'type2']:
            if text_type in embedding_stats['text_types'] and 'total_distance' in embedding_stats['text_types'][text_type]:
                data = embedding_stats['text_types'][text_type]
                
                report.append(f"  {text_type.upper()}:")
                report.append(f"    - Total distance: {data['total_distance']:.4f}")
                report.append(f"    - Average step size: {data['average_step_size']:.4f}")
                report.append(f"    - Max step size: {data['max_step_size']:.4f}")
                report.append(f"    - Direction consistency: {data['direction_consistency']:.4f}")
                report.append(f"    - Explained variance: PC1={data['explained_variance'][0]:.1%}, PC2={data['explained_variance'][1]:.1%}")
                report.append("")
        
        # Interpretation
        report.append("INTERPRETATION:")
        report.append("")
        report.append("- Higher total distance indicates more semantic drift across iterations")
        report.append("- Increasing step sizes suggest accelerating semantic change")
        report.append("- Direction consistency near 1 indicates systematic drift in one direction")
        report.append("- Direction consistency near 0 indicates random/oscillating movement")
        report.append("- Direction consistency near -1 indicates back-and-forth movement")
        report.append("")
        
        # Save report
        with open(output_dir / 'trajectory_analysis_summary.txt', 'w') as f:
            f.write('\n'.join(report))
        
        # Print to console
        print('\n'.join(report))


def main():
    """Main function to run trajectory analysis."""
    print("Trajectory Analysis for Semantic Space Experiment")
    print("=" * 50)
    print("Performing PCA-based trajectory analysis:")
    print("- PCA with n_components=2 for visualization")
    print("- Centroid tracking across iterations")
    print("- Movement statistics and visualization")
    print()
    
    try:
        analyzer = TrajectoryAnalyzer('output', max_iterations=5)
        analyzer.run_analysis()
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"\n❌ Analysis failed: {e}")


if __name__ == "__main__":
    main()
