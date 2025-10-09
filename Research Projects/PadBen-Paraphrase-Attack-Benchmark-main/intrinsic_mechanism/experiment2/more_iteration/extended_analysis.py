#!/usr/bin/env python3
"""
Integrated Extended Analysis for 10-Iteration Semantic Space Experiment

This script performs comprehensive analysis on the extended experiment data
to better observe semantic drift trends over more iterations.

Features:
- Pairwise distance analysis for up to 10 iterations
- Extended distance trend visualization
- Centroid trajectory analysis over more iterations
- Comprehensive statistical analysis
- Original sentence comparison analysis (integrated from type1_vs_original_type2_analysis.py)

"""

import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from scipy.spatial.distance import cosine, euclidean, cityblock
from sklearn.decomposition import PCA
from openai import OpenAI
import sys
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from visualization import SemanticSpaceVisualizer
from config import ExperimentConfig
from utils import load_json_data, extract_text_samples

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('default')
sns.set_palette("husl")


class IntegratedExtendedAnalyzer:
    """Integrated analyzer for extended semantic space experiment with more iterations."""
    
    def __init__(self, output_dir: str, max_iterations: int = 10, data_path: Optional[str] = None, num_samples: int = 100):
        """
        Initialize the integrated extended analyzer.
        
        Args:
            output_dir: Directory containing experiment results
            max_iterations: Maximum number of iterations to analyze
            data_path: Path to the original dataset (optional, for original sentence comparison)
            num_samples: Number of samples to use for original sentence analysis
        """
        self.output_dir = Path(output_dir)
        self.max_iterations = max_iterations
        self.data_path = Path(data_path) if data_path else None
        self.num_samples = num_samples
        self.analysis_dir = self.output_dir / 'extended_analysis'
        self.analysis_dir.mkdir(exist_ok=True)
        
        # Initialize Novita AI client for original sentence analysis
        if self.data_path:
            self.novita_client = OpenAI(
                api_key=ExperimentConfig.NOVITA_API_KEY,
                base_url=ExperimentConfig.NOVITA_BASE_URL
            )
            # Paths for saving original embeddings
            self.original_type1_embeddings_path = self.analysis_dir / 'original_type1_embeddings.npy'
            self.original_type2_embeddings_path = self.analysis_dir / 'original_type2_embeddings.npy'
        
        logger.info(f"Integrated extended analyzer initialized for {max_iterations} iterations")
    
    def get_embeddings(self, text: str) -> np.ndarray:
        """
        Extract embeddings using BGE-M3 model via Novita AI API.
        
        Args:
            text: Input text
            
        Returns:
            Text embeddings as numpy array
        """
        try:
            response = self.novita_client.embeddings.create(
                model=ExperimentConfig.EMBEDDING_MODEL,
                input=text
            )
            embeddings = np.array(response.data[0].embedding)
            return embeddings
        except Exception as e:
            logger.error(f"Error getting embeddings via Novita AI: {e}")
            return np.zeros(1024, dtype=np.float32)
    
    def load_original_sentences(self, text_type: str) -> List[str]:
        """
        Load original sentences from the dataset.
        
        Args:
            text_type: Either 'type1' or 'type2'
        
        Returns:
            List of original sentences
        """
        logger.info(f"Loading original {text_type} sentences from dataset")
        
        # Load the original dataset
        data = load_json_data(str(self.data_path))
        
        # Extract samples using the same method as the experiment
        samples_data = extract_text_samples(
            data, 
            num_samples=self.num_samples,
            seed=ExperimentConfig.RANDOM_SEED
        )
        
        # Get original sentences
        sentences = [sample['text'] for sample in samples_data[text_type]]
        
        logger.info(f"Loaded {len(sentences)} original {text_type} sentences")
        return sentences
    
    def load_type1_iteration_embeddings(self, iteration: int) -> Optional[np.ndarray]:
        """
        Load type1 embeddings for a specific iteration from saved .npy file.
        
        Args:
            iteration: Iteration number
            
        Returns:
            Numpy array of embeddings or None if not found
        """
        iteration_dir = self.output_dir / str(iteration) / "type1"
        embeddings_file = iteration_dir / "embeddings.npy"
        
        if not embeddings_file.exists():
            logger.warning(f"Type1 iteration {iteration} embeddings not found at {embeddings_file}")
            return None
        
        try:
            embeddings = np.load(embeddings_file)
            logger.info(f"Loaded type1 iteration {iteration} embeddings: shape {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Error loading type1 iteration {iteration} embeddings: {e}")
            return None
    
    def load_or_compute_original_embeddings(self, text_type: str) -> np.ndarray:
        """
        Load original embeddings from file, or compute and save them if not found.
        
        Args:
            text_type: Either 'type1' or 'type2'
        
        Returns:
            Numpy array of original embeddings
        """
        embeddings_path = getattr(self, f'original_{text_type}_embeddings_path')
        
        # Check if embeddings already exist
        if embeddings_path.exists():
            logger.info(f"Loading existing original {text_type} embeddings from {embeddings_path}")
            try:
                embeddings = np.load(embeddings_path)
                logger.info(f"Loaded original {text_type} embeddings: shape {embeddings.shape}")
                return embeddings
            except Exception as e:
                logger.error(f"Error loading existing {text_type} embeddings: {e}")
                logger.info(f"Will recompute {text_type} embeddings")
        
        # Compute embeddings
        logger.info(f"Computing original {text_type} embeddings")
        original_sentences = self.load_original_sentences(text_type)
        
        embeddings = []
        for text in tqdm(original_sentences, desc=f"Computing original {text_type} embeddings"):
            embedding = self.get_embeddings(text)
            embeddings.append(embedding)
        
        # Convert to numpy array and save
        embeddings_array = np.array(embeddings)
        np.save(embeddings_path, embeddings_array)
        logger.info(f"Saved original {text_type} embeddings to {embeddings_path}")
        logger.info(f"Original {text_type} embeddings shape: {embeddings_array.shape}")
        
        return embeddings_array
    
    def compute_distances_against_original(self, original_type: str) -> List[Dict[str, Any]]:
        """
        Compute distances between type1 iterations and original sentences.
        
        Args:
            original_type: Either 'type1' or 'type2' for the baseline
        
        Returns:
            List of distance results
        """
        logger.info(f"Computing type1 iterations vs original {original_type} distances")
        
        # Load or compute original embeddings
        original_embeddings = self.load_or_compute_original_embeddings(original_type)
        
        results = []
        
        # Process each type1 iteration
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"Processing type1 iteration {iteration} vs original {original_type}")
            
            # Load type1 embeddings for this iteration
            type1_embeddings = self.load_type1_iteration_embeddings(iteration)
            
            if type1_embeddings is None:
                logger.warning(f"No type1 embeddings found for iteration {iteration}")
                continue
            
            # Ensure we have the same number of samples
            min_samples = min(len(original_embeddings), len(type1_embeddings))
            
            if min_samples == 0:
                logger.warning(f"No samples to compare for iteration {iteration}")
                continue
            
            # Compute distances for each corresponding pair
            cosine_distances = []
            euclidean_distances = []
            manhattan_distances = []
            
            for sample_idx in range(min_samples):
                original_embedding = original_embeddings[sample_idx]
                type1_embedding = type1_embeddings[sample_idx]
                
                # Cosine distance (1 - cosine similarity)
                cosine_sim = np.dot(original_embedding, type1_embedding) / (
                    np.linalg.norm(original_embedding) * np.linalg.norm(type1_embedding)
                )
                cosine_dist = 1 - cosine_sim
                cosine_distances.append(cosine_dist)
                
                # Euclidean distance
                euclidean_dist = euclidean(original_embedding, type1_embedding)
                euclidean_distances.append(euclidean_dist)
                
                # Manhattan distance
                manhattan_dist = cityblock(original_embedding, type1_embedding)
                manhattan_distances.append(manhattan_dist)
            
            # Compute averages
            avg_cosine = np.mean(cosine_distances)
            avg_euclidean = np.mean(euclidean_distances)
            avg_manhattan = np.mean(manhattan_distances)
            
            results.append({
                'iteration': iteration,
                'comparison': f'type1_{iteration}_vs_original_{original_type}',
                'cosine_distance': float(avg_cosine),
                'euclidean_distance': float(avg_euclidean),
                'manhattan_distance': float(avg_manhattan),
                'num_samples': min_samples,
                'cosine_std': float(np.std(cosine_distances)),
                'euclidean_std': float(np.std(euclidean_distances)),
                'manhattan_std': float(np.std(manhattan_distances))
            })
            
            logger.info(f"Iteration {iteration}: cosine={avg_cosine:.4f}, euclidean={avg_euclidean:.4f}, manhattan={avg_manhattan:.4f}")
        
        return results
    
    def create_extended_pairwise_distance_analysis(self) -> Dict[str, Any]:
        """
        Create comprehensive pairwise distance analysis for extended iterations.
        
        Returns:
            Dictionary containing distance analysis results
        """
        logger.info("Creating extended pairwise distance analysis")
        
        results = {
            'hidden_states': {'type1': [], 'type2': [], 'type1_vs_type2_baseline': []},
            'embeddings': {'type1': [], 'type2': [], 'type1_vs_type2_baseline': []}
        }
        
        for data_type in ['hidden_states', 'embeddings']:
            for text_type in ['type1', 'type2']:
                logger.info(f"Processing {data_type} {text_type}")
                
                # Load all iterations
                all_data = {}
                for iteration in range(1, self.max_iterations + 1):
                    try:
                        file_path = self.output_dir / str(iteration) / text_type / f"{data_type}.npy"
                        data = np.load(file_path)
                        all_data[iteration] = data
                    except FileNotFoundError:
                        logger.warning(f"Skipping {data_type} {text_type} iteration {iteration}")
                        continue
                
                # Compute distances from iteration 1 to all others
                if 1 in all_data:
                    data_1 = all_data[1]
                    
                    for iteration in range(2, self.max_iterations + 1):
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
        
        # Add cross-type comparison: type1 iterations vs type2 baseline (iteration 1)
        logger.info("Computing cross-type comparison: type1 iterations vs type2 baseline")
        
        for data_type in ['hidden_states', 'embeddings']:
            try:
                # Load type2 iteration 1 as baseline
                type2_baseline_path = self.output_dir / "1" / "type2" / f"{data_type}.npy"
                if type2_baseline_path.exists():
                    type2_baseline = np.load(type2_baseline_path)
                    
                    # Compare type1 iterations 1-10 against type2 baseline
                    for iteration in range(1, self.max_iterations + 1):
                        type1_iter_path = self.output_dir / str(iteration) / "type1" / f"{data_type}.npy"
                        
                        if type1_iter_path.exists():
                            type1_iter_data = np.load(type1_iter_path)
                            
                            # Ensure both datasets have same number of samples
                            min_samples = min(len(type2_baseline), len(type1_iter_data))
                            
                            # Compute distances for each corresponding pair
                            cosine_distances = []
                            euclidean_distances = []
                            manhattan_distances = []
                            
                            for sample_idx in range(min_samples):
                                type2_sample = type2_baseline[sample_idx]
                                type1_sample = type1_iter_data[sample_idx]
                                
                                # Cosine distance (1 - cosine similarity)
                                cosine_sim = np.dot(type2_sample, type1_sample) / (np.linalg.norm(type2_sample) * np.linalg.norm(type1_sample))
                                cosine_dist = 1 - cosine_sim
                                cosine_distances.append(cosine_dist)
                                
                                # Euclidean distance
                                euclidean_dist = euclidean(type2_sample, type1_sample)
                                euclidean_distances.append(euclidean_dist)
                                
                                # Manhattan distance
                                manhattan_dist = cityblock(type2_sample, type1_sample)
                                manhattan_distances.append(manhattan_dist)
                            
                            # Compute averages
                            avg_cosine = np.mean(cosine_distances)
                            avg_euclidean = np.mean(euclidean_distances)
                            avg_manhattan = np.mean(manhattan_distances)
                            
                            results[data_type]['type1_vs_type2_baseline'].append({
                                'comparison': f'type1_{iteration}_vs_type2_1',
                                'cosine_distance': float(avg_cosine),
                                'euclidean_distance': float(avg_euclidean),
                                'manhattan_distance': float(avg_manhattan),
                                'num_samples': min_samples
                            })
                        else:
                            logger.warning(f"Type1 iteration {iteration} data not found for {data_type}")
                else:
                    logger.warning(f"Type2 baseline data not found for {data_type}")
                    
            except Exception as e:
                logger.error(f"Error in cross-type comparison for {data_type}: {e}")
        
        return results
    
    def create_extended_distance_trend_plots(self, distance_results: Dict[str, Any]) -> None:
        """
        Create extended line charts showing distance trends across more iterations.
        
        Args:
            distance_results: Results from pairwise distance analysis
        """
        logger.info("Creating extended distance trend plots")
        
        # Create figure with subplots (3 rows for the additional cross-type comparison)
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        fig.suptitle(f'Extended Distance Trends Across {self.max_iterations} Paraphrasing Iterations\n(Pairwise Sample Distances)', 
                    fontsize=16, fontweight='bold')
        
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
                           marker='o', linewidth=2.5, markersize=6,
                           color=colors[text_type], label=text_type.upper(),
                           alpha=0.8)
            
            ax.set_title(f'Hidden States - {label}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Iteration')
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xticks(range(2, self.max_iterations + 1, max(1, self.max_iterations // 8)))
            
            # Format y-axis based on metric scale
            if metric == 'cosine_distance':
                ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Plot embeddings (middle row)
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[1, i]
            
            for text_type in ['type1', 'type2']:
                data = distance_results['embeddings'][text_type]
                if data:
                    iterations = [int(row['comparison'].split('-')[1]) for row in data]
                    values = [row[metric] for row in data]
                    
                    ax.plot(iterations, values, 
                           marker='s', linewidth=2.5, markersize=6,
                           color=colors[text_type], label=text_type.upper(),
                           alpha=0.8)
            
            ax.set_title(f'Embeddings - {label}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Iteration')
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xticks(range(2, self.max_iterations + 1, max(1, self.max_iterations // 8)))
            
            # Format y-axis based on metric scale
            if metric == 'cosine_distance':
                ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Plot cross-type comparison (bottom row): type1 iterations vs type2 baseline
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[2, i]
            
            data = distance_results['hidden_states']['type1_vs_type2_baseline']
            if data:
                iterations = [int(row['comparison'].split('_')[1]) for row in data]
                values = [row[metric] for row in data]
                
                ax.plot(iterations, values, 
                       marker='D', linewidth=2.5, markersize=6,
                       color='#F18F01', label='Type1 vs Type2 Baseline',
                       alpha=0.8)
            
            ax.set_title(f'Cross-Type Comparison - {label}\n(Type1 Iterations vs Type2 Baseline)', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Type1 Iteration')
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xticks(range(1, self.max_iterations + 1, max(1, self.max_iterations // 8)))
            
            # Format y-axis based on metric scale
            if metric == 'cosine_distance':
                ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        plt.tight_layout()
        
        # Save the plot
        save_path = self.analysis_dir / f'extended_distance_trends_{self.max_iterations}iter.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Extended distance trends plot saved to: {save_path}")
        
        plt.show()
    
    def create_original_comparison_plots(self, original_results: Dict[str, Any]) -> None:
        """
        Create plots for original sentence comparison analysis.
        
        Args:
            original_results: Results from original sentence comparison
        """
        logger.info("Creating original sentence comparison plots")
        
        # Create figure with subplots for both comparisons
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Original Sentence Comparison Analysis\nType1 Iterations vs Original Sentences (BGE-M3 Embeddings)', 
                    fontsize=16, fontweight='bold')
        
        # Colors for different comparisons
        colors = {
            'type1_vs_original_type1': {'cosine': '#E74C3C', 'euclidean': '#3498DB', 'manhattan': '#2ECC71'},
            'type1_vs_original_type2': {'cosine': '#8E44AD', 'euclidean': '#F39C12', 'manhattan': '#1ABC9C'}
        }
        
        # Plot both comparisons
        for row_idx, (comparison_name, comparison_data) in enumerate([
            ('Type1 vs Original Type1', original_results['type1_vs_original_type1']),
            ('Type1 vs Original Type2', original_results['type1_vs_original_type2'])
        ]):
            if not comparison_data:
                logger.warning(f"No data to plot for {comparison_name}")
                continue
            
            # Extract data
            iterations = [row['iteration'] for row in comparison_data]
            cosine_distances = [row['cosine_distance'] for row in comparison_data]
            euclidean_distances = [row['euclidean_distance'] for row in comparison_data]
            manhattan_distances = [row['manhattan_distance'] for row in comparison_data]
            
            # Extract standard deviations for error bars
            cosine_stds = [row['cosine_std'] for row in comparison_data]
            euclidean_stds = [row['euclidean_std'] for row in comparison_data]
            manhattan_stds = [row['manhattan_std'] for row in comparison_data]
            
            color_key = f'type1_vs_original_type{row_idx + 1}'
            
            # Plot cosine distance
            ax = axes[row_idx, 0]
            ax.errorbar(iterations, cosine_distances, yerr=cosine_stds,
                       marker='o', linewidth=2.5, markersize=8,
                       color=colors[color_key]['cosine'], capsize=5, capthick=2,
                       alpha=0.8)
            ax.set_title(f'{comparison_name}\nCosine Distance', fontsize=12, fontweight='bold')
            ax.set_xlabel('Type1 Iteration')
            ax.set_ylabel('Cosine Distance')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(iterations)
            
            # Plot euclidean distance
            ax = axes[row_idx, 1]
            ax.errorbar(iterations, euclidean_distances, yerr=euclidean_stds,
                       marker='s', linewidth=2.5, markersize=8,
                       color=colors[color_key]['euclidean'], capsize=5, capthick=2,
                       alpha=0.8)
            ax.set_title(f'{comparison_name}\nEuclidean Distance', fontsize=12, fontweight='bold')
            ax.set_xlabel('Type1 Iteration')
            ax.set_ylabel('Euclidean Distance')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(iterations)
            
            # Plot manhattan distance
            ax = axes[row_idx, 2]
            ax.errorbar(iterations, manhattan_distances, yerr=manhattan_stds,
                       marker='^', linewidth=2.5, markersize=8,
                       color=colors[color_key]['manhattan'], capsize=5, capthick=2,
                       alpha=0.8)
            ax.set_title(f'{comparison_name}\nManhattan Distance', fontsize=12, fontweight='bold')
            ax.set_xlabel('Type1 Iteration')
            ax.set_ylabel('Manhattan Distance')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(iterations)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = self.analysis_dir / f'original_sentence_comparison_{self.max_iterations}iter.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Original sentence comparison plot saved to: {plot_path}")
        
        plt.show()
    
    def create_extended_centroid_trajectory_plots(self) -> None:
        """
        Create extended centroid trajectory visualization for more iterations.
        """
        logger.info("Creating extended centroid trajectory plots")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle(f'Extended Centroid Trajectories Across {self.max_iterations} Iterations', 
                    fontsize=16, fontweight='bold')
        
        # Data types to analyze
        data_types = ['hidden_states', 'embeddings']
        data_labels = ['Hidden States', 'Embeddings']
        
        # Extended color palette for more iterations
        colors = plt.cm.tab10(np.linspace(0, 1, self.max_iterations))
        
        # Markers for iterations (cycle through available markers)
        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', '+', 'x']
        
        for data_idx, (data_type, data_label) in enumerate(zip(data_types, data_labels)):
            try:
                # Compute centroids
                centroids = self._compute_centroids_from_files(data_type)
                
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
                           color='gray', linewidth=2, alpha=0.6, zorder=1,
                           label='Centroid Trajectory')
                    
                    # Plot centroid points for each iteration
                    for i, iteration in enumerate(iterations):
                        x, y = pca_centroids[text_type][iteration]
                        color = colors[iteration-1] if iteration <= len(colors) else colors[-1]
                        marker = markers[(iteration-1) % len(markers)]
                        
                        ax.scatter(x, y, 
                                 c=[color], 
                                 marker=marker,
                                 s=150, 
                                 edgecolors='white', 
                                 linewidth=1.5,
                                 zorder=3,
                                 label=f'Iter {iteration}' if i < 5 else "")  # Only label first 5 for clarity
                        
                        # Add iteration number as text (only for every 2nd iteration to avoid clutter)
                        if iteration % 2 == 1 or iteration <= 5:
                            ax.annotate(str(iteration), 
                                      (x, y), 
                                      xytext=(3, 3), 
                                      textcoords='offset points',
                                      fontsize=9, 
                                      fontweight='bold',
                                      color=color)
                    
                    # Set title and labels
                    ax.set_title(f'{data_label} - {text_type.upper()}', 
                               fontsize=12, fontweight='bold')
                    ax.set_xlabel(f'PC1 ({variance_ratio[0]:.1%} variance)')
                    ax.set_ylabel(f'PC2 ({variance_ratio[1]:.1%} variance)')
                    ax.grid(True, alpha=0.3)
                    
                    # Add legend only for the first subplot (with limited entries)
                    if data_idx == 0 and type_idx == 0:
                        handles, labels = ax.get_legend_handles_labels()
                        # Keep only first few entries to avoid clutter
                        ax.legend(handles[:6], labels[:6], 
                                loc='upper right', 
                                fontsize=8,
                                title='Iterations (first 5)')
                    
                    # Auto-adjust axis limits with padding
                    if trajectory_x and trajectory_y:
                        x_range = max(trajectory_x) - min(trajectory_x)
                        y_range = max(trajectory_y) - min(trajectory_y)
                        
                        x_padding = max(x_range * 0.15, 0.1)
                        y_padding = max(y_range * 0.15, 0.1)
                        
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
        save_path = self.analysis_dir / f'extended_centroid_trajectories_{self.max_iterations}iter.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Extended centroid trajectory plot saved to: {save_path}")
        
        plt.show()
    
    def _compute_centroids_from_files(self, data_type: str) -> Dict[str, Dict[int, np.ndarray]]:
        """Compute centroids from saved files."""
        centroids = {'type1': {}, 'type2': {}}
        
        for text_type in ['type1', 'type2']:
            for iteration in range(1, self.max_iterations + 1):
                try:
                    file_path = self.output_dir / str(iteration) / text_type / f"{data_type}.npy"
                    data = np.load(file_path)
                    centroid = np.mean(data, axis=0)
                    centroids[text_type][iteration] = centroid
                except FileNotFoundError:
                    continue
        
        return centroids
    
    def _apply_pca_to_centroids(self, centroids: Dict[str, Dict[int, np.ndarray]]) -> tuple:
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
    
    def save_extended_distance_tables(self, distance_results: Dict[str, Any]) -> None:
        """
        Save extended distance analysis results as CSV and JSON files.
        
        Args:
            distance_results: Results from pairwise distance analysis
        """
        logger.info("Saving extended distance analysis tables")
        
        # Save hidden states table (including cross-type comparison)
        if (distance_results['hidden_states']['type1'] or 
            distance_results['hidden_states']['type2'] or 
            distance_results['hidden_states']['type1_vs_type2_baseline']):
            
            combined_hidden = []
            
            # Add type1 and type2 comparisons
            for text_type in ['type1', 'type2']:
                for row in distance_results['hidden_states'][text_type]:
                    combined_row = row.copy()
                    combined_row['text_type'] = text_type
                    combined_hidden.append(combined_row)
            
            # Add cross-type comparison
            for row in distance_results['hidden_states']['type1_vs_type2_baseline']:
                combined_row = row.copy()
                combined_row['text_type'] = 'type1_vs_type2_baseline'
                combined_hidden.append(combined_row)
            
            if combined_hidden:
                df_hidden = pd.DataFrame(combined_hidden)
                df_hidden = df_hidden[['text_type', 'comparison', 'cosine_distance', 'euclidean_distance', 'manhattan_distance', 'num_samples']]
                df_hidden.to_csv(self.analysis_dir / f'extended_hidden_states_distance_table_{self.max_iterations}iter.csv', index=False)
        
        # Save embeddings table (including cross-type comparison)
        if (distance_results['embeddings']['type1'] or 
            distance_results['embeddings']['type2'] or 
            distance_results['embeddings']['type1_vs_type2_baseline']):
            
            combined_embeddings = []
            
            # Add type1 and type2 comparisons
            for text_type in ['type1', 'type2']:
                for row in distance_results['embeddings'][text_type]:
                    combined_row = row.copy()
                    combined_row['text_type'] = text_type
                    combined_embeddings.append(combined_row)
            
            # Add cross-type comparison
            for row in distance_results['embeddings']['type1_vs_type2_baseline']:
                combined_row = row.copy()
                combined_row['text_type'] = 'type1_vs_type2_baseline'
                combined_embeddings.append(combined_row)
            
            if combined_embeddings:
                df_embeddings = pd.DataFrame(combined_embeddings)
                df_embeddings = df_embeddings[['text_type', 'comparison', 'cosine_distance', 'euclidean_distance', 'manhattan_distance', 'num_samples']]
                df_embeddings.to_csv(self.analysis_dir / f'extended_embeddings_distance_table_{self.max_iterations}iter.csv', index=False)
        
        # Save as JSON
        with open(self.analysis_dir / f'extended_distance_tables_{self.max_iterations}iter.json', 'w') as f:
            json.dump(distance_results, f, indent=2)
        
        logger.info(f"Extended distance tables saved to: {self.analysis_dir}")
    
    def save_original_comparison_results(self, original_results: Dict[str, Any]) -> None:
        """
        Save original sentence comparison results to CSV and JSON files.
        
        Args:
            original_results: Results from original sentence comparison
        """
        logger.info("Saving original sentence comparison results")
        
        # Save type1 vs original type1 as CSV
        if original_results['type1_vs_original_type1']:
            df1 = pd.DataFrame(original_results['type1_vs_original_type1'])
            csv_path1 = self.analysis_dir / f'type1_vs_original_type1_distances_{self.max_iterations}iter.csv'
            df1.to_csv(csv_path1, index=False)
            logger.info(f"Type1 vs original type1 CSV saved to: {csv_path1}")
        
        # Save type1 vs original type2 as CSV
        if original_results['type1_vs_original_type2']:
            df2 = pd.DataFrame(original_results['type1_vs_original_type2'])
            csv_path2 = self.analysis_dir / f'type1_vs_original_type2_distances_{self.max_iterations}iter.csv'
            df2.to_csv(csv_path2, index=False)
            logger.info(f"Type1 vs original type2 CSV saved to: {csv_path2}")
        
        # Save combined CSV
        combined_data = []
        for comparison_type in ['type1_vs_original_type1', 'type1_vs_original_type2']:
            for row in original_results[comparison_type]:
                row_copy = row.copy()
                row_copy['comparison_type'] = comparison_type
                combined_data.append(row_copy)
        
        if combined_data:
            df_combined = pd.DataFrame(combined_data)
            csv_combined_path = self.analysis_dir / f'original_sentence_comparison_{self.max_iterations}iter.csv'
            df_combined.to_csv(csv_combined_path, index=False)
            logger.info(f"Combined original comparison CSV saved to: {csv_combined_path}")
        
        # Save as JSON
        json_path = self.analysis_dir / f'original_sentence_comparison_{self.max_iterations}iter.json'
        with open(json_path, 'w') as f:
            json.dump(original_results, f, indent=2)
        logger.info(f"Original comparison JSON saved to: {json_path}")
    
    def create_summary_statistics(self, distance_results: Dict[str, Any], original_results: Optional[Dict[str, Any]] = None) -> None:
        """
        Create and save comprehensive summary statistics.
        
        Args:
            distance_results: Results from pairwise distance analysis
            original_results: Results from original sentence comparison (optional)
        """
        logger.info("Creating comprehensive summary statistics")
        
        # Create summary report
        report = []
        report.append("=" * 80)
        report.append("INTEGRATED EXTENDED ANALYSIS SUMMARY")
        report.append("=" * 80)
        report.append("")
        
        # Metadata
        report.append("EXPERIMENT CONFIGURATION:")
        report.append(f"  - Maximum iterations analyzed: {self.max_iterations}")
        report.append(f"  - Analysis directory: {self.analysis_dir}")
        if self.data_path:
            report.append(f"  - Original dataset: {self.data_path}")
            report.append(f"  - Number of samples: {self.num_samples}")
        report.append("")
        
        # Extended pairwise distance analysis
        report.append("EXTENDED PAIRWISE DISTANCE ANALYSIS:")
        report.append("")
        
        for data_type in ['hidden_states', 'embeddings']:
            report.append(f"{data_type.upper()} ANALYSIS:")
            
            for text_type in ['type1', 'type2']:
                data = distance_results[data_type][text_type]
                if data:
                    report.append(f"  {text_type.upper()}:")
                    cosine_distances = [row['cosine_distance'] for row in data]
                    euclidean_distances = [row['euclidean_distance'] for row in data]
                    manhattan_distances = [row['manhattan_distance'] for row in data]
                    
                    report.append(f"    Cosine Distance - Mean: {np.mean(cosine_distances):.6f}, Std: {np.std(cosine_distances):.6f}")
                    report.append(f"    Euclidean Distance - Mean: {np.mean(euclidean_distances):.6f}, Std: {np.std(euclidean_distances):.6f}")
                    report.append(f"    Manhattan Distance - Mean: {np.mean(manhattan_distances):.6f}, Std: {np.std(manhattan_distances):.6f}")
            
            # Cross-type comparison
            cross_data = distance_results[data_type]['type1_vs_type2_baseline']
            if cross_data:
                report.append(f"  CROSS-TYPE COMPARISON (Type1 vs Type2 Baseline):")
                cosine_distances = [row['cosine_distance'] for row in cross_data]
                euclidean_distances = [row['euclidean_distance'] for row in cross_data]
                manhattan_distances = [row['manhattan_distance'] for row in cross_data]
                
                report.append(f"    Cosine Distance - Mean: {np.mean(cosine_distances):.6f}, Std: {np.std(cosine_distances):.6f}")
                report.append(f"    Euclidean Distance - Mean: {np.mean(euclidean_distances):.6f}, Std: {np.std(euclidean_distances):.6f}")
                report.append(f"    Manhattan Distance - Mean: {np.mean(manhattan_distances):.6f}, Std: {np.std(manhattan_distances):.6f}")
            
            report.append("")
        
        # Original sentence comparison analysis
        if original_results:
            report.append("ORIGINAL SENTENCE COMPARISON ANALYSIS:")
            report.append("")
            
            for comparison_name, comparison_key in [
                ('TYPE1 VS ORIGINAL TYPE1', 'type1_vs_original_type1'),
                ('TYPE1 VS ORIGINAL TYPE2', 'type1_vs_original_type2')
            ]:
                data = original_results[comparison_key]
                if data:
                    report.append(f"{comparison_name} ANALYSIS:")
                    
                    cosine_distances = [row['cosine_distance'] for row in data]
                    euclidean_distances = [row['euclidean_distance'] for row in data]
                    manhattan_distances = [row['manhattan_distance'] for row in data]
                    
                    report.append(f"  Cosine Distance:")
                    report.append(f"    - Mean: {np.mean(cosine_distances):.6f}")
                    report.append(f"    - Std:  {np.std(cosine_distances):.6f}")
                    report.append(f"    - Min:  {np.min(cosine_distances):.6f} (Iteration {data[np.argmin(cosine_distances)]['iteration']})")
                    report.append(f"    - Max:  {np.max(cosine_distances):.6f} (Iteration {data[np.argmax(cosine_distances)]['iteration']})")
                    report.append("")
                    
                    report.append(f"  Euclidean Distance:")
                    report.append(f"    - Mean: {np.mean(euclidean_distances):.6f}")
                    report.append(f"    - Std:  {np.std(euclidean_distances):.6f}")
                    report.append(f"    - Min:  {np.min(euclidean_distances):.6f} (Iteration {data[np.argmin(euclidean_distances)]['iteration']})")
                    report.append(f"    - Max:  {np.max(euclidean_distances):.6f} (Iteration {data[np.argmax(euclidean_distances)]['iteration']})")
                    report.append("")
                    
                    report.append(f"  Manhattan Distance:")
                    report.append(f"    - Mean: {np.mean(manhattan_distances):.6f}")
                    report.append(f"    - Std:  {np.std(manhattan_distances):.6f}")
                    report.append(f"    - Min:  {np.min(manhattan_distances):.6f} (Iteration {data[np.argmin(manhattan_distances)]['iteration']})")
                    report.append(f"    - Max:  {np.max(manhattan_distances):.6f} (Iteration {data[np.argmax(manhattan_distances)]['iteration']})")
                    report.append("")
        
        report.append("=" * 80)
        
        # Save report
        report_path = self.analysis_dir / f'integrated_analysis_summary_{self.max_iterations}iter.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Integrated analysis summary saved to: {report_path}")
        
        # Print key findings
        print('\n'.join(report))
    
    def run_complete_integrated_analysis(self) -> None:
        """Run the complete integrated analysis pipeline."""
        logger.info(f"Starting complete integrated analysis for {self.max_iterations} iterations")
        
        try:
            # 1. Extended pairwise distance analysis
            distance_results = self.create_extended_pairwise_distance_analysis()
            
            # 2. Save extended distance tables
            self.save_extended_distance_tables(distance_results)
            
            # 3. Create extended distance trend plots
            self.create_extended_distance_trend_plots(distance_results)
            
            # 4. Create centroid trajectory plots
            self.create_extended_centroid_trajectory_plots()
            
            # 5. Original sentence comparison analysis (if data path provided)
            original_results = None
            if self.data_path:
                logger.info("Running original sentence comparison analysis")
                original_results = {
                    'type1_vs_original_type1': self.compute_distances_against_original('type1'),
                    'type1_vs_original_type2': self.compute_distances_against_original('type2')
                }
                
                # Save original comparison results
                self.save_original_comparison_results(original_results)
                
                # Create original comparison plots
                self.create_original_comparison_plots(original_results)
            
            # 6. Create comprehensive summary statistics
            self.create_summary_statistics(distance_results, original_results)
            
            logger.info("Integrated analysis completed successfully!")
            
            print("\n" + "="*80)
            print("INTEGRATED ANALYSIS COMPLETED!")
            print("="*80)
            print(f"📁 Analysis results saved to: {self.analysis_dir}")
            print(f"🔄 Iterations analyzed: {self.max_iterations}")
            print("📊 Generated files:")
            print(f"  - extended_distance_trends_{self.max_iterations}iter.png")
            print(f"  - extended_centroid_trajectories_{self.max_iterations}iter.png")
            print(f"  - extended_hidden_states_distance_table_{self.max_iterations}iter.csv")
            print(f"  - extended_embeddings_distance_table_{self.max_iterations}iter.csv")
            print(f"  - extended_distance_tables_{self.max_iterations}iter.json")
            if original_results:
                print(f"  - original_sentence_comparison_{self.max_iterations}iter.png")
                print(f"  - type1_vs_original_type1_distances_{self.max_iterations}iter.csv")
                print(f"  - type1_vs_original_type2_distances_{self.max_iterations}iter.csv")
                print(f"  - original_sentence_comparison_{self.max_iterations}iter.csv (combined)")
                print(f"  - original_sentence_comparison_{self.max_iterations}iter.json")
                print(f"  - original_type1_embeddings.npy (cached)")
                print(f"  - original_type2_embeddings.npy (cached)")
            print(f"  - integrated_analysis_summary_{self.max_iterations}iter.txt")
            print("="*80)
            
        except Exception as e:
            logger.error(f"Integrated analysis failed: {e}")
            raise


def main():
    """Main function to run integrated analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run integrated extended analysis on multi-iteration experiment data")
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations to analyze (default: 10)')
    parser.add_argument('--output-dir', type=str, default='../output', 
                       help='Output directory containing experiment data (default: ../output)')
    parser.add_argument('--data-path', type=str, 
                       help='Path to original dataset for original sentence comparison (optional)')
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of samples for original sentence analysis (default: 100)')
    
    args = parser.parse_args()
    
    # Create and run integrated analyzer
    analyzer = IntegratedExtendedAnalyzer(
        output_dir=args.output_dir,
        max_iterations=args.iterations,
        data_path=args.data_path,
        num_samples=args.samples
    )
    
    analyzer.run_complete_integrated_analysis()


if __name__ == "__main__":
    main()
