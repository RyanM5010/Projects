#!/usr/bin/env python3
"""
Distance Analysis for Semantic Space Experiment

This script performs distance analysis between iterations using multiple distance metrics:
- Cosine similarity (1 - cosine similarity)
- Euclidean distance
- Manhattan distance

Outputs JSON files with distance measurements for both hidden states and embeddings.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from scipy.spatial.distance import cosine, euclidean, cityblock
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DistanceAnalyzer:
    """Analyzer for computing distances between iterations."""
    
    def __init__(self, output_dir: str, max_iterations: int = 5):
        """
        Initialize the distance analyzer.
        
        Args:
            output_dir: Directory containing experiment results
            max_iterations: Maximum number of iterations
        """
        self.output_dir = Path(output_dir)
        self.max_iterations = max_iterations
        
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
    
    def compute_centroid_distances(self, data_type: str) -> Dict[str, Any]:
        """
        Compute distances between iteration centroids.
        
        Args:
            data_type: 'hidden_states' or 'embeddings'
            
        Returns:
            Dictionary containing distance analysis results
        """
        logger.info(f"Computing centroid distances for {data_type}")
        
        results = {
            'data_type': data_type,
            'analysis_type': 'centroid_distances',
            'text_types': {}
        }
        
        for text_type in ['type1', 'type2']:
            logger.info(f"Processing {text_type}")
            
            # Load all iterations and compute centroids
            centroids = {}
            for iteration in range(1, self.max_iterations + 1):
                try:
                    data = self.load_data(iteration, text_type, data_type)
                    centroid = np.mean(data, axis=0)
                    centroids[iteration] = centroid
                    logger.info(f"Computed centroid for iteration {iteration}: shape {centroid.shape}")
                except FileNotFoundError as e:
                    logger.warning(f"Skipping iteration {iteration}: {e}")
                    continue
            
            # Compute pairwise distances between centroids
            distance_matrix = self._compute_pairwise_distances(centroids)
            
            # Compute sequential distances (iteration i to i+1)
            sequential_distances = self._compute_sequential_distances(centroids)
            
            results['text_types'][text_type] = {
                'centroids': {str(k): v.tolist() for k, v in centroids.items()},
                'distance_matrix': distance_matrix,
                'sequential_distances': sequential_distances,
                'num_iterations': len(centroids)
            }
        
        return results
    
    def compute_sample_distances(self, data_type: str) -> Dict[str, Any]:
        """
        Compute distances between all samples across iterations.
        
        Args:
            data_type: 'hidden_states' or 'embeddings'
            
        Returns:
            Dictionary containing sample distance analysis results
        """
        logger.info(f"Computing sample distances for {data_type}")
        
        results = {
            'data_type': data_type,
            'analysis_type': 'sample_distances',
            'text_types': {}
        }
        
        for text_type in ['type1', 'type2']:
            logger.info(f"Processing {text_type}")
            
            # Load all iterations
            all_data = {}
            for iteration in range(1, self.max_iterations + 1):
                try:
                    data = self.load_data(iteration, text_type, data_type)
                    all_data[iteration] = data
                except FileNotFoundError as e:
                    logger.warning(f"Skipping iteration {iteration}: {e}")
                    continue
            
            # Compute average pairwise distances between iterations
            avg_distances = self._compute_average_pairwise_distances(all_data)
            
            # Compute within-iteration and between-iteration statistics
            stats = self._compute_distance_statistics(all_data)
            
            results['text_types'][text_type] = {
                'average_distances': avg_distances,
                'statistics': stats,
                'num_iterations': len(all_data)
            }
        
        return results
    
    def _compute_pairwise_distances(self, centroids: Dict[int, np.ndarray]) -> Dict[str, List[List[float]]]:
        """Compute pairwise distances between centroids using multiple metrics."""
        iterations = sorted(centroids.keys())
        n = len(iterations)
        
        # Initialize distance matrices
        cosine_matrix = [[0.0] * n for _ in range(n)]
        euclidean_matrix = [[0.0] * n for _ in range(n)]
        manhattan_matrix = [[0.0] * n for _ in range(n)]
        
        for i, iter1 in enumerate(iterations):
            for j, iter2 in enumerate(iterations):
                if i != j:
                    emb1 = centroids[iter1]
                    emb2 = centroids[iter2]
                    
                    # Cosine distance (1 - cosine similarity)
                    cosine_dist = 1 - cosine_similarity([emb1], [emb2])[0][0]
                    cosine_matrix[i][j] = float(cosine_dist)
                    
                    # Euclidean distance
                    euclidean_dist = euclidean(emb1, emb2)
                    euclidean_matrix[i][j] = float(euclidean_dist)
                    
                    # Manhattan distance
                    manhattan_dist = cityblock(emb1, emb2)
                    manhattan_matrix[i][j] = float(manhattan_dist)
        
        return {
            'cosine': cosine_matrix,
            'euclidean': euclidean_matrix,
            'manhattan': manhattan_matrix,
            'iteration_order': iterations
        }
    
    def _compute_sequential_distances(self, centroids: Dict[int, np.ndarray]) -> Dict[str, List[float]]:
        """Compute sequential distances between consecutive iterations."""
        iterations = sorted(centroids.keys())
        
        cosine_seq = []
        euclidean_seq = []
        manhattan_seq = []
        
        for i in range(len(iterations) - 1):
            iter1 = iterations[i]
            iter2 = iterations[i + 1]
            
            emb1 = centroids[iter1]
            emb2 = centroids[iter2]
            
            # Cosine distance
            cosine_dist = 1 - cosine_similarity([emb1], [emb2])[0][0]
            cosine_seq.append(float(cosine_dist))
            
            # Euclidean distance
            euclidean_dist = euclidean(emb1, emb2)
            euclidean_seq.append(float(euclidean_dist))
            
            # Manhattan distance
            manhattan_dist = cityblock(emb1, emb2)
            manhattan_seq.append(float(manhattan_dist))
        
        return {
            'cosine': cosine_seq,
            'euclidean': euclidean_seq,
            'manhattan': manhattan_seq,
            'iteration_pairs': [(iterations[i], iterations[i+1]) for i in range(len(iterations)-1)]
        }
    
    def _compute_average_pairwise_distances(self, all_data: Dict[int, np.ndarray]) -> Dict[str, List[List[float]]]:
        """Compute average pairwise distances between all samples in different iterations."""
        iterations = sorted(all_data.keys())
        n = len(iterations)
        
        # Initialize distance matrices
        cosine_matrix = [[0.0] * n for _ in range(n)]
        euclidean_matrix = [[0.0] * n for _ in range(n)]
        manhattan_matrix = [[0.0] * n for _ in range(n)]
        
        for i, iter1 in enumerate(iterations):
            for j, iter2 in enumerate(iterations):
                if i != j:
                    data1 = all_data[iter1]
                    data2 = all_data[iter2]
                    
                    # Compute average distances between all pairs
                    cosine_dists = []
                    euclidean_dists = []
                    manhattan_dists = []
                    
                    # Sample a subset for efficiency (max 100 pairs)
                    n_samples = min(100, min(len(data1), len(data2)))
                    indices1 = np.random.choice(len(data1), n_samples, replace=False)
                    indices2 = np.random.choice(len(data2), n_samples, replace=False)
                    
                    for idx1 in indices1:
                        for idx2 in indices2:
                            emb1 = data1[idx1]
                            emb2 = data2[idx2]
                            
                            # Cosine distance
                            cosine_dist = 1 - cosine_similarity([emb1], [emb2])[0][0]
                            cosine_dists.append(cosine_dist)
                            
                            # Euclidean distance
                            euclidean_dist = euclidean(emb1, emb2)
                            euclidean_dists.append(euclidean_dist)
                            
                            # Manhattan distance
                            manhattan_dist = cityblock(emb1, emb2)
                            manhattan_dists.append(manhattan_dist)
                    
                    cosine_matrix[i][j] = float(np.mean(cosine_dists))
                    euclidean_matrix[i][j] = float(np.mean(euclidean_dists))
                    manhattan_matrix[i][j] = float(np.mean(manhattan_dists))
        
        return {
            'cosine': cosine_matrix,
            'euclidean': euclidean_matrix,
            'manhattan': manhattan_matrix,
            'iteration_order': iterations
        }
    
    def _compute_distance_statistics(self, all_data: Dict[int, np.ndarray]) -> Dict[str, Any]:
        """Compute within-iteration and between-iteration distance statistics."""
        iterations = sorted(all_data.keys())
        
        # Within-iteration distances (sample diversity)
        within_iteration_stats = {}
        for iteration in iterations:
            data = all_data[iteration]
            if len(data) > 1:
                # Compute pairwise distances within iteration
                n_samples = min(50, len(data))  # Sample for efficiency
                indices = np.random.choice(len(data), n_samples, replace=False)
                sampled_data = data[indices]
                
                distances = []
                for i in range(len(sampled_data)):
                    for j in range(i + 1, len(sampled_data)):
                        dist = euclidean(sampled_data[i], sampled_data[j])
                        distances.append(dist)
                
                within_iteration_stats[iteration] = {
                    'mean': float(np.mean(distances)),
                    'std': float(np.std(distances)),
                    'min': float(np.min(distances)),
                    'max': float(np.max(distances))
                }
        
        # Between-iteration distances (drift measurement)
        between_iteration_stats = {}
        for i, iter1 in enumerate(iterations[:-1]):
            iter2 = iterations[i + 1]
            
            data1 = all_data[iter1]
            data2 = all_data[iter2]
            
            # Compute distances between consecutive iterations
            n_samples = min(50, min(len(data1), len(data2)))
            indices1 = np.random.choice(len(data1), n_samples, replace=False)
            indices2 = np.random.choice(len(data2), n_samples, replace=False)
            
            distances = []
            for idx1 in indices1:
                for idx2 in indices2:
                    dist = euclidean(data1[idx1], data2[idx2])
                    distances.append(dist)
            
            between_iteration_stats[f"{iter1}_to_{iter2}"] = {
                'mean': float(np.mean(distances)),
                'std': float(np.std(distances)),
                'min': float(np.min(distances)),
                'max': float(np.max(distances))
            }
        
        return {
            'within_iteration': within_iteration_stats,
            'between_iteration': between_iteration_stats
        }
    
    def run_analysis(self) -> None:
        """Run complete distance analysis and save results."""
        logger.info("Starting distance analysis")
        
        # Create output directory
        output_dir = self.output_dir / 'distance_analysis'
        output_dir.mkdir(exist_ok=True)
        
        # Analyze hidden states
        logger.info("Analyzing hidden states...")
        hidden_states_centroid = self.compute_centroid_distances('hidden_states')
        hidden_states_sample = self.compute_sample_distances('hidden_states')
        
        # Save hidden states results
        with open(output_dir / 'hidden_states_distances.json', 'w') as f:
            json.dump({
                'centroid_analysis': hidden_states_centroid,
                'sample_analysis': hidden_states_sample
            }, f, indent=2)
        
        logger.info("Hidden states analysis saved to hidden_states_distances.json")
        
        # Analyze embeddings
        logger.info("Analyzing embeddings...")
        embeddings_centroid = self.compute_centroid_distances('embeddings')
        embeddings_sample = self.compute_sample_distances('embeddings')
        
        # Save embeddings results
        with open(output_dir / 'embeddings_distances.json', 'w') as f:
            json.dump({
                'centroid_analysis': embeddings_centroid,
                'sample_analysis': embeddings_sample
            }, f, indent=2)
        
        logger.info("Embeddings analysis saved to embeddings_distances.json")
        
        # Create summary report
        self._create_summary_report(output_dir, hidden_states_centroid, embeddings_centroid)
        
        print("\n" + "="*60)
        print("DISTANCE ANALYSIS COMPLETED!")
        print("="*60)
        print(f"📁 Results saved to: {output_dir}")
        print("📊 Files created:")
        print("  - hidden_states_distances.json")
        print("  - embeddings_distances.json")
        print("  - distance_analysis_summary.txt")
        print("="*60)
    
    def _create_summary_report(self, output_dir: Path, hidden_results: Dict, embedding_results: Dict) -> None:
        """Create a summary report of the distance analysis."""
        report = []
        report.append("=" * 60)
        report.append("DISTANCE ANALYSIS SUMMARY REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Hidden states summary
        report.append("HIDDEN STATES ANALYSIS:")
        report.append("")
        for text_type in ['type1', 'type2']:
            if text_type in hidden_results['text_types']:
                data = hidden_results['text_types'][text_type]
                seq_dist = data['sequential_distances']
                
                report.append(f"  {text_type.upper()}:")
                report.append(f"    - Sequential cosine distances: {[f'{d:.4f}' for d in seq_dist['cosine']]}")
                report.append(f"    - Sequential euclidean distances: {[f'{d:.4f}' for d in seq_dist['euclidean']]}")
                report.append(f"    - Sequential manhattan distances: {[f'{d:.4f}' for d in seq_dist['manhattan']]}")
                
                # Trend analysis
                if len(seq_dist['euclidean']) > 1:
                    trend = "increasing" if seq_dist['euclidean'][-1] > seq_dist['euclidean'][0] else "decreasing"
                    report.append(f"    - Distance trend: {trend}")
                report.append("")
        
        # Embeddings summary
        report.append("EMBEDDINGS ANALYSIS:")
        report.append("")
        for text_type in ['type1', 'type2']:
            if text_type in embedding_results['text_types']:
                data = embedding_results['text_types'][text_type]
                seq_dist = data['sequential_distances']
                
                report.append(f"  {text_type.upper()}:")
                report.append(f"    - Sequential cosine distances: {[f'{d:.4f}' for d in seq_dist['cosine']]}")
                report.append(f"    - Sequential euclidean distances: {[f'{d:.4f}' for d in seq_dist['euclidean']]}")
                report.append(f"    - Sequential manhattan distances: {[f'{d:.4f}' for d in seq_dist['manhattan']]}")
                
                # Trend analysis
                if len(seq_dist['euclidean']) > 1:
                    trend = "increasing" if seq_dist['euclidean'][-1] > seq_dist['euclidean'][0] else "decreasing"
                    report.append(f"    - Distance trend: {trend}")
                report.append("")
        
        # Save report
        with open(output_dir / 'distance_analysis_summary.txt', 'w') as f:
            f.write('\n'.join(report))
        
        # Print to console
        print('\n'.join(report))


def main():
    """Main function to run distance analysis."""
    print("Distance Analysis for Semantic Space Experiment")
    print("=" * 50)
    print("Computing distances between iterations using multiple metrics:")
    print("- Cosine distance (1 - cosine similarity)")
    print("- Euclidean distance")
    print("- Manhattan distance")
    print()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        analyzer = DistanceAnalyzer('output', max_iterations=5)
        analyzer.run_analysis()
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"\n❌ Analysis failed: {e}")


if __name__ == "__main__":
    main()
