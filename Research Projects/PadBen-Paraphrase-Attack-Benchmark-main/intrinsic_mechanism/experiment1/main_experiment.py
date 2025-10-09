#!/usr/bin/env python3
"""
Main experiment script for semantic vs paraphrase analysis.

This script implements the experiment design to explore whether semantic equivalence
equals paraphrase in LLM world by comparing Type2 (LLM generated) and Type4 (LLM paraphrased) texts.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from pathlib import Path

# OpenAI client for BGE-m3 embeddings
from openai import OpenAI

# Distance metrics
from scipy.spatial.distance import cosine, euclidean, cityblock
from sklearn.metrics.pairwise import cosine_similarity

# Dimensionality reduction and clustering
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SemanticParaphraseExperiment:
    """
    Main experiment class for semantic vs paraphrase analysis.
    
    This class handles data loading, embedding generation, distance calculations,
    and semantic space exploration for comparing Type2 and Type4 texts.
    """
    
    def __init__(self, data_path: str, api_key: str, base_url: str):
        """
        Initialize the experiment.
        
        Args:
            data_path: Path to the final_generated_data.json file
            api_key: OpenAI API key for BGE-m3 embeddings
            base_url: OpenAI base URL for the API
        """
        self.data_path = Path(data_path)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.embeddings = {}
        self.distances = {}
        self.results = {}
        
        # Set up environment variables
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = base_url
        
    def load_data(self, max_samples: int = 3000) -> Dict[str, List[str]]:
        """
        Load and extract text data from the JSON file.
        
        Args:
            max_samples: Maximum number of samples to load for each type (default: 3000)
        
        Returns:
            Dictionary containing lists of texts for each type
        """
        if max_samples is None:
            logger.info(f"Loading FULL dataset from {self.data_path} (no sample limit)")
        else:
            logger.info(f"Loading data from {self.data_path} (max {max_samples} samples per type)")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract texts by type
        texts = {
            'type1': [],  # human original
            'type2': [],  # llm generated
            'type4': []   # llm paraphrased original
        }
        
        for item in data:
            # Stop if we have enough samples for all types (only if max_samples is specified)
            if max_samples is not None and (len(texts['type1']) >= max_samples and 
                len(texts['type2']) >= max_samples and 
                len(texts['type4']) >= max_samples):
                break
                
            if 'human_original_text(type1)' in item and (max_samples is None or len(texts['type1']) < max_samples):
                texts['type1'].append(item['human_original_text(type1)'])
            if 'llm_generated_text(type2)' in item and (max_samples is None or len(texts['type2']) < max_samples):
                texts['type2'].append(item['llm_generated_text(type2)'])
            if 'llm_paraphrased_original_text(type4)-prompt-based' in item and (max_samples is None or len(texts['type4']) < max_samples):
                texts['type4'].append(item['llm_paraphrased_original_text(type4)-prompt-based'])
        
        logger.info(f"Loaded {len(texts['type1'])} Type1, {len(texts['type2'])} Type2, {len(texts['type4'])} Type4 texts")
        return texts
    
    def get_embeddings(self, texts: List[str], text_type: str, use_full_dataset: bool = False) -> np.ndarray:
        """
        Generate BGE-m3 embeddings for a list of texts.
        First checks if embeddings already exist and loads them if available.
        
        Args:
            texts: List of text strings
            text_type: Type identifier for logging
            use_full_dataset: If True, save to full_embeddings directory
            
        Returns:
            Numpy array of embeddings
        """
        # Check if embeddings already exist
        embeddings_dir = Path("full_embeddings" if use_full_dataset else "embeddings")
        embeddings_dir.mkdir(exist_ok=True)
        embeddings_file = embeddings_dir / f"{text_type}_embeddings.npy"
        
        if embeddings_file.exists():
            logger.info(f"Loading existing embeddings for {text_type} from {embeddings_file}")
            return np.load(embeddings_file)
        
        logger.info(f"Generating new embeddings for {len(texts)} {text_type} texts")
        
        embeddings = []
        batch_size = 100  # Process in batches to avoid API limits
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            try:
                response = self.client.embeddings.create(
                    model="baai/bge-m3",
                    input=batch_texts
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                # Continue with next batch
                continue
        
        embeddings_array = np.array(embeddings)
        
        # Save embeddings for future use
        logger.info(f"Saving embeddings for {text_type} to {embeddings_file}")
        np.save(embeddings_file, embeddings_array)
        
        return embeddings_array
    
    def calculate_distances(self, emb1: np.ndarray, emb2: np.ndarray) -> Dict[str, float]:
        """
        Calculate various distance metrics between two embedding arrays.
        
        Args:
            emb1: First embedding array
            emb2: Second embedding array
            
        Returns:
            Dictionary of distance metrics
        """
        # Ensure arrays have the same shape
        min_len = min(len(emb1), len(emb2))
        emb1 = emb1[:min_len]
        emb2 = emb2[:min_len]
        
        distances = {}
        
        # Calculate pairwise distances
        cosine_dists = []
        euclidean_dists = []
        manhattan_dists = []
        
        for i in range(min_len):
            # Cosine similarity (convert to distance)
            cos_sim = cosine_similarity([emb1[i]], [emb2[i]])[0][0]
            cosine_dists.append(1 - cos_sim)
            
            # Euclidean distance
            euclidean_dists.append(euclidean(emb1[i], emb2[i]))
            
            # Manhattan distance
            manhattan_dists.append(cityblock(emb1[i], emb2[i]))
        
        distances = {
            'cosine_similarity': np.mean(cosine_dists),
            'euclidean': np.mean(euclidean_dists),
            'manhattan': np.mean(manhattan_dists)
        }
        
        return distances
    
    def run_distance_analysis(self, texts: Dict[str, List[str]], use_full_dataset: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Run the main distance analysis experiment.
        
        Args:
            texts: Dictionary containing texts for each type
            use_full_dataset: If True, save to full_embeddings directory
            
        Returns:
            Dictionary of distance results
        """
        logger.info("Starting distance analysis experiment")
        
        # Generate embeddings for all types
        for text_type, text_list in texts.items():
            if text_list:  # Only process if we have texts
                self.embeddings[text_type] = self.get_embeddings(text_list, text_type, use_full_dataset)
        
        # Calculate distances between different type pairs
        distance_results = {}
        
        # Type1 vs Type2
        if 'type1' in self.embeddings and 'type2' in self.embeddings:
            distance_results['type1_vs_type2'] = self.calculate_distances(
                self.embeddings['type1'], self.embeddings['type2']
            )
        
        # Type1 vs Type4
        if 'type1' in self.embeddings and 'type4' in self.embeddings:
            distance_results['type1_vs_type4'] = self.calculate_distances(
                self.embeddings['type1'], self.embeddings['type4']
            )
        
        # Type2 vs Type4 (key comparison)
        if 'type2' in self.embeddings and 'type4' in self.embeddings:
            distance_results['type2_vs_type4'] = self.calculate_distances(
                self.embeddings['type2'], self.embeddings['type4']
            )
        
        self.distances = distance_results
        return distance_results
    
    def run_semantic_space_exploration(self) -> Dict[str, Any]:
        """
        Run semantic space exploration with dimensionality reduction and clustering.
        
        Returns:
            Dictionary containing exploration results
        """
        logger.info("Starting semantic space exploration")
        
        # Combine all embeddings
        all_embeddings = []
        labels = []
        type_mapping = {}
        
        for i, (text_type, embeddings) in enumerate(self.embeddings.items()):
            all_embeddings.append(embeddings)
            labels.extend([i] * len(embeddings))
            type_mapping[i] = text_type
        
        combined_embs = np.concatenate(all_embeddings)
        labels = np.array(labels)
        
        # Dimensionality reduction
        logger.info("Performing PCA dimensionality reduction")
        pca_2d = PCA(n_components=2, random_state=42).fit_transform(combined_embs)
        
        
        # Clustering analysis (3 clusters for 3 text types)
        logger.info("Performing KMeans clustering with 3 clusters")
        kmeans_labels = KMeans(n_clusters=3, random_state=42).fit_predict(combined_embs)
        
        exploration_results = {
            'combined_embeddings': combined_embs,
            'labels': labels,
            'type_mapping': type_mapping,
            'pca_2d': pca_2d,
            'kmeans_labels': kmeans_labels
        }
        
        return exploration_results
    
    def save_results(self, distance_results: Dict, exploration_results: Dict, output_dir: str = "results", use_full_dataset: bool = False):
        """
        Save experiment results to files.
        
        Args:
            distance_results: Results from distance analysis
            exploration_results: Results from semantic space exploration
            output_dir: Directory to save results
            use_full_dataset: If True, save to full_results directory
        """
        if use_full_dataset:
            output_dir = "full_results"
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save distance results
        with open(output_path / "distance_results.json", 'w') as f:
            json.dump(distance_results, f, indent=2)
        
        # Save exploration results (without large arrays)
        exploration_summary = {
            'type_mapping': exploration_results['type_mapping'],
            'num_samples': len(exploration_results['labels']),
            'pca_shape': exploration_results['pca_2d'].shape,
            'kmeans_clusters': len(np.unique(exploration_results['kmeans_labels']))
        }
        
        with open(output_path / "exploration_summary.json", 'w') as f:
            json.dump(exploration_summary, f, indent=2)
        
        # Save arrays separately
        np.save(output_path / "pca_2d.npy", exploration_results['pca_2d'])
        np.save(output_path / "kmeans_labels.npy", exploration_results['kmeans_labels'])
        np.save(output_path / "labels.npy", exploration_results['labels'])
        
        logger.info(f"Results saved to {output_path}")


def main():
    """Main execution function."""
    # Configuration - Get API key from environment variables
    data_path = "/Users/jonathanzha/Desktop/PADBen/data/test/final_generated_data.json"
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.probex.top/v1")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    # Initialize experiment
    experiment = SemanticParaphraseExperiment(data_path, api_key, base_url)
    
    try:
        # Load data (full dataset - no max_samples limit)
        texts = experiment.load_data(max_samples=None)
        
        # Run distance analysis with full dataset
        distance_results = experiment.run_distance_analysis(texts, use_full_dataset=True)
        logger.info("Distance analysis completed")
        logger.info(f"Results: {distance_results}")
        
        # Run semantic space exploration
        exploration_results = experiment.run_semantic_space_exploration()
        logger.info("Semantic space exploration completed")
        
        # Save results to full_results directory
        experiment.save_results(distance_results, exploration_results, use_full_dataset=True)
        
        logger.info("Full dataset experiment completed successfully!")
        logger.info("Results saved to full_results/ directory")
        logger.info("Embeddings saved to full_embeddings/ directory")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


if __name__ == "__main__":
    main()
