#!/usr/bin/env python3
"""
Semantic Space Experiment: Iterative Paraphrasing Analysis

This script conducts an experiment to analyze how paraphrased text deviates from 
original text in semantic space through multiple iterations of paraphrasing.

The experiment:
1. Extracts 100 samples each of type1 and type2 from the dataset
2. Performs 5 iterations of paraphrasing using Qwen3-4B model
3. Captures hidden states and embeddings at each iteration
4. Performs PCA analysis on both hidden states and embeddings
5. Visualizes the semantic drift across iterations
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings

import numpy as np
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    logging as transformers_logging
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from openai import OpenAI

# Import local modules
from config import ExperimentConfig, PARAPHRASE_PROMPTS, DEFAULT_PROMPT_TYPE
from utils import (
    set_random_seeds, load_json_data, save_json_data, extract_text_samples,
    create_output_directories, load_model_safe, clean_generated_text,
    log_memory_usage
)
from visualization import SemanticSpaceVisualizer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
transformers_logging.set_verbosity_error()

logger = logging.getLogger(__name__)


class SemanticSpaceExperiment:
    """
    Main experiment class for conducting semantic space analysis through 
    iterative paraphrasing.
    """
    
    def __init__(
        self, 
        data_path: str,
        output_dir: str = "output",
        max_iterations: int = 5,
        num_samples: int = 100,
        device: Optional[str] = None
    ):
        """
        Initialize the experiment.
        
        Args:
            data_path: Path to the input JSON data file
            output_dir: Directory to save experiment outputs
            max_iterations: Maximum number of paraphrasing iterations
            num_samples: Number of samples to extract per type
            device: Device to use for model inference ('cuda', 'cpu', or None for auto)
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.max_iterations = max_iterations
        self.num_samples = num_samples
        
        # Set up device
        self.device = ExperimentConfig.get_device() if device is None else torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize models (will be loaded lazily)
        self.paraphrase_model = None
        self.paraphrase_tokenizer = None
        self.novita_client = None
        
        # Data storage
        self.samples_data = {}
        self.experiment_results = {}
        
        # Set random seeds for reproducibility
        set_random_seeds(ExperimentConfig.RANDOM_SEED)
        
        # Create output directories
        self.output_dir = create_output_directories(output_dir, max_iterations)
        
        # Initialize visualizer with new analysis directory
        self.analysis_dir = self.output_dir / 'analysis'
        self.analysis_dir.mkdir(exist_ok=True)
        self.visualizer = SemanticSpaceVisualizer(self.analysis_dir)
    
    
    def load_models(self) -> None:
        """Load the required models for paraphrasing and initialize embedding API client."""
        logger.info("Loading models...")
        log_memory_usage("Before model loading: ")
        
        # Load paraphrasing model
        self.paraphrase_tokenizer = AutoTokenizer.from_pretrained(
            ExperimentConfig.PARAPHRASE_MODEL,
            **ExperimentConfig.get_model_kwargs()
        )
        self.paraphrase_model = load_model_safe(
            ExperimentConfig.PARAPHRASE_MODEL,
            AutoModelForCausalLM,
            self.device,
            **ExperimentConfig.get_model_kwargs()
        )
        
        # Initialize Novita AI client for embeddings
        self.novita_client = OpenAI(
            api_key=ExperimentConfig.NOVITA_API_KEY,
            base_url=ExperimentConfig.NOVITA_BASE_URL
        )
        logger.info("Novita AI client initialized for BGE-M3 embeddings")
        
        log_memory_usage("After model loading: ")
    
    def load_and_sample_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load data from JSON file and extract samples.
        
        Returns:
            Dictionary containing sampled data for type1 and type2
        """
        logger.info(f"Loading data from: {self.data_path}")
        
        # Load data using utility function
        data = load_json_data(str(self.data_path))
        
        # Extract samples using utility function
        self.samples_data = extract_text_samples(
            data, 
            num_samples=self.num_samples,
            seed=ExperimentConfig.RANDOM_SEED
        )
        
        logger.info(f"Sampled {len(self.samples_data['type1'])} type1 and {len(self.samples_data['type2'])} type2 texts")
        
        return self.samples_data
    
    def paraphrase_text(self, text: str, temperature: float = None) -> str:
        """
        Paraphrase a given text using the loaded model.
        
        Args:
            text: Input text to paraphrase
            temperature: Sampling temperature for generation
            
        Returns:
            Paraphrased text
        """
        if temperature is None:
            temperature = ExperimentConfig.TEMPERATURE
            
        prompt = PARAPHRASE_PROMPTS[DEFAULT_PROMPT_TYPE].format(text=text)
        
        messages = [{"role": "user", "content": prompt}]
        
        inputs = self.paraphrase_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.paraphrase_model.generate(
                **inputs,
                max_new_tokens=ExperimentConfig.MAX_NEW_TOKENS,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.paraphrase_tokenizer.eos_token_id
            )
        
        # Extract only the generated part
        generated_text = self.paraphrase_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        ).strip()
        
        return clean_generated_text(generated_text)
    
    def get_hidden_states(self, text: str) -> np.ndarray:
        """
        Extract hidden states from the paraphrasing model.
        
        Args:
            text: Input text
            
        Returns:
            Hidden states as numpy array
        """
        inputs = self.paraphrase_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=ExperimentConfig.MAX_LENGTH
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.paraphrase_model(**inputs, output_hidden_states=True)
            # Use the last hidden state, mean pooling across sequence length
            hidden_states = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
        
        return hidden_states.squeeze()
    
    def get_embeddings(self, text: str) -> np.ndarray:
        """
        Extract embeddings using BGE-M3 model via Novita AI API.
        
        Args:
            text: Input text
            
        Returns:
            Text embeddings as numpy array
        """
        try:
            # Use Novita AI API for embeddings
            response = self.novita_client.embeddings.create(
                model=ExperimentConfig.EMBEDDING_MODEL,
                input=text
            )
            
            # Extract embeddings from response
            embeddings = np.array(response.data[0].embedding)
            return embeddings
            
        except Exception as e:
            logger.error(f"Error getting embeddings via Novita AI: {e}")
            # Fallback: return zero vector of standard BGE-M3 size
            return np.zeros(1024, dtype=np.float32)
    
    def check_existing_data(self, text_type: str) -> Dict[int, bool]:
        """
        Check which iterations already have saved data.
        
        Args:
            text_type: Either 'type1' or 'type2'
            
        Returns:
            Dictionary mapping iteration numbers to existence status
        """
        existing_data = {}
        
        for iteration in range(1, self.max_iterations + 1):
            iteration_dir = self.output_dir / str(iteration) / text_type
            
            # Check if all required files exist
            files_exist = all([
                (iteration_dir / 'texts.json').exists(),
                (iteration_dir / 'hidden_states.npy').exists(),
                (iteration_dir / 'embeddings.npy').exists()
            ])
            
            existing_data[iteration] = files_exist
            
        return existing_data
    
    def load_existing_iteration_data(self, text_type: str, iteration: int) -> Dict[str, Any]:
        """
        Load existing data for a specific iteration and text type.
        
        Args:
            text_type: Either 'type1' or 'type2'
            iteration: Iteration number
            
        Returns:
            Dictionary containing loaded data
        """
        iteration_dir = self.output_dir / str(iteration) / text_type
        
        # Load texts
        texts_data = load_json_data(str(iteration_dir / 'texts.json'))
        
        # Load hidden states and embeddings
        hidden_states = np.load(iteration_dir / 'hidden_states.npy')
        embeddings = np.load(iteration_dir / 'embeddings.npy')
        
        return {
            'texts': texts_data['texts'],
            'original_texts': texts_data.get('original_texts', []),
            'hidden_states': hidden_states.tolist(),
            'embeddings': embeddings.tolist()
        }
    
    def run_iteration_experiment(self, text_type: str) -> Dict[str, Any]:
        """
        Run the iterative paraphrasing experiment for a given text type.
        Loads existing data if available, otherwise generates new data.
        
        Args:
            text_type: Either 'type1' or 'type2'
            
        Returns:
            Dictionary containing all iteration results
        """
        logger.info(f"Running experiment for {text_type}")
        
        # Check for existing data
        existing_data = self.check_existing_data(text_type)
        existing_iterations = [iter_num for iter_num, exists in existing_data.items() if exists]
        
        if existing_iterations:
            logger.info(f"Found existing data for {text_type} iterations: {existing_iterations}")
        
        samples = self.samples_data[text_type]
        results = {
            'original_texts': [],
            'iterations': {i: {'texts': [], 'hidden_states': [], 'embeddings': []} 
                          for i in range(1, self.max_iterations + 1)}
        }
        
        # Load existing data first
        for iteration in existing_iterations:
            try:
                logger.info(f"Loading existing data for {text_type} iteration {iteration}")
                loaded_data = self.load_existing_iteration_data(text_type, iteration)
                
                results['iterations'][iteration]['texts'] = loaded_data['texts']
                results['iterations'][iteration]['hidden_states'] = loaded_data['hidden_states']
                results['iterations'][iteration]['embeddings'] = loaded_data['embeddings']
                
                # Set original texts from first loaded iteration
                if not results['original_texts'] and loaded_data.get('original_texts'):
                    results['original_texts'] = loaded_data['original_texts']
                    
            except Exception as e:
                logger.error(f"Error loading existing data for {text_type} iteration {iteration}: {e}")
                # Remove from existing iterations so it gets regenerated
                existing_iterations.remove(iteration)
        
        # Generate missing iterations
        missing_iterations = [i for i in range(1, self.max_iterations + 1) if i not in existing_iterations]
        
        if missing_iterations:
            logger.info(f"Generating data for {text_type} iterations: {missing_iterations}")
            
            # If we don't have original texts yet, set them up
            if not results['original_texts']:
                results['original_texts'] = [sample['text'] for sample in samples]
            
            for sample_idx, sample in enumerate(tqdm(samples, desc=f"Processing {text_type} samples")):
                current_text = sample['text']
                
                # Process each iteration
                for iteration in range(1, self.max_iterations + 1):
                    if iteration in existing_iterations:
                        # Use existing data to update current_text for next iteration
                        if sample_idx < len(results['iterations'][iteration]['texts']):
                            current_text = results['iterations'][iteration]['texts'][sample_idx]
                        continue
                    
                    try:
                        # Paraphrase the current text
                        paraphrased_text = self.paraphrase_text(current_text)
                        
                        # Get hidden states and embeddings
                        hidden_states = self.get_hidden_states(paraphrased_text)
                        embeddings = self.get_embeddings(paraphrased_text)
                        
                        # Store results
                        results['iterations'][iteration]['texts'].append(paraphrased_text)
                        results['iterations'][iteration]['hidden_states'].append(hidden_states)
                        results['iterations'][iteration]['embeddings'].append(embeddings)
                        
                        # Update current text for next iteration
                        current_text = paraphrased_text
                        
                    except Exception as e:
                        logger.error(f"Error processing sample {sample['idx']} at iteration {iteration}: {e}")
                        # Use the previous text if paraphrasing fails
                        results['iterations'][iteration]['texts'].append(current_text)
                        results['iterations'][iteration]['hidden_states'].append(
                            results['iterations'][iteration-1]['hidden_states'][-1] if iteration > 1 
                            else self.get_hidden_states(current_text)
                        )
                        results['iterations'][iteration]['embeddings'].append(
                            results['iterations'][iteration-1]['embeddings'][-1] if iteration > 1 
                            else self.get_embeddings(current_text)
                        )
        else:
            logger.info(f"All data for {text_type} already exists, skipping generation")
        
        return results
    
    def save_iteration_results(self, text_type: str, results: Dict[str, Any]) -> None:
        """
        Save results for each iteration (only saves new/missing data).
        
        Args:
            text_type: Either 'type1' or 'type2'
            results: Results dictionary from run_iteration_experiment
        """
        logger.info(f"Saving results for {text_type}")
        
        # Check which iterations need to be saved
        existing_data = self.check_existing_data(text_type)
        
        for iteration in range(1, self.max_iterations + 1):
            # Skip if data already exists and has content
            if existing_data.get(iteration, False) and results['iterations'][iteration]['texts']:
                logger.info(f"Skipping save for {text_type} iteration {iteration} (already exists)")
                continue
            
            # Only save if we have new data
            if not results['iterations'][iteration]['texts']:
                logger.warning(f"No data to save for {text_type} iteration {iteration}")
                continue
                
            iteration_dir = self.output_dir / str(iteration) / text_type
            
            # Save texts using utility function
            texts_data = {
                'iteration': iteration,
                'text_type': text_type,
                'texts': results['iterations'][iteration]['texts'],
                'original_texts': results['original_texts']
            }
            
            save_json_data(texts_data, iteration_dir / 'texts.json')
            
            # Save hidden states
            hidden_states = np.array(results['iterations'][iteration]['hidden_states'])
            np.save(iteration_dir / 'hidden_states.npy', hidden_states)
            
            # Save embeddings
            embeddings = np.array(results['iterations'][iteration]['embeddings'])
            np.save(iteration_dir / 'embeddings.npy', embeddings)
            
            logger.info(f"Saved iteration {iteration} results for {text_type}")
    
    def perform_pca_analysis(self) -> Dict[str, Any]:
        """
        Perform PCA analysis on hidden states and embeddings across all iterations.
        
        Returns:
            Dictionary containing PCA results
        """
        logger.info("Performing PCA analysis")
        
        pca_results = {
            'hidden_states': {'type1': {}, 'type2': {}},
            'embeddings': {'type1': {}, 'type2': {}}
        }
        
        for text_type in ['type1', 'type2']:
            # Collect all hidden states and embeddings across iterations
            all_hidden_states = []
            all_embeddings = []
            iteration_labels = []
            
            for iteration in range(1, self.max_iterations + 1):
                iteration_dir = self.output_dir / str(iteration) / text_type
                
                # Load hidden states
                hidden_states = np.load(iteration_dir / 'hidden_states.npy')
                all_hidden_states.append(hidden_states)
                
                # Load embeddings
                embeddings = np.load(iteration_dir / 'embeddings.npy')
                all_embeddings.append(embeddings)
                
                # Create labels for this iteration
                iteration_labels.extend([iteration] * len(hidden_states))
            
            # Concatenate all data
            all_hidden_states = np.vstack(all_hidden_states)
            all_embeddings = np.vstack(all_embeddings)
            
            # Perform PCA on hidden states
            scaler_hidden = StandardScaler()
            hidden_states_scaled = scaler_hidden.fit_transform(all_hidden_states)
            
            pca_hidden = PCA(n_components=ExperimentConfig.PCA_COMPONENTS)
            hidden_states_pca = pca_hidden.fit_transform(hidden_states_scaled)
            
            pca_results['hidden_states'][text_type] = {
                'pca_components': hidden_states_pca,
                'iteration_labels': iteration_labels,
                'explained_variance_ratio': pca_hidden.explained_variance_ratio_
            }
            
            # Perform PCA on embeddings
            scaler_embed = StandardScaler()
            embeddings_scaled = scaler_embed.fit_transform(all_embeddings)
            
            pca_embed = PCA(n_components=ExperimentConfig.PCA_COMPONENTS)
            embeddings_pca = pca_embed.fit_transform(embeddings_scaled)
            
            pca_results['embeddings'][text_type] = {
                'pca_components': embeddings_pca,
                'iteration_labels': iteration_labels,
                'explained_variance_ratio': pca_embed.explained_variance_ratio_
            }
            
            logger.info(f"PCA completed for {text_type}")
        
        # Save PCA results
        pca_save_path = self.output_dir / 'visualizations' / 'pca_results.npz'
        np.savez(pca_save_path, **{
            f"{feature_type}_{text_type}_components": data['pca_components']
            for feature_type in ['hidden_states', 'embeddings']
            for text_type in ['type1', 'type2']
            for data in [pca_results[feature_type][text_type]]
        })
        logger.info(f"PCA results saved to: {pca_save_path}")
        
        return pca_results
    
    def create_visualizations(self, pca_results: Dict[str, Any]) -> None:
        """
        Create visualizations for PCA results using the visualization module.
        
        Args:
            pca_results: Results from perform_pca_analysis
        """
        logger.info("Creating integrated analysis and visualizations")
        
        # Create only the essential visualizations
        # 1. PCA comprehensive visualization
        self.visualizer.create_pca_visualization(pca_results, self.max_iterations)
        
        # 2. Integrated distance analysis
        logger.info("Performing pairwise distance analysis")
        distance_results = self.visualizer.create_pairwise_distance_analysis(
            self.output_dir, self.max_iterations
        )
        
        # 3. Save distance tables
        self.visualizer.save_distance_tables(distance_results, self.analysis_dir)
        
        # 4. Create distance trend plots
        distance_trends_path = self.analysis_dir / 'distance_trends.png'
        self.visualizer.create_distance_trend_plots(distance_results, distance_trends_path)
        
        # 5. Create centroid trajectory plots
        centroid_trajectory_path = self.analysis_dir / 'centroid_trajectories.png'
        self.visualizer.create_centroid_trajectory_plots(
            self.output_dir, centroid_trajectory_path, self.max_iterations
        )
        
        # Create summary report
        experiment_config = {
            'num_samples': self.num_samples,
            'max_iterations': self.max_iterations,
            'paraphrase_model': ExperimentConfig.PARAPHRASE_MODEL,
            'embedding_model': ExperimentConfig.EMBEDDING_MODEL,
            'device': str(self.device),
            'random_seed': ExperimentConfig.RANDOM_SEED
        }
        
        self.visualizer.create_summary_report(pca_results, experiment_config, self.max_iterations)
        
        logger.info("Integrated analysis completed successfully!")
        logger.info(f"Analysis results saved to: {self.analysis_dir}")
        print("\n" + "="*60)
        print("SEMANTIC SPACE ANALYSIS COMPLETED!")
        print("="*60)
        print(f"📁 Analysis results saved to: {self.analysis_dir}")
        print("📊 Generated files:")
        print("  - pca_comprehensive.png")
        print("  - distance_trends.png")
        print("  - centroid_trajectories.png")
        print("  - hidden_states_distance_table.csv")
        print("  - embeddings_distance_table.csv")
        print("  - distance_tables.json")
        print("  - enhanced_summary_report.txt")
        print("="*60)
    
    def run_full_experiment(self) -> None:
        """Run the complete experiment pipeline."""
        logger.info("Starting semantic space experiment")
        log_memory_usage("Starting experiment: ")
        
        try:
            # Load models
            self.load_models()
            
            # Load and sample data
            self.load_and_sample_data()
            
            # Run experiments for both text types
            for text_type in ['type1', 'type2']:
                logger.info(f"Processing {text_type} samples...")
                results = self.run_iteration_experiment(text_type)
                self.save_iteration_results(text_type, results)
                self.experiment_results[text_type] = results
                log_memory_usage(f"After {text_type} processing: ")
            
            # Perform PCA analysis
            pca_results = self.perform_pca_analysis()
            
            # Create visualizations
            self.create_visualizations(pca_results)
            
            log_memory_usage("Experiment completed: ")
            logger.info("Experiment completed successfully!")
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise


def main():
    """Main function to run the experiment."""
    # Use configuration from config module
    experiment = SemanticSpaceExperiment(
        data_path=ExperimentConfig.DATA_PATH,
        output_dir=ExperimentConfig.OUTPUT_DIR,
        max_iterations=ExperimentConfig.MAX_ITERATIONS,
        num_samples=ExperimentConfig.NUM_SAMPLES
    )
    
    experiment.run_full_experiment()


if __name__ == "__main__":
    main()
