#!/usr/bin/env python3
"""
Extended Semantic Space Experiment: 10 Iterations

This script extends the semantic space experiment to 10 iterations to better
observe the general trend of semantic drift over more paraphrasing cycles.

The experiment:
1. Extracts 100 samples each of type1 and type2 from the dataset
2. Performs 10 iterations of paraphrasing using Qwen3-4B model
3. Captures hidden states and embeddings at each iteration
4. Saves results in the same format as the original experiment

Output Structure:
semantic_space/output/{iteration}/{text_type}/
├── texts.json
├── hidden_states.npy  (from Qwen)
└── embeddings.npy     (from BGE-M3)

"""

import json
import logging
import os
import sys
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
from tqdm import tqdm
from openai import OpenAI

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import ExperimentConfig, PARAPHRASE_PROMPTS, DEFAULT_PROMPT_TYPE
from utils import (
    set_random_seeds, load_json_data, save_json_data, extract_text_samples,
    create_output_directories, load_model_safe, clean_generated_text,
    log_memory_usage
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
transformers_logging.set_verbosity_error()

logger = logging.getLogger(__name__)


class ExtendedSemanticSpaceExperiment:
    """
    Extended experiment class for conducting 10-iteration semantic space analysis.
    """
    
    def __init__(
        self, 
        data_path: str,
        output_dir: str = "../output",
        max_iterations: int = 10,
        num_samples: int = 100,
        device: Optional[str] = None,
        use_foundation: bool = True
    ):
        """
        Initialize the extended experiment.
        
        Args:
            data_path: Path to the input JSON data file
            output_dir: Directory to save experiment outputs (relative to parent)
            max_iterations: Maximum number of paraphrasing iterations (default: 10)
            num_samples: Number of samples to extract per type
            device: Device to use for model inference ('cuda', 'cpu', or None for auto)
            use_foundation: Whether to use existing 1-5 iteration data as foundation
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.max_iterations = max_iterations
        self.num_samples = num_samples
        self.use_foundation = use_foundation
        
        # Set up device
        self.device = ExperimentConfig.get_device() if device is None else torch.device(device)
        
        logger.info(f"Extended experiment initialized with {max_iterations} iterations")
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
        
        # Create output directories for extended iterations
        self.output_dir = self._create_extended_output_directories(output_dir, max_iterations)
    
    def _create_extended_output_directories(self, output_dir: str, max_iterations: int) -> Path:
        """Create output directories for extended experiment."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create directories for each iteration and text type
        for iteration in range(1, max_iterations + 1):
            for text_type in ['type1', 'type2']:
                iter_dir = output_path / str(iteration) / text_type
                iter_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created output directories for {max_iterations} iterations")
        return output_path
    
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
        Extract hidden states from the paraphrasing model (Qwen).
        
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
    
    def load_foundation_data(self, text_type: str) -> Dict[str, Any]:
        """
        Load existing 1-5 iteration data as foundation for extended experiment.
        This ensures consistency with previous experiments and avoids randomization.
        
        Args:
            text_type: Either 'type1' or 'type2'
            
        Returns:
            Dictionary containing foundation data (iterations 1-5)
        """
        logger.info(f"Loading foundation data (iterations 1-5) for {text_type}")
        
        foundation_results = {
            'original_texts': [],
            'iterations': {i: {'texts': [], 'hidden_states': [], 'embeddings': []} 
                          for i in range(1, 6)}  # Only 1-5 for foundation
        }
        
        foundation_loaded = False
        
        # Try to load iterations 1-5 as foundation
        for iteration in range(1, 6):
            try:
                iteration_dir = self.output_dir / str(iteration) / text_type
                
                # Check if all required files exist
                if all([
                    (iteration_dir / 'texts.json').exists(),
                    (iteration_dir / 'hidden_states.npy').exists(),
                    (iteration_dir / 'embeddings.npy').exists()
                ]):
                    logger.info(f"Loading foundation data for {text_type} iteration {iteration}")
                    loaded_data = self.load_existing_iteration_data(text_type, iteration)
                    
                    foundation_results['iterations'][iteration]['texts'] = loaded_data['texts']
                    foundation_results['iterations'][iteration]['hidden_states'] = loaded_data['hidden_states']
                    foundation_results['iterations'][iteration]['embeddings'] = loaded_data['embeddings']
                    
                    # Set original texts from first loaded iteration
                    if not foundation_results['original_texts'] and loaded_data.get('original_texts'):
                        foundation_results['original_texts'] = loaded_data['original_texts']
                    
                    foundation_loaded = True
                else:
                    logger.warning(f"Foundation data incomplete for {text_type} iteration {iteration}")
                    break
                    
            except Exception as e:
                logger.error(f"Error loading foundation data for {text_type} iteration {iteration}: {e}")
                break
        
        if foundation_loaded:
            logger.info(f"Successfully loaded foundation data (1-5) for {text_type}")
        else:
            logger.warning(f"Could not load complete foundation data for {text_type}")
        
        return foundation_results
    
    def run_extended_iteration_experiment(self, text_type: str) -> Dict[str, Any]:
        """
        Run the extended iterative paraphrasing experiment for a given text type.
        First loads existing 1-5 iteration data as foundation, then extends to max_iterations.
        
        Args:
            text_type: Either 'type1' or 'type2'
            
        Returns:
            Dictionary containing all iteration results
        """
        logger.info(f"Running extended experiment for {text_type} ({self.max_iterations} iterations)")
        
        # Initialize results structure
        results = {
            'original_texts': [],
            'iterations': {i: {'texts': [], 'hidden_states': [], 'embeddings': []} 
                          for i in range(1, self.max_iterations + 1)}
        }
        
        # Step 1: Load foundation data (iterations 1-5) if available and requested
        if self.use_foundation:
            foundation_data = self.load_foundation_data(text_type)
        else:
            logger.info(f"Skipping foundation data loading for {text_type} (use_foundation=False)")
            foundation_data = {
                'original_texts': [],
                'iterations': {i: {'texts': [], 'hidden_states': [], 'embeddings': []} 
                              for i in range(1, 6)}
            }
        
        # Copy foundation data to results
        foundation_loaded_iterations = []
        for iteration in range(1, 6):
            if foundation_data['iterations'][iteration]['texts']:
                results['iterations'][iteration] = foundation_data['iterations'][iteration]
                foundation_loaded_iterations.append(iteration)
        
        if foundation_data['original_texts']:
            results['original_texts'] = foundation_data['original_texts']
        
        if foundation_loaded_iterations:
            logger.info(f"Foundation loaded for {text_type} iterations: {foundation_loaded_iterations}")
        
        # Step 2: Check for existing extended data (6+)
        existing_data = self.check_existing_data(text_type)
        existing_extended_iterations = [iter_num for iter_num in range(6, self.max_iterations + 1) 
                                      if existing_data.get(iter_num, False)]
        
        if existing_extended_iterations:
            logger.info(f"Found existing extended data for {text_type} iterations: {existing_extended_iterations}")
        
        # Step 3: Load existing extended data (6+)
        for iteration in existing_extended_iterations:
            try:
                logger.info(f"Loading existing extended data for {text_type} iteration {iteration}")
                loaded_data = self.load_existing_iteration_data(text_type, iteration)
                
                results['iterations'][iteration]['texts'] = loaded_data['texts']
                results['iterations'][iteration]['hidden_states'] = loaded_data['hidden_states']
                results['iterations'][iteration]['embeddings'] = loaded_data['embeddings']
                    
            except Exception as e:
                logger.error(f"Error loading existing extended data for {text_type} iteration {iteration}: {e}")
                # Remove from existing iterations so it gets regenerated
                if iteration in existing_extended_iterations:
                    existing_extended_iterations.remove(iteration)
        
        # Step 4: Determine missing iterations
        all_loaded_iterations = foundation_loaded_iterations + existing_extended_iterations
        missing_iterations = [i for i in range(1, self.max_iterations + 1) if i not in all_loaded_iterations]
        
        if missing_iterations:
            logger.info(f"Generating data for {text_type} iterations: {missing_iterations}")
            
            # If we don't have original texts yet, set them up from samples
            if not results['original_texts']:
                if foundation_loaded_iterations:
                    # Use original texts from foundation if available
                    logger.info(f"Using original texts from foundation data for {text_type}")
                else:
                    # Fall back to sample texts
                    logger.info(f"Using sample texts as original texts for {text_type}")
                    results['original_texts'] = [sample['text'] for sample in self.samples_data[text_type]]
            
            # Determine the number of samples to process
            num_samples = len(results['original_texts']) if results['original_texts'] else len(self.samples_data[text_type])
            
            for sample_idx in tqdm(range(num_samples), desc=f"Processing {text_type} samples"):
                # Start with original text
                if results['original_texts']:
                    current_text = results['original_texts'][sample_idx]
                else:
                    current_text = self.samples_data[text_type][sample_idx]['text']
                
                # Process each iteration
                for iteration in range(1, self.max_iterations + 1):
                    if iteration in all_loaded_iterations:
                        # Use existing data to update current_text for next iteration
                        if sample_idx < len(results['iterations'][iteration]['texts']):
                            current_text = results['iterations'][iteration]['texts'][sample_idx]
                        continue
                    
                    if iteration not in missing_iterations:
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
            results: Results dictionary from run_extended_iteration_experiment
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
            
            # Save hidden states (from Qwen)
            hidden_states = np.array(results['iterations'][iteration]['hidden_states'])
            np.save(iteration_dir / 'hidden_states.npy', hidden_states)
            
            # Save embeddings (from BGE-M3)
            embeddings = np.array(results['iterations'][iteration]['embeddings'])
            np.save(iteration_dir / 'embeddings.npy', embeddings)
            
            logger.info(f"Saved iteration {iteration} results for {text_type}")
    
    def run_full_extended_experiment(self) -> None:
        """Run the complete extended experiment pipeline."""
        logger.info(f"Starting extended semantic space experiment ({self.max_iterations} iterations)")
        log_memory_usage("Starting extended experiment: ")
        
        try:
            # Load models
            self.load_models()
            
            # Load and sample data
            self.load_and_sample_data()
            
            # Run experiments for both text types
            for text_type in ['type1', 'type2']:
                logger.info(f"Processing {text_type} samples...")
                results = self.run_extended_iteration_experiment(text_type)
                self.save_iteration_results(text_type, results)
                self.experiment_results[text_type] = results
                log_memory_usage(f"After {text_type} processing: ")
            
            log_memory_usage("Extended experiment completed: ")
            logger.info("Extended experiment completed successfully!")
            
            print("\n" + "="*60)
            print("EXTENDED SEMANTIC SPACE EXPERIMENT COMPLETED!")
            print("="*60)
            print(f"📁 Results saved to: {self.output_dir}")
            print(f"🔄 Iterations completed: {self.max_iterations}")
            print("📊 Generated files per iteration:")
            print("  - texts.json")
            print("  - hidden_states.npy (from Qwen)")
            print("  - embeddings.npy (from BGE-M3)")
            print("="*60)
            
        except Exception as e:
            logger.error(f"Extended experiment failed: {e}")
            raise


def main():
    """Main function to run the extended experiment."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run extended semantic space experiment (10 iterations)")
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations (default: 10)')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples per type (default: 100)')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--data-path', type=str, default='../../data/test/final_generated_data.json', 
                       help='Path to data file')
    parser.add_argument('--output-dir', type=str, default='../output', 
                       help='Output directory (relative to semantic_space/)')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run extended experiment
    experiment = ExtendedSemanticSpaceExperiment(
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_iterations=args.iterations,
        num_samples=args.samples,
        device=args.device
    )
    
    experiment.run_full_extended_experiment()


if __name__ == "__main__":
    main()
