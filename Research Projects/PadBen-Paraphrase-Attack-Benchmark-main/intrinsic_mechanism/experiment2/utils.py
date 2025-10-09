"""
Utility functions for the semantic space experiment.

This module contains helper functions for data processing, model operations,
and file management used throughout the experiment.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from tqdm import tqdm

logger = logging.getLogger(__name__)


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load JSON data from file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List of data items
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} items from {file_path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {e}")
        raise


def save_json_data(data: Any, file_path: str, indent: int = 2) -> None:
    """
    Save data as JSON file.
    
    Args:
        data: Data to save
        file_path: Output file path
        indent: JSON indentation level
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    
    logger.info(f"Saved data to {file_path}")


def extract_text_samples(
    data: List[Dict[str, Any]], 
    num_samples: int = 100,
    seed: int = 42
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract samples of type1 and type2 texts from the dataset.
    
    Args:
        data: List of data items
        num_samples: Number of samples to extract per type
        seed: Random seed for sampling
        
    Returns:
        Dictionary with 'type1' and 'type2' sample lists
    """
    type1_samples = []
    type2_samples = []
    
    for item in data:
        # Extract type1 (human original text)
        if 'human_original_text(type1)' in item and item['human_original_text(type1)']:
            type1_samples.append({
                'idx': item['idx'],
                'text': item['human_original_text(type1)'],
                'dataset_source': item.get('dataset_source', 'unknown')
            })
        
        # Extract type2 (LLM generated text)
        if 'llm_generated_text(type2)' in item and item['llm_generated_text(type2)']:
            type2_samples.append({
                'idx': item['idx'],
                'text': item['llm_generated_text(type2)'],
                'dataset_source': item.get('dataset_source', 'unknown')
            })
    
    logger.info(f"Found {len(type1_samples)} type1 and {len(type2_samples)} type2 samples")
    
    # Sample the required number
    random.seed(seed)
    sampled_type1 = random.sample(type1_samples, min(num_samples, len(type1_samples)))
    sampled_type2 = random.sample(type2_samples, min(num_samples, len(type2_samples)))
    
    return {
        'type1': sampled_type1,
        'type2': sampled_type2
    }


def create_output_directories(output_dir: str, max_iterations: int) -> Path:
    """
    Create the output directory structure.
    
    Args:
        output_dir: Base output directory
        max_iterations: Maximum number of iterations
        
    Returns:
        Path object for the output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create iteration directories
    for iteration in range(1, max_iterations + 1):
        for text_type in ['type1', 'type2']:
            iteration_path = output_path / str(iteration) / text_type
            iteration_path.mkdir(parents=True, exist_ok=True)
    
    # Create visualization directory
    (output_path / 'visualizations').mkdir(exist_ok=True)
    
    logger.info(f"Created output directories at: {output_path}")
    return output_path


def load_model_safe(
    model_name: str, 
    model_class: type,
    device: torch.device,
    **kwargs
) -> Any:
    """
    Safely load a model with error handling.
    
    Args:
        model_name: Name/path of the model
        model_class: Model class to instantiate
        device: Device to load model on
        **kwargs: Additional arguments for model loading
        
    Returns:
        Loaded model
        
    Raises:
        RuntimeError: If model loading fails
    """
    try:
        model = model_class.from_pretrained(model_name, **kwargs).to(device)
        logger.info(f"Successfully loaded {model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise RuntimeError(f"Model loading failed: {e}")


def batch_process(
    items: List[Any], 
    process_func: callable, 
    batch_size: int = 32,
    desc: str = "Processing"
) -> List[Any]:
    """
    Process items in batches with progress tracking.
    
    Args:
        items: List of items to process
        process_func: Function to apply to each item
        batch_size: Size of each batch
        desc: Description for progress bar
        
    Returns:
        List of processed results
    """
    results = []
    
    for i in tqdm(range(0, len(items), batch_size), desc=desc):
        batch = items[i:i + batch_size]
        batch_results = []
        
        for item in batch:
            try:
                result = process_func(item)
                batch_results.append(result)
            except Exception as e:
                logger.warning(f"Failed to process item: {e}")
                batch_results.append(None)
        
        results.extend(batch_results)
    
    return results


def clean_generated_text(text: str) -> str:
    """
    Clean and normalize generated text.
    
    Args:
        text: Raw generated text
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove common artifacts
    text = text.strip()
    
    # Remove potential prompt artifacts
    if text.lower().startswith('paraphrased text:'):
        text = text[17:].strip()
    if text.lower().startswith('rewritten text:'):
        text = text[15:].strip()
    
    return text


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple similarity between two texts.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score (0-1)
    """
    # Simple word overlap similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)


def validate_experiment_data(data: Dict[str, Any]) -> bool:
    """
    Validate experiment data structure.
    
    Args:
        data: Experiment data to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = ['type1', 'type2']
    
    if not all(key in data for key in required_keys):
        logger.error(f"Missing required keys: {required_keys}")
        return False
    
    for text_type in required_keys:
        if not isinstance(data[text_type], list):
            logger.error(f"{text_type} should be a list")
            return False
        
        if not data[text_type]:
            logger.error(f"{text_type} list is empty")
            return False
        
        # Check sample structure
        sample = data[text_type][0]
        if not isinstance(sample, dict) or 'text' not in sample:
            logger.error(f"Invalid sample structure in {text_type}")
            return False
    
    return True


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage information.
    
    Returns:
        Dictionary with memory usage statistics
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        usage = {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        }
        
        if torch.cuda.is_available():
            usage['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            usage['gpu_cached_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        
        return usage
    except ImportError:
        return {'error': 'psutil not available'}
    except Exception as e:
        return {'error': str(e)}


def log_memory_usage(prefix: str = "") -> None:
    """
    Log current memory usage.
    
    Args:
        prefix: Prefix for log message
    """
    usage = get_memory_usage()
    if 'error' not in usage:
        msg = f"{prefix}Memory usage - RSS: {usage['rss_mb']:.1f}MB"
        if 'gpu_allocated_mb' in usage:
            msg += f", GPU: {usage['gpu_allocated_mb']:.1f}MB"
        logger.info(msg)
    else:
        logger.warning(f"{prefix}Could not get memory usage: {usage['error']}")


def setup_logging(log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_file: Optional log file path
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers
    )
    
    # Suppress some verbose logs
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
