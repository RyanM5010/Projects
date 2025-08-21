"""
OUTFOX Data Processing Module.

This module provides functionality to load, explore, and analyze the OUTFOX dataset,
which contains human and AI-generated text for synthetic text detection tasks.
The dataset is split into train/test/valid sets with pickle files.
"""

import logging
import warnings
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OUTFOXDataLoader:
    """
    A comprehensive loader and analyzer for the OUTFOX dataset.
    
    The OUTFOX dataset contains human and AI-generated text stored in pickle files
    across train, test, and validation splits. Each split contains:
    - contexts.pkl: Context information
    - humans.pkl: Human-written text (Type 1)
    - problem_statements.pkl: Problem statements or prompts
    """
    
    def __init__(self, data_dir: str = "./data/OUTFOX-data") -> None:
        """
        Initialize the OUTFOX data loader.
        
        Args:
            data_dir: Directory containing OUTFOX data splits.
        """
        self.data_dir = Path(data_dir)
        self.raw_datasets: Dict[str, Dict[str, Any]] = {}
        self.combined_dataset: Optional[pd.DataFrame] = None
        self.human_text_dataset: Optional[pd.DataFrame] = None
        self.generation_prompts_dataset: Optional[pd.DataFrame] = None
        
        # Define expected splits and file types
        self.expected_splits = ['train', 'test', 'valid']
        self.expected_file_types = ['contexts', 'humans', 'problem_statements']
        
    def discover_pickle_files(self) -> Dict[str, Dict[str, str]]:
        """
        Discover all pickle files in the OUTFOX data directory.
        
        Returns:
            Dictionary mapping splits to file types to file paths.
        """
        logger.info(f"Discovering pickle files in {self.data_dir}...")
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"OUTFOX data directory not found: {self.data_dir}")
        
        file_mapping = {}
        
        for split in self.expected_splits:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                logger.warning(f"Split directory not found: {split_dir}")
                continue
                
            file_mapping[split] = {}
            
            for file_type in self.expected_file_types:
                pickle_file = split_dir / f"{split}_{file_type}.pkl"
                if pickle_file.exists():
                    file_mapping[split][file_type] = str(pickle_file)
                    logger.info(f"  • Found: {split}/{file_type}")
                else:
                    logger.warning(f"  • Missing: {split}/{file_type}")
        
        logger.info(f"Discovery complete. Found {len(file_mapping)} splits")
        return file_mapping
    
    def load_pickle_file(self, file_path: str) -> Any:
        """
        Load a single pickle file.
        
        Args:
            file_path: Path to the pickle file.
            
        Returns:
            Loaded data from pickle file.
        """
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Loaded pickle file: {Path(file_path).name}")
            return data
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return None
    
    def load_individual_splits(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all pickle files from each split.
        
        Returns:
            Dictionary of loaded data organized by split and file type.
        """
        logger.info("Loading individual OUTFOX splits...")
        
        file_mapping = self.discover_pickle_files()
        datasets = {}
        
        for split, files in file_mapping.items():
            logger.info(f"Loading {split} split...")
            datasets[split] = {}
            
            for file_type, file_path in files.items():
                data = self.load_pickle_file(file_path)
                if data is not None:
                    datasets[split][file_type] = data
                    
                    # Log data structure info
                    if isinstance(data, (list, tuple)):
                        logger.info(f"  • {file_type}: {len(data)} items (list/tuple)")
                    elif isinstance(data, dict):
                        logger.info(f"  • {file_type}: {len(data)} items (dict)")
                    elif isinstance(data, pd.DataFrame):
                        logger.info(f"  • {file_type}: {len(data):,} rows, {len(data.columns)} cols (DataFrame)")
                    elif hasattr(data, '__len__'):
                        logger.info(f"  • {file_type}: {len(data)} items ({type(data).__name__})")
                    else:
                        logger.info(f"  • {file_type}: {type(data).__name__}")
        
        self.raw_datasets = datasets
        logger.info(f"Successfully loaded {len(datasets)} splits")
        return datasets
    
    def create_human_text_dataset(self) -> pd.DataFrame:
        """
        Create a dataset containing only index and human text (Type 1).
        
        Returns:
            DataFrame with index and human text.
        """
        logger.info("Creating human text dataset (Type 1)...")
        
        if not self.raw_datasets:
            self.load_individual_splits()
        
        all_records = []
        global_index = 0
        
        for split, split_data in self.raw_datasets.items():
            if 'humans' not in split_data:
                logger.warning(f"No 'humans' data found in {split} split")
                continue
            
            humans_data = split_data['humans']
            logger.info(f"Processing {split} split: {len(humans_data)} human texts")
            
            # Handle different data structures
            if isinstance(humans_data, (list, tuple)):
                for i, human_text in enumerate(humans_data):
                    all_records.append({
                        'index': global_index,
                        'split': split,
                        'split_index': i,
                        'human_text': human_text
                    })
                    global_index += 1
            
            elif isinstance(humans_data, dict):
                for key, human_text in humans_data.items():
                    all_records.append({
                        'index': global_index,
                        'split': split,
                        'split_index': key,
                        'human_text': human_text
                    })
                    global_index += 1
        
        self.human_text_dataset = pd.DataFrame(all_records)
        logger.info(f"Human text dataset created: {len(self.human_text_dataset):,} samples")
        
        return self.human_text_dataset
    
    def create_generation_prompts_dataset(self) -> pd.DataFrame:
        """
        Create a dataset with index and combined contexts + problem_statements for Type 2 generation.
        
        Returns:
            DataFrame with index and generation prompts.
        """
        logger.info("Creating generation prompts dataset (for Type 2)...")
        
        if not self.raw_datasets:
            self.load_individual_splits()
        
        all_records = []
        global_index = 0
        
        for split, split_data in self.raw_datasets.items():
            contexts_data = split_data.get('contexts')
            problem_statements_data = split_data.get('problem_statements')
            
            if contexts_data is None and problem_statements_data is None:
                logger.warning(f"No contexts or problem_statements data found in {split} split")
                continue
            
            # Determine the length for this split
            split_length = 0
            if contexts_data is not None and hasattr(contexts_data, '__len__'):
                split_length = max(split_length, len(contexts_data))
            if problem_statements_data is not None and hasattr(problem_statements_data, '__len__'):
                split_length = max(split_length, len(problem_statements_data))
            
            logger.info(f"Processing {split} split: {split_length} prompt pairs")
            
            for i in range(split_length):
                # Extract context
                context = ""
                if contexts_data is not None:
                    if isinstance(contexts_data, (list, tuple)) and i < len(contexts_data):
                        context = str(contexts_data[i]) if contexts_data[i] is not None else ""
                    elif isinstance(contexts_data, dict):
                        context = str(contexts_data.get(i, ""))
                
                # Extract problem statement
                problem_statement = ""
                if problem_statements_data is not None:
                    if isinstance(problem_statements_data, (list, tuple)) and i < len(problem_statements_data):
                        problem_statement = str(problem_statements_data[i]) if problem_statements_data[i] is not None else ""
                    elif isinstance(problem_statements_data, dict):
                        problem_statement = str(problem_statements_data.get(i, ""))
                
                # Combine context and problem statement
                combined_prompt = ""
                if context and problem_statement:
                    combined_prompt = f"Context: {context}\n\nProblem: {problem_statement}"
                elif context:
                    combined_prompt = f"Context: {context}"
                elif problem_statement:
                    combined_prompt = f"Problem: {problem_statement}"
                
                all_records.append({
                    'index': global_index,
                    'split': split,
                    'split_index': i,
                    'context': context,
                    'problem_statement': problem_statement,
                    'combined_prompt': combined_prompt
                })
                global_index += 1
        
        self.generation_prompts_dataset = pd.DataFrame(all_records)
        logger.info(f"Generation prompts dataset created: {len(self.generation_prompts_dataset):,} samples")
        
        return self.generation_prompts_dataset
    
    def analyze_data_structure(self) -> Dict[str, Any]:
        """
        Analyze the structure of loaded OUTFOX data.
        
        Returns:
            Dictionary containing comprehensive structure analysis.
        """
        if not self.raw_datasets:
            self.load_individual_splits()
        
        print("=" * 80)
        print("OUTFOX Dataset Structure Analysis")
        print("=" * 80)
        
        structure_info = {}
        
        for split, split_data in self.raw_datasets.items():
            print(f"\n📊 {split.upper()} Split:")
            structure_info[split] = {}
            
            for file_type, data in split_data.items():
                print(f"   • {file_type}:")
                
                # Analyze different data types
                if isinstance(data, (list, tuple)):
                    structure_info[split][file_type] = {
                        'type': type(data).__name__,
                        'length': len(data),
                        'sample_items': []
                    }
                    
                    print(f"     - Type: {type(data).__name__}")
                    print(f"     - Length: {len(data):,}")
                    
                    # Show sample items
                    if len(data) > 0:
                        print(f"     - Sample items (first 3):")
                        for i, item in enumerate(data[:3]):
                            if isinstance(item, str):
                                preview = item[:100] + "..." if len(item) > 100 else item
                                print(f"       [{i}]: {repr(preview)}")
                                structure_info[split][file_type]['sample_items'].append(preview)
                            else:
                                print(f"       [{i}]: {type(item).__name__} - {str(item)[:100]}")
                                structure_info[split][file_type]['sample_items'].append(str(type(item).__name__))
                
                elif isinstance(data, dict):
                    structure_info[split][file_type] = {
                        'type': 'dict',
                        'length': len(data),
                        'keys': list(data.keys())[:10],  # First 10 keys
                        'sample_values': {}
                    }
                    
                    print(f"     - Type: Dictionary")
                    print(f"     - Keys: {len(data):,}")
                    print(f"     - Sample keys: {list(data.keys())[:5]}")
                    
                    # Show sample values
                    for key in list(data.keys())[:3]:
                        value = data[key]
                        if isinstance(value, str):
                            preview = value[:100] + "..." if len(value) > 100 else value
                            print(f"       {key}: {repr(preview)}")
                            structure_info[split][file_type]['sample_values'][key] = preview
                        else:
                            print(f"       {key}: {type(value).__name__}")
                            structure_info[split][file_type]['sample_values'][key] = str(type(value).__name__)
                
                elif isinstance(data, pd.DataFrame):
                    structure_info[split][file_type] = {
                        'type': 'DataFrame',
                        'shape': data.shape,
                        'columns': list(data.columns),
                        'dtypes': data.dtypes.to_dict()
                    }
                    
                    print(f"     - Type: DataFrame")
                    print(f"     - Shape: {data.shape}")
                    print(f"     - Columns: {list(data.columns)}")
                
                else:
                    structure_info[split][file_type] = {
                        'type': type(data).__name__,
                        'info': str(data)[:200]
                    }
                    print(f"     - Type: {type(data).__name__}")
                    print(f"     - Info: {str(data)[:100]}")
        
        return structure_info
    
    def display_sample_data(self, num_samples: int = 3) -> None:
        """
        Display sample data from the OUTFOX dataset.
        
        Args:
            num_samples: Number of samples to display from each split.
        """
        if not self.raw_datasets:
            self.load_individual_splits()
        
        print("\n" + "=" * 80)
        print("OUTFOX Sample Data Examples")
        print("=" * 80)
        
        for split, split_data in self.raw_datasets.items():
            print(f"\n🔍 {split.upper()} Split Samples:")
            
            for file_type, data in split_data.items():
                print(f"\n   📋 {file_type}:")
                
                if isinstance(data, (list, tuple)):
                    for i, item in enumerate(data[:num_samples]):
                        if isinstance(item, str):
                            preview = item[:150] + "..." if len(item) > 150 else item
                            print(f"     [{i}]: {repr(preview)}")
                        else:
                            print(f"     [{i}]: {type(item).__name__} - {str(item)[:100]}")
                
                elif isinstance(data, dict):
                    sample_keys = list(data.keys())[:num_samples]
                    for key in sample_keys:
                        value = data[key]
                        if isinstance(value, str):
                            preview = value[:150] + "..." if len(value) > 150 else value
                            print(f"     {key}: {repr(preview)}")
                        else:
                            print(f"     {key}: {type(value).__name__}")
                
                elif isinstance(data, pd.DataFrame):
                    print(f"     DataFrame preview:")
                    print(data.head(num_samples).to_string(max_colwidth=50))
    
    def get_text_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate text statistics for string data in the dataset.
        
        Returns:
            Dictionary containing text statistics.
        """
        if not self.raw_datasets:
            self.load_individual_splits()
        
        print("\n" + "=" * 80)
        print("OUTFOX Text Statistics Analysis")
        print("=" * 80)
        
        text_stats = {}
        
        for split, split_data in self.raw_datasets.items():
            print(f"\n📊 {split.upper()} Split Text Statistics:")
            text_stats[split] = {}
            
            for file_type, data in split_data.items():
                if isinstance(data, (list, tuple)):
                    # Check if items are strings
                    string_items = [item for item in data if isinstance(item, str)]
                    
                    if string_items:
                        lengths = [len(item) for item in string_items]
                        word_counts = [len(item.split()) for item in string_items]
                        
                        stats = {
                            'total_items': len(data),
                            'string_items': len(string_items),
                            'char_length_mean': np.mean(lengths),
                            'char_length_std': np.std(lengths),
                            'char_length_min': min(lengths),
                            'char_length_max': max(lengths),
                            'word_count_mean': np.mean(word_counts),
                            'word_count_std': np.std(word_counts),
                            'word_count_min': min(word_counts),
                            'word_count_max': max(word_counts)
                        }
                        
                        text_stats[split][file_type] = stats
                        
                        print(f"   • {file_type}:")
                        print(f"     - Total items: {stats['total_items']:,}")
                        print(f"     - String items: {stats['string_items']:,}")
                        print(f"     - Char length: {stats['char_length_mean']:.1f} ± {stats['char_length_std']:.1f}")
                        print(f"       Range: {stats['char_length_min']}-{stats['char_length_max']}")
                        print(f"     - Word count: {stats['word_count_mean']:.1f} ± {stats['word_count_std']:.1f}")
                        print(f"       Range: {stats['word_count_min']}-{stats['word_count_max']}")
                
                elif isinstance(data, dict):
                    # Check dictionary values
                    string_values = [v for v in data.values() if isinstance(v, str)]
                    
                    if string_values:
                        lengths = [len(v) for v in string_values]
                        word_counts = [len(v.split()) for v in string_values]
                        
                        stats = {
                            'total_keys': len(data),
                            'string_values': len(string_values),
                            'char_length_mean': np.mean(lengths),
                            'char_length_std': np.std(lengths),
                            'char_length_min': min(lengths),
                            'char_length_max': max(lengths),
                            'word_count_mean': np.mean(word_counts),
                            'word_count_std': np.std(word_counts),
                            'word_count_min': min(word_counts),
                            'word_count_max': max(word_counts)
                        }
                        
                        text_stats[split][file_type] = stats
                        
                        print(f"   • {file_type} (dict values):")
                        print(f"     - Total keys: {stats['total_keys']:,}")
                        print(f"     - String values: {stats['string_values']:,}")
                        print(f"     - Char length: {stats['char_length_mean']:.1f} ± {stats['char_length_std']:.1f}")
                        print(f"       Range: {stats['char_length_min']}-{stats['char_length_max']}")
                        print(f"     - Word count: {stats['word_count_mean']:.1f} ± {stats['word_count_std']:.1f}")
                        print(f"       Range: {stats['word_count_min']}-{stats['word_count_max']}")
        
        return text_stats
    
    def save_processed_datasets(self, output_dir: str = './data/OUTFOX-data/processed') -> None:
        """
        Save the processed OUTFOX datasets.
        
        Args:
            output_dir: Directory to save processed data.
        """
        logger.info("Saving processed OUTFOX datasets...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create datasets if not already created
        if self.human_text_dataset is None:
            self.create_human_text_dataset()
        if self.generation_prompts_dataset is None:
            self.create_generation_prompts_dataset()
        
        # Save human text dataset (Type 1)
        if self.human_text_dataset is not None and not self.human_text_dataset.empty:
            # CSV format
            human_csv_path = os.path.join(output_dir, 'outfox_human_text.csv')
            self.human_text_dataset.to_csv(human_csv_path, index=False)
            logger.info(f"Saved human text dataset to: {human_csv_path}")
            
            # JSON format  
            human_json_path = os.path.join(output_dir, 'outfox_human_text.json')
            self.human_text_dataset.to_json(human_json_path, orient='records', indent=2)
            logger.info(f"Saved human text dataset to: {human_json_path}")
        
        # Save generation prompts dataset (for Type 2)
        if self.generation_prompts_dataset is not None and not self.generation_prompts_dataset.empty:
            # CSV format
            prompts_csv_path = os.path.join(output_dir, 'outfox_generation_prompts.csv')
            self.generation_prompts_dataset.to_csv(prompts_csv_path, index=False)
            logger.info(f"Saved generation prompts dataset to: {prompts_csv_path}")
            
            # JSON format
            prompts_json_path = os.path.join(output_dir, 'outfox_generation_prompts.json')
            self.generation_prompts_dataset.to_json(prompts_json_path, orient='records', indent=2)
            logger.info(f"Saved generation prompts dataset to: {prompts_json_path}")
        
        # Save summary statistics
        stats_path = os.path.join(output_dir, 'outfox_processing_stats.txt')
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("OUTFOX Dataset Processing Statistics\n")
            f.write("=" * 45 + "\n\n")
            
            if self.human_text_dataset is not None:
                f.write(f"Human Text Dataset (Type 1):\n")
                f.write(f"  • Total samples: {len(self.human_text_dataset):,}\n")
                f.write(f"  • Columns: {list(self.human_text_dataset.columns)}\n")
                if 'human_text' in self.human_text_dataset.columns:
                    avg_len = self.human_text_dataset['human_text'].str.len().mean()
                    f.write(f"  • Average text length: {avg_len:.1f} chars\n")
                f.write("\n")
            
            if self.generation_prompts_dataset is not None:
                f.write(f"Generation Prompts Dataset (for Type 2):\n")
                f.write(f"  • Total samples: {len(self.generation_prompts_dataset):,}\n")
                f.write(f"  • Columns: {list(self.generation_prompts_dataset.columns)}\n")
                if 'combined_prompt' in self.generation_prompts_dataset.columns:
                    avg_len = self.generation_prompts_dataset['combined_prompt'].str.len().mean()
                    f.write(f"  • Average prompt length: {avg_len:.1f} chars\n")
                f.write("\n")
            
            f.write("Processing Notes:\n")
            f.write("• Human text dataset contains Type 1 text (human-original)\n")
            f.write("• Generation prompts combine contexts + problem_statements\n")
            f.write("• Use generation prompts to create Type 2 text with LLMs\n")
            f.write("• Type 3, 4, 5 need to be generated separately\n")
        
        logger.info(f"Saved processing statistics to: {stats_path}")
        
        # Print summary
        print(f"\n💾 OUTFOX processed datasets saved to: {output_dir}")
        
        if self.human_text_dataset is not None:
            print(f"   • Human text (Type 1):")
            print(f"     - CSV: outfox_human_text.csv ({len(self.human_text_dataset):,} samples)")
            print(f"     - JSON: outfox_human_text.json")
        
        if self.generation_prompts_dataset is not None:
            print(f"   • Generation prompts (for Type 2):")
            print(f"     - CSV: outfox_generation_prompts.csv ({len(self.generation_prompts_dataset):,} samples)")
            print(f"     - JSON: outfox_generation_prompts.json")
        
        print(f"   • Statistics: outfox_processing_stats.txt")
        
        print(f"\n📋 Dataset Structure:")
        if self.human_text_dataset is not None:
            print(f"   • Human text columns: {list(self.human_text_dataset.columns)}")
        if self.generation_prompts_dataset is not None:
            print(f"   • Prompts columns: {list(self.generation_prompts_dataset.columns)}")


def load_and_process_outfox_data(data_dir: str = "./data/OUTFOX-data",
                                output_dir: str = './data/OUTFOX-data/processed',
                                num_samples: int = 3) -> OUTFOXDataLoader:
    """
    Load OUTFOX dataset, perform analysis, and save processed datasets.
    
    Args:
        data_dir: Directory containing OUTFOX data splits.
        output_dir: Directory to save processed data.
        num_samples: Number of sample examples to display.
        
    Returns:
        Configured OUTFOXDataLoader instance with processed data.
    """
    # Initialize loader
    loader = OUTFOXDataLoader(data_dir=data_dir)
    
    try:
        # Load all splits
        raw_datasets = loader.load_individual_splits()
        
        # Perform comprehensive analysis
        print("\n" + "=" * 80)
        print("🔍 OUTFOX Dataset Analysis")
        print("=" * 80)
        
        structure_info = loader.analyze_data_structure()
        loader.display_sample_data(num_samples=num_samples)
        text_stats = loader.get_text_statistics()
        
        # Create processed datasets
        print("\n" + "=" * 80)
        print("🔄 Creating Processed Datasets")
        print("=" * 80)
        
        human_dataset = loader.create_human_text_dataset()
        prompts_dataset = loader.create_generation_prompts_dataset()
        
        # Save processed datasets
        loader.save_processed_datasets(output_dir=output_dir)
        
        print("\n" + "=" * 80)
        print("✅ OUTFOX Processing Complete!")
        print("=" * 80)
        print(f"📁 Raw data directory: {data_dir}")
        print(f"💾 Processed data saved to: {output_dir}")
        
        # Summary of processed datasets
        print(f"\n📊 Processed Datasets Summary:")
        if human_dataset is not None:
            print(f"   • Human text dataset: {len(human_dataset):,} samples")
            print(f"     - Contains Type 1 text (human-original)")  
        if prompts_dataset is not None:
            print(f"   • Generation prompts dataset: {len(prompts_dataset):,} samples")
            print(f"     - Contains contexts + problem_statements for Type 2 generation")
        
        print(f"\n📋 Next Steps:")
        print(f"   • Use human text dataset as Type 1 input")
        print(f"   • Use generation prompts with LLMs to create Type 2 text")
        print(f"   • Generate Type 3 (human paraphrased) separately")
        print(f"   • Generate Type 4 & 5 (LLM paraphrased) separately")
        
        return loader
        
    except Exception as e:
        logger.error(f"Failed to process OUTFOX dataset: {e}")
        raise


if __name__ == "__main__":
    """
    Example usage of the OUTFOX data processor.
    """
    # Process OUTFOX data
    loader = load_and_process_outfox_data(
        data_dir="./data/OUTFOX-data",
        output_dir="./data/OUTFOX-data/processed",
        num_samples=5
    )
    
    print(f"\n🎯 OUTFOX Processing Complete!")
    print(f"   • Raw datasets loaded: {len(loader.raw_datasets)}")
    if loader.human_text_dataset is not None:
        print(f"   • Human text dataset: {len(loader.human_text_dataset):,} samples")
    if loader.generation_prompts_dataset is not None:
        print(f"   • Generation prompts dataset: {len(loader.generation_prompts_dataset):,} samples")
    print(f"   • Ready for Type 2 generation and further processing!") 