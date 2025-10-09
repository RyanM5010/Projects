"""
MRPC (Microsoft Research Paraphrase Corpus) Data Processing Module.

This module provides functionality to load, explore, and analyze the MRPC dataset,
which contains pairs of sentences labeled for paraphrase detection.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from datasets import Dataset, load_dataset, concatenate_datasets
from sklearn.model_selection import train_test_split
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MRPCDataLoader:
    """
    A comprehensive loader and analyzer for the MRPC dataset.
    
    The Microsoft Research Paraphrase Corpus (MRPC) contains pairs of sentences
    extracted from online news sources, with human annotations indicating whether
    each pair captures a paraphrase/semantic equivalence relationship.
    
    This loader can combine all splits and filter for paraphrase pairs only (label=1).
    """
    
    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        """
        Initialize the MRPC data loader.
        
        Args:
            cache_dir: Optional directory to cache downloaded datasets.
                      If None, uses default cache directory.
        """
        self.cache_dir = cache_dir
        self.dataset: Optional[Dict[str, Dataset]] = None
        self.df_train: Optional[pd.DataFrame] = None
        self.df_validation: Optional[pd.DataFrame] = None
        self.df_test: Optional[pd.DataFrame] = None
        self.combined_dataset: Optional[Dataset] = None
        self.paraphrase_df: Optional[pd.DataFrame] = None
        
    def load_dataset(self, use_auth_token: Optional[str] = None) -> Dict[str, Dataset]:
        """
        Load the MRPC dataset from Hugging Face datasets.
        
        Args:
            use_auth_token: Optional Hugging Face authentication token.
                           Required if dataset access is restricted.
                           
        Returns:
            Dictionary containing train, validation, and test splits.
            
        Raises:
            Exception: If dataset loading fails.
        """
        try:
            logger.info("Loading MRPC dataset from Hugging Face...")
            
            # Load the dataset
            self.dataset = load_dataset(
                "glue", 
                "mrpc",
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
                use_auth_token=use_auth_token
            )
            
            logger.info("Successfully loaded MRPC dataset")
            return self.dataset
            
        except Exception as e:
            logger.error(f"Failed to load MRPC dataset: {e}")
            raise Exception(f"Could not load MRPC dataset: {e}") from e
    
    def convert_to_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Convert the loaded dataset to pandas DataFrames for easier analysis.
        
        Returns:
            Tuple of (train_df, validation_df, test_df) DataFrames.
            
        Raises:
            ValueError: If dataset hasn't been loaded yet.
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        logger.info("Converting datasets to pandas DataFrames...")
        
        # Convert to DataFrames
        self.df_train = pd.DataFrame(self.dataset['train'])
        self.df_validation = pd.DataFrame(self.dataset['validation'])
        self.df_test = pd.DataFrame(self.dataset['test'])
        
        logger.info("Successfully converted to DataFrames")
        return self.df_train, self.df_validation, self.df_test
    
    def combine_and_filter_paraphrases(self) -> Dataset:
        """
        Combine all splits and filter for paraphrase pairs only (label=1).
        
        Returns:
            Combined dataset containing only paraphrase pairs.
            
        Raises:
            ValueError: If dataset hasn't been loaded yet.
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        logger.info("Combining all splits and filtering for paraphrase pairs...")
        
        # Concatenate all splits
        all_splits = [
            self.dataset['train'],
            self.dataset['validation'], 
            self.dataset['test']
        ]
        combined = concatenate_datasets(all_splits)
        
        # Filter for paraphrase pairs only (label=1)
        self.combined_dataset = combined.filter(lambda example: example['label'] == 1)
        
        logger.info(f"Filtered dataset: {len(self.combined_dataset):,} paraphrase pairs")
        return self.combined_dataset
    
    def convert_paraphrases_to_dataframe(self) -> pd.DataFrame:
        """
        Convert the combined paraphrase dataset to a pandas DataFrame.
        
        Returns:
            DataFrame containing only paraphrase pairs.
            
        Raises:
            ValueError: If combined dataset hasn't been created yet.
        """
        if self.combined_dataset is None:
            raise ValueError("Combined dataset not created. Call combine_and_filter_paraphrases() first.")
        
        logger.info("Converting combined paraphrase dataset to pandas DataFrame...")
        
        # Convert to DataFrame
        self.paraphrase_df = pd.DataFrame(self.combined_dataset)
        
        logger.info(f"Successfully converted to DataFrame with {len(self.paraphrase_df):,} paraphrase pairs")
        return self.paraphrase_df
    
    def save_processed_data(self, output_dir: str = './data/mrpc') -> None:
        """
        Save the processed paraphrase data to files.
        
        Args:
            output_dir: Directory to save the processed data.
        """
        if self.paraphrase_df is None:
            raise ValueError("Paraphrase DataFrame not created. Call convert_paraphrases_to_dataframe() first.")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as CSV
        csv_path = os.path.join(output_dir, 'mrpc_paraphrases.csv')
        self.paraphrase_df.to_csv(csv_path, index=False)
        logger.info(f"Saved paraphrase pairs to: {csv_path}")
        
        # Save as JSON for better text preservation
        json_path = os.path.join(output_dir, 'mrpc_paraphrases.json')
        self.paraphrase_df.to_json(json_path, orient='records', indent=2)
        logger.info(f"Saved paraphrase pairs to: {json_path}")
        
        # Save just the sentence pairs as a simple text file
        pairs_path = os.path.join(output_dir, 'sentence_pairs.txt')
        with open(pairs_path, 'w', encoding='utf-8') as f:
            for _, row in self.paraphrase_df.iterrows():
                f.write(f"{row['sentence1']}\t{row['sentence2']}\n")
        logger.info(f"Saved sentence pairs to: {pairs_path}")
        
        # Save summary statistics
        stats_path = os.path.join(output_dir, 'dataset_stats.txt')
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("MRPC Paraphrase Dataset Statistics\n")
            f.write("=" * 40 + "\n")
            f.write(f"Total paraphrase pairs: {len(self.paraphrase_df):,}\n")
            f.write(f"Average sentence1 length: {self.paraphrase_df['sentence1'].str.len().mean():.1f} chars\n")
            f.write(f"Average sentence2 length: {self.paraphrase_df['sentence2'].str.len().mean():.1f} chars\n")
            f.write(f"Average sentence1 words: {self.paraphrase_df['sentence1'].str.split().str.len().mean():.1f}\n")
            f.write(f"Average sentence2 words: {self.paraphrase_df['sentence2'].str.split().str.len().mean():.1f}\n")
        logger.info(f"Saved dataset statistics to: {stats_path}")
        
        print(f"\n💾 Processed data saved to: {output_dir}")
        print(f"   • CSV format: mrpc_paraphrases.csv")
        print(f"   • JSON format: mrpc_paraphrases.json") 
        print(f"   • Text pairs: sentence_pairs.txt")
        print(f"   • Statistics: dataset_stats.txt")
    
    def get_sentence_pairs(self) -> List[Tuple[str, str]]:
        """
        Extract just the sentence pairs as a list of tuples.
        
        Returns:
            List of (sentence1, sentence2) tuples for all paraphrase pairs.
        """
        if self.paraphrase_df is None:
            raise ValueError("Paraphrase DataFrame not created. Call convert_paraphrases_to_dataframe() first.")
        
        pairs = list(zip(self.paraphrase_df['sentence1'], self.paraphrase_df['sentence2']))
        logger.info(f"Extracted {len(pairs):,} sentence pairs")
        return pairs

    def explore_data_structure(self) -> Dict[str, Dict[str, Union[int, List[str], str]]]:
        """
        Analyze and display the structure of the MRPC dataset.
        
        Returns:
            Dictionary containing comprehensive dataset statistics and structure info.
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        structure_info = {}
        
        print("=" * 60)
        print("MRPC Dataset Structure Analysis")
        print("=" * 60)
        
        for split_name, split_data in self.dataset.items():
            print(f"\n📊 {split_name.upper()} Split:")
            print(f"   • Sample count: {len(split_data):,}")
            print(f"   • Features: {list(split_data.features.keys())}")
            
            # Convert to DataFrame for detailed analysis
            df = pd.DataFrame(split_data)
            
            # Feature analysis
            feature_info = {}
            for feature in df.columns:
                if df[feature].dtype == 'object':
                    feature_info[feature] = {
                        'type': 'text',
                        'unique_count': df[feature].nunique(),
                        'sample_length': df[feature].str.len().describe().to_dict() if df[feature].dtype == 'object' else None
                    }
                else:
                    feature_info[feature] = {
                        'type': 'numeric',
                        'unique_count': df[feature].nunique(),
                        'value_counts': df[feature].value_counts().to_dict()
                    }
            
            structure_info[split_name] = {
                'sample_count': len(split_data),
                'features': list(split_data.features.keys()),
                'feature_details': feature_info
            }
            
            # Display feature details
            for feature, details in feature_info.items():
                print(f"   • {feature}:")
                print(f"     - Type: {details['type']}")
                print(f"     - Unique values: {details['unique_count']}")
                
                if feature == 'label' and 'value_counts' in details:
                    print(f"     - Label distribution: {details['value_counts']}")
                elif feature in ['sentence1', 'sentence2'] and 'sample_length' in details:
                    lengths = details['sample_length']
                    print(f"     - Text length stats: mean={lengths['mean']:.1f}, "
                          f"min={lengths['min']:.0f}, max={lengths['max']:.0f}")
        
        return structure_info
    
    def display_sample_data(self, num_samples: int = 5) -> None:
        """
        Display sample data from each split to understand the data format.
        
        Args:
            num_samples: Number of samples to display from each split.
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        print("\n" + "=" * 60)
        print("Sample Data Examples")
        print("=" * 60)
        
        for split_name, split_data in self.dataset.items():
            print(f"\n🔍 {split_name.upper()} Split Samples:")
            
            df_sample = pd.DataFrame(split_data[:num_samples])
            
            for idx, row in df_sample.iterrows():
                print(f"\nSample {idx + 1}:")
                print(f"  Sentence 1: {row['sentence1']}")
                print(f"  Sentence 2: {row['sentence2']}")
                print(f"  Label: {row['label']} ({'Paraphrase' if row['label'] == 1 else 'Not Paraphrase'})")
                print(f"  Index: {row['idx']}")
    
    def analyze_label_distribution(self) -> Dict[str, Dict[int, int]]:
        """
        Analyze the distribution of labels across all splits.
        
        Returns:
            Dictionary containing label distribution for each split.
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        print("\n" + "=" * 60)
        print("Label Distribution Analysis")
        print("=" * 60)
        
        distribution = {}
        
        for split_name, split_data in self.dataset.items():
            df = pd.DataFrame(split_data)
            label_counts = df['label'].value_counts().sort_index()
            distribution[split_name] = label_counts.to_dict()
            
            total = len(df)
            print(f"\n📈 {split_name.upper()} Split:")
            print(f"   Total samples: {total:,}")
            
            for label, count in label_counts.items():
                label_name = "Paraphrase" if label == 1 else "Not Paraphrase"
                percentage = (count / total) * 100
                print(f"   • {label_name} (Label {label}): {count:,} ({percentage:.1f}%)")
        
        return distribution
    
    def get_text_statistics(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Calculate comprehensive text statistics for sentences.
        
        Returns:
            Dictionary containing text statistics for each split and sentence column.
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        print("\n" + "=" * 60)
        print("Text Statistics Analysis")
        print("=" * 60)
        
        text_stats = {}
        
        for split_name, split_data in self.dataset.items():
            df = pd.DataFrame(split_data)
            split_stats = {}
            
            print(f"\n📝 {split_name.upper()} Split Text Statistics:")
            
            for sentence_col in ['sentence1', 'sentence2']:
                # Calculate statistics
                lengths = df[sentence_col].str.len()
                word_counts = df[sentence_col].str.split().str.len()
                
                stats = {
                    'char_length_mean': lengths.mean(),
                    'char_length_std': lengths.std(),
                    'char_length_min': lengths.min(),
                    'char_length_max': lengths.max(),
                    'word_count_mean': word_counts.mean(),
                    'word_count_std': word_counts.std(),
                    'word_count_min': word_counts.min(),
                    'word_count_max': word_counts.max()
                }
                
                split_stats[sentence_col] = stats
                
                print(f"   {sentence_col}:")
                print(f"     • Character length: {stats['char_length_mean']:.1f} ± {stats['char_length_std']:.1f} "
                      f"(range: {stats['char_length_min']}-{stats['char_length_max']})")
                print(f"     • Word count: {stats['word_count_mean']:.1f} ± {stats['word_count_std']:.1f} "
                      f"(range: {stats['word_count_min']:.0f}-{stats['word_count_max']:.0f})")
            
            text_stats[split_name] = split_stats
        
        return text_stats


def load_and_process_mrpc_paraphrases(cache_dir: Optional[str] = None,
                                     output_dir: str = './data/mrpc',
                                     num_samples: int = 5) -> MRPCDataLoader:
    """
    Load MRPC dataset, filter for paraphrases, and save processed data.
    
    Args:
        cache_dir: Optional cache directory for dataset storage.
        output_dir: Directory to save processed data.
        num_samples: Number of sample examples to display.
        
    Returns:
        Configured MRPCDataLoader instance with processed paraphrase data.
    """
    # Initialize loader
    cache_path = Path(cache_dir) if cache_dir else None
    loader = MRPCDataLoader(cache_dir=cache_path)
    
    try:
        # Load dataset
        dataset = loader.load_dataset()
        
        # Perform original analysis (optional, for exploration)
        structure_info = loader.explore_data_structure()
        loader.display_sample_data(num_samples=num_samples)
        distribution = loader.analyze_label_distribution()
        text_stats = loader.get_text_statistics()
        
        # Process for paraphrases only
        print("\n" + "=" * 60)
        print("Processing for Paraphrase Pairs Only")
        print("=" * 60)
        
        combined_dataset = loader.combine_and_filter_paraphrases()
        paraphrase_df = loader.convert_paraphrases_to_dataframe()
        
        # Save processed data
        loader.save_processed_data(output_dir=output_dir)
        
        print("\n" + "=" * 60)
        print("✅ MRPC Paraphrase Processing Complete!")
        print("=" * 60)
        print(f"📁 Raw dataset cached at: {loader.cache_dir or 'default cache location'}")
        print(f"💾 Processed data saved at: {output_dir}")
        print(f"🎯 Total paraphrase pairs: {len(paraphrase_df):,}")
        
        return loader
        
    except Exception as e:
        logger.error(f"Failed to process MRPC paraphrase dataset: {e}")
        raise


if __name__ == "__main__":
    """
    Example usage of the MRPC paraphrase data processor.
    """
    # Create the cache_dir if it doesn't exist
    cache_dir = './data/mrpc'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # Load, process, and save paraphrase data
    loader = load_and_process_mrpc_paraphrases(
        cache_dir=cache_dir, 
        output_dir=cache_dir,
        num_samples=3
    )
    
    # Get processed data
    paraphrase_df = loader.paraphrase_df
    sentence_pairs = loader.get_sentence_pairs()
    
    print(f"\n🎯 Ready for paraphrase analysis!")
    print(f"   • Total paraphrase pairs: {len(paraphrase_df):,}")
    print(f"   • Data saved in: {cache_dir}")
    print(f"   • Sample pair: {sentence_pairs[0]}")