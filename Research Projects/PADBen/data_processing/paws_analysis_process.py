"""
PAWS (Paraphrase Adversaries from Word Scrambling) Data Processing Module.

This module provides functionality to load, explore, and analyze the PAWS dataset,
which contains paraphrase pairs with human annotations from Wikipedia and Quora,
designed for paraphrase identification tasks.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset, concatenate_datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PAWSDataLoader:
    """
    A comprehensive loader and analyzer for the PAWS dataset.
    
    The PAWS (Paraphrase Adversaries from Word Scrambling) dataset contains pairs of 
    sentences with human annotations indicating whether each pair captures a paraphrase 
    relationship. The dataset is designed to be challenging for paraphrase identification
    models and contains examples from Wikipedia and Quora.
    
    This loader can combine all splits and filter for paraphrase pairs only (label=1).
    """
    
    def __init__(self, subset: str = "labeled_final", cache_dir: Optional[Path] = None) -> None:
        """
        Initialize the PAWS data loader.
        
        Args:
            subset: PAWS dataset subset to load ("labeled_final" or "labeled_swap").
            cache_dir: Optional directory to cache downloaded datasets.
                      If None, uses default cache directory.
        """
        self.subset = subset
        self.cache_dir = cache_dir
        self.dataset: Optional[Dict[str, Dataset]] = None
        self.df_train: Optional[pd.DataFrame] = None
        self.df_validation: Optional[pd.DataFrame] = None
        self.df_test: Optional[pd.DataFrame] = None
        self.combined_dataset: Optional[Dataset] = None
        self.paraphrase_df: Optional[pd.DataFrame] = None
        self.clean_dataset: Optional[pd.DataFrame] = None
        
    def load_dataset(self, use_auth_token: Optional[str] = None) -> Dict[str, Dataset]:
        """
        Load the PAWS dataset from Hugging Face datasets.
        
        Args:
            use_auth_token: Optional Hugging Face authentication token.
                           Required if dataset access is restricted.
                           
        Returns:
            Dictionary containing train, validation, and test splits.
            
        Raises:
            Exception: If dataset loading fails.
        """
        try:
            logger.info(f"Loading PAWS dataset (subset: {self.subset}) from Hugging Face...")
            
            # Load the dataset
            self.dataset = load_dataset(
                "paws", 
                self.subset,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
                use_auth_token=use_auth_token
            )
            
            logger.info("Successfully loaded PAWS dataset")
            return self.dataset
            
        except Exception as e:
            logger.error(f"Failed to load PAWS dataset: {e}")
            raise Exception(f"Could not load PAWS dataset: {e}") from e
    
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
    
    def remove_high_similarity_duplicates(self, similarity_threshold: float = 0.95) -> pd.DataFrame:
        """
        Remove samples with high cosine similarity to reduce duplicates.
        
        Args:
            similarity_threshold: Cosine similarity threshold for duplicate removal.
            
        Returns:
            DataFrame with high-similarity duplicates removed.
        """
        if self.paraphrase_df is None:
            raise ValueError("Paraphrase DataFrame not created. Call convert_paraphrases_to_dataframe() first.")
        
        logger.info(f"Removing high-similarity duplicates (threshold: {similarity_threshold})...")
        
        df = self.paraphrase_df.copy()
        original_length = len(df)
        
        # Remove rows with missing text
        df = df.dropna(subset=['sentence1', 'sentence2'])
        
        if len(df) < 2:
            logger.warning("Not enough samples for similarity comparison")
            return df
        
        try:
            # Method 1: Exact duplicates on sentence1
            exact_duplicates = df.duplicated(subset=['sentence1'], keep='first')
            df_no_exact = df[~exact_duplicates].copy()
            exact_removed = exact_duplicates.sum()
            
            logger.info(f"Removed {exact_removed:,} exact duplicates")
            
            # Method 2: Similarity-based duplicates on sentence1
            if similarity_threshold < 1.0 and len(df_no_exact) > 1:
                logger.info(f"Computing similarity-based duplicates (threshold: {similarity_threshold})...")
                
                # Compute TF-IDF vectors for sentence1
                vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(df_no_exact['sentence1'])
                
                # Compute cosine similarity matrix
                cosine_sim = cosine_similarity(tfidf_matrix)
                
                # Find indices to remove (upper triangle to avoid duplicates)
                indices_to_remove = set()
                for i in range(len(cosine_sim)):
                    for j in range(i + 1, len(cosine_sim)):
                        if cosine_sim[i][j] > similarity_threshold:
                            indices_to_remove.add(j)  # Remove the later occurrence
                
                # Remove high-similarity samples
                df_filtered = df_no_exact.drop(df_no_exact.index[list(indices_to_remove)])
                similarity_removed = len(indices_to_remove)
                
                logger.info(f"Removed {similarity_removed:,} similarity-based duplicates")
            else:
                df_filtered = df_no_exact
                similarity_removed = 0
            
            total_removed = original_length - len(df_filtered)
            logger.info(f"Total samples removed: {total_removed:,}")
            logger.info(f"Final dataset size: {len(df_filtered):,}")
            
            return df_filtered
            
        except Exception as e:
            logger.error(f"Error in similarity computation: {e}")
            logger.warning("Returning original DataFrame without duplicate removal")
            return df
    
    def create_clean_dataset(self) -> pd.DataFrame:
        """
        Create a clean dataset with only Type 1 and Type 3 text columns.
        
        This creates a clean version with only:
        - sentence1 (Type 1: Human original text)
        - sentence2 (Type 3: Human paraphrased text)
        - Metadata for tracking
        
        Returns:
            Clean DataFrame ready for modern LLM generation.
        """
        logger.info("Creating clean dataset with only Type 1 and Type 3 text...")
        
        if self.paraphrase_df is None:
            raise ValueError("Paraphrase DataFrame not created. Call convert_paraphrases_to_dataframe() first.")
        
        # Select only essential columns
        essential_columns = ['sentence1', 'sentence2', 'id']
        
        # Filter to only include rows with both sentences available
        clean_df = self.paraphrase_df[essential_columns].copy()
        clean_df = clean_df.dropna(subset=['sentence1', 'sentence2'])
        
        # Rename columns to standard format
        clean_df = clean_df.rename(columns={
            'sentence1': 'human_original_text',      # Type 1
            'sentence2': 'human_paraphrased_text',   # Type 3
            'id': 'original_id'
        })
        
        # Add sample ID for tracking
        clean_df['sample_id'] = range(1, len(clean_df) + 1)
        
        # Reorder columns for better readability
        column_order = [
            'sample_id',
            'original_id',
            'human_original_text',     # Type 1
            'human_paraphrased_text'   # Type 3
        ]
        
        clean_df = clean_df[column_order]
        
        self.clean_dataset = clean_df
        
        logger.info(f"Clean dataset created: {len(clean_df):,} samples")
        logger.info(f"Columns: {list(clean_df.columns)}")
        
        return clean_df
    
    def explore_data_structure(self) -> Dict[str, Dict[str, Union[int, List[str], str]]]:
        """
        Analyze and display the structure of the PAWS dataset.
        
        Returns:
            Dictionary containing comprehensive dataset statistics and structure info.
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        structure_info = {}
        
        print("=" * 80)
        print("PAWS Dataset Structure Analysis")
        print("=" * 80)
        
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
        
        print("\n" + "=" * 80)
        print("PAWS Sample Data Examples")
        print("=" * 80)
        
        for split_name, split_data in self.dataset.items():
            print(f"\n🔍 {split_name.upper()} Split Samples:")
            
            df_sample = pd.DataFrame(split_data[:num_samples])
            
            for idx, row in df_sample.iterrows():
                print(f"\nSample {idx + 1}:")
                print(f"  Sentence 1: {row['sentence1']}")
                print(f"  Sentence 2: {row['sentence2']}")
                print(f"  Label: {row['label']} ({'Paraphrase' if row['label'] == 1 else 'Not Paraphrase'})")
                if 'id' in row:
                    print(f"  ID: {row['id']}")
    
    def analyze_label_distribution(self) -> Dict[str, Dict[int, int]]:
        """
        Analyze the distribution of labels across all splits.
        
        Returns:
            Dictionary containing label distribution for each split.
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        print("\n" + "=" * 80)
        print("PAWS Label Distribution Analysis")
        print("=" * 80)
        
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
        
        print("\n" + "=" * 80)
        print("PAWS Text Statistics Analysis")
        print("=" * 80)
        
        text_stats = {}
        
        for split_name, split_data in self.dataset.items():
            df = pd.DataFrame(split_data)
            split_stats = {}
            
            print(f"\n📝 {split_name.upper()} Split Text Statistics:")
            
            for sentence_col in ['sentence1', 'sentence2']:
                if sentence_col in df.columns:
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
    
    def analyze_paraphrase_quality(self) -> Dict[str, Any]:
        """
        Analyze the quality and characteristics of paraphrase pairs.
        
        Returns:
            Dictionary containing paraphrase quality analysis.
        """
        if self.paraphrase_df is None:
            raise ValueError("Paraphrase DataFrame not created. Call convert_paraphrases_to_dataframe() first.")
        
        print("\n" + "=" * 80)
        print("PAWS Paraphrase Quality Analysis")
        print("=" * 80)
        
        df = self.paraphrase_df
        quality_stats = {}
        
        # Calculate similarity metrics
        print(f"📊 Paraphrase Pair Analysis:")
        
        # Length differences
        len_diff = abs(df['sentence1'].str.len() - df['sentence2'].str.len())
        word_diff = abs(df['sentence1'].str.split().str.len() - df['sentence2'].str.split().str.len())
        
        quality_stats['length_differences'] = {
            'char_diff_mean': len_diff.mean(),
            'char_diff_std': len_diff.std(),
            'char_diff_max': len_diff.max(),
            'word_diff_mean': word_diff.mean(),
            'word_diff_std': word_diff.std(),
            'word_diff_max': word_diff.max()
        }
        
        print(f"   • Character length differences:")
        print(f"     - Mean: {quality_stats['length_differences']['char_diff_mean']:.1f}")
        print(f"     - Max: {quality_stats['length_differences']['char_diff_max']}")
        
        print(f"   • Word count differences:")
        print(f"     - Mean: {quality_stats['length_differences']['word_diff_mean']:.1f}")
        print(f"     - Max: {quality_stats['length_differences']['word_diff_max']}")
        
        # Check for identical pairs
        identical_pairs = (df['sentence1'] == df['sentence2']).sum()
        quality_stats['identical_pairs'] = int(identical_pairs)
        identical_pct = (identical_pairs / len(df)) * 100
        
        print(f"   • Identical pairs: {identical_pairs:,} ({identical_pct:.2f}%)")
        
        return quality_stats
    
    def save_processed_data(self, output_dir: str = './data/PAWS-data/processed') -> None:
        """
        Save the processed PAWS data to files.
        
        Args:
            output_dir: Directory to save the processed data.
        """
        if self.paraphrase_df is None:
            raise ValueError("Paraphrase DataFrame not created. Call convert_paraphrases_to_dataframe() first.")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save full processed dataset
        csv_path = os.path.join(output_dir, 'paws_processed.csv')
        self.paraphrase_df.to_csv(csv_path, index=False)
        logger.info(f"Saved PAWS data to: {csv_path}")
        
        # Save as JSON for better text preservation
        json_path = os.path.join(output_dir, 'paws_processed.json')
        self.paraphrase_df.to_json(json_path, orient='records', indent=2)
        logger.info(f"Saved PAWS data to: {json_path}")
        
        # Create and save clean dataset (Type 1 and Type 3 only)
        clean_df = self.create_clean_dataset()
        clean_csv_path = os.path.join(output_dir, 'paws_processed_clean.csv')
        clean_df.to_csv(clean_csv_path, index=False)
        logger.info(f"Saved clean PAWS data to: {clean_csv_path}")
        
        # Save clean dataset as JSON as well
        clean_json_path = os.path.join(output_dir, 'paws_processed_clean.json')
        clean_df.to_json(clean_json_path, orient='records', indent=2)
        logger.info(f"Saved clean PAWS data to: {clean_json_path}")
        
        # Save summary statistics
        stats_path = os.path.join(output_dir, 'paws_stats.txt')
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("PAWS Dataset Statistics\n")
            f.write("=" * 40 + "\n")
            f.write(f"Dataset subset: {self.subset}\n")
            f.write(f"Total paraphrase pairs: {len(self.paraphrase_df):,}\n")
            f.write(f"Clean dataset samples: {len(clean_df):,}\n")
            
            # Text statistics
            avg_len1 = self.paraphrase_df['sentence1'].str.len().mean()
            avg_len2 = self.paraphrase_df['sentence2'].str.len().mean()
            f.write(f"Average sentence1 length: {avg_len1:.1f} chars\n")
            f.write(f"Average sentence2 length: {avg_len2:.1f} chars\n")
            
            f.write("\nClean Dataset Info:\n")
            f.write("- Contains only Type 1 (sentence1) and Type 3 (sentence2) text\n")
            f.write("- Ready for modern LLM generation of Types 2, 4, and 5\n")
            f.write("- Source: Wikipedia and Quora paraphrase pairs\n")
        
        logger.info(f"Saved PAWS statistics to: {stats_path}")
        
        print(f"\n💾 PAWS processed data saved to: {output_dir}")
        print(f"   • Full dataset CSV: paws_processed.csv ({len(self.paraphrase_df):,} samples)")
        print(f"   • Full dataset JSON: paws_processed.json")
        print(f"   • Clean dataset CSV: paws_processed_clean.csv ({len(clean_df):,} samples)")
        print(f"   • Clean dataset JSON: paws_processed_clean.json")
        print(f"   • Statistics: paws_stats.txt")
        
        print(f"\n🧹 Clean Dataset Info:")
        print(f"   • Contains only Type 1 and Type 3 text columns")
        print(f"   • Perfect for modern LLM generation")
        print(f"   • Columns: {list(clean_df.columns)}")


def load_and_process_paws_data(subset: str = "labeled_final",
                              cache_dir: Optional[str] = None,
                              output_dir: str = './data/PAWS-data/processed',
                              similarity_threshold: float = 0.95,
                              num_samples: int = 5) -> PAWSDataLoader:
    """
    Load PAWS dataset, filter for paraphrases, remove duplicates, and analyze.
    
    Args:
        subset: PAWS dataset subset to load.
        cache_dir: Optional cache directory for dataset storage.
        output_dir: Directory to save processed data.
        similarity_threshold: Similarity threshold for duplicate removal.
        num_samples: Number of sample examples to display.
        
    Returns:
        Configured PAWSDataLoader instance with processed paraphrase data.
    """
    # Initialize loader
    cache_path = Path(cache_dir) if cache_dir else None
    loader = PAWSDataLoader(subset=subset, cache_dir=cache_path)
    
    try:
        # Load dataset
        dataset = loader.load_dataset()
        
        # Perform original analysis (optional, for exploration)
        structure_info = loader.explore_data_structure()
        loader.display_sample_data(num_samples=num_samples)
        distribution = loader.analyze_label_distribution()
        text_stats = loader.get_text_statistics()
        
        # Process for paraphrases only
        print("\n" + "=" * 80)
        print("Processing for Paraphrase Pairs Only")
        print("=" * 80)
        
        combined_dataset = loader.combine_and_filter_paraphrases()
        paraphrase_df = loader.convert_paraphrases_to_dataframe()
        
        # Analyze paraphrase quality
        quality_analysis = loader.analyze_paraphrase_quality()
        
        # Remove duplicates
        print("\n" + "=" * 80)
        print("🧹 Removing Duplicates")
        print("=" * 80)
        
        deduplicated_df = loader.remove_high_similarity_duplicates(
            similarity_threshold=similarity_threshold
        )
        
        # Update paraphrase_df with deduplicated version
        loader.paraphrase_df = deduplicated_df
        
        # Save processed data
        loader.save_processed_data(output_dir=output_dir)
        
        print("\n" + "=" * 80)
        print("✅ PAWS Processing Complete!")
        print("=" * 80)
        print(f"📁 Raw dataset cached at: {loader.cache_dir or 'default cache location'}")
        print(f"💾 Processed data saved at: {output_dir}")
        print(f"🎯 Total paraphrase pairs: {len(paraphrase_df):,}")
        print(f"📊 After deduplication: {len(deduplicated_df):,}")
        print(f"🧹 Clean dataset: {len(loader.clean_dataset):,} samples")
        
        # Print quality summary
        print(f"\n📋 Quality Summary:")
        print(f"   • Dataset subset: {subset}")
        print(f"   • Available text types: 1, 3 (human original & paraphrased)")
        print(f"   • Missing text types: 2, 4, 5 (need LLM generation)")
        print(f"   • Clean dataset ready for modern LLM generation")
        print(f"   • Source: Wikipedia and Quora paraphrase pairs")
        
        return loader
        
    except Exception as e:
        logger.error(f"Failed to process PAWS dataset: {e}")
        raise


if __name__ == "__main__":
    """
    Example usage of the PAWS data processor.
    """
    # Create output directory
    output_dir = './data/PAWS-data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process PAWS data
    loader = load_and_process_paws_data(
        subset="labeled_final",
        cache_dir=None,
        output_dir=output_dir,
        similarity_threshold=0.95,
        num_samples=5
    )
    
    print(f"\n🎯 PAWS Analysis Complete!")
    print(f"   • Total paraphrase pairs: {len(loader.paraphrase_df):,}")
    print(f"   • Clean dataset: {len(loader.clean_dataset):,} samples")
    print(f"   • Ready for modern LLM generation!") 