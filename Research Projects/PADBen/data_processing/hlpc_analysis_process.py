"""
HLPC (Human & LLM Paraphrase Collection) Data Processing Module.

This module provides functionality to load, explore, and analyze the HLPC dataset,
which contains human original text, human paraphrases, and LLM-generated/paraphrased text
from multiple sources (MRPC, XSum, QQP, Multi-PIT) with different models (BART, DIPPER).
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import glob

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HLPCDataLoader:
    """
    A comprehensive loader and analyzer for the HLPC dataset.
    
    The Human & LLM Paraphrase Collection (HLPC) contains multiple datasets
    (MRPC, XSum, QQP, Multi-PIT) with human original text, human paraphrases,
    and LLM-generated/paraphrased text using different models.
    """
    
    def __init__(self, data_dir: str = "./data/HLPC-data") -> None:
        """
        Initialize the HLPC data loader.
        
        Args:
            data_dir: Directory containing HLPC CSV files.
        """
        self.data_dir = Path(data_dir)
        self.raw_datasets: Dict[str, pd.DataFrame] = {}
        self.combined_dataset: Optional[pd.DataFrame] = None
        self.deduplicated_dataset: Optional[pd.DataFrame] = None
        self.clean_dataset: Optional[pd.DataFrame] = None
        
        # Define expected columns
        self.expected_columns = [
            'originalSentence1',  # Type 1: Human original
            'originalSentence2',  # Type 3: Human paraphrased
            'gpt_text',          # Type 2: GPT-generated (outdated)
            'gpt_p_text_1',      # Type 4: GPT-paraphrased original (outdated)
            'gpt_p_text_2',      
            'gpt_p_text_3',      
            'gpt_p_text_4',      
            'gpt_p_text_5',      # Type 5: GPT-paraphrased generated (outdated)
            'opt_text',          # Type 2: OPT-generated (outdated)
            'opt_p_text_1',      # Type 4: OPT-paraphrased original (outdated)
            'opt_p_text_2',      
            'opt_p_text_3',      
            'opt_p_text_4',      
            'opt_p_text_5'       # Type 5: OPT-paraphrased generated (outdated)
        ]
    
    def discover_csv_files(self) -> Dict[str, str]:
        """
        Discover all CSV files in the HLPC data directory.
        
        Returns:
            Dictionary mapping file identifiers to file paths.
        """
        logger.info(f"Discovering CSV files in {self.data_dir}...")
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"HLPC data directory not found: {self.data_dir}")
        
        # Find all CSV files
        csv_files = list(self.data_dir.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
        
        # Create mapping of file identifiers to paths
        file_mapping = {}
        for csv_file in csv_files:
            file_id = csv_file.stem  # Filename without extension
            file_mapping[file_id] = str(csv_file)
        
        logger.info(f"Found {len(csv_files)} CSV files:")
        for file_id, path in file_mapping.items():
            logger.info(f"  • {file_id}")
        
        return file_mapping
    
    def load_individual_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all individual CSV files into separate DataFrames.
        
        Returns:
            Dictionary of DataFrames for each CSV file.
        """
        logger.info("Loading individual HLPC datasets...")
        
        csv_files = self.discover_csv_files()
        datasets = {}
        
        for file_id, file_path in csv_files.items():
            try:
                logger.info(f"Loading {file_id}...")
                df = pd.read_csv(file_path)
                
                # Validate columns
                missing_cols = set(self.expected_columns) - set(df.columns)
                if missing_cols:
                    logger.warning(f"Missing columns in {file_id}: {missing_cols}")
                
                # Add metadata
                df['source_file'] = file_id
                df['dataset_source'] = self._extract_dataset_name(file_id)
                df['model_source'] = self._extract_model_name(file_id)
                
                datasets[file_id] = df
                logger.info(f"  • Loaded {len(df):,} samples from {file_id}")
                
            except Exception as e:
                logger.error(f"Failed to load {file_id}: {e}")
                continue
        
        self.raw_datasets = datasets
        logger.info(f"Successfully loaded {len(datasets)} datasets")
        return datasets
    
    def _extract_dataset_name(self, file_id: str) -> str:
        """Extract dataset name from file identifier."""
        if 'MRPC' in file_id:
            return 'MRPC'
        elif 'XSUM' in file_id:
            return 'XSUM'
        elif 'QQP' in file_id:
            return 'QQP'
        elif 'PIT' in file_id:
            return 'PIT'
        else:
            return 'UNKNOWN'
    
    def _extract_model_name(self, file_id: str) -> str:
        """Extract model name from file identifier."""
        if 'BART' in file_id:
            return 'BART'
        elif 'DIPPER' in file_id:
            return 'DIPPER'
        else:
            return 'UNKNOWN'
    
    def combine_all_datasets(self) -> pd.DataFrame:
        """
        Combine all individual datasets into a single DataFrame.
        
        Returns:
            Combined DataFrame with all HLPC data.
        """
        logger.info("Combining all HLPC datasets...")
        
        if not self.raw_datasets:
            self.load_individual_datasets()
        
        # Combine all datasets
        all_dfs = list(self.raw_datasets.values())
        self.combined_dataset = pd.concat(all_dfs, ignore_index=True)
        
        logger.info(f"Combined dataset: {len(self.combined_dataset):,} total samples")
        return self.combined_dataset
    
    def remove_duplicates_by_first_sentence(self, similarity_threshold: float = 0.95) -> pd.DataFrame:
        """
        Remove duplicates based on the first sentence (originalSentence1).
        
        Args:
            similarity_threshold: Cosine similarity threshold for duplicate detection.
            
        Returns:
            Deduplicated DataFrame.
        """
        logger.info("Removing duplicates based on first sentence...")
        
        if self.combined_dataset is None:
            self.combine_all_datasets()
        
        df = self.combined_dataset.copy()
        original_count = len(df)
        
        # Remove rows with missing first sentences
        df = df.dropna(subset=['originalSentence1'])
        
        if len(df) < 2:
            logger.warning("Not enough samples for duplicate detection")
            self.deduplicated_dataset = df
            return df
        
        try:
            # Method 1: Exact duplicates
            exact_duplicates = df.duplicated(subset=['originalSentence1'], keep='first')
            df_no_exact = df[~exact_duplicates].copy()
            exact_removed = exact_duplicates.sum()
            
            logger.info(f"Removed {exact_removed:,} exact duplicates")
            
            # Method 2: Similarity-based duplicates
            if similarity_threshold < 1.0 and len(df_no_exact) > 1:
                logger.info(f"Computing similarity-based duplicates (threshold: {similarity_threshold})...")
                
                # Compute TF-IDF vectors
                vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(df_no_exact['originalSentence1'])
                
                # Compute cosine similarity matrix
                cosine_sim = cosine_similarity(tfidf_matrix)
                
                # Find indices to remove (upper triangle to avoid duplicates)
                indices_to_remove = set()
                for i in range(len(cosine_sim)):
                    for j in range(i + 1, len(cosine_sim)):
                        if cosine_sim[i][j] > similarity_threshold:
                            indices_to_remove.add(j)  # Remove the later occurrence
                
                # Remove high-similarity samples
                df_final = df_no_exact.drop(df_no_exact.index[list(indices_to_remove)])
                similarity_removed = len(indices_to_remove)
                
                logger.info(f"Removed {similarity_removed:,} similarity-based duplicates")
            else:
                df_final = df_no_exact
                similarity_removed = 0
            
            total_removed = original_count - len(df_final)
            logger.info(f"Total samples removed: {total_removed:,}")
            logger.info(f"Final dataset size: {len(df_final):,}")
            
            self.deduplicated_dataset = df_final
            return df_final
            
        except Exception as e:
            logger.error(f"Error in duplicate removal: {e}")
            logger.warning("Returning dataset with only exact duplicates removed")
            self.deduplicated_dataset = df_no_exact
            return df_no_exact
    
    def create_clean_dataset(self) -> pd.DataFrame:
        """
        Create a clean dataset with only Type 1 and Type 3 text columns.
        
        This removes all outdated generated/paraphrased columns, keeping only:
        - originalSentence1 (Type 1: Human original text)
        - originalSentence2 (Type 3: Human paraphrased text)
        - Metadata columns for tracking
        
        Returns:
            Clean DataFrame ready for modern LLM regeneration.
        """
        logger.info("Creating clean dataset with only Type 1 and Type 3 text...")
        
        # Use deduplicated dataset if available, otherwise combined dataset
        source_dataset = self.deduplicated_dataset if self.deduplicated_dataset is not None else self.combined_dataset
        
        if source_dataset is None:
            raise ValueError("No dataset available. Load and process data first.")
        
        # Select only essential columns
        essential_columns = [
            'originalSentence1',  # Type 1: Human original text
            'originalSentence2',  # Type 3: Human paraphrased text
            'source_file',        # Metadata: source file
            'dataset_source',     # Metadata: dataset source (MRPC, XSUM, etc.)
            'model_source'        # Metadata: model source (BART, DIPPER)
        ]
        
        # Filter to only include rows with both sentences available
        clean_df = source_dataset[essential_columns].copy()
        clean_df = clean_df.dropna(subset=['originalSentence1', 'originalSentence2'])
        
        # Add unique ID for tracking
        clean_df['sample_id'] = range(1, len(clean_df) + 1)
        
        # Reorder columns for better readability
        column_order = [
            'sample_id',
            'originalSentence1',  # Type 1
            'originalSentence2',  # Type 3  
            'dataset_source',
            'model_source',
            'source_file'
        ]
        
        clean_df = clean_df[column_order]
        
        self.clean_dataset = clean_df
        
        logger.info(f"Clean dataset created: {len(clean_df):,} samples")
        logger.info(f"Columns retained: {list(clean_df.columns)}")
        
        return clean_df
    
    def explore_data_structure(self) -> Dict[str, Dict[str, Union[int, List[str], str]]]:
        """
        Analyze and display the structure of the HLPC dataset.
        
        Returns:
            Dictionary containing comprehensive dataset statistics and structure info.
        """
        if self.combined_dataset is None:
            self.combine_all_datasets()
        
        structure_info = {}
        
        print("=" * 80)
        print("HLPC Dataset Structure Analysis")
        print("=" * 80)
        
        df = self.combined_dataset
        
        print(f"\n📊 Combined Dataset Overview:")
        print(f"   • Total samples: {len(df):,}")
        print(f"   • Total columns: {len(df.columns)}")
        print(f"   • Source files: {df['source_file'].nunique()}")
        print(f"   • Dataset sources: {sorted(df['dataset_source'].unique())}")
        print(f"   • Model sources: {sorted(df['model_source'].unique())}")
        
        # Analyze by source dataset
        print(f"\n📈 Breakdown by Dataset Source:")
        for dataset in sorted(df['dataset_source'].unique()):
            count = len(df[df['dataset_source'] == dataset])
            print(f"   • {dataset}: {count:,} samples")
        
        # Analyze by model
        print(f"\n🤖 Breakdown by Model Source:")
        for model in sorted(df['model_source'].unique()):
            count = len(df[df['model_source'] == model])
            print(f"   • {model}: {count:,} samples")
        
        # Analyze text columns
        text_columns = [
            ('originalSentence1', 'Type 1: Human Original'),
            ('originalSentence2', 'Type 3: Human Paraphrased'),
            ('gpt_text', 'Type 2: GPT Generated'),
            ('gpt_p_text_1', 'Type 4: GPT Paraphrased (iter 1)'),
            ('gpt_p_text_5', 'Type 5: GPT Paraphrased (iter 5)'),
            ('opt_text', 'Type 2: OPT Generated'),
            ('opt_p_text_1', 'Type 4: OPT Paraphrased (iter 1)'),
            ('opt_p_text_5', 'Type 5: OPT Paraphrased (iter 5)')
        ]
        
        print(f"\n📝 Text Column Analysis:")
        column_stats = {}
        
        for col, description in text_columns:
            if col in df.columns:
                non_null_count = df[col].notna().sum()
                non_null_pct = (non_null_count / len(df)) * 100
                
                if non_null_count > 0:
                    avg_length = df[col].dropna().str.len().mean()
                    print(f"   • {description}")
                    print(f"     - Available: {non_null_count:,}/{len(df):,} ({non_null_pct:.1f}%)")
                    print(f"     - Avg length: {avg_length:.1f} chars")
                    
                    column_stats[col] = {
                        'description': description,
                        'available_count': int(non_null_count),
                        'availability_pct': float(non_null_pct),
                        'avg_length': float(avg_length)
                    }
                else:
                    print(f"   • {description}: No data available")
                    column_stats[col] = {
                        'description': description,
                        'available_count': 0,
                        'availability_pct': 0.0,
                        'avg_length': 0.0
                    }
        
        structure_info = {
            'total_samples': len(df),
            'total_columns': len(df.columns),
            'source_files': df['source_file'].nunique(),
            'dataset_breakdown': df['dataset_source'].value_counts().to_dict(),
            'model_breakdown': df['model_source'].value_counts().to_dict(),
            'column_stats': column_stats
        }
        
        return structure_info
    
    def display_sample_data(self, num_samples: int = 3) -> None:
        """
        Display sample data from the HLPC dataset.
        
        Args:
            num_samples: Number of samples to display.
        """
        if self.combined_dataset is None:
            self.combine_all_datasets()
        
        print("\n" + "=" * 80)
        print("HLPC Sample Data Examples")
        print("=" * 80)
        
        df = self.combined_dataset
        sample_df = df.head(num_samples)
        
        for idx, row in sample_df.iterrows():
            print(f"\n🔍 Sample {idx + 1} (Source: {row['source_file']}):")
            print(f"  Dataset: {row['dataset_source']} | Model: {row['model_source']}")
            print(f"  Original (Type 1): {row['originalSentence1'][:100]}...")
            print(f"  Human Para (Type 3): {row['originalSentence2'][:100]}...")
            
            if pd.notna(row['gpt_text']):
                print(f"  GPT Generated (Type 2): {str(row['gpt_text'])[:100]}...")
            if pd.notna(row['gpt_p_text_1']):
                print(f"  GPT Para (Type 4): {str(row['gpt_p_text_1'])[:100]}...")
            if pd.notna(row['opt_text']):
                print(f"  OPT Generated (Type 2): {str(row['opt_text'])[:100]}...")
    
    def analyze_text_quality_issues(self) -> Dict[str, any]:
        """
        Analyze potential quality issues in the HLPC dataset.
        
        Returns:
            Dictionary containing quality analysis results.
        """
        if self.combined_dataset is None:
            self.combine_all_datasets()
        
        print("\n" + "=" * 80)
        print("HLPC Data Quality Analysis")
        print("=" * 80)
        
        df = self.combined_dataset
        quality_issues = {}
        
        # Check for truncated text (text that seems cut off)
        text_columns = ['gpt_text', 'gpt_p_text_1', 'gpt_p_text_5', 'opt_text', 'opt_p_text_1', 'opt_p_text_5']
        
        print(f"\n🔍 Quality Issues Analysis:")
        
        for col in text_columns:
            if col in df.columns:
                non_null_data = df[col].dropna()
                if len(non_null_data) > 0:
                    # Check for very short text (potential truncation)
                    very_short = (non_null_data.str.len() < 20).sum()
                    very_short_pct = (very_short / len(non_null_data)) * 100
                    
                    # Check for repeated text (potential quality issues)
                    duplicated = non_null_data.duplicated().sum()
                    duplicated_pct = (duplicated / len(non_null_data)) * 100
                    
                    # Check for incomplete sentences (ending without punctuation)
                    incomplete = (~non_null_data.str.endswith(('.', '!', '?', '"'))).sum()
                    incomplete_pct = (incomplete / len(non_null_data)) * 100
                    
                    print(f"   • {col}:")
                    print(f"     - Very short texts (<20 chars): {very_short} ({very_short_pct:.1f}%)")
                    print(f"     - Duplicated texts: {duplicated} ({duplicated_pct:.1f}%)")
                    print(f"     - Incomplete sentences: {incomplete} ({incomplete_pct:.1f}%)")
                    
                    quality_issues[col] = {
                        'very_short_count': int(very_short),
                        'very_short_pct': float(very_short_pct),
                        'duplicated_count': int(duplicated),
                        'duplicated_pct': float(duplicated_pct),
                        'incomplete_count': int(incomplete),
                        'incomplete_pct': float(incomplete_pct)
                    }
        
        return quality_issues
    
    def get_text_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate comprehensive text statistics for all text columns.
        
        Returns:
            Dictionary containing text statistics for each column.
        """
        if self.combined_dataset is None:
            self.combine_all_datasets()
        
        print("\n" + "=" * 80)
        print("HLPC Text Statistics Analysis")
        print("=" * 80)
        
        df = self.combined_dataset
        text_stats = {}
        
        text_columns = [
            'originalSentence1', 'originalSentence2', 'gpt_text', 'gpt_p_text_1', 
            'gpt_p_text_5', 'opt_text', 'opt_p_text_1', 'opt_p_text_5'
        ]
        
        for col in text_columns:
            if col in df.columns:
                non_null_data = df[col].dropna()
                if len(non_null_data) > 0:
                    lengths = non_null_data.str.len()
                    word_counts = non_null_data.str.split().str.len()
                    
                    stats = {
                        'count': len(non_null_data),
                        'char_length_mean': float(lengths.mean()),
                        'char_length_std': float(lengths.std()),
                        'char_length_min': int(lengths.min()),
                        'char_length_max': int(lengths.max()),
                        'word_count_mean': float(word_counts.mean()),
                        'word_count_std': float(word_counts.std()),
                        'word_count_min': int(word_counts.min()),
                        'word_count_max': int(word_counts.max())
                    }
                    
                    text_stats[col] = stats
                    
                    print(f"\n📊 {col}:")
                    print(f"   • Sample count: {stats['count']:,}")
                    print(f"   • Character length: {stats['char_length_mean']:.1f} ± {stats['char_length_std']:.1f}")
                    print(f"     Range: {stats['char_length_min']}-{stats['char_length_max']} chars")
                    print(f"   • Word count: {stats['word_count_mean']:.1f} ± {stats['word_count_std']:.1f}")
                    print(f"     Range: {stats['word_count_min']}-{stats['word_count_max']} words")
        
        return text_stats
    
    def save_processed_data(self, output_dir: str = './data/processed') -> None:
        """
        Save the processed HLPC data to files.
        
        Args:
            output_dir: Directory to save the processed data.
        """
        if self.deduplicated_dataset is None:
            logger.warning("No deduplicated dataset available. Using combined dataset.")
            dataset_to_save = self.combined_dataset
        else:
            dataset_to_save = self.deduplicated_dataset
        
        if dataset_to_save is None:
            raise ValueError("No dataset available to save. Load and process data first.")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save full processed dataset as CSV
        csv_path = os.path.join(output_dir, 'hlpc_processed.csv')
        dataset_to_save.to_csv(csv_path, index=False)
        logger.info(f"Saved HLPC data to: {csv_path}")
        
        # Save full processed dataset as JSON for better text preservation
        json_path = os.path.join(output_dir, 'hlpc_processed.json')
        dataset_to_save.to_json(json_path, orient='records', indent=2)
        logger.info(f"Saved HLPC data to: {json_path}")
        
        # Create and save clean dataset (Type 1 and Type 3 only)
        clean_df = self.create_clean_dataset()
        clean_csv_path = os.path.join(output_dir, 'hlpc_processed_clean.csv')
        clean_df.to_csv(clean_csv_path, index=False)
        logger.info(f"Saved clean HLPC data to: {clean_csv_path}")
        
        # Save clean dataset as JSON as well
        clean_json_path = os.path.join(output_dir, 'hlpc_processed_clean.json')
        clean_df.to_json(clean_json_path, orient='records', indent=2)
        logger.info(f"Saved clean HLPC data to: {clean_json_path}")
        
        # Save summary statistics
        stats_path = os.path.join(output_dir, 'hlpc_stats.txt')
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("HLPC Dataset Statistics\n")
            f.write("=" * 40 + "\n")
            f.write(f"Total samples (full): {len(dataset_to_save):,}\n")
            f.write(f"Total samples (clean): {len(clean_df):,}\n")
            f.write(f"Dataset sources: {sorted(dataset_to_save['dataset_source'].unique())}\n")
            f.write(f"Model sources: {sorted(dataset_to_save['model_source'].unique())}\n")
            
            # Text statistics
            if 'originalSentence1' in dataset_to_save.columns:
                avg_len = dataset_to_save['originalSentence1'].str.len().mean()
                f.write(f"Average original text length: {avg_len:.1f} chars\n")
            
            f.write("\nClean Dataset Info:\n")
            f.write("- Contains only Type 1 (originalSentence1) and Type 3 (originalSentence2) text\n")
            f.write("- All outdated generated/paraphrased columns removed\n")
            f.write("- Ready for modern LLM regeneration of Types 2, 4, and 5\n")
        
        logger.info(f"Saved HLPC statistics to: {stats_path}")
        
        print(f"\n💾 HLPC processed data saved to: {output_dir}")
        print(f"   • Full dataset CSV: hlpc_processed.csv ({len(dataset_to_save):,} samples)")
        print(f"   • Full dataset JSON: hlpc_processed.json")
        print(f"   • Clean dataset CSV: hlpc_processed_clean.csv ({len(clean_df):,} samples)")
        print(f"   • Clean dataset JSON: hlpc_processed_clean.json")  
        print(f"   • Statistics: hlpc_stats.txt")
        
        print(f"\n🧹 Clean Dataset Info:")
        print(f"   • Contains only Type 1 and Type 3 text columns")
        print(f"   • Removes all outdated generated/paraphrased text")
        print(f"   • Perfect for modern LLM regeneration")
        print(f"   • Columns: {list(clean_df.columns)}")


def load_and_process_hlpc_data(data_dir: str = "./data/HLPC-data",
                              output_dir: str = './data/HLPC-data/processed',
                              similarity_threshold: float = 0.95,
                              num_samples: int = 3) -> HLPCDataLoader:
    """
    Load HLPC dataset, combine all files, remove duplicates, and analyze.
    
    Args:
        data_dir: Directory containing HLPC CSV files.
        output_dir: Directory to save processed data.
        similarity_threshold: Similarity threshold for duplicate removal.
        num_samples: Number of sample examples to display.
        
    Returns:
        Configured HLPCDataLoader instance with processed data.
    """
    # Initialize loader
    loader = HLPCDataLoader(data_dir=data_dir)
    
    try:
        # Load individual datasets
        individual_datasets = loader.load_individual_datasets()
        
        # Combine all datasets
        combined_df = loader.combine_all_datasets()
        
        # Perform comprehensive analysis
        print("\n" + "=" * 80)
        print("🔍 HLPC Dataset Analysis")
        print("=" * 80)
        
        structure_info = loader.explore_data_structure()
        loader.display_sample_data(num_samples=num_samples)
        quality_issues = loader.analyze_text_quality_issues()
        text_stats = loader.get_text_statistics()
        
        # Remove duplicates
        print("\n" + "=" * 80)
        print("🧹 Removing Duplicates")
        print("=" * 80)
        
        deduplicated_df = loader.remove_duplicates_by_first_sentence(
            similarity_threshold=similarity_threshold
        )
        
        # Save processed data (includes both full and clean versions)
        loader.save_processed_data(output_dir=output_dir)
        
        print("\n" + "=" * 80)
        print("✅ HLPC Processing Complete!")
        print("=" * 80)
        print(f"📁 Raw data directory: {data_dir}")
        print(f"💾 Processed data saved at: {output_dir}")
        print(f"🎯 Final dataset size: {len(deduplicated_df):,} samples")
        print(f"📊 Original vs. Final: {len(combined_df):,} → {len(deduplicated_df):,}")
        print(f"🧹 Clean dataset size: {len(loader.clean_dataset):,} samples")
        
        # Print quality summary
        print(f"\n📋 Quality Summary:")
        print(f"   • Available text types: 1, 2, 3, 4, 5 (all types)")
        print(f"   • Note: Types 2, 4, 5 use outdated models (GPT-2, OPT-1.3B)")
        print(f"   • Clean dataset ready for modern LLM regeneration")
        print(f"   • Recommendation: Use clean dataset for Types 2, 4, 5 regeneration")
        
        return loader
        
    except Exception as e:
        logger.error(f"Failed to process HLPC dataset: {e}")
        raise


if __name__ == "__main__":
    """
    Example usage of the HLPC data processor.
    """
    # Process HLPC data
    loader = load_and_process_hlpc_data(
        data_dir="./data/HLPC-data",
        output_dir="./data/HLPC-data/processed",
        similarity_threshold=0.95,
        num_samples=3
    )
    
    print(f"\n🎯 HLPC Analysis Complete!")
    print(f"   • Combined dataset: {len(loader.combined_dataset):,} samples")
    print(f"   • Deduplicated dataset: {len(loader.deduplicated_dataset):,} samples")
    print(f"   • Clean dataset: {len(loader.clean_dataset):,} samples")
    print(f"   • Ready for modern LLM regeneration!") 