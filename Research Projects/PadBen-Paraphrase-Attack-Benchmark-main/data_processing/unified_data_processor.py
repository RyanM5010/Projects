"""
Unified Data Processing Module for PADBen Benchmark.

This module provides a comprehensive framework for processing multiple datasets
(MRPC, HLPC, PAWS) and standardizing them to a unified format with all 5 text types:
- Type 1: Human original text
- Type 2: LLM-generated text
- Type 3: Human-paraphrased human original text  
- Type 4: LLM-paraphrased human original text
- Type 5: LLM-paraphrased LLM-generated text
Plus metadata for tracking dataset sources and indices.
"""

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class UnifiedDataProcessor:
    """
    Unified data processor for PADBen benchmark datasets.
    
    Handles loading, cleaning, and standardizing multiple datasets (MRPC, HLPC, PAWS)
    to a common format with all 5 text types for comprehensive evaluation.
    """
    
    def __init__(self, output_dir: str = "./data/processed") -> None:
        """
        Initialize the unified data processor.
        
        Args:
            output_dir: Directory to save processed datasets.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define standard column names for unified format
        self.standard_columns = [
            'idx',
            'dataset_source', 
            'human_original_text',           # Type 1
            'llm_generated_text',            # Type 2
            'human_paraphrased_text',        # Type 3
            'llm_paraphrased_original_text', # Type 4
            'llm_paraphrased_generated_text' # Type 5
        ]
        
        # Dataset cache directories
        self.cache_dirs = {
            'mrpc': Path('./data/mrpc'),
            'hlpc': Path('./data/HLPC-data/processed'),
            'paws': Path('./data/PAWS-data/processed')
        }
        
        # Similarity threshold for duplicate detection
        self.similarity_threshold = 0.85
        
        logger.info(f"Initialized UnifiedDataProcessor with output directory: {self.output_dir}")
    
    def load_mrpc_data(self) -> Optional[pd.DataFrame]:
        """
        Load and process MRPC dataset directly from cache file.
        
        Returns:
            DataFrame with standardized columns or None if loading fails.
        """
        try:
            logger.info("Loading MRPC dataset...")
            cache_file = self.cache_dirs['mrpc'] / 'mrpc_paraphrases.csv'
            
            if not cache_file.exists():
                logger.warning(f"MRPC cache file not found at {cache_file}")
                return None
            
            # Load directly from cache file
            mrpc_df = pd.read_csv(cache_file)
            
            if mrpc_df.empty:
                logger.warning("MRPC data is empty")
                return None
            
            # Filter only label=1 (paraphrases) as specified in requirements
            mrpc_df = mrpc_df[mrpc_df['label'] == 1].copy()
            
            if mrpc_df.empty:
                logger.warning("No paraphrase pairs (label=1) found in MRPC data")
                return None
            
            # Standardize to unified format
            unified_df = pd.DataFrame()
            unified_df['idx'] = range(len(mrpc_df))
            unified_df['dataset_source'] = 'mrpc'
            unified_df['human_original_text'] = mrpc_df['sentence1']      # Type 1
            unified_df['llm_generated_text'] = None                       # Type 2 - to be generated
            unified_df['human_paraphrased_text'] = mrpc_df['sentence2']   # Type 3
            unified_df['llm_paraphrased_original_text'] = None            # Type 4 - to be generated
            unified_df['llm_paraphrased_generated_text'] = None           # Type 5 - to be generated
            
            logger.info(f"Successfully loaded {len(unified_df)} MRPC samples")
            return unified_df
            
        except Exception as e:
            logger.error(f"Error loading MRPC data: {e}")
            return None
    
    def load_hlpc_data(self) -> Optional[pd.DataFrame]:
        """
        Load and process HLPC dataset directly from cache file.
        
        Returns:
            DataFrame with standardized columns or None if loading fails.
        """
        try:
            logger.info("Loading HLPC dataset...")
            cache_file = self.cache_dirs['hlpc'] / 'hlpc_processed_clean.csv'
            
            if not cache_file.exists():
                logger.warning(f"HLPC clean cache file not found at {cache_file}")
                # Try to load the full processed file instead
                full_cache_file = self.cache_dirs['hlpc'] / 'hlpc_processed_full.csv'
                if full_cache_file.exists():
                    cache_file = full_cache_file
                else:
                    logger.warning(f"No HLPC cache files found")
                    return None
            
            # Load directly from cache file
            hlpc_df = pd.read_csv(cache_file)
            
            if hlpc_df.empty:
                logger.warning("HLPC data is empty")
                return None
            
            # Standardize to unified format
            unified_df = pd.DataFrame()
            unified_df['idx'] = range(len(hlpc_df))
            unified_df['dataset_source'] = 'hlpc'
            
            # Use the correct column names from HLPC cache
            unified_df['human_original_text'] = hlpc_df['originalSentence1']     # Type 1
            unified_df['human_paraphrased_text'] = hlpc_df['originalSentence2']  # Type 3
            unified_df['llm_generated_text'] = None                              # Type 2 - to be generated
            unified_df['llm_paraphrased_original_text'] = None                   # Type 4 - to be generated  
            unified_df['llm_paraphrased_generated_text'] = None                  # Type 5 - to be generated
            
            logger.info(f"Successfully loaded {len(unified_df)} HLPC samples")
            return unified_df
            
        except Exception as e:
            logger.error(f"Error loading HLPC data: {e}")
            return None
    
    def load_paws_data(self) -> Optional[pd.DataFrame]:
        """
        Load and process PAWS dataset directly from cache file.
        
        Returns:
            DataFrame with standardized columns or None if loading fails.
        """
        try:
            logger.info("Loading PAWS dataset...")
            cache_file = self.cache_dirs['paws'] / 'paws_processed_clean.csv'
            
            if not cache_file.exists():
                logger.warning(f"PAWS clean cache file not found at {cache_file}")
                # Try to load the full processed file instead
                full_cache_file = self.cache_dirs['paws'] / 'paws_processed_full.csv'
                if full_cache_file.exists():
                    cache_file = full_cache_file
                else:
                    logger.warning(f"No PAWS cache files found")
                    return None
            
            # Load directly from cache file
            paws_df = pd.read_csv(cache_file)
            
            if paws_df.empty:
                logger.warning("PAWS data is empty")
                return None
            
            # Standardize to unified format
            unified_df = pd.DataFrame()
            unified_df['idx'] = range(len(paws_df))
            unified_df['dataset_source'] = 'paws'
            
            # Use the correct column names from PAWS cache
            unified_df['human_original_text'] = paws_df['human_original_text']      # Type 1
            unified_df['human_paraphrased_text'] = paws_df['human_paraphrased_text'] # Type 3
            unified_df['llm_generated_text'] = None                                  # Type 2 - to be generated
            unified_df['llm_paraphrased_original_text'] = None                       # Type 4 - to be generated
            unified_df['llm_paraphrased_generated_text'] = None                      # Type 5 - to be generated
            
            logger.info(f"Successfully loaded {len(unified_df)} PAWS samples")
            return unified_df
            
        except Exception as e:
            logger.error(f"Error loading PAWS data: {e}")
            return None
    
    
    def remove_duplicates_by_similarity(self, df: pd.DataFrame, 
                                      text_column: str = 'human_original_text',
                                      threshold: float = None) -> pd.DataFrame:
        """
        Remove duplicate rows based on text similarity in specified column.
        
        Args:
            df: Input DataFrame.
            text_column: Column name to check for similarity.
            threshold: Similarity threshold (0-1). Uses class default if None.
            
        Returns:
            DataFrame with duplicates removed.
        """
        if threshold is None:
            threshold = self.similarity_threshold
            
        if df.empty or text_column not in df.columns:
            logger.warning(f"DataFrame is empty or missing column '{text_column}'")
            return df
        
        logger.info(f"Removing duplicates based on similarity in '{text_column}' column (threshold: {threshold})")
        
        # Filter out rows with null values in the text column
        valid_df = df.dropna(subset=[text_column]).copy()
        if valid_df.empty:
            logger.warning("No valid text data found after removing null values")
            return df
        
        # Calculate TF-IDF vectors
        try:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                max_features=10000,
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform(valid_df[text_column].astype(str))
            
            # Calculate pairwise cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Find duplicates
            duplicates_to_remove = set()
            n_samples = len(valid_df)
            
            for i in range(n_samples):
                if i in duplicates_to_remove:
                    continue
                    
                for j in range(i + 1, n_samples):
                    if j in duplicates_to_remove:
                        continue
                        
                    if similarity_matrix[i, j] >= threshold:
                        duplicates_to_remove.add(j)  # Remove the later occurrence
            
            # Remove duplicates
            indices_to_keep = [i for i in range(n_samples) if i not in duplicates_to_remove]
            deduplicated_df = valid_df.iloc[indices_to_keep].copy()
            
            logger.info(f"Removed {len(duplicates_to_remove)} similar samples out of {len(valid_df)}")
            logger.info(f"Remaining samples: {len(deduplicated_df)}")
            
            return deduplicated_df
            
        except Exception as e:
            logger.error(f"Error during similarity-based deduplication: {e}")
            return df
    
    def concatenate_datasets(self, datasets: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Concatenate multiple datasets and reassign global indices.
        
        Args:
            datasets: List of DataFrames to concatenate.
            
        Returns:
            Concatenated DataFrame with reassigned indices.
        """
        if not datasets:
            logger.warning("No datasets provided for concatenation")
            return pd.DataFrame(columns=self.standard_columns)
        
        # Filter out None/empty datasets
        valid_datasets = [df for df in datasets if df is not None and not df.empty]
        
        if not valid_datasets:
            logger.warning("No valid datasets found for concatenation")
            return pd.DataFrame(columns=self.standard_columns)
        
        logger.info(f"Concatenating {len(valid_datasets)} datasets")
        
        # Concatenate datasets
        concatenated_df = pd.concat(valid_datasets, ignore_index=True)
        
        # Reassign global indices
        concatenated_df['idx'] = range(len(concatenated_df))
        
        # Log dataset composition
        dataset_counts = concatenated_df['dataset_source'].value_counts()
        logger.info(f"Dataset composition after concatenation:")
        for dataset, count in dataset_counts.items():
            logger.info(f"  {dataset}: {count} samples")
        
        return concatenated_df
    
    def analyze_unified_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the unified dataset and provide comprehensive statistics.
        
        Args:
            df: Unified DataFrame to analyze.
            
        Returns:
            Dictionary containing analysis results.
        """
        if df.empty:
            return {"error": "Dataset is empty"}
        
        logger.info("Analyzing unified dataset...")
        
        analysis = {
            "total_samples": len(df),
            "datasets": {},
            "text_statistics": {},
            "missing_data": {},
            "data_quality": {}
        }
        
        # Dataset composition
        dataset_counts = df['dataset_source'].value_counts()
        for dataset, count in dataset_counts.items():
            analysis["datasets"][dataset] = {
                "count": int(count),
                "percentage": float(count / len(df) * 100)
            }
        
        # Text statistics for each type
        text_columns = [col for col in self.standard_columns if 'text' in col]
        
        for col in text_columns:
            if col in df.columns:
                valid_texts = df[col].dropna()
                if not valid_texts.empty:
                    text_lengths = valid_texts.astype(str).str.len()
                    analysis["text_statistics"][col] = {
                        "count": len(valid_texts),
                        "mean_length": float(text_lengths.mean()),
                        "median_length": float(text_lengths.median()),
                        "min_length": int(text_lengths.min()),
                        "max_length": int(text_lengths.max()),
                        "std_length": float(text_lengths.std())
                    }
        
        # Missing data analysis
        for col in self.standard_columns:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                analysis["missing_data"][col] = {
                    "missing_count": int(missing_count),
                    "missing_percentage": float(missing_count / len(df) * 100)
                }
        
        # Data quality checks - convert numpy bool to Python bool for JSON serialization
        analysis["data_quality"] = {
            "has_type1": bool("human_original_text" in df.columns and df['human_original_text'].notna().any()),
            "has_type3": bool("human_paraphrased_text" in df.columns and df['human_paraphrased_text'].notna().any()),
            "ready_for_generation": True,  # Type 2, 4, 5 will be generated
        }
        
        return analysis
    
    def save_unified_data(self, df: pd.DataFrame, filename: str = "unified_padben_data") -> None:
        """
        Save the unified dataset in multiple formats.
        
        Args:
            df: Unified DataFrame to save.
            filename: Base filename (without extension).
        """
        if df.empty:
            logger.warning("Cannot save empty dataset")
            return
        
        logger.info(f"Saving unified dataset with {len(df)} samples...")
        
        try:
            # Save as CSV
            csv_path = self.output_dir / f"{filename}.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved CSV to: {csv_path}")
            
            # Save as JSON
            json_path = self.output_dir / f"{filename}.json"
            df.to_json(json_path, orient='records', indent=2)
            logger.info(f"Saved JSON to: {json_path}")
            
            # Save analysis
            analysis = self.analyze_unified_data(df)
            analysis_path = self.output_dir / f"{filename}_analysis.json"
            with open(analysis_path, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)  # Use default=str for JSON serialization
            logger.info(f"Saved analysis to: {analysis_path}")
            
        except Exception as e:
            logger.error(f"Error saving unified data: {e}")
    
    def process_all_datasets(self, remove_duplicates: bool = True) -> pd.DataFrame:
        """
        Load, process, and unify all available datasets.
        
        Args:
            remove_duplicates: Whether to remove similar duplicates based on Type 1 text.
            
        Returns:
            Unified DataFrame containing all processed datasets.
        """
        logger.info("Starting unified data processing pipeline...")
        
        # Load individual datasets
        datasets = []
        
        # Load MRPC
        mrpc_df = self.load_mrpc_data()
        if mrpc_df is not None:
            datasets.append(mrpc_df)
        
        # Load HLPC
        hlpc_df = self.load_hlpc_data()
        if hlpc_df is not None:
            datasets.append(hlpc_df)
        
        # Load PAWS
        paws_df = self.load_paws_data()
        if paws_df is not None:
            datasets.append(paws_df)
        
        
        if not datasets:
            logger.error("No datasets were successfully loaded")
            return pd.DataFrame(columns=self.standard_columns)
        
        # Concatenate datasets
        unified_df = self.concatenate_datasets(datasets)
        
        if unified_df.empty:
            logger.error("Failed to concatenate datasets")
            return unified_df
        
        # Remove duplicates if requested
        if remove_duplicates:
            unified_df = self.remove_duplicates_by_similarity(
                unified_df, 
                text_column='human_original_text'
            )
        
        # Save unified data
        self.save_unified_data(unified_df, "unified_padben_base")
        
        logger.info("Unified data processing pipeline completed successfully!")
        return unified_df


def main():
    """
    Main function to run the unified data processing pipeline.
    """
    processor = UnifiedDataProcessor()
    
    # Process all datasets
    unified_data = processor.process_all_datasets(remove_duplicates=True)
    
    if not unified_data.empty:
        print(f"\n{'='*60}")
        print("UNIFIED DATA PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total samples: {len(unified_data)}")
        print(f"Dataset composition:")
        dataset_counts = unified_data['dataset_source'].value_counts()
        for dataset, count in dataset_counts.items():
            print(f"  {dataset}: {count} samples ({count/len(unified_data)*100:.1f}%)")
        print(f"{'='*60}")
    else:
        print("Failed to process datasets")


if __name__ == "__main__":
    main() 