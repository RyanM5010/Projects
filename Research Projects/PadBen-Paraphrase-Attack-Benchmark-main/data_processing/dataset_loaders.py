"""
Specialized Dataset Loaders for PADBen Benchmark.

This module provides specialized loaders for each dataset type with
detailed processing and validation capabilities.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


class MRPCLoader:
    """Specialized loader for MRPC dataset."""
    
    def __init__(self, data_path: str = "./data/mrpc/mrpc_paraphrases.csv") -> None:
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None
    
    def load_and_validate(self) -> pd.DataFrame:
        """Load and validate MRPC dataset."""
        logger.info("Loading MRPC dataset...")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"MRPC data not found at {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        
        # Validate expected columns
        expected_columns = ['sentence1', 'sentence2', 'label', 'idx']
        missing_columns = set(expected_columns) - set(self.df.columns)
        if missing_columns:
            raise ValueError(f"Missing expected columns: {missing_columns}")
        
        # Validate all samples are paraphrases (label=1)
        if not (self.df['label'] == 1).all():
            logger.warning("Found non-paraphrase samples in MRPC data")
        
        logger.info(f"MRPC loaded: {len(self.df):,} samples")
        return self.df
    
    def get_text_types(self) -> Dict[int, pd.Series]:
        """Get available text types from MRPC."""
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_and_validate() first.")
        
        return {
            1: self.df['sentence1'],  # Human original
            3: self.df['sentence2']   # Human paraphrased
        }


class PAWSLoader:
    """Specialized loader for PAWS dataset."""
    
    def __init__(self, subset: str = "labeled_final") -> None:
        self.subset = subset
        self.dataset: Optional[Dataset] = None
        self.df: Optional[pd.DataFrame] = None
    
    def load_and_validate(self) -> pd.DataFrame:
        """Load and validate PAWS dataset."""
        logger.info(f"Loading PAWS dataset (subset: {self.subset})...")
        
        try:
            # Load from Hugging Face
            self.dataset = load_dataset("paws", self.subset)
            
            # Combine all splits
            all_data = []
            for split_name, split_data in self.dataset.items():
                df = pd.DataFrame(split_data)
                # Filter for paraphrases only (label=1)
                paraphrase_df = df[df['label'] == 1].copy()
                paraphrase_df['split'] = split_name
                all_data.append(paraphrase_df)
            
            self.df = pd.concat(all_data, ignore_index=True)
            
            logger.info(f"PAWS loaded: {len(self.df):,} paraphrase samples")
            return self.df
            
        except Exception as e:
            logger.error(f"Failed to load PAWS dataset: {e}")
            raise
    
    def get_text_types(self) -> Dict[int, pd.Series]:
        """Get available text types from PAWS."""
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_and_validate() first.")
        
        return {
            1: self.df['sentence1'],  # Human original
            3: self.df['sentence2']   # Human paraphrased
        }


class HLPCLoader:
    """Specialized loader for HLPC dataset."""
    
    def __init__(self, data_path: Optional[str] = None) -> None:
        self.data_path = Path(data_path) if data_path else None
        self.df: Optional[pd.DataFrame] = None
    
    def load_and_validate(self) -> pd.DataFrame:
        """Load and validate HLPC dataset."""
        logger.info("Loading HLPC dataset...")
        
        # TODO: Implement actual HLPC loading when path is provided
        if self.data_path and self.data_path.exists():
            # Load actual HLPC data
            logger.info(f"Loading HLPC from {self.data_path}")
            # Implementation depends on actual HLPC format
            pass
        else:
            logger.warning("HLPC data path not provided or doesn't exist")
            # Return empty DataFrame with proper structure
            self.df = pd.DataFrame({
                'originalSentence1': [],
                'originalSentence2': [],
                'gpt_text': [],
                'gpt_p_text_1': [],
                'gpt_p_text_5': [],
                'opt_text': [],
                'opt_p_text_1': [],
                'opt_p_text_5': []
            })
        
        return self.df
    
    def get_text_types(self) -> Dict[int, pd.Series]:
        """Get available text types from HLPC."""
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_and_validate() first.")
        
        if self.df.empty:
            return {i: pd.Series([], dtype='object') for i in range(1, 6)}
        
        return {
            1: self.df['originalSentence1'],  # Human original
            2: self.df['gpt_text'],          # LLM generated (outdated)
            3: self.df['originalSentence2'],  # Human paraphrased
            4: self.df['gpt_p_text_1'],      # LLM paraphrased original (outdated)
            5: self.df['gpt_p_text_5']       # LLM paraphrased generated (outdated)
        }

