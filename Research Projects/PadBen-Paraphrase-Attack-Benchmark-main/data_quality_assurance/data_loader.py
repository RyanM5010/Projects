"""Data loading utilities for PADBen quality examination."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

try:
    from .config import DATA_PATH, TEXT_TYPES
except ImportError:
    from config import DATA_PATH, TEXT_TYPES

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads and preprocesses PADBen dataset for quality examination."""
    
    def __init__(self, data_path: Optional[Path] = None) -> None:
        """
        Initialize DataLoader.
        
        Args:
            data_path: Path to the JSON data file. Defaults to config DATA_PATH.
        """
        self.data_path = data_path or DATA_PATH
        self.data: List[Dict[str, Any]] = []
        self.text_types = TEXT_TYPES
        
    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load data from JSON file.
        
        Returns:
            List of data records.
            
        Raises:
            FileNotFoundError: If data file doesn't exist.
            json.JSONDecodeError: If JSON is malformed.
        """
        try:
            logger.info(f"Loading data from {self.data_path}")
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            logger.info(f"Successfully loaded {len(self.data)} records")
            return self.data
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.data_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {e}")
            raise
            
    def extract_texts_by_type(self, text_type: str) -> List[str]:
        """
        Extract all texts of a specific type.
        
        Args:
            text_type: Type of text to extract (e.g., 'type1', 'type2', etc.).
            
        Returns:
            List of texts for the specified type.
            
        Raises:
            ValueError: If text_type is not valid.
        """
        if text_type not in self.text_types:
            raise ValueError(f"Invalid text type: {text_type}. Valid types: {list(self.text_types.keys())}")
            
        if not self.data:
            self.load_data()
            
        field_name = self.text_types[text_type]
        texts = []
        
        for record in self.data:
            if field_name in record and record[field_name]:
                texts.append(str(record[field_name]).strip())
                
        logger.info(f"Extracted {len(texts)} texts for type {text_type}")
        return texts
    
    def extract_all_texts(self) -> Dict[str, List[str]]:
        """
        Extract all texts grouped by type.
        
        Returns:
            Dictionary mapping text types to lists of texts.
        """
        if not self.data:
            self.load_data()
            
        all_texts = {}
        for text_type in self.text_types.keys():
            all_texts[text_type] = self.extract_texts_by_type(text_type)
            
        return all_texts
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get basic statistics about the dataset.
        
        Returns:
            Dictionary containing dataset statistics.
        """
        if not self.data:
            self.load_data()
            
        stats = {
            "total_records": len(self.data),
            "dataset_sources": {},
            "text_type_counts": {},
            "avg_text_lengths": {}
        }
        
        # Count dataset sources
        for record in self.data:
            source = record.get("dataset_source", "unknown")
            stats["dataset_sources"][source] = stats["dataset_sources"].get(source, 0) + 1
            
        # Count texts per type and calculate average lengths
        all_texts = self.extract_all_texts()
        for text_type, texts in all_texts.items():
            stats["text_type_counts"][text_type] = len(texts)
            if texts:
                avg_length = sum(len(text.split()) for text in texts) / len(texts)
                stats["avg_text_lengths"][text_type] = round(avg_length, 2)
            else:
                stats["avg_text_lengths"][text_type] = 0
                
        return stats
    
    def create_dataframe(self) -> pd.DataFrame:
        """
        Create a pandas DataFrame from the loaded data.
        
        Returns:
            DataFrame with all text types as columns.
        """
        if not self.data:
            self.load_data()
            
        df_data = []
        for record in self.data:
            row = {"idx": record.get("idx"), "dataset_source": record.get("dataset_source")}
            for text_type, field_name in self.text_types.items():
                row[text_type] = record.get(field_name, "")
            df_data.append(row)
            
        return pd.DataFrame(df_data)
    
    def validate_data_completeness(self) -> Dict[str, float]:
        """
        Check data completeness for each text type.
        
        Returns:
            Dictionary mapping text types to completeness percentages.
        """
        if not self.data:
            self.load_data()
            
        completeness = {}
        total_records = len(self.data)
        
        for text_type, field_name in self.text_types.items():
            non_empty_count = sum(
                1 for record in self.data 
                if record.get(field_name) and str(record[field_name]).strip()
            )
            completeness[text_type] = (non_empty_count / total_records) * 100 if total_records > 0 else 0
            
        return completeness
