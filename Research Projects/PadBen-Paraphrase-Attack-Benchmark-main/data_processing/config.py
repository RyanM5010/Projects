"""
Configuration module for PADBen data processing pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class DatasetConfig:
    """Configuration for individual datasets."""
    name: str
    path: Optional[str] = None
    available_types: List[int] = field(default_factory=list)
    missing_types: List[int] = field(default_factory=list)
    enabled: bool = True


@dataclass
class ProcessingConfig:
    """Configuration for data processing pipeline."""
    
    # Output settings
    output_dir: str = "./data/processed"
    save_formats: List[str] = field(default_factory=lambda: ["csv", "json"])
    
    # Duplicate removal settings
    similarity_threshold: float = 0.95
    similarity_method: str = "cosine"  # cosine, jaccard, etc.
    
    # Text preprocessing settings
    min_text_length: int = 10
    max_text_length: int = 1000
    remove_empty: bool = True
    
    # Dataset configurations
    datasets: Dict[str, DatasetConfig] = field(default_factory=lambda: {
        "MRPC": DatasetConfig(
            name="MRPC",
            path="./data/mrpc/mrpc_paraphrases.csv",
            available_types=[1, 3],
            missing_types=[2, 4, 5]
        ),
        "HLPC": DatasetConfig(
            name="HLPC", 
            path=None,
            available_types=[1, 2, 3, 4, 5],
            missing_types=[]
        ),
        "PAWS": DatasetConfig(
            name="PAWS",
            path=None,  # Loaded from HuggingFace
            available_types=[1, 3],
            missing_types=[2, 4, 5]
        )
    })
    
    # Text type definitions
    text_type_names: Dict[int, str] = field(default_factory=lambda: {
        1: "human_original_text",
        2: "llm_generated_text", 
        3: "human_paraphrased_text",
        4: "llm_paraphrased_original_text",
        5: "llm_paraphrased_generated_text"
    })


# Default configuration instance
DEFAULT_CONFIG = ProcessingConfig() 