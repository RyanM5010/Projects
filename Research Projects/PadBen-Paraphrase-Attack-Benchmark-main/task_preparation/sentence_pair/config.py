"""
Configuration for sentence pair task preparation pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SentencePairTaskConfig:
    """Configuration for sentence pair task preparation pipeline.
    
    Attributes:
        input_path: Path to input JSON file
        output_dir: Output directory for processed tasks
        random_seed: Random seed for reproducibility
        min_length: Minimum text length for filtering
        batch_size: Batch size for processing
        enable_task1: Whether to enable Task1 processing
        enable_task2: Whether to enable Task2 processing
        enable_task3: Whether to enable Task3 processing
        enable_task4: Whether to enable Task4 processing
        enable_task5: Whether to enable Task5 processing
    """
    input_path: Path
    output_dir: Path
    random_seed: int = 42
    min_length: int = 5
    batch_size: int = 1000
    enable_task1: bool = True
    enable_task2: bool = True
    enable_task3: bool = True
    enable_task4: bool = True
    enable_task5: bool = True
