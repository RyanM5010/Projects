"""
Configuration management for task preparation pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TaskPreparationConfig:
    """Configuration parameters for task data preparation pipeline.
    
    Attributes:
        random_seed: Random seed for reproducibility
        input_file_path: Path to input JSON file
        output_dir: Directory for output files
        label_balance_tolerance: Tolerance for label balance validation (0.02 = 2%)
        min_text_length: Minimum text length in characters
        log_level: Logging level
        batch_size: Batch size for processing large datasets
    """
    
    # Core settings
    random_seed: int = 42
    input_file_path: Optional[Path] = None
    output_dir: Optional[Path] = None
    
    # Validation settings
    label_balance_tolerance: float = 0.02  # 2%
    min_text_length: int = 5
    
    # Processing settings
    batch_size: int = 1000
    log_level: str = "INFO"
    
    # Task selection flags
    enable_task1: bool = True
    enable_task2: bool = True
    enable_task3: bool = True
    enable_task4: bool = True
    enable_task5: bool = True
    
    def __post_init__(self) -> None:
        """Post-initialization validation."""
        if self.input_file_path is not None:
            self.input_file_path = Path(self.input_file_path)
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
            
        # Validate tolerance range
        if not 0.0 <= self.label_balance_tolerance <= 0.1:
            raise ValueError("label_balance_tolerance must be between 0.0 and 0.1")
            
        # Validate minimum text length
        if self.min_text_length < 1:
            raise ValueError("min_text_length must be at least 1")
            
        # Validate batch size
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
