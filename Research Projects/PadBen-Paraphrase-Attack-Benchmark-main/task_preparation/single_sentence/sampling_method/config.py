"""
Configuration management for dynamically-adjusted task preparation pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DynamicTaskConfig:
    """Configuration parameters for dynamic task data preparation pipeline.
    
    This configuration supports selecting one sample per index with
    configurable label balance ratios.
    
    Attributes:
        random_seed: Random seed for reproducibility
        input_file_path: Path to input JSON file
        output_dir: Directory for output files
        min_text_length: Minimum text length in characters
        log_level: Logging level
        batch_size: Batch size for processing large datasets
        
        # Dynamic label balance settings
        label_1_ratio: Desired ratio of label 1 samples (0.0 to 1.0)
        enable_task1: Enable Task 1 processing
        enable_task2: Enable Task 2 processing
        enable_task3: Enable Task 3 processing
        enable_task4: Enable Task 4 processing
    """
    
    # Core settings
    random_seed: int = 42
    input_file_path: Optional[Path] = None
    output_dir: Optional[Path] = None
    
    # Validation settings
    min_text_length: int = 5
    
    # Processing settings
    batch_size: int = 1000
    log_level: str = "INFO"
    
    # Dynamic label balance settings
    label_1_ratio: float = 0.5  # 50% label 1, 50% label 0
    
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
            
        # Validate label_1_ratio range
        if not 0.0 <= self.label_1_ratio <= 1.0:
            raise ValueError("label_1_ratio must be between 0.0 and 1.0")
            
        # Validate minimum text length
        if self.min_text_length < 1:
            raise ValueError("min_text_length must be at least 1")
            
        # Validate batch size
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
    
    @property
    def label_0_ratio(self) -> float:
        """Get the ratio of label 0 samples.
        
        Returns:
            Ratio of label 0 samples (1.0 - label_1_ratio)
        """
        return 1.0 - self.label_1_ratio
