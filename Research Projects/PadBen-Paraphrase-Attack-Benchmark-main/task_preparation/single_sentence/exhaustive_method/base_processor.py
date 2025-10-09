"""
Abstract base class for task processors.
"""

from abc import ABC, abstractmethod
from typing import List

from data_models import InputSample, ProcessedSample


class AbstractTaskProcessor(ABC):
    """Abstract base class for task-specific processors.
    
    Each task processor implements the logic to convert input samples
    into task-specific processed samples with appropriate labels.
    """
    
    @abstractmethod
    def process(self, input_samples: List[InputSample]) -> List[ProcessedSample]:
        """Process input samples into task-specific format.
        
        Args:
            input_samples: List of input samples to process
            
        Returns:
            List of processed samples with appropriate labels
            
        Raises:
            ValueError: If input data is invalid
        """
        pass
    
    @abstractmethod
    def get_task_name(self) -> str:
        """Get the name of the task.
        
        Returns:
            Human-readable task name
        """
        pass
    
    @abstractmethod
    def get_expected_output_size(self, input_size: int) -> int:
        """Get expected output size given input size.
        
        Args:
            input_size: Number of input samples
            
        Returns:
            Expected number of output samples
        """
        pass
    
    @abstractmethod
    def get_output_filename(self) -> str:
        """Get the output filename for this task.
        
        Returns:
            Filename for the output JSON file
        """
        pass
    
    def validate_output(self, processed_samples: List[ProcessedSample]) -> bool:
        """Validate the processed output.
        
        Args:
            processed_samples: List of processed samples to validate
            
        Returns:
            True if output is valid, False otherwise
        """
        if not processed_samples:
            return False
            
        # Check label distribution (should be roughly 50-50)
        labels = [sample.label for sample in processed_samples]
        label_0_count = labels.count(0)
        label_1_count = labels.count(1)
        
        total_count = len(labels)
        if total_count == 0:
            return False
            
        balance_ratio = abs(label_0_count - label_1_count) / total_count
        
        # Allow 2% tolerance for label imbalance
        return balance_ratio <= 0.02
