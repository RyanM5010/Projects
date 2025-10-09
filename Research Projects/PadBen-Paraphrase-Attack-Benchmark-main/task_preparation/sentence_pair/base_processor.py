"""
Base processor for sentence pair task preparation.
"""

from abc import ABC, abstractmethod
from typing import List
from data_models import InputSample, ProcessedSample


class AbstractSentencePairProcessor(ABC):
    """Abstract base class for sentence pair task processors.
    
    Each task processor implements the logic for creating sentence pairs
    from input samples, where the model needs to determine which sentence
    in the pair is machine-generated vs human-written.
    """
    
    @abstractmethod
    def process(self, input_samples: List[InputSample]) -> List[ProcessedSample]:
        """Process input samples to create sentence pair samples.
        
        Args:
            input_samples: List of input samples to process
            
        Returns:
            List of processed sentence pair samples
            
        Raises:
            ValueError: If processing fails
        """
        pass
    
    @abstractmethod
    def get_task_name(self) -> str:
        """Get the name of the task.
        
        Returns:
            Task name string
        """
        pass
    
    @abstractmethod
    def get_expected_output_size(self, input_size: int) -> int:
        """Get the expected output size for a given input size.
        
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
            Output filename string
        """
        pass
