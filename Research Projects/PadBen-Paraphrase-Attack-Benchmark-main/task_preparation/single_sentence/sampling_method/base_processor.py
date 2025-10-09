"""
Abstract base class for dynamic task processors.
"""

import random
from abc import ABC, abstractmethod
from typing import List, Tuple

from data_models import DynamicInputSample, DynamicProcessedSample


class AbstractDynamicTaskProcessor(ABC):
    """Abstract base class for dynamic task-specific processors.
    
    Each task processor implements the logic to convert input samples
    into task-specific processed samples with configurable label balance,
    selecting only one sample per input index.
    """
    
    def __init__(self, random_seed: int = 42, label_1_ratio: float = 0.5) -> None:
        """Initialize processor with configuration.
        
        Args:
            random_seed: Seed for reproducible random selection
            label_1_ratio: Desired ratio of label 1 samples (0.0 to 1.0)
        """
        self.random_seed = random_seed
        self.label_1_ratio = label_1_ratio
        self.label_0_ratio = 1.0 - label_1_ratio
    
    @abstractmethod
    def get_text_pairs(self, input_sample: DynamicInputSample) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        """Get the two text type pairs for this task.
        
        Args:
            input_sample: Input sample containing all text types
            
        Returns:
            Tuple of ((text_for_label_0, type_name_0), (text_for_label_1, type_name_1))
        """
        pass
    
    def process(self, input_samples: List[DynamicInputSample]) -> List[DynamicProcessedSample]:
        """Process input samples into task-specific format with dynamic balance.
        
        For each input sample, randomly selects either the label 0 or label 1 text
        based on the configured label_1_ratio.
        
        Args:
            input_samples: List of input samples to process
            
        Returns:
            List of processed samples with configured label balance
            
        Raises:
            ValueError: If input data is invalid or empty
        """
        if not input_samples:
            raise ValueError("Input samples cannot be empty")
        
        random.seed(self.random_seed)
        processed_samples: List[DynamicProcessedSample] = []
        
        # Calculate target counts for each label
        total_samples = len(input_samples)
        target_label_1_count = int(total_samples * self.label_1_ratio)
        target_label_0_count = total_samples - target_label_1_count
        
        # Create a list of label assignments
        label_assignments = [1] * target_label_1_count + [0] * target_label_0_count
        random.shuffle(label_assignments)
        
        # Process each input sample
        for i, input_sample in enumerate(input_samples):
            # Get text pairs for this task
            (text_0, type_0), (text_1, type_1) = self.get_text_pairs(input_sample)
            
            # Validate texts exist and meet minimum length
            if not text_0 or not text_1:
                raise ValueError(f"Sample {input_sample.idx} missing required text types")
            
            if len(text_0.strip()) < 5 or len(text_1.strip()) < 5:
                raise ValueError(f"Sample {input_sample.idx} has text shorter than minimum length")
            
            # Select text based on assigned label
            assigned_label = label_assignments[i]
            if assigned_label == 0:
                selected_text = text_0.strip()
                selected_type = type_0
            else:
                selected_text = text_1.strip()
                selected_type = type_1
            
            # Create processed sample
            processed_sample = DynamicProcessedSample(
                idx=input_sample.idx,  # Keep original index
                sentence=selected_text,
                label=assigned_label,
                text_type=selected_type
            )
            processed_samples.append(processed_sample)
        
        return processed_samples
    
    @abstractmethod
    def get_task_name(self) -> str:
        """Get the name of the task.
        
        Returns:
            Human-readable task name
        """
        pass
    
    def get_expected_output_size(self, input_size: int) -> int:
        """Get expected output size given input size.
        
        For dynamic processors, output size equals input size (one per index).
        
        Args:
            input_size: Number of input samples
            
        Returns:
            Expected number of output samples (same as input size)
        """
        return input_size
    
    @abstractmethod
    def get_output_filename(self) -> str:
        """Get the output filename for this task.
        
        Returns:
            Filename for the output JSON file
        """
        pass
    
    def validate_output(self, processed_samples: List[DynamicProcessedSample]) -> bool:
        """Validate the processed output.
        
        Args:
            processed_samples: List of processed samples to validate
            
        Returns:
            True if output is valid, False otherwise
        """
        if not processed_samples:
            return False
        
        # Check that all samples have valid indices
        indices = [sample.idx for sample in processed_samples]
        if len(set(indices)) != len(indices):
            return False  # Duplicate indices
        
        # Check label distribution matches target ratio (with small tolerance)
        labels = [sample.label for sample in processed_samples]
        label_0_count = labels.count(0)
        label_1_count = labels.count(1)
        
        total_count = len(labels)
        if total_count == 0:
            return False
        
        actual_label_1_ratio = label_1_count / total_count
        ratio_difference = abs(actual_label_1_ratio - self.label_1_ratio)
        
        # Allow tolerance based on sample size (larger tolerance for smaller datasets)
        tolerance = max(0.05, 2.0 / total_count)  # At least 5% or 2 samples worth
        
        return ratio_difference <= tolerance
    
    def get_statistics(self, processed_samples: List[DynamicProcessedSample]) -> dict:
        """Get statistics about the processed samples.
        
        Args:
            processed_samples: List of processed samples
            
        Returns:
            Dictionary containing statistics
        """
        if not processed_samples:
            return {}
        
        labels = [sample.label for sample in processed_samples]
        text_types = [sample.text_type for sample in processed_samples]
        
        label_0_count = labels.count(0)
        label_1_count = labels.count(1)
        total_count = len(labels)
        
        # Count text type usage
        type_counts = {}
        for text_type in text_types:
            type_counts[text_type] = type_counts.get(text_type, 0) + 1
        
        return {
            "total_samples": total_count,
            "label_0_count": label_0_count,
            "label_1_count": label_1_count,
            "label_0_ratio": label_0_count / total_count if total_count > 0 else 0.0,
            "label_1_ratio": label_1_count / total_count if total_count > 0 else 0.0,
            "target_label_1_ratio": self.label_1_ratio,
            "text_type_counts": type_counts
        }
