"""
Utility classes for task preparation pipeline.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from data_models import InputSample, ProcessedSample, SingleSample, PairSample


logger = logging.getLogger(__name__)


class DataValidator:
    """Validator for input and output data quality control."""
    
    def __init__(self, min_text_length: int = 5) -> None:
        """Initialize validator with configuration.
        
        Args:
            min_text_length: Minimum required text length in characters
        """
        self.min_text_length = min_text_length
    
    def validate_input_sample(self, sample_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate a single input sample.
        
        Args:
            sample_data: Dictionary containing sample data
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check required fields presence
            required_fields = [
                "idx", "dataset_source",
                "human_original_text(type1)",
                "llm_generated_text(type2)", 
                "human_paraphrased_text(type3)",
                "llm_paraphrased_original_text(type4)-prompt-based",
                "llm_paraphrased_generated_text(type5)-1st",
                "llm_paraphrased_generated_text(type5)-3rd"
            ]
            
            missing_fields = [field for field in required_fields if field not in sample_data]
            if missing_fields:
                return False, f"Missing required fields: {missing_fields}"
            
            # Validate data types
            if not isinstance(sample_data["idx"], int):
                return False, "Field 'idx' must be an integer"
                
            if not isinstance(sample_data["dataset_source"], str):
                return False, "Field 'dataset_source' must be a string"
            
            # Validate text fields
            text_fields = [field for field in required_fields if field.startswith(("human_", "llm_"))]
            for field in text_fields:
                text_value = sample_data[field]
                if not isinstance(text_value, str):
                    return False, f"Field '{field}' must be a string"
                    
                if len(text_value.strip()) < self.min_text_length:
                    return False, f"Field '{field}' text too short (minimum {self.min_text_length} characters)"
            
            return True, ""
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def validate_input_data(self, input_data: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """Validate entire input dataset.
        
        Args:
            input_data: List of input sample dictionaries
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        if not input_data:
            return False, ["Input data is empty"]
            
        errors = []
        
        # Validate each sample
        for i, sample_data in enumerate(input_data):
            is_valid, error_msg = self.validate_input_sample(sample_data)
            if not is_valid:
                errors.append(f"Sample {i}: {error_msg}")
        
        # Check for duplicate indices
        indices = [sample.get("idx", -1) for sample in input_data]
        if len(set(indices)) != len(indices):
            errors.append("Duplicate indices found in input data")
        
        return len(errors) == 0, errors
    
    def validate_processed_samples(self, samples: List[ProcessedSample], 
                                 expected_size: int,
                                 tolerance: float = 0.02) -> Tuple[bool, List[str]]:
        """Validate processed samples.
        
        Args:
            samples: List of processed samples
            expected_size: Expected number of samples
            tolerance: Label balance tolerance (default 2%)
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not samples:
            errors.append("Processed samples list is empty")
            return False, errors
            
        # Check sample count
        if len(samples) != expected_size:
            errors.append(f"Expected {expected_size} samples, got {len(samples)}")
        
        # Check label distribution
        labels = [sample.label for sample in samples]
        label_counts = Counter(labels)
        
        if set(labels) != {0, 1}:
            errors.append(f"Expected labels 0 and 1, found labels: {set(labels)}")
        
        if 0 in label_counts and 1 in label_counts:
            total = len(samples)
            balance_ratio = abs(label_counts[0] - label_counts[1]) / total
            if balance_ratio > tolerance:
                errors.append(f"Label imbalance: {balance_ratio:.3f} > {tolerance:.3f}")
        
        # Check index continuity
        indices = [sample.idx for sample in samples]
        expected_indices = list(range(len(samples)))
        if sorted(indices) != expected_indices:
            errors.append("Indices are not continuous from 0 to N-1")
        
        # Check for duplicate indices
        if len(set(indices)) != len(indices):
            errors.append("Duplicate indices found in processed samples")
        
        return len(errors) == 0, errors


class LabelBalancer:
    """Utility for checking and reporting label balance."""
    
    @staticmethod
    def get_label_distribution(samples: List[ProcessedSample]) -> Dict[int, int]:
        """Get label distribution counts.
        
        Args:
            samples: List of processed samples
            
        Returns:
            Dictionary mapping labels to counts
        """
        labels = [sample.label for sample in samples]
        return dict(Counter(labels))
    
    @staticmethod
    def calculate_balance_ratio(samples: List[ProcessedSample]) -> float:
        """Calculate label balance ratio (0.0 = perfect balance, 1.0 = completely imbalanced).
        
        Args:
            samples: List of processed samples
            
        Returns:
            Balance ratio between 0.0 and 1.0
        """
        if not samples:
            return 1.0
            
        distribution = LabelBalancer.get_label_distribution(samples)
        
        if len(distribution) != 2:
            return 1.0  # Not binary classification
            
        counts = list(distribution.values())
        total = sum(counts)
        
        if total == 0:
            return 1.0
            
        return abs(counts[0] - counts[1]) / total
    
    @staticmethod
    def is_balanced(samples: List[ProcessedSample], tolerance: float = 0.02) -> bool:
        """Check if labels are balanced within tolerance.
        
        Args:
            samples: List of processed samples
            tolerance: Maximum allowed imbalance ratio
            
        Returns:
            True if balanced within tolerance
        """
        return LabelBalancer.calculate_balance_ratio(samples) <= tolerance


class StatisticsReporter:
    """Utility for generating statistics reports."""
    
    @staticmethod
    def generate_task_report(task_name: str, 
                           input_samples: List[InputSample],
                           processed_samples: List[ProcessedSample]) -> Dict[str, Any]:
        """Generate comprehensive task processing report.
        
        Args:
            task_name: Name of the processed task
            input_samples: Original input samples
            processed_samples: Processed output samples
            
        Returns:
            Dictionary containing detailed statistics
        """
        # Basic counts
        input_count = len(input_samples)
        output_count = len(processed_samples)
        expansion_ratio = output_count / input_count if input_count > 0 else 0
        
        # Label distribution
        label_dist = LabelBalancer.get_label_distribution(processed_samples)
        balance_ratio = LabelBalancer.calculate_balance_ratio(processed_samples)
        
        # Dataset source distribution
        source_dist = Counter([sample.dataset_source for sample in input_samples])
        
        # Text length statistics
        if processed_samples and isinstance(processed_samples[0], SingleSample):
            text_lengths = [len(sample.sentence) for sample in processed_samples 
                          if isinstance(sample, SingleSample)]
        elif processed_samples and isinstance(processed_samples[0], PairSample):
            text_lengths = [len(sample.paraphrased_sentence) for sample in processed_samples 
                          if isinstance(sample, PairSample)]
        else:
            text_lengths = []
        
        text_stats = {}
        if text_lengths:
            text_stats = {
                "min_length": min(text_lengths),
                "max_length": max(text_lengths),
                "avg_length": sum(text_lengths) / len(text_lengths),
                "median_length": sorted(text_lengths)[len(text_lengths) // 2]
            }
        
        return {
            "task_name": task_name,
            "processing_summary": {
                "input_samples": input_count,
                "output_samples": output_count,
                "expansion_ratio": expansion_ratio
            },
            "label_distribution": label_dist,
            "label_balance": {
                "balance_ratio": balance_ratio,
                "is_balanced": balance_ratio <= 0.02
            },
            "source_distribution": dict(source_dist),
            "text_statistics": text_stats,
            "validation_status": "passed" if balance_ratio <= 0.02 else "failed"
        }
    
    @staticmethod
    def save_report(report: Dict[str, Any], output_path: Path) -> None:
        """Save report to JSON file.
        
        Args:
            report: Report dictionary to save
            output_path: Path for output file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Report saved to {output_path}")
    
    @staticmethod
    def print_summary(report: Dict[str, Any]) -> None:
        """Print a formatted summary of the report.
        
        Args:
            report: Report dictionary to summarize
        """
        print(f"\n=== {report['task_name']} - Processing Summary ===")
        print(f"Input samples: {report['processing_summary']['input_samples']:,}")
        print(f"Output samples: {report['processing_summary']['output_samples']:,}")
        print(f"Expansion ratio: {report['processing_summary']['expansion_ratio']:.1f}x")
        
        print(f"\nLabel Distribution:")
        for label, count in report['label_distribution'].items():
            percentage = count / report['processing_summary']['output_samples'] * 100
            print(f"  Label {label}: {count:,} ({percentage:.1f}%)")
        
        balance_status = "✓ BALANCED" if report['label_balance']['is_balanced'] else "✗ IMBALANCED"
        print(f"Label Balance: {balance_status} (ratio: {report['label_balance']['balance_ratio']:.3f})")
        
        if report['text_statistics']:
            stats = report['text_statistics']
            print(f"\nText Length Statistics:")
            print(f"  Min: {stats['min_length']} chars")
            print(f"  Max: {stats['max_length']} chars") 
            print(f"  Average: {stats['avg_length']:.1f} chars")
            print(f"  Median: {stats['median_length']} chars")
        
        print(f"\nValidation Status: {report['validation_status'].upper()}")
        print("=" * 50)
