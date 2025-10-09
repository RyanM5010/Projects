"""
Utility functions for sentence pair task preparation pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List
from data_models import InputSample, ProcessedSample, SentencePairSample


class DataValidator:
    """Utility class for validating input data and processed samples."""
    
    @staticmethod
    def validate_input_samples(samples: List[InputSample], min_length: int = 5) -> List[InputSample]:
        """Validate input samples and filter out invalid ones.
        
        Args:
            samples: List of input samples to validate
            min_length: Minimum text length for filtering
            
        Returns:
            List of valid input samples
            
        Raises:
            ValueError: If validation fails
        """
        if not samples:
            raise ValueError("Input samples list cannot be empty")
        
        valid_samples = []
        for sample in samples:
            # Check if all required text fields are present and long enough
            text_fields = [sample.type1, sample.type2, sample.type3, sample.type4, sample.type5_1st, sample.type5_3rd]
            if all(text and len(text.strip()) >= min_length for text in text_fields):
                valid_samples.append(sample)
            else:
                logging.warning(f"Skipping sample {sample.idx} due to insufficient text length")
        
        if not valid_samples:
            raise ValueError("No valid samples found after filtering")
        
        logging.info(f"Validated {len(valid_samples)} out of {len(samples)} input samples")
        return valid_samples
    
    @staticmethod
    def validate_processed_samples(samples: List[ProcessedSample]) -> None:
        """Validate processed samples.
        
        Args:
            samples: List of processed samples to validate
            
        Raises:
            ValueError: If validation fails
        """
        if not samples:
            raise ValueError("Processed samples list cannot be empty")
        
        for sample in samples:
            if not isinstance(sample, SentencePairSample):
                raise ValueError(f"Invalid sample type: {type(sample)}")
            
            if len(sample.sentence_pair) != 2:
                raise ValueError(f"Invalid sentence pair length: {len(sample.sentence_pair)}")
            
            if len(sample.label_pair) != 2:
                raise ValueError(f"Invalid label pair length: {len(sample.label_pair)}")
            
            if not all(isinstance(label, int) and label in [0, 1] for label in sample.label_pair):
                raise ValueError(f"Invalid labels in label pair: {sample.label_pair}")


class StatisticsReporter:
    """Utility class for generating and reporting statistics."""
    
    @staticmethod
    def generate_task_report(task_name: str, input_samples: List[InputSample], 
                           processed_samples: List[ProcessedSample]) -> Dict[str, Any]:
        """Generate a comprehensive report for a task.
        
        Args:
            task_name: Name of the task
            input_samples: List of input samples
            processed_samples: List of processed samples
            
        Returns:
            Dictionary containing task statistics
        """
        if not processed_samples:
            return {"error": "No processed samples to report on"}
        
        # Calculate label distribution
        label_0_count = sum(1 for sample in processed_samples if sample.label == 0)
        label_1_count = sum(1 for sample in processed_samples if sample.label == 1)
        total_samples = len(processed_samples)
        
        # Calculate text length statistics
        all_lengths = []
        for sample in processed_samples:
            if isinstance(sample, SentencePairSample):
                for sentence in sample.sentence_pair:
                    all_lengths.append(len(sentence))
        
        return {
            "task_name": task_name,
            "input_samples": len(input_samples),
            "output_samples": total_samples,
            "expansion_ratio": total_samples / len(input_samples) if input_samples else 0,
            "label_distribution": {
                "label_0_count": label_0_count,
                "label_1_count": label_1_count,
                "label_0_ratio": label_0_count / total_samples if total_samples > 0 else 0,
                "label_1_ratio": label_1_count / total_samples if total_samples > 0 else 0
            },
            "text_length_statistics": {
                "min_length": min(all_lengths) if all_lengths else 0,
                "max_length": max(all_lengths) if all_lengths else 0,
                "avg_length": sum(all_lengths) / len(all_lengths) if all_lengths else 0,
                "median_length": sorted(all_lengths)[len(all_lengths) // 2] if all_lengths else 0
            },
            "validation_status": "PASSED"
        }
    
    @staticmethod
    def save_report(report: Dict[str, Any], output_path: Path) -> None:
        """Save report to JSON file.
        
        Args:
            report: Report dictionary to save
            output_path: Path to save the report
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Report saved to {output_path}")
    
    @staticmethod
    def print_summary(report: Dict[str, Any]) -> None:
        """Print a formatted summary of the report.
        
        Args:
            report: Report dictionary to print
        """
        if "error" in report:
            print(f"Error: {report['error']}")
            return
        
        print(f"\n=== {report['task_name']} - Processing Summary ===")
        print(f"Input samples: {report['input_samples']:,}")
        print(f"Output samples: {report['output_samples']:,}")
        print(f"Expansion ratio: {report['expansion_ratio']:.1f}x")
        
        print(f"\nLabel Distribution:")
        print(f"  Label 0: {report['label_distribution']['label_0_count']:,} ({report['label_distribution']['label_0_ratio']:.1%})")
        print(f"  Label 1: {report['label_distribution']['label_1_count']:,} ({report['label_distribution']['label_1_ratio']:.1%})")
        
        print(f"\nText Length Statistics:")
        print(f"  Min: {report['text_length_statistics']['min_length']} chars")
        print(f"  Max: {report['text_length_statistics']['max_length']} chars")
        print(f"  Average: {report['text_length_statistics']['avg_length']:.1f} chars")
        print(f"  Median: {report['text_length_statistics']['median_length']} chars")
        
        print(f"\nValidation Status: {report['validation_status']}")
        print("=" * 50)
