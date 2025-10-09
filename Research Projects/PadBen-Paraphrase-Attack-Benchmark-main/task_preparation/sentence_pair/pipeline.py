"""
Pipeline for sentence pair task preparation.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from base_processor import AbstractSentencePairProcessor
from config import SentencePairTaskConfig
from data_models import InputSample, ProcessedSample
from utils import DataValidator, StatisticsReporter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentencePairTaskPipeline:
    """Pipeline for processing sentence pair tasks."""
    
    def __init__(self, config: SentencePairTaskConfig) -> None:
        """Initialize the pipeline with configuration.
        
        Args:
            config: Configuration for the pipeline
        """
        self.config = config
        self.processors: List[AbstractSentencePairProcessor] = []
        
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def add_processor(self, processor: AbstractSentencePairProcessor) -> None:
        """Add a task processor to the pipeline.
        
        Args:
            processor: Task processor to add
        """
        self.processors.append(processor)
        logger.info(f"Added processor: {processor.get_task_name()}")
    
    def load_input_data(self, input_path: Optional[Path] = None) -> List[InputSample]:
        """Load input data from JSON file.
        
        Args:
            input_path: Path to input JSON file (uses config if None)
            
        Returns:
            List of input samples
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If data loading fails
        """
        path = input_path or self.config.input_path
        
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        
        logger.info(f"Loading input data from {path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            logger.info(f"Loaded {len(raw_data)} raw samples")
            
            # Convert to InputSample objects
            input_samples = []
            for item in raw_data:
                try:
                    sample = InputSample.from_dict(item)
                    input_samples.append(sample)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping invalid sample: {e}")
                    continue
            
            # Validate samples
            input_samples = DataValidator.validate_input_samples(
                input_samples, self.config.min_length
            )
            
            logger.info(f"Successfully loaded and validated {len(input_samples)} input samples")
            return input_samples
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in input file: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load input data: {e}")
    
    def process_task(self, processor: AbstractSentencePairProcessor, 
                    input_samples: List[InputSample]) -> List[ProcessedSample]:
        """Process a single task.
        
        Args:
            processor: Task processor to use
            input_samples: Input samples to process
            
        Returns:
            List of processed samples
            
        Raises:
            ValueError: If processing fails
        """
        task_name = processor.get_task_name()
        logger.info(f"Processing task: {task_name}")
        
        try:
            processed_samples = processor.process(input_samples)
            
            # Validate processed samples
            DataValidator.validate_processed_samples(processed_samples)
            
            expected_size = processor.get_expected_output_size(len(input_samples))
            if len(processed_samples) != expected_size:
                logger.warning(f"Unexpected output size: got {len(processed_samples)}, expected {expected_size}")
            
            logger.info(f"Successfully processed {len(processed_samples)} samples for {task_name}")
            return processed_samples
            
        except Exception as e:
            logger.error(f"Failed to process task {task_name}: {e}")
            raise ValueError(f"Task processing failed: {e}")
    
    def save_task_output(self, processor: AbstractSentencePairProcessor,
                        processed_samples: List[ProcessedSample],
                        output_dir: Optional[Path] = None) -> Path:
        """Save processed samples to JSON file.
        
        Args:
            processor: Task processor that generated the samples
            processed_samples: Samples to save
            output_dir: Output directory (uses config if None)
            
        Returns:
            Path to saved file
            
        Raises:
            OSError: If file cannot be written
        """
        base_dir = output_dir or self.config.output_dir
        if not base_dir:
            # Default to data/tasks/ structure
            base_dir = Path("data/tasks")
            
        # Create task-specific directory based on processor type
        task_name = processor.get_task_name().lower()
        if "paraphrase source attribution" in task_name:
            task_dir = base_dir / "task1"
        elif "general text authorship" in task_name:
            task_dir = base_dir / "task2"
        elif "ai text laundering" in task_name:
            task_dir = base_dir / "task3"
        elif "iterative paraphrase depth" in task_name:
            task_dir = base_dir / "task4"
        elif "original vs deep paraphrase attack" in task_name:
            task_dir = base_dir / "task5"
        else:
            # Fallback to base directory
            task_dir = base_dir
            
        task_dir.mkdir(parents=True, exist_ok=True)
        
        filename = processor.get_output_filename()
        output_path = task_dir / filename
        
        # Convert samples to dictionaries
        output_data = [sample.to_dict() for sample in processed_samples]
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(processed_samples)} samples to {output_path}")
            return output_path
            
        except OSError as e:
            logger.error(f"Failed to save output to {output_path}: {e}")
            raise OSError(f"Cannot write output file: {e}")
    
    def run(self, input_path: Optional[Path] = None, 
            output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Run the complete pipeline.
        
        Args:
            input_path: Path to input JSON file (uses config if None)
            output_dir: Output directory (uses config if None)
            
        Returns:
            Dictionary containing pipeline execution summary
            
        Raises:
            ValueError: If pipeline execution fails
        """
        if not self.processors:
            raise ValueError("No processors added to pipeline")
        
        logger.info("Starting sentence pair task data preparation pipeline")
        
        # Load input data
        input_samples = self.load_input_data(input_path)
        
        # Process each task
        results = {}
        total_output_samples = 0
        
        for processor in self.processors:
            task_name = processor.get_task_name()
            
            try:
                # Process task
                processed_samples = self.process_task(processor, input_samples)
                
                # Save output
                output_path = self.save_task_output(processor, processed_samples, output_dir)
                
                # Generate report
                report = StatisticsReporter.generate_task_report(
                    task_name, input_samples, processed_samples
                )
                
                # Save individual task report
                report_path = output_path.parent / f"{output_path.stem}_report.json"
                StatisticsReporter.save_report(report, report_path)
                
                # Print summary
                StatisticsReporter.print_summary(report)
                
                # Store results
                results[task_name] = {
                    "output_file": str(output_path),
                    "report_file": str(report_path),
                    "sample_count": len(processed_samples),
                    "status": "success"
                }
                
                total_output_samples += len(processed_samples)
                
            except Exception as e:
                logger.error(f"Task {task_name} failed: {e}")
                results[task_name] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        # Generate pipeline summary
        successful_tasks = sum(1 for r in results.values() if r["status"] == "success")
        failed_tasks = len(results) - successful_tasks
        
        summary = {
            "pipeline_status": "completed" if failed_tasks == 0 else "partial_failure",
            "input_samples": len(input_samples),
            "total_output_samples": total_output_samples,
            "expansion_ratio": total_output_samples / len(input_samples) if input_samples else 0,
            "tasks_successful": successful_tasks,
            "tasks_failed": failed_tasks,
            "task_results": results
        }
        
        # Save pipeline summary
        summary_path = (output_dir or self.config.output_dir) / "pipeline_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Pipeline summary saved to {summary_path}")
        
        # Print final summary
        print("\n" + "=" * 60)
        print("SENTENCE PAIR PIPELINE EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Status: {summary['pipeline_status'].upper()}")
        print(f"Input samples: {summary['input_samples']:,}")
        print(f"Total output samples: {summary['total_output_samples']:,}")
        print(f"Expansion ratio: {summary['expansion_ratio']:.1f}x")
        print(f"Tasks successful: {summary['tasks_successful']}")
        print(f"Tasks failed: {summary['tasks_failed']}")
        print(f"\nOutput directory: {output_dir or self.config.output_dir}")
        print("=" * 60)
        
        return summary
