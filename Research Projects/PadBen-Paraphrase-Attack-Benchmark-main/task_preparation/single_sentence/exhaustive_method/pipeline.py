"""
Main pipeline orchestrator for task data preparation.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from base_processor import AbstractTaskProcessor
from config import TaskPreparationConfig
from data_models import InputSample, ProcessedSample
from utils import DataValidator, StatisticsReporter


logger = logging.getLogger(__name__)


class TaskDataPreparationPipeline:
    """Main pipeline for converting organized JSON data into task-specific datasets.
    
    This pipeline orchestrates the entire process of loading input data,
    validating it, processing it through task-specific processors, and
    generating output files with comprehensive reporting.
    """
    
    def __init__(self, config: TaskPreparationConfig) -> None:
        """Initialize pipeline with configuration.
        
        Args:
            config: Configuration object containing all pipeline settings
        """
        self.config = config
        self.validator = DataValidator(min_text_length=config.min_text_length)
        self.processors: List[AbstractTaskProcessor] = []
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def add_processor(self, processor: AbstractTaskProcessor) -> None:
        """Add a task processor to the pipeline.
        
        Args:
            processor: Task processor to add
        """
        self.processors.append(processor)
        logger.info(f"Added processor: {processor.get_task_name()}")
    
    def load_input_data(self, input_path: Optional[Path] = None) -> List[InputSample]:
        """Load and validate input JSON data.
        
        Args:
            input_path: Path to input JSON file (uses config if None)
            
        Returns:
            List of validated InputSample objects
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If input data is invalid
            json.JSONDecodeError: If JSON format is invalid
        """
        file_path = input_path or self.config.input_file_path
        if not file_path:
            raise ValueError("No input file path specified")
            
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        logger.info(f"Loading input data from {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON format in {file_path}: {e}")
        
        if not isinstance(raw_data, list):
            raise ValueError("Input JSON must be a list of samples")
        
        logger.info(f"Loaded {len(raw_data)} raw samples")
        
        # Validate input data
        is_valid, errors = self.validator.validate_input_data(raw_data)
        if not is_valid:
            error_msg = f"Input validation failed with {len(errors)} errors:\n" + "\n".join(errors[:10])
            if len(errors) > 10:
                error_msg += f"\n... and {len(errors) - 10} more errors"
            raise ValueError(error_msg)
        
        # Convert to InputSample objects
        input_samples = []
        for i, sample_data in enumerate(raw_data):
            try:
                input_sample = InputSample.from_dict(sample_data)
                input_samples.append(input_sample)
            except (KeyError, ValueError) as e:
                raise ValueError(f"Failed to parse sample {i}: {e}")
        
        logger.info(f"Successfully loaded and validated {len(input_samples)} input samples")
        return input_samples
    
    def process_task(self, processor: AbstractTaskProcessor, 
                    input_samples: List[InputSample]) -> List[ProcessedSample]:
        """Process a single task.
        
        Args:
            processor: Task processor to use
            input_samples: Input samples to process
            
        Returns:
            List of processed samples
            
        Raises:
            ValueError: If processing fails or validation fails
        """
        task_name = processor.get_task_name()
        logger.info(f"Processing task: {task_name}")
        
        # Process samples
        try:
            processed_samples = processor.process(input_samples)
        except Exception as e:
            raise ValueError(f"Task processing failed for {task_name}: {e}")
        
        # Validate output
        expected_size = processor.get_expected_output_size(len(input_samples))
        is_valid, errors = self.validator.validate_processed_samples(
            processed_samples, expected_size, self.config.label_balance_tolerance
        )
        
        if not is_valid:
            error_msg = f"Output validation failed for {task_name}:\n" + "\n".join(errors)
            raise ValueError(error_msg)
        
        # Additional processor-specific validation
        if not processor.validate_output(processed_samples):
            raise ValueError(f"Processor-specific validation failed for {task_name}")
        
        logger.info(f"Successfully processed {len(processed_samples)} samples for {task_name}")
        return processed_samples
    
    def save_task_output(self, processor: AbstractTaskProcessor,
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
        except OSError as e:
            raise OSError(f"Failed to write output file {output_path}: {e}")
        
        logger.info(f"Saved {len(processed_samples)} samples to {output_path}")
        return output_path
    
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
        
        logger.info("Starting task data preparation pipeline")
        
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
                    "validation_status": report["validation_status"]
                }
                
                total_output_samples += len(processed_samples)
                
            except Exception as e:
                logger.error(f"Failed to process task {task_name}: {e}")
                results[task_name] = {
                    "error": str(e),
                    "validation_status": "failed"
                }
        
        # Generate overall summary
        summary = {
            "pipeline_status": "completed",
            "input_samples": len(input_samples),
            "total_output_samples": total_output_samples,
            "expansion_ratio": total_output_samples / len(input_samples) if input_samples else 0,
            "tasks_processed": len([r for r in results.values() if "error" not in r]),
            "tasks_failed": len([r for r in results.values() if "error" in r]),
            "task_results": results
        }
        
        # Save overall summary
        if output_dir or self.config.output_dir:
            summary_path = (output_dir or self.config.output_dir) / "pipeline_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info(f"Pipeline summary saved to {summary_path}")
        
        logger.info(f"Pipeline completed: {summary['tasks_processed']} tasks successful, "
                   f"{summary['tasks_failed']} tasks failed")
        
        return summary
