"""
Pipeline for dynamically-adjusted task data preparation.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any

from config import DynamicTaskConfig
from data_models import DynamicInputSample, DynamicProcessedSample
from base_processor import AbstractDynamicTaskProcessor


class DynamicTaskPipeline:
    """Pipeline for processing task data with dynamic label balance.
    
    This pipeline processes input JSON data through multiple task processors,
    each selecting one sample per index with configurable label ratios.
    """
    
    def __init__(self, config: DynamicTaskConfig) -> None:
        """Initialize pipeline with configuration.
        
        Args:
            config: Configuration object containing all pipeline settings
        """
        self.config = config
        self.processors: List[AbstractDynamicTaskProcessor] = []
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Ensure output directory exists
        if config.output_dir:
            config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def add_processor(self, processor: AbstractDynamicTaskProcessor) -> None:
        """Add a task processor to the pipeline.
        
        Args:
            processor: Task processor to add
        """
        self.processors.append(processor)
        self.logger.info(f"Added processor: {processor.get_task_name()}")
    
    def load_input_data(self) -> List[DynamicInputSample]:
        """Load and validate input data from JSON file.
        
        Returns:
            List of input samples
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If input data is invalid
        """
        if not self.config.input_file_path or not self.config.input_file_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.config.input_file_path}")
        
        self.logger.info(f"Loading input data from: {self.config.input_file_path}")
        
        try:
            with open(self.config.input_file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in input file: {e}")
        
        if not isinstance(raw_data, list):
            raise ValueError("Input data must be a list of samples")
        
        # Convert to DynamicInputSample objects
        input_samples = []
        for i, sample_data in enumerate(raw_data):
            try:
                sample = DynamicInputSample.from_dict(sample_data)
                input_samples.append(sample)
            except (KeyError, ValueError) as e:
                self.logger.warning(f"Skipping invalid sample at index {i}: {e}")
        
        if not input_samples:
            raise ValueError("No valid input samples found")
        
        self.logger.info(f"Loaded {len(input_samples)} valid input samples")
        return input_samples
    
    def process_task(self, processor: AbstractDynamicTaskProcessor, 
                    input_samples: List[DynamicInputSample]) -> Dict[str, Any]:
        """Process a single task.
        
        Args:
            processor: Task processor to use
            input_samples: Input samples to process
            
        Returns:
            Dictionary containing processing results and statistics
        """
        task_name = processor.get_task_name()
        self.logger.info(f"Processing {task_name}...")
        
        start_time = time.time()
        
        try:
            # Process samples
            processed_samples = processor.process(input_samples)
            
            # Validate output
            if not processor.validate_output(processed_samples):
                raise ValueError("Output validation failed")
            
            # Generate statistics
            stats = processor.get_statistics(processed_samples)
            
            # Save output
            task_dir = self._get_task_directory(processor)
            task_dir.mkdir(parents=True, exist_ok=True)
            output_file = task_dir / processor.get_output_filename()
            self._save_output(processed_samples, output_file)
            
            # Save report
            report_file = output_file.with_suffix('.report.json')
            self._save_report(stats, report_file, task_name)
            
            processing_time = time.time() - start_time
            
            self.logger.info(f"Completed {task_name} in {processing_time:.2f}s")
            
            return {
                "status": "success",
                "output_file": str(output_file),
                "report_file": str(report_file),
                "processing_time": processing_time,
                "statistics": stats
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Failed to process {task_name}: {e}")
            
            return {
                "status": "error",
                "error": str(e),
                "processing_time": processing_time
            }
    
    def _save_output(self, processed_samples: List[DynamicProcessedSample], 
                    output_file: Path) -> None:
        """Save processed samples to JSON file.
        
        Args:
            processed_samples: Processed samples to save
            output_file: Output file path
        """
        # Convert to dictionaries (excluding text_type from output)
        output_data = [sample.to_dict() for sample in processed_samples]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(processed_samples)} samples to {output_file}")
    
    def _save_report(self, stats: Dict[str, Any], report_file: Path, task_name: str) -> None:
        """Save processing report to JSON file.
        
        Args:
            stats: Statistics dictionary
            report_file: Report file path
            task_name: Name of the task
        """
        report = {
            "task_name": task_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "random_seed": self.config.random_seed,
                "label_1_ratio": self.config.label_1_ratio,
                "min_text_length": self.config.min_text_length
            },
            "statistics": stats
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved processing report to {report_file}")
    
    def run(self) -> Dict[str, Any]:
        """Run the complete pipeline.
        
        Returns:
            Dictionary containing pipeline execution summary
        """
        self.logger.info("Starting dynamic task preparation pipeline")
        pipeline_start_time = time.time()
        
        try:
            # Load input data
            input_samples = self.load_input_data()
            
            # Process each task
            task_results = {}
            successful_tasks = 0
            failed_tasks = 0
            
            for processor in self.processors:
                task_name = processor.get_task_name()
                result = self.process_task(processor, input_samples)
                task_results[task_name] = result
                
                if result["status"] == "success":
                    successful_tasks += 1
                else:
                    failed_tasks += 1
            
            # Generate pipeline summary
            pipeline_time = time.time() - pipeline_start_time
            
            summary = {
                "pipeline_status": "completed",
                "input_samples": len(input_samples),
                "total_output_samples": len(input_samples) * successful_tasks,  # One per task
                "tasks_processed": successful_tasks,
                "tasks_failed": failed_tasks,
                "pipeline_time": pipeline_time,
                "config": {
                    "random_seed": self.config.random_seed,
                    "label_1_ratio": self.config.label_1_ratio,
                    "input_file": str(self.config.input_file_path),
                    "output_dir": str(self.config.output_dir)
                },
                "task_results": task_results
            }
            
            # Save pipeline summary
            if self.config.output_dir:
                summary_file = self.config.output_dir / "dynamic_pipeline_summary.json"
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                self.logger.info(f"Saved pipeline summary to {summary_file}")
            
            self.logger.info(f"Pipeline completed in {pipeline_time:.2f}s")
            return summary
            
        except Exception as e:
            pipeline_time = time.time() - pipeline_start_time
            self.logger.error(f"Pipeline failed: {e}")
            
            return {
                "pipeline_status": "failed",
                "error": str(e),
                "pipeline_time": pipeline_time,
                "tasks_processed": 0,
                "tasks_failed": len(self.processors)
            }
    
    def _get_task_directory(self, processor: AbstractDynamicTaskProcessor) -> Path:
        """Get the task-specific directory for a processor.
        
        Args:
            processor: Task processor
            
        Returns:
            Path to task-specific directory
        """
        task_name = processor.get_task_name().lower()
        if "paraphrase source attribution" in task_name:
            task_dir = self.config.output_dir / "task1"
        elif "general text authorship" in task_name:
            task_dir = self.config.output_dir / "task2"
        elif "ai text laundering" in task_name:
            task_dir = self.config.output_dir / "task3"
        elif "iterative paraphrase depth" in task_name:
            task_dir = self.config.output_dir / "task4"
        elif "original vs deep paraphrase attack" in task_name:
            task_dir = self.config.output_dir / "task5"
        else:
            # Fallback to base directory
            task_dir = self.config.output_dir
            
        return task_dir
