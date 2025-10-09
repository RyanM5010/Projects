"""
Main script for running task data preparation pipeline.

This script demonstrates how to use the TaskDataPreparationPipeline
to convert organized JSON data into task-specific datasets.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from config import TaskPreparationConfig
from pipeline import TaskDataPreparationPipeline
from task_processors import Task1Processor, Task2Processor, Task3Processor, Task4Processor, Task5Processor


def setup_argparser() -> argparse.ArgumentParser:
    """Set up command line argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Convert organized JSON data into task-specific datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process TASK1 with default settings
  python -m task_preparation.main --input data/merged_data.json --output results/

  # Process with custom random seed and validation tolerance
  python -m task_preparation.main --input data/merged_data.json --output results/ \\
                                  --seed 123 --tolerance 0.01

  # Enable debug logging
  python -m task_preparation.main --input data/merged_data.json --output results/ \\
                                  --log-level DEBUG
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to input JSON file containing organized data"
    )
    
    parser.add_argument(
        "--output", "-o", 
        type=Path,
        required=True,
        help="Output directory for generated task files"
    )
    
    # Optional arguments
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducible results (default: 42)"
    )
    
    parser.add_argument(
        "--tolerance", "-t",
        type=float,
        default=0.02,
        help="Label balance tolerance (default: 0.02 = 2%%)"
    )
    
    parser.add_argument(
        "--min-length",
        type=int,
        default=5,
        help="Minimum text length in characters (default: 5)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for processing (default: 1000)"
    )
    
    # Task selection arguments
    parser.add_argument(
        "--task1",
        action="store_true",
        default=True,
        help="Enable TASK1: Paraphrase Source Attribution without Context"
    )
    
    parser.add_argument(
        "--task2",
        action="store_true", 
        default=True,
        help="Enable TASK2: General Text Authorship Detection"
    )
    
    parser.add_argument(
        "--task3",
        action="store_true",
        default=True,
        help="Enable TASK3: AI Text Laundering Detection"
    )
    
    parser.add_argument(
        "--task4",
        action="store_true",
        default=True,
        help="Enable TASK4: Iterative Paraphrase Depth Detection"
    )
    
    parser.add_argument(
        "--task5",
        action="store_true",
        default=True,
        help="Enable TASK5: Original vs Deep Paraphrase Attack Detection"
    )
    
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Enable all available tasks"
    )
    
    return parser


def main(input_path: Path, output_dir: Path, 
         random_seed: int = 42,
         tolerance: float = 0.02,
         min_length: int = 5,
         log_level: str = "INFO",
         batch_size: int = 1000,
         enable_task1: bool = True,
         enable_task2: bool = True,
         enable_task3: bool = True,
         enable_task4: bool = True,
         enable_task5: bool = True) -> Optional[int]:
    """Main function to run the task preparation pipeline.
    
    Args:
        input_path: Path to input JSON file
        output_dir: Output directory for results
        random_seed: Random seed for reproducibility
        tolerance: Label balance tolerance
        min_length: Minimum text length
        log_level: Logging level
        batch_size: Processing batch size
        enable_task1: Whether to enable TASK1
        enable_task2: Whether to enable TASK2
        enable_task3: Whether to enable TASK3
        enable_task4: Whether to enable TASK4
        enable_task5: Whether to enable TASK5
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Create configuration
        config = TaskPreparationConfig(
            random_seed=random_seed,
            input_file_path=input_path,
            output_dir=output_dir / "single-sentence" / "exhaustive_method",
            label_balance_tolerance=tolerance,
            min_text_length=min_length,
            log_level=log_level,
            batch_size=batch_size,
            enable_task1=enable_task1,
            enable_task2=enable_task2,
            enable_task3=enable_task3,
            enable_task4=enable_task4,
            enable_task5=enable_task5
        )
        
        # Initialize pipeline
        pipeline = TaskDataPreparationPipeline(config)
        
        # Add processors based on configuration
        if enable_task1:
            task1_processor = Task1Processor(random_seed=random_seed)
            pipeline.add_processor(task1_processor)
            
        if enable_task2:
            task2_processor = Task2Processor(random_seed=random_seed)
            pipeline.add_processor(task2_processor)
            
        if enable_task3:
            task3_processor = Task3Processor(random_seed=random_seed)
            pipeline.add_processor(task3_processor)
            
        if enable_task4:
            task4_processor = Task4Processor(random_seed=random_seed)
            pipeline.add_processor(task4_processor)
            
        if enable_task5:
            task5_processor = Task5Processor(random_seed=random_seed)
            pipeline.add_processor(task5_processor)
        
        # Run pipeline
        summary = pipeline.run()
        
        # Print final summary
        print(f"\n{'='*60}")
        print("PIPELINE EXECUTION SUMMARY")
        print(f"{'='*60}")
        print(f"Status: {summary['pipeline_status'].upper()}")
        print(f"Input samples: {summary['input_samples']:,}")
        print(f"Total output samples: {summary['total_output_samples']:,}")
        print(f"Expansion ratio: {summary['expansion_ratio']:.1f}x")
        print(f"Tasks successful: {summary['tasks_processed']}")
        print(f"Tasks failed: {summary['tasks_failed']}")
        
        if summary['tasks_failed'] > 0:
            print(f"\nFailed tasks:")
            for task_name, result in summary['task_results'].items():
                if 'error' in result:
                    print(f"  - {task_name}: {result['error']}")
        
        print(f"\nOutput directory: {output_dir}")
        print(f"{'='*60}")
        
        return 0 if summary['tasks_failed'] == 0 else 1
        
    except Exception as e:
        print(f"Pipeline execution failed: {e}", file=sys.stderr)
        return 1


def cli_main() -> None:
    """Command line interface main function."""
    parser = setup_argparser()
    args = parser.parse_args()
    
    # Validate input file exists
    if not args.input.exists():
        print(f"Error: Input file does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Determine which tasks to enable
    enable_task1 = args.task1 or args.all_tasks
    enable_task2 = args.task2 or args.all_tasks
    enable_task3 = args.task3 or args.all_tasks
    enable_task4 = args.task4 or args.all_tasks
    enable_task5 = args.task5 or args.all_tasks
    
    # Ensure at least one task is enabled
    if not (enable_task1 or enable_task2 or enable_task3 or enable_task4 or enable_task5):
        print("Error: At least one task must be enabled", file=sys.stderr)
        sys.exit(1)
    
    # Run main function
    exit_code = main(
        input_path=args.input,
        output_dir=args.output,
        random_seed=args.seed,
        tolerance=args.tolerance,
        min_length=args.min_length,
        log_level=args.log_level,
        batch_size=args.batch_size,
        enable_task1=enable_task1,
        enable_task2=enable_task2,
        enable_task3=enable_task3,
        enable_task4=enable_task4,
        enable_task5=enable_task5
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    cli_main()
