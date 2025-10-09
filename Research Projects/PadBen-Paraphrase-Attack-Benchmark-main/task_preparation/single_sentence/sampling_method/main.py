"""
Main script for running dynamically-adjusted task data preparation pipeline.

This script demonstrates how to use the DynamicTaskPipeline to convert organized JSON data
into task-specific datasets with configurable label balance ratios.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from config import DynamicTaskConfig
from pipeline import DynamicTaskPipeline
from task_processors import (
    DynamicTask1Processor,
    DynamicTask2Processor, 
    DynamicTask3Processor,
    DynamicTask4Processor,
    DynamicTask5Processor
)


def setup_argparser() -> argparse.ArgumentParser:
    """Set up command line argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Convert organized JSON data into task-specific datasets with dynamic label balance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all tasks with 50% label 1 ratio (balanced)
  python -m task_preparation.dynamically-adjusted_task_prep.main \\
      --input data/test/final_generated_data.json --output results/

  # Process with 30% label 1 ratio (more label 0)
  python -m task_preparation.dynamically-adjusted_task_prep.main \\
      --input data/test/final_generated_data.json --output results/ \\
      --label-1-ratio 0.3

  # Process with 80% label 1 ratio (more label 1)
  python -m task_preparation.dynamically-adjusted_task_prep.main \\
      --input data/test/final_generated_data.json --output results/ \\
      --label-1-ratio 0.8

  # Process only specific tasks
  python -m task_preparation.dynamically-adjusted_task_prep.main \\
      --input data/test/final_generated_data.json --output results/ \\
      --task1 --task2 --label-1-ratio 0.6

  # Enable debug logging
  python -m task_preparation.dynamically-adjusted_task_prep.main \\
      --input data/test/final_generated_data.json --output results/ \\
      --log-level DEBUG --label-1-ratio 0.5
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
    
    # Dynamic label balance argument (main feature)
    parser.add_argument(
        "--label-1-ratio", "--coefficient",
        type=float,
        default=0.5,
        help="Ratio of label 1 samples (0.0 to 1.0). E.g., 0.3 = 30%% label 1, 70%% label 0; "
             "0.8 = 80%% label 1, 20%% label 0 (default: 0.5 for balanced)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducible results (default: 42)"
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
        help="Enable TASK1: Paraphrase Source Attribution without Context"
    )
    
    parser.add_argument(
        "--task2",
        action="store_true", 
        help="Enable TASK2: General Text Authorship Detection"
    )
    
    parser.add_argument(
        "--task3",
        action="store_true",
        help="Enable TASK3: AI Text Laundering Detection"
    )
    
    parser.add_argument(
        "--task4",
        action="store_true",
        help="Enable TASK4: Iterative Paraphrase Depth Detection"
    )
    
    parser.add_argument(
        "--task5",
        action="store_true",
        help="Enable TASK5: Original vs Deep Paraphrase Attack Detection"
    )
    
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        default=True,
        help="Enable all available tasks (default: True if no specific tasks selected)"
    )
    
    return parser


def main(input_path: Path, output_dir: Path, 
         label_1_ratio: float = 0.5,
         random_seed: int = 42,
         min_length: int = 5,
         log_level: str = "INFO",
         batch_size: int = 1000,
         enable_task1: bool = True,
         enable_task2: bool = True,
         enable_task3: bool = True,
         enable_task4: bool = True,
         enable_task5: bool = True) -> Optional[int]:
    """Main function to run the dynamic task preparation pipeline.
    
    Args:
        input_path: Path to input JSON file
        output_dir: Output directory for results
        label_1_ratio: Ratio of label 1 samples (0.0 to 1.0)
        random_seed: Random seed for reproducibility
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
        # Determine output subdirectory based on label ratio
        if label_1_ratio == 0.3:
            ratio_dir = "30-70"
        elif label_1_ratio == 0.8:
            ratio_dir = "80-20"
        else:
            ratio_dir = "50-50"
        
        config = DynamicTaskConfig(
            random_seed=random_seed,
            input_file_path=input_path,
            output_dir=output_dir / "single-sentence" / "sampling_method" / ratio_dir,
            label_1_ratio=label_1_ratio,
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
        pipeline = DynamicTaskPipeline(config)
        
        # Add processors based on configuration
        if enable_task1:
            task1_processor = DynamicTask1Processor(
                random_seed=random_seed, 
                label_1_ratio=label_1_ratio
            )
            pipeline.add_processor(task1_processor)
            
        if enable_task2:
            task2_processor = DynamicTask2Processor(
                random_seed=random_seed, 
                label_1_ratio=label_1_ratio
            )
            pipeline.add_processor(task2_processor)
            
        if enable_task3:
            task3_processor = DynamicTask3Processor(
                random_seed=random_seed, 
                label_1_ratio=label_1_ratio
            )
            pipeline.add_processor(task3_processor)
            
        if enable_task4:
            task4_processor = DynamicTask4Processor(
                random_seed=random_seed, 
                label_1_ratio=label_1_ratio
            )
            pipeline.add_processor(task4_processor)
            
        if enable_task5:
            task5_processor = DynamicTask5Processor(
                random_seed=random_seed, 
                label_1_ratio=label_1_ratio
            )
            pipeline.add_processor(task5_processor)
        
        # Run pipeline
        summary = pipeline.run()
        
        # Print final summary
        print(f"\n{'='*70}")
        print("DYNAMIC PIPELINE EXECUTION SUMMARY")
        print(f"{'='*70}")
        print(f"Status: {summary['pipeline_status'].upper()}")
        print(f"Input samples: {summary['input_samples']:,}")
        print(f"Label 1 ratio: {label_1_ratio:.1%} (Label 0 ratio: {1-label_1_ratio:.1%})")
        print(f"Output samples per task: {summary['input_samples']:,}")
        print(f"Tasks successful: {summary['tasks_processed']}")
        print(f"Tasks failed: {summary['tasks_failed']}")
        
        if summary['tasks_failed'] > 0:
            print(f"\nFailed tasks:")
            for task_name, result in summary['task_results'].items():
                if 'error' in result:
                    print(f"  - {task_name}: {result['error']}")
        
        # Print task-specific statistics
        print(f"\nTask Statistics:")
        for task_name, result in summary['task_results'].items():
            if result.get('status') == 'success':
                stats = result.get('statistics', {})
                actual_ratio = stats.get('label_1_ratio', 0.0)
                print(f"  {task_name}:")
                print(f"    - Actual label 1 ratio: {actual_ratio:.1%}")
                print(f"    - Label 0 count: {stats.get('label_0_count', 0):,}")
                print(f"    - Label 1 count: {stats.get('label_1_count', 0):,}")
                if 'text_type_counts' in stats:
                    print(f"    - Text type usage: {stats['text_type_counts']}")
        
        print(f"\nOutput directory: {output_dir}")
        print(f"{'='*70}")
        
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
    
    # Validate label_1_ratio
    if not 0.0 <= args.label_1_ratio <= 1.0:
        print(f"Error: label-1-ratio must be between 0.0 and 1.0, got {args.label_1_ratio}", file=sys.stderr)
        sys.exit(1)
    
    # Determine which tasks to enable
    if args.task1 or args.task2 or args.task3 or args.task4 or args.task5:
        # Specific tasks selected
        enable_task1 = args.task1
        enable_task2 = args.task2
        enable_task3 = args.task3
        enable_task4 = args.task4
        enable_task5 = args.task5
    else:
        # No specific tasks selected, enable all (default behavior)
        enable_task1 = True
        enable_task2 = True
        enable_task3 = True
        enable_task4 = True
        enable_task5 = True
    
    # Ensure at least one task is enabled
    if not (enable_task1 or enable_task2 or enable_task3 or enable_task4 or enable_task5):
        print("Error: At least one task must be enabled", file=sys.stderr)
        sys.exit(1)
    
    # Run main function
    exit_code = main(
        input_path=args.input,
        output_dir=args.output,
        label_1_ratio=args.label_1_ratio,
        random_seed=args.seed,
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
