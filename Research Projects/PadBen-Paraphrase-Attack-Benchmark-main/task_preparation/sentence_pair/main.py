#!/usr/bin/env python3
"""
Main script for sentence pair task data preparation pipeline.

This script processes organized JSON data to create sentence pair classification tasks
where models need to determine which sentence in a pair is machine-generated vs human-written.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from config import SentencePairTaskConfig
from pipeline import SentencePairTaskPipeline
from task_processors import (
    Task1SentencePairProcessor,
    Task2SentencePairProcessor,
    Task3SentencePairProcessor,
    Task4SentencePairProcessor,
    Task5SentencePairProcessor
)


def setup_logging(log_level: str) -> None:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main(input_path: Path, output_dir: Path,
         random_seed: int = 42,
         min_length: int = 5,
         log_level: str = "INFO",
         batch_size: int = 1000,
         enable_task1: bool = True,
         enable_task2: bool = True,
         enable_task3: bool = True,
         enable_task4: bool = True,
         enable_task5: bool = True) -> Optional[int]:
    """Main function to run the sentence pair task preparation pipeline.
    
    Args:
        input_path: Path to input JSON file
        output_dir: Output directory for processed tasks
        random_seed: Random seed for reproducibility
        min_length: Minimum text length for filtering
        log_level: Logging level
        batch_size: Batch size for processing
        enable_task1: Whether to enable Task1 processing
        enable_task2: Whether to enable Task2 processing
        enable_task3: Whether to enable Task3 processing
        enable_task4: Whether to enable Task4 processing
        enable_task5: Whether to enable Task5 processing
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    setup_logging(log_level)
    
    # Create configuration
    config = SentencePairTaskConfig(
        input_path=input_path,
        output_dir=output_dir / "sentence-pair",
        random_seed=random_seed,
        min_length=min_length,
        batch_size=batch_size,
        enable_task1=enable_task1,
        enable_task2=enable_task2,
        enable_task3=enable_task3,
        enable_task4=enable_task4,
        enable_task5=enable_task5
    )
    
    # Create pipeline
    pipeline = SentencePairTaskPipeline(config)
    
    # Add processors based on configuration
    if enable_task1:
        task1_processor = Task1SentencePairProcessor(random_seed=random_seed)
        pipeline.add_processor(task1_processor)
    
    if enable_task2:
        task2_processor = Task2SentencePairProcessor(random_seed=random_seed)
        pipeline.add_processor(task2_processor)
    
    if enable_task3:
        task3_processor = Task3SentencePairProcessor(random_seed=random_seed)
        pipeline.add_processor(task3_processor)
    
    if enable_task4:
        task4_processor = Task4SentencePairProcessor(random_seed=random_seed)
        pipeline.add_processor(task4_processor)
    
    if enable_task5:
        task5_processor = Task5SentencePairProcessor(random_seed=random_seed)
        pipeline.add_processor(task5_processor)
    
    # Run pipeline
    try:
        summary = pipeline.run()
        
        if summary["pipeline_status"] == "completed":
            print("\n✅ Pipeline completed successfully!")
            return 0
        else:
            print(f"\n⚠️  Pipeline completed with {summary['tasks_failed']} failed tasks")
            return 1
            
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sentence Pair Task Data Preparation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all tasks
  python main.py --input data.json --output results/ --all-tasks
  
  # Process specific tasks
  python main.py --input data.json --output results/ --task1 --task2
  
  # Process with custom parameters
  python main.py --input data.json --output results/ --task1 --seed 123 --min-length 10
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input JSON file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for processed tasks"
    )
    
    # Task selection arguments
    parser.add_argument(
        "--task1",
        action="store_true",
        help="Enable TASK1: Paraphrase Source Attribution without Context (Sentence Pair)"
    )
    parser.add_argument(
        "--task2",
        action="store_true",
        help="Enable TASK2: General Text Authorship Detection (Sentence Pair)"
    )
    parser.add_argument(
        "--task3",
        action="store_true",
        help="Enable TASK3: AI Text Laundering Detection (Sentence Pair)"
    )
    parser.add_argument(
        "--task4",
        action="store_true",
        help="Enable TASK4: Iterative Paraphrase Depth Detection (Sentence Pair)"
    )
    parser.add_argument(
        "--task5",
        action="store_true",
        help="Enable TASK5: Original vs Deep Paraphrase Attack Detection (Sentence Pair)"
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Enable all tasks (default if no specific tasks are selected)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=5,
        help="Minimum text length for filtering (default: 5)"
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
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.input.exists():
        print(f"Error: Input file does not exist: {args.input}", file=sys.stderr)
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
    
    # Check if at least one task is enabled
    if not (enable_task1 or enable_task2 or enable_task3 or enable_task4 or enable_task5):
        print("Error: At least one task must be enabled", file=sys.stderr)
        sys.exit(1)
    
    # Run main function
    exit_code = main(
        input_path=args.input,
        output_dir=args.output,
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
