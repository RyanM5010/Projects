#!/usr/bin/env python3
"""
Runner script for Extended Semantic Space Experiment

This script provides a streamlined interface to run the extended experiment
with 10 iterations to better observe semantic drift trends.

Usage:
    python run_extended_experiment.py [--iterations N] [--samples N] [--device DEVICE]

Example:
    python run_extended_experiment.py --iterations 10 --samples 100 --device cuda
    python run_extended_experiment.py --iterations 15 --samples 50
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import ExperimentConfig
from extended_experiment import ExtendedSemanticSpaceExperiment
from utils import setup_logging, log_memory_usage


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run extended semantic space experiment with more iterations"
    )
    
    parser.add_argument(
        '--iterations', 
        type=int, 
        default=10,
        help='Number of iterations (default: 10)'
    )
    
    parser.add_argument(
        '--samples', 
        type=int, 
        default=ExperimentConfig.NUM_SAMPLES,
        help=f'Number of samples per type (default: {ExperimentConfig.NUM_SAMPLES})'
    )
    
    parser.add_argument(
        '--device', 
        type=str, 
        default=None,
        choices=['cuda', 'cpu', 'auto'],
        help='Device to use for inference (default: auto)'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='../output',
        help='Output directory relative to semantic_space/ (default: ../output)'
    )
    
    parser.add_argument(
        '--data-path', 
        type=str, 
        default='../../data/test/final_generated_data.json',
        help='Path to data file (default: ../../data/test/final_generated_data.json)'
    )
    
    parser.add_argument(
        '--log-file', 
        type=str, 
        default=None,
        help='Path to log file (default: console only)'
    )
    
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Validate configuration without running experiment'
    )
    
    parser.add_argument(
        '--force-regenerate', 
        action='store_true',
        help='Force regeneration of all data (ignore existing files)'
    )
    
    parser.add_argument(
        '--use-foundation', 
        action='store_true',
        default=True,
        help='Use existing 1-5 iteration data as foundation (default: True)'
    )
    
    parser.add_argument(
        '--no-foundation', 
        action='store_true',
        help='Do not use existing 1-5 iteration data, start fresh'
    )
    
    return parser.parse_args()


def main():
    """Main function to run the extended experiment."""
    args = parse_arguments()
    
    # Set up logging
    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting extended semantic space experiment runner")
    log_memory_usage("Initial ")
    
    # Validate arguments
    if args.iterations < 1:
        logger.error("Number of iterations must be at least 1")
        sys.exit(1)
    
    if args.samples < 1:
        logger.error("Number of samples must be at least 1")
        sys.exit(1)
    
    logger.info(f"Configuration: {args.samples} samples, {args.iterations} iterations")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device: {args.device or 'auto'}")
    
    if args.force_regenerate:
        logger.info("Mode: Force regenerate (will overwrite existing data)")
    else:
        logger.info("Mode: Smart loading (will use existing data if available)")
    
    if args.no_foundation:
        logger.info("Foundation: Will NOT use existing 1-5 iteration data")
    else:
        logger.info("Foundation: Will use existing 1-5 iteration data if available")
    
    if args.dry_run:
        logger.info("Dry run completed successfully")
        return
    
    try:
        # Handle force regeneration
        if args.force_regenerate:
            logger.info("Force regeneration mode - removing existing data")
            import shutil
            output_path = Path(args.output_dir)
            if output_path.exists():
                for iteration_dir in output_path.glob('[0-9]*'):
                    if iteration_dir.is_dir():
                        shutil.rmtree(iteration_dir)
                        logger.info(f"Removed existing data for iteration {iteration_dir.name}")
        
        # Create and run extended experiment
        experiment = ExtendedSemanticSpaceExperiment(
            data_path=args.data_path,
            output_dir=args.output_dir,
            max_iterations=args.iterations,
            num_samples=args.samples,
            device=args.device if args.device != 'auto' else None,
            use_foundation=not args.no_foundation  # Use foundation unless explicitly disabled
        )
        
        log_memory_usage("Before extended experiment: ")
        experiment.run_full_extended_experiment()
        log_memory_usage("After extended experiment: ")
        
        logger.info("Extended experiment completed successfully!")
        print("\n" + "="*70)
        print("EXTENDED EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Results saved to: {args.output_dir}")
        print(f"Iterations completed: {args.iterations}")
        print(f"Samples per type: {args.samples}")
        print("\n📊 Data Structure:")
        print(f"  semantic_space/output/{{1-{args.iterations}}}/{{type1,type2}}/")
        print("    ├── texts.json")
        print("    ├── hidden_states.npy  (from Qwen)")
        print("    └── embeddings.npy     (from BGE-M3)")
        print("\n🔍 Next Steps:")
        print("  - Use the extended data for trend analysis")
        print("  - Run analysis tools with --iterations", args.iterations)
        print("  - Compare trends between 5 and", args.iterations, "iterations")
        print("="*70)
        
    except KeyboardInterrupt:
        logger.info("Extended experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Extended experiment failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
