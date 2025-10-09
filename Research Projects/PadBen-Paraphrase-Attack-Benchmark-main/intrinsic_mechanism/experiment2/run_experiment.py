#!/usr/bin/env python3
"""
Simple runner script for the semantic space experiment.

This script provides a streamlined interface to run the complete experiment
with minimal configuration required.

Usage:
    python run_experiment.py [--samples N] [--iterations N] [--device DEVICE]

Example:
    python run_experiment.py --samples 50 --iterations 3 --device cuda
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from config import ExperimentConfig
from main_experiment import SemanticSpaceExperiment
from utils import setup_logging, log_memory_usage


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run semantic space experiment with iterative paraphrasing"
    )
    
    parser.add_argument(
        '--samples', 
        type=int, 
        default=ExperimentConfig.NUM_SAMPLES,
        help=f'Number of samples per type (default: {ExperimentConfig.NUM_SAMPLES})'
    )
    
    parser.add_argument(
        '--iterations', 
        type=int, 
        default=ExperimentConfig.MAX_ITERATIONS,
        help=f'Maximum iterations (default: {ExperimentConfig.MAX_ITERATIONS})'
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
        default=ExperimentConfig.OUTPUT_DIR,
        help=f'Output directory (default: {ExperimentConfig.OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--data-path', 
        type=str, 
        default=ExperimentConfig.DATA_PATH,
        help=f'Path to data file (default: {ExperimentConfig.DATA_PATH})'
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
        '--analysis-only', 
        action='store_true',
        help='Skip data generation and only run analysis on existing data'
    )
    
    return parser.parse_args()


def main():
    """Main function to run the experiment."""
    args = parse_arguments()
    
    # Set up logging
    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting semantic space experiment runner")
    log_memory_usage("Initial ")
    
    # Update configuration with command line arguments
    if args.samples:
        ExperimentConfig.NUM_SAMPLES = args.samples
    if args.iterations:
        ExperimentConfig.MAX_ITERATIONS = args.iterations
    if args.device:
        ExperimentConfig.DEVICE = args.device if args.device != 'auto' else None
    if args.output_dir:
        ExperimentConfig.OUTPUT_DIR = args.output_dir
    if args.data_path:
        ExperimentConfig.DATA_PATH = args.data_path
    
    # Validate configuration
    if not ExperimentConfig.validate_config():
        logger.error("Configuration validation failed")
        sys.exit(1)
    
    logger.info(f"Configuration: {args.samples} samples, {args.iterations} iterations")
    logger.info(f"Data path: {ExperimentConfig.DATA_PATH}")
    logger.info(f"Output directory: {ExperimentConfig.OUTPUT_DIR}")
    logger.info(f"Device: {ExperimentConfig.get_device()}")
    
    if args.analysis_only:
        logger.info("Mode: Analysis only (using existing data)")
    elif args.force_regenerate:
        logger.info("Mode: Force regenerate (will overwrite existing data)")
    else:
        logger.info("Mode: Smart loading (will use existing data if available)")
    
    if args.dry_run:
        logger.info("Dry run completed successfully")
        return
    
    try:
        # Create experiment
        experiment = SemanticSpaceExperiment(
            data_path=ExperimentConfig.DATA_PATH,
            output_dir=ExperimentConfig.OUTPUT_DIR,
            max_iterations=ExperimentConfig.MAX_ITERATIONS,
            num_samples=ExperimentConfig.NUM_SAMPLES,
            device=ExperimentConfig.DEVICE
        )
        
        log_memory_usage("Before experiment: ")
        
        if args.analysis_only:
            logger.info("Running analysis-only mode")
            # Load and sample data for analysis
            experiment.load_and_sample_data()
            
            # Perform PCA analysis on existing data
            pca_results = experiment.perform_pca_analysis()
            
            # Create visualizations
            experiment.create_visualizations(pca_results)
            
        elif args.force_regenerate:
            logger.info("Force regeneration mode - will overwrite existing data")
            # Remove existing data directories
            import shutil
            for iteration in range(1, ExperimentConfig.MAX_ITERATIONS + 1):
                iter_dir = Path(ExperimentConfig.OUTPUT_DIR) / str(iteration)
                if iter_dir.exists():
                    shutil.rmtree(iter_dir)
                    logger.info(f"Removed existing data for iteration {iteration}")
            
            # Run full experiment
            experiment.run_full_experiment()
        else:
            # Normal mode - load existing data if available
            experiment.run_full_experiment()
        
        log_memory_usage("After experiment: ")
        
        logger.info("Experiment completed successfully!")
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Results saved to: {ExperimentConfig.OUTPUT_DIR}")
        print(f"Analysis results available in: {ExperimentConfig.OUTPUT_DIR}/analysis/")
        print("\n📊 Generated Analysis Files:")
        print("  - pca_comprehensive.png")
        print("  - distance_trends.png") 
        print("  - centroid_trajectories.png")
        print("  - hidden_states_distance_table.csv")
        print("  - embeddings_distance_table.csv")
        print("  - distance_tables.json")
        print("  - enhanced_summary_report.txt")
        
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
