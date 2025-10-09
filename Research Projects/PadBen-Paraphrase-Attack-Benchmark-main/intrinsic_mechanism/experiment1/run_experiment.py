#!/usr/bin/env python3
"""
Convenience script to run the complete semantic vs paraphrase experiment.

This script orchestrates the entire experiment pipeline from data loading
to result visualization.
"""

import os
import sys
import logging
from pathlib import Path
import subprocess
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies() -> bool:
    """
    Check if all required dependencies are installed.
    
    Returns:
        True if all dependencies are available, False otherwise
    """
    required_packages = [
        'numpy', 'pandas', 'scipy', 'scikit-learn',
        'openai', 'umap', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.error("Please install them using: pip install -r requirements.txt")
        return False
    
    return True


def check_data_file(data_path: str) -> bool:
    """
    Check if the data file exists and is accessible.
    
    Args:
        data_path: Path to the data file
        
    Returns:
        True if file exists and is readable, False otherwise
    """
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        return False
    
    try:
        with open(data_path, 'r') as f:
            # Just check if we can read the file
            f.read(100)
        return True
    except Exception as e:
        logger.error(f"Cannot read data file: {e}")
        return False


def check_api_credentials() -> bool:
    """
    Check if API credentials are properly configured.
    Note: Using hardcoded Novitas AI credentials for this experiment.
    
    Returns:
        True if credentials are available, False otherwise
    """
    # Using hardcoded Novitas AI credentials
    api_key = "sk_UAor5zd9GsqksXircDQutvoSK1tWGTnW407fV8tIdMA"
    base_url = "https://api.novita.ai/openai"
    
    if not api_key:
        logger.error("Novitas AI API key not configured")
        return False
    
    if not base_url:
        logger.error("Novitas AI base URL not configured")
        return False
    
    logger.info(f"Using Novitas AI: {base_url}")
    return True


def run_main_experiment() -> bool:
    """
    Run the main experiment script.
    
    Returns:
        True if experiment completed successfully, False otherwise
    """
    logger.info("Starting main experiment...")
    
    try:
        result = subprocess.run([
            sys.executable, "main_experiment.py"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            logger.info("Main experiment completed successfully")
            return True
        else:
            logger.error(f"Main experiment failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error running main experiment: {e}")
        return False


def run_visualization(use_full_dataset: bool = False) -> bool:
    """
    Run the visualization script.
    
    Args:
        use_full_dataset: If True, visualize full dataset results
        
    Returns:
        True if visualization completed successfully, False otherwise
    """
    logger.info("Starting visualization...")
    
    try:
        cmd = [sys.executable, "visualization.py"]
        if use_full_dataset:
            cmd.append("--full")
            logger.info("Visualizing FULL dataset results")
        else:
            logger.info("Visualizing sample results")
            
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            logger.info("Visualization completed successfully")
            return True
        else:
            logger.error(f"Visualization failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error running visualization: {e}")
        return False


def create_results_summary() -> None:
    """
    Create a summary of the experiment results.
    """
    results_dir = Path("results")
    
    if not results_dir.exists():
        logger.warning("Results directory not found")
        return
    
    summary_file = results_dir / "experiment_summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write("Semantic vs Paraphrase Experiment Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Experiment completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # List result files
        f.write("Generated Files:\n")
        for file_path in sorted(results_dir.glob("*")):
            if file_path.is_file():
                f.write(f"  - {file_path.name}\n")
        
        f.write("\nNext Steps:\n")
        f.write("1. Review the distance_results.json for quantitative analysis\n")
        f.write("2. Examine the visualization plots for qualitative insights\n")
        f.write("3. Analyze cluster patterns in the semantic space plots\n")
        f.write("4. Compare Type2 vs Type4 distances to answer research questions\n")
    
    logger.info(f"Results summary created: {summary_file}")


def main():
    """
    Main execution function for the complete experiment pipeline.
    """
    import sys
    
    # Check if we should use full dataset
    use_full_dataset = "--full" in sys.argv or "full" in sys.argv
    
    if use_full_dataset:
        logger.info("Starting FULL DATASET Semantic vs Paraphrase Experiment Pipeline")
        logger.info("Using Novitas AI for BGE-m3 embeddings")
        logger.info("Processing entire dataset - no sample limits")
    else:
        logger.info("Starting Semantic vs Paraphrase Experiment Pipeline")
        logger.info("Using Novitas AI for BGE-m3 embeddings")
        logger.info("Using 3000 samples per type")
    
    logger.info("=" * 60)
    
    # Configuration
    data_path = "/Users/jonathanzha/Desktop/PADBen/data/test/final_generated_data.json"
    
    # Pre-flight checks
    logger.info("Performing pre-flight checks...")
    
    if not check_dependencies():
        logger.error("Dependency check failed. Exiting.")
        sys.exit(1)
    
    if not check_data_file(data_path):
        logger.error("Data file check failed. Exiting.")
        sys.exit(1)
    
    if not check_api_credentials():
        logger.error("API credentials check failed. Exiting.")
        sys.exit(1)
    
    logger.info("All pre-flight checks passed!")
    logger.info("Using Novitas AI provider for embedding generation")
    
    # Run experiment
    logger.info("Running main experiment...")
    if not run_main_experiment():
        logger.error("Main experiment failed. Exiting.")
        sys.exit(1)
    
    # Run visualization
    logger.info("Generating visualizations...")
    if not run_visualization(use_full_dataset):
        logger.error("Visualization failed. Exiting.")
        sys.exit(1)
    
    # Create summary
    create_results_summary()
    
    logger.info("=" * 60)
    logger.info("Experiment pipeline completed successfully!")
    logger.info("Using Novitas AI for BGE-m3 embeddings")
    
    if use_full_dataset:
        logger.info("Check the 'full_results/' directory for all outputs.")
        logger.info("Review 'full_results/experiment_summary.txt' for next steps.")
    else:
        logger.info("Check the 'results/' directory for all outputs.")
        logger.info("Review 'results/experiment_summary.txt' for next steps.")


if __name__ == "__main__":
    main()
