#!/usr/bin/env python3
"""
Convenient script to run data quality examination with different configurations.

Usage:
    python run_examination.py --sample-size 1000 --no-viz
    python run_examination.py --full --output-dir ./custom_output
    python run_examination.py --quality_type jaccard_similarity --full
    python run_examination.py --quality_type self-BLEU --sample-size 500
    python run_examination.py --quality_type perplexity --full --no-viz
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

try:
    from .main import DataQualityExaminer
except ImportError:
    from main import DataQualityExaminer


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration with proper encoding support."""
    import sys
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create a stream handler with UTF-8 encoding for Windows compatibility
    stream_handler = logging.StreamHandler(sys.stdout)
    try:
        # Try to set UTF-8 encoding if possible
        if hasattr(stream_handler.stream, 'reconfigure'):
            stream_handler.stream.reconfigure(encoding='utf-8')
        elif hasattr(stream_handler.stream, 'buffer'):
            # For older Python versions, wrap the stream
            import io
            stream_handler.stream = io.TextIOWrapper(
                stream_handler.stream.buffer, encoding='utf-8', errors='replace'
            )
    except (AttributeError, OSError):
        # If encoding configuration fails, we'll fall back to ASCII-safe messages
        pass
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[stream_handler],
        force=True  # Override any existing configuration
    )


class SelectiveDataQualityExaminer(DataQualityExaminer):
    """Extended DataQualityExaminer with selective metric calculation."""
    
    def run_selective_examination(self, quality_type: str, sample_size: Optional[int] = 1000,
                                generate_visualizations: bool = True) -> dict:
        """
        Run selective data quality examination for specific metrics.
        
        Args:
            quality_type: Type of quality metric to calculate
                         ('jaccard_similarity', 'self-BLEU', 'perplexity', or 'full')
            sample_size: Number of texts to sample for computationally expensive metrics.
            generate_visualizations: Whether to generate visualization plots.
            
        Returns:
            Dictionary containing examination results.
        """
        from datetime import datetime
        import logging
        
        logger = logging.getLogger(__name__)
        logger.info(f"Starting selective data quality examination: {quality_type}")
        start_time = datetime.now()
        
        try:
            # Step 1: Load and validate data
            logger.info("Step 1: Loading and validating data...")
            self._load_and_validate_data()
            
            # Initialize results structure
            self.padben_metrics = {
                "jaccard_similarity_matrix": {},
                "self_bleu_scores": {},
                "perplexity_scores": {},
                "dataset_statistics": self.dataset_stats
            }
            
            # Step 2: Calculate selected metrics
            if quality_type == 'full':
                logger.info("Step 2: Running ALL metric calculations sequentially...")
                self._calculate_jaccard_similarity()
                self._save_jaccard_results()
                
                self._calculate_self_bleu(sample_size)
                self._save_self_bleu_results()
                
                self._calculate_perplexity(sample_size)
                self._save_perplexity_results()
                
            elif quality_type == 'jaccard_similarity':
                logger.info("Step 2: Calculating Jaccard similarity matrix...")
                self._calculate_jaccard_similarity()
                self._save_jaccard_results()
                
            elif quality_type == 'self-BLEU':
                logger.info("Step 2: Calculating self-BLEU scores...")
                self._calculate_self_bleu(sample_size)
                self._save_self_bleu_results()
                
            elif quality_type == 'perplexity':
                logger.info("Step 2: Calculating perplexity scores...")
                self._calculate_perplexity(sample_size)
                self._save_perplexity_results()
                
            else:
                raise ValueError(f"Unknown quality_type: {quality_type}")
            
            # Step 3: Compare with RAID benchmark (only if we have relevant metrics)
            if quality_type in ['full', 'self-BLEU', 'perplexity']:
                logger.info("Step 3: Comparing with RAID benchmark...")
                self._compare_with_raid()
            else:
                logger.info("Step 3: Skipping RAID comparison (not applicable for Jaccard similarity)")
                self.comparison_results = {}
            
            # Step 4: Generate tables and reports
            logger.info("Step 4: Generating tables and reports...")
            self._generate_selective_tables_and_reports(quality_type)
            
            # Step 5: Create visualizations
            if generate_visualizations:
                logger.info("Step 5: Creating visualizations...")
                self._create_selective_visualizations(quality_type)
            
            # Step 6: Generate final summary
            logger.info("Step 6: Generating final summary...")
            final_results = self._generate_selective_final_summary(quality_type)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Selective examination ({quality_type}) finished in {duration:.2f} seconds")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error during selective examination: {e}")
            raise
    
    def _generate_selective_tables_and_reports(self, quality_type: str) -> None:
        """Generate tables and reports for selected metrics."""
        from config import TEXT_TYPES
        import pandas as pd
        import numpy as np
        import logging
        
        logger = logging.getLogger(__name__)
        logger.info(f"[TABLE] Generating tables for {quality_type}...")
        
        if quality_type == 'jaccard_similarity':
            # Generate Jaccard-specific table
            if self.padben_metrics["jaccard_similarity_matrix"]:
                jaccard_df = pd.DataFrame.from_dict(
                    self.padben_metrics["jaccard_similarity_matrix"], 
                    orient='index', 
                    columns=['jaccard_similarity']
                )
                jaccard_df.index.name = 'text_type_pair'
                jaccard_df.to_csv(self.output_dir / "jaccard_similarity_results.csv")
                logger.info("  [OK] Jaccard similarity table saved")
        
        elif quality_type == 'self-BLEU':
            # Generate self-BLEU specific table
            if self.padben_metrics["self_bleu_scores"]:
                bleu_data = []
                for text_type in TEXT_TYPES:
                    if text_type == 'source':
                        continue
                    score = self.padben_metrics["self_bleu_scores"].get(text_type, 'N/A')
                    bleu_data.append({
                        'Text Type': text_type.replace('_', ' ').title(),
                        'Self-BLEU Score': score
                    })
                
                # Add average
                valid_scores = [s for s in self.padben_metrics["self_bleu_scores"].values() 
                              if s is not None and not np.isnan(s)]
                if valid_scores:
                    avg_score = np.mean(valid_scores)
                    bleu_data.append({
                        'Text Type': 'Average',
                        'Self-BLEU Score': f"{avg_score:.4f}"
                    })
                
                bleu_df = pd.DataFrame(bleu_data)
                bleu_df.to_csv(self.output_dir / "self_bleu_results.csv", index=False)
                logger.info("  [OK] Self-BLEU table saved")
        
        elif quality_type == 'perplexity':
            # Generate perplexity specific table
            if self.padben_metrics["perplexity_scores"]:
                ppl_data = []
                for text_type in TEXT_TYPES:
                    if text_type == 'source':
                        continue
                    score = self.padben_metrics["perplexity_scores"].get(text_type, 'N/A')
                    ppl_data.append({
                        'Text Type': text_type.replace('_', ' ').title(),
                        'Perplexity (GPT-2 XL)': score
                    })
                
                # Add average
                valid_scores = [s for s in self.padben_metrics["perplexity_scores"].values() 
                              if s is not None and not np.isinf(s) and not np.isnan(s)]
                if valid_scores:
                    avg_score = np.mean(valid_scores)
                    ppl_data.append({
                        'Text Type': 'Average',
                        'Perplexity (GPT-2 XL)': f"{avg_score:.4f}"
                    })
                
                ppl_df = pd.DataFrame(ppl_data)
                ppl_df.to_csv(self.output_dir / "perplexity_results.csv", index=False)
                logger.info("  [OK] Perplexity table saved")
        
        elif quality_type == 'full':
            # Generate comprehensive table (same as original)
            self._generate_tables_and_reports()
    
    def _create_selective_visualizations(self, quality_type: str) -> None:
        """Create visualizations for selected metrics."""
        import logging
        
        logger = logging.getLogger(__name__)
        try:
            if quality_type == 'jaccard_similarity' and self.padben_metrics["jaccard_similarity_matrix"]:
                logger.info("  [CHART] Generating Jaccard similarity heatmap...")
                self.visualizer.create_similarity_heatmap(
                    self.padben_metrics["jaccard_similarity_matrix"],
                    title="PADBen Jaccard Similarity Matrix",
                    filename="jaccard_similarity_heatmap.png"
                )
                logger.info("    [OK] Jaccard heatmap saved")
            
            elif quality_type in ['self-BLEU', 'perplexity', 'full']:
                # Create comparison charts for BLEU/perplexity
                from config import RAID_METRICS
                
                if quality_type in ['self-BLEU', 'full'] and self.padben_metrics["self_bleu_scores"]:
                    logger.info("  [CHART] Generating self-BLEU comparison chart...")
                    # Create a simple bar chart for self-BLEU
                    import matplotlib.pyplot as plt
                    
                    text_types = list(self.padben_metrics["self_bleu_scores"].keys())
                    scores = list(self.padben_metrics["self_bleu_scores"].values())
                    
                    plt.figure(figsize=(10, 6))
                    plt.bar(text_types, scores)
                    plt.title('Self-BLEU Scores by Text Type')
                    plt.xlabel('Text Type')
                    plt.ylabel('Self-BLEU Score')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(self.output_dir / "self_bleu_chart.png", dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info("    [OK] Self-BLEU chart saved")
                
                if quality_type in ['perplexity', 'full'] and self.padben_metrics["perplexity_scores"]:
                    logger.info("  [CHART] Generating perplexity comparison chart...")
                    import matplotlib.pyplot as plt
                    
                    text_types = list(self.padben_metrics["perplexity_scores"].keys())
                    scores = list(self.padben_metrics["perplexity_scores"].values())
                    
                    plt.figure(figsize=(10, 6))
                    plt.bar(text_types, scores)
                    plt.title('Perplexity Scores by Text Type (GPT-2 XL)')
                    plt.xlabel('Text Type')
                    plt.ylabel('Perplexity Score')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(self.output_dir / "perplexity_chart.png", dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info("    [OK] Perplexity chart saved")
            
            if quality_type == 'full':
                # Create comprehensive dashboard
                self._create_visualizations()
                
        except Exception as e:
            logger.warning(f"[WARNING] Visualization generation failed: {e}")
            logger.info("[INFO] Metrics calculation completed successfully, visualizations skipped")
    
    def _generate_selective_final_summary(self, quality_type: str) -> dict:
        """Generate final summary for selected metrics."""
        from datetime import datetime
        import json
        import numpy as np
        
        # Custom JSON serializer
        def json_serializer(obj):
            import pandas as pd
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
                return str(obj)
            else:
                return str(obj)
        
        # Generate summary report
        summary_report = {
            "examination_metadata": {
                "timestamp": datetime.now().isoformat(),
                "quality_type": quality_type,
                "selective_processing": True if quality_type != 'full' else False
            },
            "dataset_statistics": self.dataset_stats,
            "data_completeness": self.data_completeness,
            "selected_metrics": {}
        }
        
        # Add relevant metrics
        if quality_type in ['jaccard_similarity', 'full']:
            # Convert tuple keys to string keys for JSON serialization
            jaccard_matrix = self.padben_metrics["jaccard_similarity_matrix"]
            if jaccard_matrix:
                jaccard_json = {}
                for key, value in jaccard_matrix.items():
                    if isinstance(key, tuple):
                        string_key = f"{key[0]}_vs_{key[1]}"
                        jaccard_json[string_key] = value
                    else:
                        jaccard_json[key] = value
                summary_report["selected_metrics"]["jaccard_similarity_matrix"] = jaccard_json
            else:
                summary_report["selected_metrics"]["jaccard_similarity_matrix"] = {}
        
        if quality_type in ['self-BLEU', 'full']:
            summary_report["selected_metrics"]["self_bleu_scores"] = self.padben_metrics["self_bleu_scores"]
        
        if quality_type in ['perplexity', 'full']:
            summary_report["selected_metrics"]["perplexity_scores"] = self.padben_metrics["perplexity_scores"]
        
        # Add RAID comparison if available
        if hasattr(self, 'comparison_results') and self.comparison_results:
            summary_report["raid_comparison"] = self.comparison_results
        
        # Save summary report
        filename = f"{quality_type}_examination_report.json" if quality_type != 'full' else "quality_examination_report.json"
        with open(self.output_dir / filename, 'w') as f:
            json.dump(summary_report, f, indent=2, default=json_serializer)
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[SAVE] Results saved to: {self.output_dir.absolute()}")
        
        return summary_report


def main() -> None:
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run PADBen data quality examination with selective metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full examination with all metrics (sequential)
  python run_examination.py --quality_type full --full
  
  # Only Jaccard similarity calculation
  python run_examination.py --quality_type jaccard_similarity --full
  
  # Only self-BLEU calculation with sampling
  python run_examination.py --quality_type self-BLEU --sample-size 500
  
  # Only perplexity calculation without visualizations
  python run_examination.py --quality_type perplexity --full --no-viz
  
  # Traditional full examination (backward compatibility)
  python run_examination.py --full --output-dir ./my_results
        """
    )
    
    parser.add_argument(
        '--quality_type',
        type=str,
        choices=['jaccard_similarity', 'self-BLEU', 'perplexity', 'full'],
        default='full',
        help='Type of quality metric to calculate (default: full)'
    )
    
    parser.add_argument(
        '--sample-size', 
        type=int, 
        default=1000,
        help='Number of texts to sample for expensive metrics (default: 1000)'
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help='Use full dataset (no sampling) - may be slow'
    )
    
    parser.add_argument(
        '--no-viz', 
        action='store_true',
        help='Skip visualization generation'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Custom output directory (default: ./outputs)'
    )
    
    parser.add_argument(
        '--data-path',
        type=Path,
        help='Custom path to data file (default: from config)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--perplexity-model',
        type=str,
        choices=[
            'gpt2-xl', 
            'llama3-8b-full', 'llama3-8b-4bit', 
            'llama2-7b-full', 'llama2-7b-4bit'
        ],
        default='gpt2-xl',
        help='Perplexity model to use: gpt2-xl (default, fastest), llama3-8b-full (requires auth, high VRAM), llama3-8b-4bit (requires auth, moderate VRAM), llama2-7b-full (ungated, high VRAM), llama2-7b-4bit (ungated, moderate VRAM)'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Determine sample size
    sample_size = None if args.full else args.sample_size
    
    # Log configuration
    logger.info("Starting PADBen Data Quality Examination")
    logger.info(f"Quality type: {args.quality_type}")
    logger.info(f"Sample size: {'Full dataset' if sample_size is None else sample_size}")
    logger.info(f"Generate visualizations: {not args.no_viz}")
    if args.output_dir:
        logger.info(f"Output directory: {args.output_dir}")
    if args.data_path:
        logger.info(f"Data path: {args.data_path}")
    
    try:
        # Initialize selective examiner
        examiner = SelectiveDataQualityExaminer(
            data_path=args.data_path,
            output_dir=args.output_dir,
            perplexity_model=args.perplexity_model  # New parameter
        )
        
        # Run selective examination
        results = examiner.run_selective_examination(
            quality_type=args.quality_type,
            sample_size=sample_size,
            generate_visualizations=not args.no_viz
        )
        
        logger.info("Examination completed successfully!")
        logger.info(f"Results saved to: {examiner.output_dir}")
        
        # Display summary based on quality type
        if args.quality_type == 'jaccard_similarity':
            logger.info("[COMPLETE] Jaccard Similarity Matrix calculated and saved")
        elif args.quality_type == 'self-BLEU':
            logger.info("[COMPLETE] Self-BLEU scores calculated and saved")
        elif args.quality_type == 'perplexity':
            logger.info("[COMPLETE] Perplexity scores calculated and saved")
        else:
            logger.info("[COMPLETE] All metrics calculated and saved")
        
    except KeyboardInterrupt:
        logger.info("Examination interrupted by user")
    except Exception as e:
        logger.error(f"Examination failed: {e}")
        raise


if __name__ == "__main__":
    main()
