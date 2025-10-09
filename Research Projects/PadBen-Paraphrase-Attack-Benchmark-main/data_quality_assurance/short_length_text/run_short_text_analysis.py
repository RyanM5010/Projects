#!/usr/bin/env python3
"""
Convenient script to run short text analysis with different configurations.

Usage:
    python run_short_text_analysis.py --threshold 10
    python run_short_text_analysis.py --threshold 5 --max-examples 100
"""

import argparse
import logging
from pathlib import Path

try:
    from .short_text_analyzer import ShortTextAnalyzer
except ImportError:
    from short_text_analyzer import ShortTextAnalyzer


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def main() -> None:
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Analyze abnormally short texts in PADBen dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze texts shorter than 10 tokens (default)
  python run_short_text_analysis.py
  
  # Use custom threshold
  python run_short_text_analysis.py --threshold 5
  
  # Limit examples for human evaluation
  python run_short_text_analysis.py --max-examples 50
  
  # Verbose logging
  python run_short_text_analysis.py --verbose
        """
    )
    
    parser.add_argument(
        '--threshold', 
        type=int, 
        default=10,
        help='Maximum token length to consider as "short" (default: 10)'
    )
    
    parser.add_argument(
        '--max-examples',
        type=int,
        default=200,
        help='Maximum examples to include in human evaluation file (default: 200)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default="./outputs/short_text_analysis",
        help='Custom output directory (default: ./outputs)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Only print summary report, do not save files'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting PADBen Short Text Analysis")
    logger.info(f"Threshold: {args.threshold} tokens")
    logger.info(f"Max examples for evaluation: {args.max_examples}")
    if args.output_dir:
        logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Initialize analyzer
        analyzer = ShortTextAnalyzer(threshold=args.threshold)
        if args.output_dir:
            analyzer.output_dir = args.output_dir
            analyzer.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run analysis
        logger.info("Running short text analysis...")
        results = analyzer.analyze_short_texts()
        
        # Print summary report
        print("\n" + analyzer.generate_summary_report())
        
        if not args.summary_only:
            # Save detailed results
            analysis_file = analyzer.save_analysis_results()
            
            # Export for human evaluation
            eval_file = analyzer.export_for_human_evaluation(max_examples=args.max_examples)
            
            print(f"\n📁 Files Generated:")
            print(f"   • Detailed Analysis: {analysis_file}")
            print(f"   • Human Evaluation: {eval_file}")
            print(f"\n💡 Next Steps:")
            print(f"   1. Review the summary above")
            print(f"   2. Open {eval_file} for human evaluation")
            print(f"   3. Check {analysis_file} for detailed statistics")
        
        logger.info("Short text analysis completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
