#!/usr/bin/env python3
"""
Main entry point for PADBen Task Preparation Module.

This module provides three main approaches for task preparation:
1. Single-sentence exhaustive method
2. Single-sentence sampling method  
3. Sentence-pair method

Usage:
    # Single-sentence exhaustive method
    python -m task_preparation.single_sentence.exhaustive_method.main --help
    
    # Single-sentence sampling method
    python -m task_preparation.single_sentence.sampling_method.main --help
    
    # Sentence-pair method
    python -m task_preparation.sentence_pair.main --help
"""

import argparse
import sys
from pathlib import Path


def setup_argparser() -> argparse.ArgumentParser:
    """Set up command line argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="PADBen Task Preparation Module - Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Methods:

1. Single-Sentence Exhaustive Method:
   Uses ALL instances from both relevant text types exhaustively.
   Dataset size: 2x original size (e.g., 16k + 16k = 32k samples)
   Label distribution: Always balanced (50-50)
   
   Usage: python -m task_preparation.single_sentence.exhaustive_method.main --help

2. Single-Sentence Sampling Method:
   Randomly samples ONE instance per original sample.
   Dataset size: Same as original (e.g., 16k samples)
   Label distribution: Configurable via sampling probabilities (30-70, 50-50, 80-20)
   
   Usage: python -m task_preparation.single_sentence.sampling_method.main --help

3. Sentence-Pair Method:
   Creates sentence pairs for classification tasks.
   Models determine which sentence in a pair is machine-generated vs human-written.
   
   Usage: python -m task_preparation.sentence_pair.main --help

Available Tasks:
- Task1: Paraphrase Source Attribution without Context (Type3 vs Type4)
- Task2: General Text Authorship Detection (Type1 vs Type2)
- Task3: AI Text Laundering Detection (Type1 vs Type5)
- Task4: Iterative Paraphrase Depth Detection (Type5-1st vs Type5-3rd)
- Task5: Original vs Deep Paraphrase Attack Detection (Type1 vs Type5-3rd)
        """
    )
    
    parser.add_argument(
        "--method", "-m",
        choices=["exhaustive", "sampling", "sentence_pair"],
        help="Choose the task preparation method"
    )
    
    parser.add_argument(
        "--help-method",
        choices=["exhaustive", "sampling", "sentence_pair"],
        help="Show detailed help for a specific method"
    )
    
    return parser


def show_method_help(method: str) -> None:
    """Show detailed help for a specific method.
    
    Args:
        method: The method to show help for
    """
    if method == "exhaustive":
        print("Single-Sentence Exhaustive Method")
        print("=" * 40)
        print("This method uses ALL instances from both relevant text types exhaustively.")
        print("For example, for Task1 (Type3 vs Type4):")
        print("- Takes ALL 16k Type3 sentences")
        print("- Takes ALL 16k Type4 sentences") 
        print("- Creates 32k total sentences (16k with label 0, 16k with label 1)")
        print("- Result: Balanced 50-50 distribution")
        print()
        print("Usage:")
        print("python -m task_preparation.single_sentence.exhaustive_method.main --help")
        
    elif method == "sampling":
        print("Single-Sentence Sampling Method")
        print("=" * 40)
        print("This method randomly samples ONE instance per original sample.")
        print("For example, for Task1 (Type3 vs Type4):")
        print("- For each of 16k original samples, randomly sample Type3 OR Type4")
        print("- Creates 16k total sentences with configurable label distribution")
        print("- Supports 30-70, 50-50, 80-20 distributions")
        print()
        print("Usage:")
        print("python -m task_preparation.single_sentence.sampling_method.main --help")
        
    elif method == "sentence_pair":
        print("Sentence-Pair Method")
        print("=" * 40)
        print("This method creates sentence pairs for classification tasks.")
        print("Models determine which sentence in a pair is machine-generated vs human-written.")
        print("Creates pairs like (Type1, Type2), (Type3, Type4), etc.")
        print()
        print("Usage:")
        print("python -m task_preparation.sentence_pair.main --help")


def main() -> None:
    """Main entry point."""
    parser = setup_argparser()
    args = parser.parse_args()
    
    if args.help_method:
        show_method_help(args.help_method)
        return
    
    if args.method:
        if args.method == "exhaustive":
            print("Redirecting to exhaustive method...")
            print("Run: python -m task_preparation.single_sentence.exhaustive_method.main --help")
        elif args.method == "sampling":
            print("Redirecting to sampling method...")
            print("Run: python -m task_preparation.single_sentence.sampling_method.main --help")
        elif args.method == "sentence_pair":
            print("Redirecting to sentence-pair method...")
            print("Run: python -m task_preparation.sentence_pair.main --help")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()