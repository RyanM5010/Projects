#!/usr/bin/env python3
"""
Test Type 2 Generation with Configurable Sample Size

This script tests the Type 2 generation pipeline using the orchestrator
with a configurable number of samples from the unified dataset.
"""

import asyncio
import json
import pandas as pd
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Progress bar for better UX
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    # Fallback: create dummy tqdm class
    class tqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable) if self.iterable else iter([])
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, n=1):
            pass
        def set_description(self, desc):
            pass
        def close(self):
            pass
    HAS_TQDM = False
    print("Warning: tqdm not installed. Install with: pip install tqdm")

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Now import the orchestrator and configuration
from data_generation.orchestrator import PADBenOrchestrator, GenerationOptions
from data_generation.config.generation_model_config import DEFAULT_CONFIG
from data_generation.type2_generation.type2_generation import EnhancedType2Generator, GenerationMethod, EnvironmentMode

def create_test_dataset(input_file: str, output_file: str, num_samples: int = None, start_idx: int = None, end_idx: int = None):
    """Create a test dataset with the specified number of samples or index range.
    
    Args:
        input_file: Path to the input dataset
        output_file: Path to save the test dataset
        num_samples: Number of samples to use from the beginning. If None, uses all samples.
        start_idx: Starting index for range selection (inclusive)
        end_idx: Ending index for range selection (exclusive)
    """
    # Load the unified dataset first to check bounds
    input_path = Path(input_file)
    if input_path.suffix == '.json':
        df = pd.read_json(input_file)
    elif input_path.suffix == '.csv':
        df = pd.read_csv(input_file)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    total_samples = len(df)
    print(f"Loaded {total_samples} total samples from {input_file}")
    
    # Determine selection method and validate parameters
    if start_idx is not None or end_idx is not None:
        # Range-based selection
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = total_samples
        
        # Validate range bounds
        if start_idx < 0:
            raise ValueError(f"Start index cannot be negative: {start_idx}")
        if end_idx > total_samples:
            raise ValueError(f"End index {end_idx} exceeds dataset size {total_samples}")
        if start_idx >= end_idx:
            raise ValueError(f"Start index {start_idx} must be less than end index {end_idx}")
        if start_idx >= total_samples:
            raise ValueError(f"Start index {start_idx} exceeds dataset size {total_samples}")
        
        test_df = df.iloc[start_idx:end_idx].copy()
        actual_samples = len(test_df)
        print(f"Creating test dataset with samples from index {start_idx} to {end_idx-1} ({actual_samples} samples)...")
        print(f"Using samples from index range [{start_idx}:{end_idx}) - {actual_samples} samples for generation")
        
    elif num_samples is None:
        # All samples
        test_df = df.copy()
        print("Creating test dataset with ALL samples...")
        print("Using ALL samples for generation")
        
    else:
        # First N samples (original behavior)
        if num_samples > total_samples:
            print(f"⚠️ Warning: Requested {num_samples} samples but dataset only has {total_samples} samples")
            num_samples = total_samples
        
        test_df = df.head(num_samples).copy()
        print(f"Creating test dataset with first {num_samples} samples...")
        print(f"Using first {num_samples} samples for generation")
    
    # Ensure the test dataset has the required columns
    required_columns = ['idx', 'dataset_source', 'human_original_text', 'human_paraphrased_text']
    for col in required_columns:
        if col not in test_df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Add missing columns that Type 2 will fill
    if 'llm_generated_text' not in test_df.columns:
        test_df['llm_generated_text'] = None
    if 'llm_generated_text_method' not in test_df.columns:
        test_df['llm_generated_text_method'] = None
    
    # Save test dataset
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == '.json':
        test_df.to_json(output_file, orient='records', indent=2)
    else:
        test_df.to_csv(output_file, index=False)
    
    print(f"Created test dataset: {output_file}")
    print(f"Test dataset shape: {test_df.shape}")
    print(f"Dataset sources in test data: {test_df['dataset_source'].value_counts().to_dict()}")
    
    return test_df

async def test_type2_generation(num_samples: int = None, methods: list = None, start_idx: int = None, end_idx: int = None):
    """Test Type 2 generation with the orchestrator.
    
    Args:
        num_samples: Number of samples to process from the beginning. If None, processes all samples.
        methods: List of methods to test. If None, tests both methods.
        start_idx: Starting index for range selection (inclusive)
        end_idx: Ending index for range selection (exclusive)
    """
    # Determine description for progress display
    if start_idx is not None or end_idx is not None:
        range_desc = f"RANGE [{start_idx or 0}:{end_idx or 'end'})"
        print(f"🧪 Starting Type 2 Generation Test - {range_desc}")
    elif num_samples is None:
        print("🧪 Starting Type 2 Generation Test - ALL SAMPLES")
    else:
        print(f"🧪 Starting Type 2 Generation Test - {num_samples} SAMPLES")
    print("=" * 60)
    
    # Paths (relative to project root)
    original_dataset = str(project_root / "data/processed/unified_padben_base.json")
    
    # Generate appropriate test dataset filename
    if start_idx is not None or end_idx is not None:
        start_str = start_idx or 0
        end_str = end_idx or "end"
        test_dataset = str(project_root / f"data/test/test_unified_padben_range_{start_str}_{end_str}.json")
    elif num_samples is None:
        test_dataset = str(project_root / "data/test/test_unified_padben_all_samples.json")
    else:
        test_dataset = str(project_root / f"data/test/test_unified_padben_{num_samples}_samples.json")
    
    output_dir = str(project_root / "data/test/type2_generation_test")
    
    try:
        # Step 1: Create test dataset
        print("Step 1: Creating test dataset...")
        test_df = create_test_dataset(original_dataset, test_dataset, num_samples=num_samples, start_idx=start_idx, end_idx=end_idx)
        
        # Step 2: Initialize test generator directly (not using orchestrator for test mode)
        print("\nStep 2: Initializing test generator...")
        generator = EnhancedType2Generator(
            DEFAULT_CONFIG.type2_config, 
            environment_mode=EnvironmentMode.TEST
        )
        print("✅ Test generator initialized successfully")
        
        # Step 3: Test specified generation methods
        print("\nStep 3: Testing generation methods...")
        
        # Default to both methods if none specified
        if methods is None:
            methods_to_test = [
                (GenerationMethod.SENTENCE_COMPLETION, "sentence_completion"),
                (GenerationMethod.QUESTION_ANSWER, "question_answer")
            ]
        else:
            method_map = {
                "sentence_completion": (GenerationMethod.SENTENCE_COMPLETION, "sentence_completion"),
                "question_answer": (GenerationMethod.QUESTION_ANSWER, "question_answer")
            }
            methods_to_test = [method_map[method] for method in methods if method in method_map]
        
        results = {}
        
        # Main progress bar for methods
        method_pbar = tqdm(
            methods_to_test,
            desc="Testing methods",
            unit="method",
            disable=not HAS_TQDM
        )
        
        for method_enum, method_name in method_pbar:
            method_pbar.set_description(f"Testing {method_name}")
            print(f"\n--- Testing {method_name} method ---")
            print(f"Processing {len(test_df)} samples...")
            
            start_time = datetime.now()
            
            # Run generation for this method
            results_df = await generator.generate_for_dataset(
                test_df.copy(), 
                method=method_enum, 
                output_dir=output_dir
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Get the correct column name for this method
            method_column = generator.get_method_column_name(method_enum)
            
            # Collect results using the correct column name
            if method_column in results_df.columns:
                generated_count = results_df[method_column].notna().sum()
            else:
                # Fallback: check if there's a generic llm_generated_text column
                if 'llm_generated_text' in results_df.columns:
                    generated_count = results_df['llm_generated_text'].notna().sum()
                else:
                    generated_count = 0
                    print(f"⚠️ Warning: Neither {method_column} nor llm_generated_text column found")
                    print(f"Available columns: {list(results_df.columns)}")
            
            total_count = len(results_df)
            success_rate = (generated_count / total_count) * 100
            
            results[method_name] = {
                'generated_count': generated_count,
                'total_count': total_count,
                'success_rate': success_rate,
                'duration': duration,
                'results_df': results_df,
                'method_column': method_column
            }
            
            print(f"✅ {method_name} completed:")
            print(f"   Generated: {generated_count}/{total_count} ({success_rate:.1f}%)")
            print(f"   Duration: {duration:.2f} seconds")
            print(f"   Avg time per sample: {duration/total_count:.2f} seconds")
            print(f"   Method column: {method_column}")
            
            # Update method progress bar
            method_pbar.set_postfix({
                'Success': f'{success_rate:.1f}%',
                'Time': f'{duration:.1f}s'
            })
        
        method_pbar.close()
        
        # Step 4: Analyze and compare results
        print("\n" + "=" * 60)
        print("🎉 TYPE 2 GENERATION TEST COMPLETED")
        print("=" * 60)
        
        print("📊 Method Comparison:")
        for method_name, result in results.items():
            print(f"  {method_name}:")
            print(f"    Success rate: {result['success_rate']:.1f}%")
            print(f"    Processing time: {result['duration']:.2f} seconds")
            print(f"    Avg time per sample: {result['duration']/result['total_count']:.2f} seconds")
            print(f"    Total samples: {result['total_count']}")
        
        # Show sample outputs from each method (limit to 2 samples for readability)
        print(f"\n📋 Sample Generated Texts:")
        print("-" * 40)
        
        for method_name, result in results.items():
            print(f"\n--- {method_name.title()} Method ---")
            df = result['results_df']
            method_column = result['method_column']
            sample_count = 0
            
            # Use tqdm for sample iteration if showing many samples
            sample_iterator = df.iterrows()
            if len(df) > 10:  # Only show progress bar if many samples
                sample_iterator = tqdm(
                    sample_iterator, 
                    desc=f"Showing {method_name} samples",
                    total=len(df),
                    disable=not HAS_TQDM,
                    leave=False
                )
            
            for idx, row in sample_iterator:
                # Check the correct column for generated text
                generated_text = None
                if method_column in df.columns and pd.notna(row[method_column]):
                    generated_text = row[method_column]
                elif 'llm_generated_text' in df.columns and pd.notna(row['llm_generated_text']):
                    generated_text = row['llm_generated_text']
                
                if generated_text and sample_count < 2:
                    sample_count += 1
                    print(f"Sample {sample_count}:")
                    print(f"  Dataset: {row['dataset_source']}")
                    print(f"  Original: {row['human_original_text'][:100]}{'...' if len(row['human_original_text']) > 100 else ''}")
                    print(f"  Generated: {generated_text[:100]}{'...' if len(generated_text) > 100 else ''}")
                    print()
        
        # Check output directories with progress
        print(f"📁 Output Directories:")
        output_path = Path(output_dir)
        if output_path.exists():
            method_names = ["sentence_completion", "question_answer"]
            
            # Use progress bar for directory checking if processing many methods
            dir_iterator = method_names
            if len(method_names) > 1:
                dir_iterator = tqdm(
                    method_names,
                    desc="Checking output directories",
                    disable=not HAS_TQDM,
                    leave=False
                )
            
            for method_name in dir_iterator:
                method_dirs = list(output_path.glob(f"{method_name}_*"))
                if method_dirs:
                    latest_dir = max(method_dirs, key=lambda p: p.name)
                    print(f"  {method_name}: {latest_dir}")
                    
                    # Check for midpoint directory
                    midpoint_dir = latest_dir / "midpoint"
                    if midpoint_dir.exists():
                        midpoint_files = list(midpoint_dir.glob("*"))
                        print(f"    Midpoint files: {len(midpoint_files)}")
                        for file in midpoint_files[:3]:  # Show first 3 files
                            print(f"      📄 {file.name}")
        
        print(f"\n✅ Test completed successfully!")
        print(f"Results saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run the test with command-line arguments."""
    global HAS_TQDM  # Fix: Access the global HAS_TQDM variable
    
    parser = argparse.ArgumentParser(description="PADBen Type 2 Generation Test")
    
    # Sample size options
    parser.add_argument("--samples", type=int, default=20, 
                       help="Number of samples to process from the beginning (default: 20)")
    parser.add_argument("--all", action="store_true", 
                       help="Process all samples in the dataset")
    
    # NEW: Range-based selection options
    parser.add_argument("--start", type=int, 
                       help="Starting index for range selection (inclusive, 0-based)")
    parser.add_argument("--end", type=int, 
                       help="Ending index for range selection (exclusive, 0-based)")
    
    # Method selection
    parser.add_argument("--methods", nargs="+", 
                       choices=["sentence_completion", "question_answer"],
                       help="Methods to test (default: both)")
    
    # Input dataset override
    parser.add_argument("--input", type=str,
                       help="Override input dataset path (default: data/processed/unified_padben_base.json)")
    
    # Progress options
    parser.add_argument("--no-progress", action="store_true",
                       help="Disable progress bars")
    
    args = parser.parse_args()
    
    # Validate argument combinations
    range_specified = args.start is not None or args.end is not None
    if args.all and range_specified:
        print("❌ Error: Cannot use --all with --start/--end range options")
        return
    
    if args.all and args.samples != 20:  # 20 is the default
        print("❌ Error: Cannot use --all with --samples option")
        return
    
    if range_specified and args.samples != 20:  # 20 is the default
        print("❌ Error: Cannot use --samples with --start/--end range options")
        return
    
    # Check if the unified dataset exists first to validate ranges
    if args.input:
        dataset_path = Path(args.input)
    else:
        dataset_path = project_root / "data/processed/unified_padben_base.json"
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please ensure the unified dataset exists before running the test.")
        return
    
    # Load dataset to check size and validate range arguments
    try:
        if dataset_path.suffix == '.json':
            df_check = pd.read_json(dataset_path)
        else:
            df_check = pd.read_csv(dataset_path)
        
        total_samples = len(df_check)
        print(f"📊 Dataset contains {total_samples} total samples")
        
        # Validate range arguments
        if range_specified:
            start_idx = args.start
            end_idx = args.end
            
            if start_idx is not None and start_idx < 0:
                print(f"❌ Error: Start index cannot be negative: {start_idx}")
                return
            
            if end_idx is not None and end_idx > total_samples:
                print(f"❌ Error: End index {end_idx} exceeds dataset size {total_samples}")
                return
            
            if start_idx is not None and start_idx >= total_samples:
                print(f"❌ Error: Start index {start_idx} exceeds dataset size {total_samples}")
                return
            
            if start_idx is not None and end_idx is not None and start_idx >= end_idx:
                print(f"❌ Error: Start index {start_idx} must be less than end index {end_idx}")
                return
    
    except Exception as e:
        print(f"❌ Error reading dataset for validation: {e}")
        return
    
    # Determine processing parameters
    if range_specified:
        num_samples = None
        start_idx = args.start
        end_idx = args.end
        
        # Calculate actual number of samples for display
        actual_start = start_idx or 0
        actual_end = end_idx or total_samples
        actual_count = actual_end - actual_start
        
        print("🚀 PADBen Type 2 Generation Test")
        print(f"Processing samples from index {actual_start} to {actual_end-1} ({actual_count} samples)")
        
    elif args.all:
        num_samples = None
        start_idx = None
        end_idx = None
        print("🚀 PADBen Type 2 Generation Test")
        print("Processing ALL samples from unified dataset")
        
    else:
        num_samples = args.samples
        start_idx = None
        end_idx = None
        print("🚀 PADBen Type 2 Generation Test")
        print(f"Processing first {num_samples} samples from unified dataset")
    
    if args.methods:
        print(f"Testing methods: {', '.join(args.methods)}")
    else:
        print("Testing both sentence_completion and question_answer methods")
    
    if args.no_progress:
        print("Progress bars disabled")
    
    print("=" * 60)
    
    # Check API key configuration
    try:
        from data_generation.config.secrets_manager import validate_all_api_keys
        if not validate_all_api_keys():
            print("❌ API keys not configured properly.")
            print("Please set up your API keys first:")
            print("1. Create a .env file with GEMINI_API_KEY=your_key")
            print("2. Or set environment variable: set GEMINI_API_KEY=your_key")  # Windows command
            return
        print("✅ API keys validated successfully")
    except Exception as e:
        print(f"⚠️ Warning: Could not validate API keys: {e}")
    
    # Show warning for large datasets with progress estimation
    if num_samples is None and not range_specified:
        # Processing all samples
        estimated_time = total_samples * 2.5  # Rough estimate: 2.5 seconds per sample
        
        print(f"\n⚠️ WARNING: You're about to process {total_samples} samples")
        print(f"Estimated processing time: {estimated_time/60:.1f} minutes ({estimated_time/3600:.1f} hours)")
        print("This will consume API credits and take significant time.")
        
        # Show progress estimation
        if HAS_TQDM:
            print("Progress tracking will be available during generation.")
        else:
            print("Install tqdm for progress tracking: pip install tqdm")
        
        response = input("Do you want to continue? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Operation cancelled.")
            return
    
    elif range_specified:
        # Processing range
        actual_start = start_idx or 0
        actual_end = end_idx or total_samples
        actual_count = actual_end - actual_start
        estimated_time = actual_count * 2.5  # Rough estimate: 2.5 seconds per sample
        
        if actual_count > 100:  # Only show warning for larger ranges
            print(f"\n📊 You're about to process {actual_count} samples (range {actual_start}:{actual_end})")
            print(f"Estimated processing time: {estimated_time/60:.1f} minutes")
            
            if actual_count > 1000:
                print("This will consume API credits and take significant time.")
                response = input("Do you want to continue? (y/N): ")
                if response.lower() not in ['y', 'yes']:
                    print("Operation cancelled.")
                    return
    
    # Disable progress bars if requested
    if args.no_progress:
        HAS_TQDM = False
    
    # Run the test with progress tracking
    print("🚀 Starting generation with progress tracking...")
    success = asyncio.run(test_type2_generation(
        num_samples=num_samples, 
        methods=args.methods, 
        start_idx=start_idx, 
        end_idx=end_idx
    ))
    
    if success:
        print("\n🎉 All tests passed! Type 2 generation is working correctly.")
    else:
        print("\n❌ Test failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
