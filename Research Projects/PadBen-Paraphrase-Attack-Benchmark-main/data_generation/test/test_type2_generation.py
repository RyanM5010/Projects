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
from typing import Optional, List, Dict, Any

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
    """Create a test dataset with specified number of samples or range."""
    try:
        # Load the unified dataset
        if Path(input_file).suffix == '.json':
            df = pd.read_json(input_file)
        else:
            df = pd.read_csv(input_file)
        
        print(f"📊 Loaded {len(df)} samples from {input_file}")
        
        # Apply range or sample selection
        if start_idx is not None or end_idx is not None:
            start = start_idx or 0
            end = end_idx or len(df)
            test_df = df.iloc[start:end].copy()
            print(f"✂️ Selected range [{start}:{end}] = {len(test_df)} samples")
        elif num_samples is not None:
            test_df = df.head(num_samples).copy()
            print(f"✂️ Selected first {num_samples} samples")
        else:
            test_df = df.copy()
            print(f"✂️ Using all {len(test_df)} samples")
        
        # Save test dataset
        if Path(output_file).suffix == '.json':
            test_df.to_json(output_file, orient='records', indent=2)
        else:
            test_df.to_csv(output_file, index=False)
        
        print(f"💾 Saved test dataset to {output_file}")
        return test_df
        
    except Exception as e:
        print(f"❌ Error creating test dataset: {e}")
        return None

def validate_null_fields(df: pd.DataFrame, target_fields: List[str]) -> Dict[str, Any]:
    """Validate that target fields are not null after generation."""
    validation_results = {
        'total_records': len(df),
        'field_results': {},
        'failed_records': []
    }
    
    for field in target_fields:
        if field in df.columns:
            null_count = df[field].isnull().sum()
            empty_count = (df[field] == '').sum() if df[field].dtype == 'object' else 0
            total_invalid = null_count + empty_count
            
            validation_results['field_results'][field] = {
                'null_count': int(null_count),
                'empty_count': int(empty_count),
                'total_invalid': int(total_invalid),
                'success_rate': float((len(df) - total_invalid) / len(df) * 100)
            }
            
            # Collect failed record indices
            if total_invalid > 0:
                failed_mask = df[field].isnull() | (df[field] == '')
                failed_indices = df[failed_mask]['idx'].tolist() if 'idx' in df.columns else df[failed_mask].index.tolist()
                validation_results['failed_records'].extend(failed_indices)
        else:
            validation_results['field_results'][field] = {
                'error': f'Field {field} not found in dataset'
            }
    
    return validation_results

async def process_individual_null_records(generator: EnhancedType2Generator, 
                                        df: pd.DataFrame, 
                                        target_fields: List[str],
                                        max_retries: int = 3) -> pd.DataFrame:
    """
    Process null records individually with immediate retry until each record is filled.
    
    This approach ensures that each null record is processed individually and retried
    until successful, rather than processing the entire batch multiple times.
    """
    print(f"🔄 Starting individual null record processing (max {max_retries} retries per record)")
    
    updated_df = df.copy()
    
    for target_field in target_fields:
        print(f"\n🎯 Processing field: {target_field}")
        
        # Determine the method based on the target field
        if 'sentence_completion' in target_field:
            method = GenerationMethod.SENTENCE_COMPLETION
        elif 'question_answer' in target_field:
            method = GenerationMethod.QUESTION_ANSWER
        else:
            method = GenerationMethod.SENTENCE_COMPLETION  # Default
        
        print(f"📝 Using method: {method.value}")
        
        # Use the new individual processing method
        updated_df = await generator.process_null_records_individually(
            updated_df, 
            target_field, 
            method, 
            max_retries
        )
        
        # Validate after processing this field
        remaining_nulls = updated_df[target_field].isnull().sum() + (updated_df[target_field] == '').sum()
        print(f"📊 Remaining null values in {target_field}: {remaining_nulls}")
    
    return updated_df

async def test_type2_generation(num_samples: int = None, methods: list = None, start_idx: int = None, 
                               end_idx: int = None, input_dataset: str = None, null_process: bool = False):
    """Test Type 2 generation with the orchestrator.
    
    Args:
        num_samples: Number of samples to process from the beginning. If None, processes all samples.
        methods: List of methods to test. If None, tests both methods.
        start_idx: Starting index for range selection (inclusive)
        end_idx: Ending index for range selection (exclusive)
        input_dataset: Path to input dataset (overrides default)
        null_process: If True, validates and retries null generation until successful
    """
    # Determine description for progress display
    if start_idx is not None or end_idx is not None:
        range_desc = f"RANGE [{start_idx or 0}:{end_idx or 'end'})"
        print(f"🧪 Starting Type 2 Generation Test - {range_desc}")
    elif num_samples is None:
        print("🧪 Starting Type 2 Generation Test - ALL SAMPLES")
    else:
        print(f"🧪 Starting Type 2 Generation Test - {num_samples} SAMPLES")
    
    if null_process:
        print("🔄 Null processing enabled - will retry individual failed records until successful")
    
    print("=" * 60)
    
    # Paths (use provided input_dataset or default)
    if input_dataset:
        original_dataset = input_dataset
        print(f"📁 Using custom input dataset: {original_dataset}")
    else:
        original_dataset = str(project_root / "data/processed/unified_padben_base.json")
        print(f"📁 Using default input dataset: {original_dataset}")
    
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
        
        # Track all results for validation
        all_results = []
        target_fields = []
        
        for generation_method, method_name in methods_to_test:
            print(f"\n🎯 Testing {method_name} generation...")
            
            try:
                # Generate using the correct method: generate_for_dataset
                results_df = await generator.generate_for_dataset(
                    test_df, 
                    method=generation_method, 
                    output_dir=output_dir
                )
                
                if results_df is not None and len(results_df) > 0:
                    print(f"✅ {method_name} generation completed: {len(results_df)} samples processed")
                    
                    # Determine target field name
                    target_field = generator.get_method_column_name(generation_method)
                    target_fields.append(target_field)
                    
                    # Check for successful generations
                    if target_field in results_df.columns:
                        successful_count = results_df[target_field].notna().sum()
                        null_count = results_df[target_field].isnull().sum()
                        empty_count = (results_df[target_field] == '').sum()
                        total_failed = null_count + empty_count
                        
                        print(f"📊 Successful generations: {successful_count}/{len(results_df)}")
                        print(f"📊 Failed generations: {total_failed} (null: {null_count}, empty: {empty_count})")
                    
                    all_results.append(results_df)
                    
                    # Save individual method results with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    method_output_dir = Path(output_dir) / f"{method_name}_{timestamp}"
                    method_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    output_file = method_output_dir / f"padben_with_type2_{method_name}_{timestamp}.json"
                    results_df.to_json(output_file, orient='records', indent=2)
                    print(f"💾 Results saved to {output_file}")
                    
                else:
                    print(f"❌ {method_name} generation failed - no results returned")
                    return False
                    
            except Exception as e:
                print(f"❌ Error during {method_name} generation: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        # Step 4: Individual null processing if enabled
        if null_process and all_results:
            print(f"\n🔄 Step 4: Individual Null Record Processing")
            
            # Use the last results dataframe for null processing
            final_results = all_results[-1]
            
            # Initial validation
            initial_validation = validate_null_fields(final_results, target_fields)
            
            print("📊 Initial validation results:")
            for field, result in initial_validation['field_results'].items():
                if isinstance(result, dict) and 'total_invalid' in result:
                    print(f"  {field}: {result['total_invalid']} null/empty ({result['success_rate']:.1f}% success)")
            
            # Check if individual processing is needed
            total_nulls = sum(
                result.get('total_invalid', 0) 
                for result in initial_validation['field_results'].values()
                if isinstance(result, dict) and 'total_invalid' in result
            )
            
            if total_nulls > 0:
                print(f"⚠️ Found {total_nulls} null values - starting individual record processing")
                
                # Process individual null records with immediate retry
                final_results = await process_individual_null_records(
                    generator, 
                    final_results, 
                    target_fields,
                    max_retries=3
                )
                
                # Final validation
                final_validation = validate_null_fields(final_results, target_fields)
                print("\n📊 Final validation results after individual processing:")
                for field, result in final_validation['field_results'].items():
                    if isinstance(result, dict) and 'total_invalid' in result:
                        print(f"  {field}: {result['total_invalid']} null/empty ({result['success_rate']:.1f}% success)")
                
                # Save final results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                final_output_dir = Path(output_dir) / f"individual_null_processed_{timestamp}"
                final_output_dir.mkdir(parents=True, exist_ok=True)
                
                final_output_file = final_output_dir / f"padben_individual_null_processed_{timestamp}.json"
                final_results.to_json(final_output_file, orient='records', indent=2)
                print(f"💾 Final individually processed results saved to {final_output_file}")
                
                # Save validation report
                validation_file = final_output_dir / f"individual_null_validation_report_{timestamp}.json"
                validation_report = {
                    "initial_validation": initial_validation,
                    "final_validation": final_validation,
                    "processing_method": "individual_record_retry",
                    "timestamp": timestamp
                }
                with open(validation_file, 'w') as f:
                    json.dump(validation_report, f, indent=2)
                print(f"📋 Validation report saved to {validation_file}")
                
                # Check final success
                final_total_nulls = sum(
                    result.get('total_invalid', 0) 
                    for result in final_validation['field_results'].values()
                    if isinstance(result, dict) and 'total_invalid' in result
                )
                
                if final_total_nulls == 0:
                    print("🎉 All null values successfully filled through individual processing!")
                else:
                    print(f"⚠️ {final_total_nulls} null values still remain after individual processing")
                
            else:
                print("✅ No null values found - no individual processing needed")
        
        print(f"\n🎉 Type 2 generation test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function with enhanced argument parsing and safety checks."""
    global HAS_TQDM  # Add this line at the beginning of main()
    
    parser = argparse.ArgumentParser(
        description="Test Type 2 Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Test with 20 samples (default)
  %(prog)s --samples 100             # Test with 100 samples
  %(prog)s --all                     # Test with all samples
  %(prog)s --start 1000 --end 1100   # Test with samples 1000-1099
  %(prog)s --methods sentence_completion  # Test only sentence completion
  %(prog)s --input data/test/null_records.json --null-process  # Process null records with individual retry
        """
    )
    
    # Sample selection options (mutually exclusive)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--samples", type=int, default=20,
                      help="Number of samples to process from the beginning (default: 20)")
    group.add_argument("--all", action="store_true",
                      help="Process all samples in the dataset")
    
    # Range selection options
    parser.add_argument("--start", type=int,
                       help="Starting index for range selection (inclusive)")
    parser.add_argument("--end", type=int,
                       help="Ending index for range selection (exclusive)")
    
    # Method selection
    parser.add_argument("--methods", nargs='+', 
                       choices=["sentence_completion", "question_answer"],
                       help="Methods to test (default: both)")
    
    # Input dataset override
    parser.add_argument("--input", type=str,
                       help="Override input dataset path (default: data/processed/unified_padben_base.json)")
    
    # Null processing option
    parser.add_argument("--null-process", action="store_true",
                       help="Enable individual null processing - retry each failed record until successful")
    
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
    
    if args.null_process:
        print("🔄 Individual null processing enabled - will retry each failed record until successful")
    
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
        end_idx=end_idx,
        input_dataset=args.input,
        null_process=args.null_process
    ))
    
    if success:
        print("\n🎉 All tests passed! Type 2 generation is working correctly.")
    else:
        print("\n❌ Test failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
