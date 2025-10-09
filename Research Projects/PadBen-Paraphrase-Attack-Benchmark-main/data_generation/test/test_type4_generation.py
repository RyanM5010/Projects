#!/usr/bin/env python3
"""
Test Type 4 Generation with Configurable Sample Size

This script tests the Type 4 generation pipeline (LLM-paraphrased original text)
with a configurable number of samples from the unified dataset.
Tests both DIPPER and prompt-based paraphrasing methods with individual method selection.
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

# Now import the required modules
from data_generation.type4_generation import Type4Generator, EnvironmentMode
from data_generation.config.generation_model_config import DEFAULT_CONFIG
from data_generation.config.type4_config import Type4ParaphraseMethod

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
    
    # Add missing columns that Type 4 will fill
    if 'llm_paraphrased_original_text' not in test_df.columns:
        test_df['llm_paraphrased_original_text'] = None
    if 'llm_paraphrased_original_text(DIPPER_based)' not in test_df.columns:
        test_df['llm_paraphrased_original_text(DIPPER_based)'] = None
    if 'llm_paraphrased_original_text(Prompt_based)' not in test_df.columns:
        test_df['llm_paraphrased_original_text(Prompt_based)'] = None
    
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

async def retry_single_record_paraphrasing(generator: Type4Generator, 
                                          row: pd.Series, 
                                          method: Type4ParaphraseMethod,
                                          max_retries: int = 3):
    """
    Retry paraphrasing for a single record until successful or max retries reached.
    
    Args:
        generator: Type4Generator instance
        row: DataFrame row containing original text
        method: Paraphrasing method to use
        max_retries: Maximum number of retry attempts
        
    Returns:
        ParaphraseResult with paraphrased text and metadata
    """
    sample_idx = row.get('idx', row.name if hasattr(row, 'name') else -1)
    
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            print(f"Attempt {attempt + 1}/{max_retries + 1} for sample {sample_idx}")
            
            result = await generator.paraphrase_text(row, method)
            
            if result.success and result.paraphrased_text and result.paraphrased_text.strip():
                print(f"✅ Successfully paraphrased text for sample {sample_idx} on attempt {attempt + 1}")
                result.metadata["retry_attempt"] = attempt + 1
                result.metadata["retry_successful"] = True
                return result
            else:
                failure_reason = result.metadata.get("failure_reason", "Unknown failure")
                print(f"⚠️ Attempt {attempt + 1} failed for sample {sample_idx}: {failure_reason}")
                
                if attempt < max_retries:
                    # Brief delay before retry to avoid rate limiting
                    await asyncio.sleep(1.0)
                
        except Exception as e:
            print(f"❌ Exception during attempt {attempt + 1} for sample {sample_idx}: {str(e)}")
            if attempt < max_retries:
                await asyncio.sleep(1.0)
    
    # All attempts failed
    print(f"❌ All {max_retries + 1} attempts failed for sample {sample_idx}")
    from data_generation.type4_generation import ParaphraseResult
    return ParaphraseResult(
        paraphrased_text=None,
        method_used=method,
        success=False,
        metadata={
            "failure_reason": f"Failed after {max_retries + 1} attempts",
            "retry_attempt": max_retries + 1,
            "retry_successful": False,
            "sample_idx": sample_idx
        }
    )

async def process_individual_null_records(generator: Type4Generator, 
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
        if 'DIPPER_based' in target_field:
            method = Type4ParaphraseMethod.DIPPER
        elif 'Prompt_based' in target_field:
            method = Type4ParaphraseMethod.PROMPT_BASED
        else:
            method = Type4ParaphraseMethod.DIPPER  # Default
        
        print(f"📝 Using method: {method.value}")
        
        # Find records with null values in the target field
        null_mask = updated_df[target_field].isnull() | (updated_df[target_field] == '')
        null_records = updated_df[null_mask].copy()
        
        if len(null_records) == 0:
            print("✅ No null records found")
            continue
        
        print(f"🎯 Found {len(null_records)} records with null values")
        
        # Process each null record individually
        successful_fills = 0
        failed_fills = 0
        
        # Progress tracking
        if HAS_TQDM:
            progress_bar = tqdm(total=len(null_records), desc=f"Processing {target_field} null records")
        
        for idx, row in null_records.iterrows():
            try:
                # Retry until successful or max attempts reached
                result = await retry_single_record_paraphrasing(generator, row, method, max_retries)
                
                if result.success and result.paraphrased_text:
                    # Update the DataFrame with the successful result
                    updated_df.at[idx, target_field] = result.paraphrased_text
                    successful_fills += 1
                    
                    if HAS_TQDM:
                        progress_bar.set_postfix({
                            'Success': successful_fills,
                            'Failed': failed_fills,
                            'Rate': f'{(successful_fills/(successful_fills+failed_fills)*100):.1f}%' if (successful_fills+failed_fills) > 0 else '0%'
                        })
                else:
                    failed_fills += 1
                    print(f"❌ Failed to fill record {row.get('idx', idx)} after all retries")
                
                if HAS_TQDM:
                    progress_bar.update(1)
                    
            except Exception as e:
                failed_fills += 1
                print(f"❌ Exception processing record {row.get('idx', idx)}: {str(e)}")
                if HAS_TQDM:
                    progress_bar.update(1)
        
        if HAS_TQDM:
            progress_bar.close()
        
        print(f"📊 Individual processing completed for {target_field}:")
        print(f"  Successfully filled: {successful_fills}")
        print(f"  Failed to fill: {failed_fills}")
        print(f"  Success rate: {(successful_fills/(successful_fills+failed_fills)*100):.1f}%" if (successful_fills+failed_fills) > 0 else "0%")
        
        # Validate after processing this field
        remaining_nulls = updated_df[target_field].isnull().sum() + (updated_df[target_field] == '').sum()
        print(f"📊 Remaining null values in {target_field}: {remaining_nulls}")
    
    return updated_df

async def test_single_method(method_enum, method_name, test_df, output_dir, method_pbar=None):
    """Test a single paraphrasing method with progress tracking."""
    print(f"\n--- Testing {method_name} method ---")
    print(f"Processing {len(test_df)} samples...")
    
    if method_pbar and HAS_TQDM:
        method_pbar.set_description(f"Testing {method_name}")
    
    start_time = datetime.now()
    
    # Initialize Type 4 generator with test environment mode
    generator = Type4Generator(DEFAULT_CONFIG.type4_config, environment_mode=EnvironmentMode.TEST)
    
    # Create a clean copy for this test
    test_df_copy = test_df.copy()
    
    # Run generation for this method
    results_df = await generator.generate_for_dataset(
        test_df_copy,
        method=method_enum,
        output_dir=output_dir
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Collect results - check for both column name formats
    dipper_column = 'llm_paraphrased_original_text(DIPPER_based)'
    prompt_column = 'llm_paraphrased_original_text(Prompt_based)'
    
    # Count generated texts from both possible columns
    dipper_count = results_df[dipper_column].notna().sum() if dipper_column in results_df.columns else 0
    prompt_count = results_df[prompt_column].notna().sum() if prompt_column in results_df.columns else 0
    
    # Also check the generic column
    generic_count = results_df['llm_paraphrased_original_text'].notna().sum() if 'llm_paraphrased_original_text' in results_df.columns else 0
    
    # Use the maximum count (some methods might use different column names)
    generated_count = max(dipper_count + prompt_count, generic_count)
    
    total_count = len(results_df)
    success_rate = (generated_count / total_count) * 100
    
    result_data = {
        'generated_count': generated_count,
        'total_count': total_count,
        'success_rate': success_rate,
        'duration': duration,
        'results_df': results_df,
        'method_name': method_name,
        'target_field': generator.get_method_column_name(method_enum)
    }
    
    print(f"✅ {method_name} completed:")
    print(f"   Generated: {generated_count}/{total_count} ({success_rate:.1f}%)")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Avg time per sample: {duration/total_count:.2f} seconds")
    
    # Update method progress bar
    if method_pbar and HAS_TQDM:
        method_pbar.set_postfix({
            'Success': f'{success_rate:.1f}%',
            'Time': f'{duration:.1f}s'
        })
    
    return result_data

async def test_type4_generation(num_samples: int = None, methods: list = None, start_idx: int = None, 
                               end_idx: int = None, input_dataset: str = None, null_process: bool = False):
    """Test Type 4 generation with configurable sample selection.
    
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
        print(f"🧪 Starting Type 4 Generation Test - {range_desc}")
    elif num_samples is None:
        print("🧪 Starting Type 4 Generation Test - ALL SAMPLES")
    else:
        print(f"🧪 Starting Type 4 Generation Test - {num_samples} SAMPLES")
    
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
        test_dataset = str(project_root / f"data/test/test_unified_padben_type4_range_{start_str}_{end_str}.json")
    elif num_samples is None:
        test_dataset = str(project_root / "data/test/test_unified_padben_type4_all_samples.json")
    else:
        test_dataset = str(project_root / f"data/test/test_unified_padben_type4_{num_samples}_samples.json")
    
    output_dir = str(project_root / "data/test/type4_generation_test")
    
    try:
        # Step 1: Create test dataset
        print("Step 1: Creating test dataset...")
        test_df = create_test_dataset(original_dataset, test_dataset, num_samples=num_samples, start_idx=start_idx, end_idx=end_idx)
        
        # Step 2: Determine which methods to test
        if methods is None:
            methods_to_test = [
                (Type4ParaphraseMethod.DIPPER, "dipper"),
                (Type4ParaphraseMethod.PROMPT_BASED, "prompt_based")
            ]
            print("\nStep 2: Testing both paraphrasing methods...")
        else:
            method_map = {
                "dipper": (Type4ParaphraseMethod.DIPPER, "dipper"),
                "prompt_based": (Type4ParaphraseMethod.PROMPT_BASED, "prompt_based")
            }
            methods_to_test = [method_map[method] for method in methods if method in method_map]
            print(f"\nStep 2: Testing selected methods: {', '.join(methods)}")
        
        # Step 3: Test methods with progress tracking
        print("\nStep 3: Testing paraphrasing methods...")
        
        results = {}
        target_fields = []
        all_results = []
        
        # Main progress bar for methods
        method_pbar = tqdm(
            methods_to_test,
            desc="Testing methods",
            unit="method",
            disable=not HAS_TQDM
        )
        
        for method_enum, method_name in method_pbar:
            result_data = await test_single_method(method_enum, method_name, test_df, output_dir, method_pbar)
            results[method_name] = result_data
            target_fields.append(result_data['target_field'])
            all_results.append(result_data['results_df'])
        
        method_pbar.close()
        
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
                
                # Initialize generator for null processing
                generator = Type4Generator(DEFAULT_CONFIG.type4_config, environment_mode=EnvironmentMode.TEST)
                
                # Process individual null records with immediate retry using the new method
                for target_field in target_fields:
                    # Determine the method based on the target field
                    if 'DIPPER_based' in target_field:
                        method = Type4ParaphraseMethod.DIPPER
                    elif 'Prompt_based' in target_field:
                        method = Type4ParaphraseMethod.PROMPT_BASED
                    else:
                        method = Type4ParaphraseMethod.DIPPER  # Default
                    
                    print(f"\n🎯 Processing field: {target_field} using method: {method.value}")
                    
                    # Use the new individual processing method
                    final_results = await generator.process_null_records_individually(
                        final_results, 
                        target_field, 
                        method, 
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
                
                final_output_file = final_output_dir / f"padben_type4_individual_null_processed_{timestamp}.json"
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
        
        # Step 5: Analyze and compare results
        print("\n" + "=" * 60)
        print("🎉 TYPE 4 GENERATION TEST COMPLETED")
        print("=" * 60)
        
        if len(results) > 1:
            print("📊 Method Comparison:")
        else:
            print(f"📊 {list(results.keys())[0].title()} Method Results:")
        
        for method_name, result in results.items():
            print(f"  {method_name}:")
            print(f"    Success rate: {result['success_rate']:.1f}%")
            print(f"    Processing time: {result['duration']:.2f} seconds")
            print(f"    Avg time per sample: {result['duration']/result['total_count']:.2f} seconds")
            print(f"    Total samples: {result['total_count']}")
        
        # Show sample outputs from each method (limit to 2 samples for readability)
        print(f"\n📋 Sample Paraphrased Texts:")
        print("-" * 40)
        
        for method_name, result in results.items():
            print(f"\n--- {method_name.title()} Method ---")
            df = result['results_df']
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
                # Check for paraphrased text in different possible columns
                paraphrased_text = None
                if pd.notna(row.get('llm_paraphrased_original_text')):
                    paraphrased_text = row['llm_paraphrased_original_text']
                elif pd.notna(row.get('llm_paraphrased_original_text(DIPPER_based)')):
                    paraphrased_text = row['llm_paraphrased_original_text(DIPPER_based)']
                elif pd.notna(row.get('llm_paraphrased_original_text(Prompt_based)')):
                    paraphrased_text = row['llm_paraphrased_original_text(Prompt_based)']
                
                if paraphrased_text and sample_count < 2:
                    sample_count += 1
                    print(f"Sample {sample_count}:")
                    print(f"  Dataset: {row['dataset_source']}")
                    print(f"  Original: {row['human_original_text'][:100]}{'...' if len(row['human_original_text']) > 100 else ''}")
                    print(f"  Paraphrased: {paraphrased_text[:100]}{'...' if len(paraphrased_text) > 100 else ''}")
                    print()
        
        # Check output directories with progress
        print(f"📁 Output Directories:")
        output_path = Path(output_dir)
        if output_path.exists():
            method_names = [m[1] for m in methods_to_test]
            
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
                method_dirs = list(output_path.glob(f"{method_name}_based_*"))
                if method_dirs:
                    latest_dir = max(method_dirs, key=lambda p: p.name)
                    print(f"  {method_name}_based: {latest_dir}")
                    
                    # Check for files in the directory
                    method_files = list(latest_dir.glob("*"))
                    if method_files:
                        print(f"    Files created: {len(method_files)}")
                        for file in sorted(method_files)[:3]:  # Show first 3 files
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
    """Main function to run the test with enhanced command-line arguments."""
    global HAS_TQDM  # Fix: Access the global HAS_TQDM variable
    
    parser = argparse.ArgumentParser(description="PADBen Type 4 Generation Test")
    
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
                       choices=["dipper", "prompt_based"],
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
        
        print("🚀 PADBen Type 4 Generation Test")
        print("Testing LLM-paraphrased original text generation")
        print(f"Processing samples from index {actual_start} to {actual_end-1} ({actual_count} samples)")
        
    elif args.all:
        num_samples = None
        start_idx = None
        end_idx = None
        print("🚀 PADBen Type 4 Generation Test")
        print("Testing LLM-paraphrased original text generation")
        print("Processing ALL samples from unified dataset")
        
    else:
        num_samples = args.samples
        start_idx = None
        end_idx = None
        print("🚀 PADBen Type 4 Generation Test")
        print("Testing LLM-paraphrased original text generation")
        print(f"Processing first {num_samples} samples from unified dataset")
    
    if args.methods:
        print(f"Testing methods: {', '.join(args.methods)}")
    else:
        print("Testing both DIPPER and prompt-based methods")
    
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
        estimated_time = total_samples * 3.0  # Rough estimate: 3.0 seconds per sample for Type 4
        
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
        estimated_time = actual_count * 3.0  # Rough estimate: 3.0 seconds per sample for Type 4
        
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
    success = asyncio.run(test_type4_generation(
        num_samples=num_samples, 
        methods=args.methods, 
        start_idx=start_idx, 
        end_idx=end_idx,
        input_dataset=args.input,  # Add this line
        null_process=args.null_process
    ))
    
    if success:
        if args.methods and len(args.methods) == 1:
            print(f"\n🎉 {args.methods[0].title()} method test passed! Type 4 generation is working correctly.")
        else:
            print("\n🎉 All tests passed! Type 4 generation is working correctly.")
    else:
        print("\n❌ Test failed. Please check the error messages above.")

if __name__ == "__main__":
    main()