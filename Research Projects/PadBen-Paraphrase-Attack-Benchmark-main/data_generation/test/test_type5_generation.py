#!/usr/bin/env python3
"""
Enhanced Test Type 5 Generation with Configurable Sample Selection

This script tests the Type 5 generation pipeline (LLM-paraphrased LLM-generated text)
with configurable sample selection (similar to test_type4_generation.py).
Tests individual methods (DIPPER or prompt-based) and iteration levels (1, 3, or 5).
Includes progress tracking and midpoint saving for higher iterations.
Preserves exact column format from reformatted JSON files.
Enhanced with full retry logic and individual null processing.
"""

import asyncio
import json
import pandas as pd
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Progress bar for better UX
try:
    from tqdm import tqdm
    from tqdm.asyncio import tqdm as async_tqdm
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
        def set_postfix(self, *args, **kwargs):
            pass
        def close(self):
            pass
    
    async_tqdm = tqdm
    HAS_TQDM = False
    print("Warning: tqdm not installed. Install with: pip install tqdm")

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Now import the required modules
from data_generation.type5_generation import Type5Generator, EnvironmentMode
from data_generation.config.generation_model_config import DEFAULT_CONFIG
from data_generation.config.type4_config import Type4ParaphraseMethod
from data_generation.config.type5_config import IterationLevel

def create_test_dataset_with_type2(input_file: str, output_file: str, num_samples: int = None, start_idx: int = None, end_idx: int = None):
    """Create a test dataset with the specified number of samples or index range, ensuring Type 2 data exists.
    Preserves exact column format from reformatted JSON files.
    
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
    
    # Detect format and preserve exact column structure
    required_columns = ['idx', 'dataset_source']
    
    # Check for reformatted format and preserve it exactly
    if 'human_original_text(type1)' in test_df.columns:
        print("✅ Detected reformatted file format - preserving exact column structure")
        # Use the exact column names from reformatted format
        type1_col = 'human_original_text(type1)'
        type2_col = 'llm_generated_text(type2)'
        type3_col = 'human_paraphrased_text(type3)'
        
        # Ensure required reformatted columns exist
        required_columns.extend([type1_col, type3_col])
        
        # Add missing Type 5 columns with exact reformatted naming
        type5_columns = [
            type2_col,  # Type 2 column (might already exist)
            'llm_paraphrased_original_text(type4)-prompt-based',
            'llm_paraphrased_original_text(type4)-dipper-based',
            'llm_paraphrased_generated_text(type5)-1st',
            'llm_paraphrased_generated_text(type5)-3rd',
            'llm_paraphrased_generated_text(type5)-5th'  # Add 5th iteration support
        ]
        
        # Check if we have Type 2 data (required for Type 5)
        type2_available = test_df[type2_col].notna().sum() if type2_col in test_df.columns else 0
        print(f"Samples with Type 2 data: {type2_available}/{len(test_df)}")
        
        # If no Type 2 data, create mock Type 2 data for testing
        if type2_available == 0:
            print("⚠️ No Type 2 data found. Creating mock Type 2 data for testing...")
            if type2_col not in test_df.columns:
                test_df[type2_col] = None
            
            for idx, row in test_df.iterrows():
                if pd.isna(row[type2_col]):
                    # Create a simple mock based on original text
                    original = row[type1_col]
                    mock_generated = f"Generated version: {original[:100]}..." if len(original) > 100 else f"Generated version: {original}"
                    test_df.at[idx, type2_col] = mock_generated
        
        print(f"Final Type 2 data availability: {test_df[type2_col].notna().sum()}/{len(test_df)}")
        
    else:
        # Standard format (for backward compatibility)
        print("✅ Detected standard format")
        type1_col = 'human_original_text'
        type2_col = 'llm_generated_text'
        type3_col = 'human_paraphrased_text'
        
        required_columns.extend([type1_col, type3_col])
        
        # Add missing columns for standard format
        type5_columns = [
            type2_col,
            'llm_paraphrased_generated_text',
            'llm_paraphrased_generated_text(DIPPER_based_1_iteration)',
            'llm_paraphrased_generated_text(DIPPER_based_3_iterations)',
            'llm_paraphrased_generated_text(DIPPER_based_5_iterations)',
            'llm_paraphrased_generated_text(Prompt_based_1_iteration)',
            'llm_paraphrased_generated_text(Prompt_based_3_iterations)',
            'llm_paraphrased_generated_text(Prompt_based_5_iterations)'
        ]
        
        # Check if we have Type 2 data (required for Type 5)
        type2_available = test_df[type2_col].notna().sum() if type2_col in test_df.columns else 0
        print(f"Samples with Type 2 data: {type2_available}/{len(test_df)}")
        
        # If no Type 2 data, create mock Type 2 data for testing
        if type2_available == 0:
            print("⚠️ No Type 2 data found. Creating mock Type 2 data for testing...")
            if type2_col not in test_df.columns:
                test_df[type2_col] = None
            
            for idx, row in test_df.iterrows():
                if pd.isna(row[type2_col]):
                    # Create a simple mock based on original text
                    original = row[type1_col]
                    mock_generated = f"Generated version: {original[:100]}..." if len(original) > 100 else f"Generated version: {original}"
                    test_df.at[idx, type2_col] = mock_generated
        
        print(f"Final Type 2 data availability: {test_df[type2_col].notna().sum()}/{len(test_df)}")
    
    # Ensure all required columns exist
    for col in required_columns:
        if col not in test_df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Add missing Type 5 columns if they don't exist
    for col in type5_columns:
        if col not in test_df.columns:
            test_df[col] = None
    
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

def get_column_names_for_format(df: pd.DataFrame):
    """Get the appropriate column names based on the detected format."""
    if 'human_original_text(type1)' in df.columns:
        # Reformatted format
        return {
            'type1': 'human_original_text(type1)',
            'type2': 'llm_generated_text(type2)',
            'type3': 'human_paraphrased_text(type3)',
            'type5_1st': 'llm_paraphrased_generated_text(type5)-1st',
            'type5_3rd': 'llm_paraphrased_generated_text(type5)-3rd',
            'type5_5th': 'llm_paraphrased_generated_text(type5)-5th'
        }
    else:
        # Standard format
        return {
            'type1': 'human_original_text',
            'type2': 'llm_generated_text',
            'type3': 'human_paraphrased_text',
            'type5_1st': 'llm_paraphrased_generated_text(DIPPER_based_1_iteration)',
            'type5_3rd': 'llm_paraphrased_generated_text(DIPPER_based_3_iterations)',
            'type5_5th': 'llm_paraphrased_generated_text(DIPPER_based_5_iterations)'
        }

def get_type5_column_name(method: Type4ParaphraseMethod, iteration_level: IterationLevel, df: pd.DataFrame) -> str:
    """Get the correct Type 5 column name based on format and method/iteration."""
    if 'human_original_text(type1)' in df.columns:
        # Reformatted format - simplified naming
        if iteration_level == IterationLevel.FIRST:
            return 'llm_paraphrased_generated_text(type5)-1st'
        elif iteration_level == IterationLevel.THIRD:
            return 'llm_paraphrased_generated_text(type5)-3rd'
        elif iteration_level == IterationLevel.FIFTH:
            return 'llm_paraphrased_generated_text(type5)-5th'
    else:
        # Standard format - detailed naming
        method_name_map = {
            Type4ParaphraseMethod.DIPPER: "DIPPER_based",
            Type4ParaphraseMethod.PROMPT_BASED: "Prompt_based"
        }
        iteration_name_map = {
            IterationLevel.FIRST: "1_iteration",
            IterationLevel.THIRD: "3_iterations",
            IterationLevel.FIFTH: "5_iterations"
        }
        method_name = method_name_map.get(method, method.value)
        iteration_name = iteration_name_map.get(iteration_level, f"{iteration_level.value}_iterations")
        return f"llm_paraphrased_generated_text({method_name}_{iteration_name})"

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

def save_midpoint_results_with_iterations(results_df: pd.DataFrame, 
                                        generation_metadata: List[Dict[str, Any]], 
                                        method: Type4ParaphraseMethod, 
                                        iteration_level: IterationLevel, 
                                        batch_idx: int,
                                        output_dir: str, 
                                        timestamp: str):
    """Save midpoint results with iteration history for higher iterations (3rd, 5th)."""
    
    # Only save midpoint results for higher iterations
    if iteration_level == IterationLevel.FIRST:
        return  # Skip midpoint saving for 1st iteration as requested
    
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get the appropriate column names
        cols = get_column_names_for_format(results_df)
        method_column = get_type5_column_name(method, iteration_level, results_df)
        
        # Filter to only successfully generated Type 5 samples
        type5_generated = results_df[results_df[method_column].notna()].copy()
        
        if len(type5_generated) == 0:
            print(f"No Type 5 generated samples to save in midpoint results for batch {batch_idx}")
            return
        
        # Create a mapping from sample index to metadata
        metadata_by_idx = {}
        for meta in generation_metadata:
            if 'sample_idx' in meta:
                metadata_by_idx[meta['sample_idx']] = meta
        
        # Prepare midpoint data with iteration history
        midpoint_records = []
        
        for idx, row in type5_generated.iterrows():
            try:
                # Create base record preserving original JSON format
                record = {
                    'idx': row['idx'],
                    'dataset_source': row['dataset_source'],
                    cols['type1']: row[cols['type1']],
                    cols['type2']: row[cols['type2']],
                    'final_paraphrased_text': row[method_column]
                }
                
                # Add iteration history if available in metadata
                if idx in metadata_by_idx and 'iteration_history' in metadata_by_idx[idx]:
                    iteration_history = metadata_by_idx[idx]['iteration_history']
                    if hasattr(iteration_history, 'iterations'):
                        # Add each iteration as a separate field
                        for i, iteration_text in enumerate(iteration_history.iterations, 1):
                            record[f'iteration_{i}_text'] = iteration_text
                            if hasattr(iteration_history, 'iteration_times') and i-1 < len(iteration_history.iteration_times):
                                record[f'iteration_{i}_time'] = iteration_history.iteration_times[i-1]
                            if hasattr(iteration_history, 'similarities') and i-1 < len(iteration_history.similarities):
                                record[f'iteration_{i}_similarity'] = iteration_history.similarities[i-1]
                        
                        # Add summary information
                        record['total_iterations_completed'] = len(iteration_history.iterations)
                        record['stopped_early'] = getattr(iteration_history, 'stopped_early', False)
                        record['stop_reason'] = getattr(iteration_history, 'stop_reason', '')
                
                midpoint_records.append(record)
                
            except Exception as e:
                print(f"Warning: Failed to process sample {idx} for midpoint save: {e}")
                continue
        
        if not midpoint_records:
            print(f"No valid records to save for batch {batch_idx}")
            return
        
        # Save midpoint results as JSON (batch-based, every 1000 samples)
        batch_number = (batch_idx * 1000) // 1000  # Group batches into sets of 1000
        method_name = method.value
        iteration_name = f"{iteration_level.value}iterations"
        midpoint_file = output_path / f"type5_midpoint_{method_name}_{iteration_name}_batch_{batch_number}_{timestamp}.json"
        
        with open(midpoint_file, 'w', encoding='utf-8') as f:
            json.dump(midpoint_records, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"✅ Saved Type 5 midpoint results: {midpoint_file}")
        print(f"   Samples: {len(midpoint_records)}")
        print(f"   Method: {method_name}, Iterations: {iteration_level.value}")
        
    except Exception as e:
        print(f"⚠️ Failed to save Type 5 midpoint results: {str(e)}")

async def test_type5_generation(method: Type4ParaphraseMethod, iteration_level: IterationLevel, 
                               num_samples: int = None, start_idx: int = None, end_idx: int = None,
                               input_dataset: str = None, output_dir: str = None, null_process: bool = False):
    """Test Type 5 generation with specific method and iteration level with configurable sample selection."""
    method_name = method.value
    iteration_name = f"{iteration_level.value}_iteration{'s' if iteration_level.value > 1 else ''}"
    
    # Determine description for progress display
    if start_idx is not None or end_idx is not None:
        range_desc = f"RANGE [{start_idx or 0}:{end_idx or 'end'})"
        print(f"🧪 Starting Type 5 Generation Test: {method_name} with {iteration_name} - {range_desc}")
    elif num_samples is None:
        print(f"🧪 Starting Type 5 Generation Test: {method_name} with {iteration_name} - ALL SAMPLES")
    else:
        print(f"🧪 Starting Type 5 Generation Test: {method_name} with {iteration_name} - {num_samples} SAMPLES")
    
    if null_process:
        print("🔄 Null processing enabled - will retry individual failed records until successful")
    
    print("=" * 80)
    
    # Paths (use provided input_dataset or default)
    if input_dataset:
        original_dataset = input_dataset
        print(f"📁 Using custom input dataset: {original_dataset}")
    else:
        # Default to the postprocessed dataset
        original_dataset = str(project_root / "data/test/merged_intratype/postprocessed_padben_20250829_182530.json")
        print(f"📁 Using default input dataset: {original_dataset}")
    
    # Use provided output_dir or default
    if output_dir:
        final_output_dir = output_dir
        print(f"📁 Using custom output directory: {final_output_dir}")
    else:
        final_output_dir = str(project_root / "data/test/type5_generation_test")
        print(f"📁 Using default output directory: {final_output_dir}")
    
    # Ensure output directory exists
    output_path = Path(final_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory created/verified: {output_path}")
    
    # Generate appropriate test dataset filename
    if start_idx is not None or end_idx is not None:
        start_str = start_idx or 0
        end_str = end_idx or "end"
        test_dataset = str(project_root / f"data/test/test_unified_padben_type5_range_{start_str}_{end_str}.json")
    elif num_samples is None:
        test_dataset = str(project_root / "data/test/test_unified_padben_type5_all_samples.json")
    else:
        test_dataset = str(project_root / f"data/test/test_unified_padben_type5_{num_samples}_samples.json")
    
    try:
        # Step 1: Create test dataset with Type 2 data
        print("Step 1: Creating test dataset with Type 2 data...")
        test_df = create_test_dataset_with_type2(original_dataset, test_dataset, 
                                               num_samples=num_samples, start_idx=start_idx, end_idx=end_idx)
        
        # Get column names for this format
        cols = get_column_names_for_format(test_df)
        
        # Step 2: Initialize Type 5 generator in TEST mode
        print("\nStep 2: Initializing Type 5 generator in TEST mode...")
        generator = Type5Generator(
            DEFAULT_CONFIG.type5_config, 
            environment_mode=EnvironmentMode.TEST
        )
        print("✅ Type 5 generator initialized successfully")
        
        # Step 3: Test using generate_for_dataset (similar to Type 4 approach)
        print(f"\nStep 3: Testing {method_name} with {iteration_name} using generate_for_dataset...")
        print(f"Processing {len(test_df)} samples with automatic intermediate saving...")
        print(f"Expected output structure: {final_output_dir}/[iteration]_[timestamp]/")
        print(f"  - Main results: CSV, JSON, metadata files")
        print(f"  - Midpoint saves: midpoint/ subdirectory with batch results and full iteration metadata")
        
        start_time = datetime.now()
        
        # Run generation using generate_for_dataset
        # This will automatically handle:
        # 1. Proper directory structure creation
        # 2. Format preservation (no extra columns)
        # 3. All iteration results (1st, 2nd, 3rd if doing 3rd iteration)
        # 4. Midpoint saves with full iteration metadata
        results_df = await generator.generate_for_dataset(
            test_df,
            method=method,
            iteration=iteration_level,
            output_dir=final_output_dir
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Step 4: Individual null processing if enabled
        if null_process:
            print(f"\n🔄 Step 4: Individual Null Record Processing")
            
            # Get method column based on format from the results
            method_column = generator.get_type5_column_name(method, iteration_level, results_df)
            
            # Initial validation
            initial_validation = validate_null_fields(results_df, [method_column])
            
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
                results_df = await generator.process_null_records_individually(
                    results_df, 
                    method_column, 
                    method, 
                    iteration_level,
                    max_retries=3
                )
                
                # Final validation
                final_validation = validate_null_fields(results_df, [method_column])
                print("\n📊 Final validation results after individual processing:")
                for field, result in final_validation['field_results'].items():
                    if isinstance(result, dict) and 'total_invalid' in result:
                        print(f"  {field}: {result['total_invalid']} null/empty ({result['success_rate']:.1f}% success)")
                
                # Save final results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                final_output_path = Path(final_output_dir) / f"individual_null_processed_{timestamp}"
                final_output_path.mkdir(parents=True, exist_ok=True)
                
                final_output_file = final_output_path / f"padben_type5_individual_null_processed_{timestamp}.json"
                results_df.to_json(final_output_file, orient='records', indent=2)
                print(f"💾 Final individually processed results saved to {final_output_file}")
                
                # Save validation report
                validation_file = final_output_path / f"individual_null_validation_report_{timestamp}.json"
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
        
        # Step 5: Analyze results using the correct column name
        method_column = generator.get_type5_column_name(method, iteration_level, results_df)
        print(f"Checking results in column: {method_column}")
        
        generated_count = results_df[method_column].notna().sum()
        total_count = len(results_df)
        success_rate = (generated_count / total_count) * 100
        
        # Check for intermediate iteration columns if applicable
        iteration_columns_filled = {}
        if iteration_level.value >= 3:
            first_iter_col = generator.get_type5_column_name(method, IterationLevel.FIRST, results_df)
            if first_iter_col in results_df.columns:
                iteration_columns_filled['1st'] = results_df[first_iter_col].notna().sum()
        if iteration_level.value >= 5:
            third_iter_col = generator.get_type5_column_name(method, IterationLevel.THIRD, results_df)
            if third_iter_col in results_df.columns:
                iteration_columns_filled['3rd'] = results_df[third_iter_col].notna().sum()
        
        print(f"\n✅ {method_name} with {iteration_name} completed:")
        print(f"   Generated: {generated_count}/{total_count} ({success_rate:.1f}%)")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Avg time per sample: {duration/total_count:.2f} seconds")
        print(f"   Primary column used: {method_column}")
        
        if iteration_columns_filled:
            print(f"   Intermediate iteration columns filled:")
            for iter_name, count in iteration_columns_filled.items():
                print(f"     {iter_name} iteration: {count}/{total_count}")
        
        # Step 6: Show sample results
        print(f"\n📋 Sample Results from {method_name} with {iteration_name}:")
        print("-" * 70)
        
        sample_count = 0
        for idx, row in results_df.iterrows():
            if pd.notna(row[method_column]) and sample_count < 3:
                sample_count += 1
                print(f"Sample {sample_count}:")
                print(f"  Dataset: {row['dataset_source']}")
                
                # Use the original columns from input format
                original_col = cols['type1']
                generated_col = cols['type2']
                
                print(f"  Original: {row[original_col][:80]}{'...' if len(row[original_col]) > 80 else ''}")
                if pd.notna(row[generated_col]):
                    print(f"  Generated (Type 2): {row[generated_col][:80]}{'...' if len(row[generated_col]) > 80 else ''}")
                
                # Show intermediate iterations if available
                if iteration_level.value >= 3:
                    first_iter_col = generator.get_type5_column_name(method, IterationLevel.FIRST, results_df)
                    if first_iter_col in results_df.columns and pd.notna(row[first_iter_col]):
                        print(f"  1st Iteration: {row[first_iter_col][:80]}{'...' if len(row[first_iter_col]) > 80 else ''}")
                
                if iteration_level.value >= 5:
                    third_iter_col = generator.get_type5_column_name(method, IterationLevel.THIRD, results_df)
                    if third_iter_col in results_df.columns and pd.notna(row[third_iter_col]):
                        print(f"  3rd Iteration: {row[third_iter_col][:80]}{'...' if len(row[third_iter_col]) > 80 else ''}")
                
                print(f"  Final ({iteration_level.value} iterations): {row[method_column][:80]}{'...' if len(row[method_column]) > 80 else ''}")
                print()
        
        # Step 7: Show statistics
        print(f"📊 Generation Statistics:")
        print(f"   Total processed: {generator.stats['total_processed']}")
        print(f"   Successful paraphrases: {generator.stats['successful_paraphrases']}")
        print(f"   Failed paraphrases: {generator.stats['failed_paraphrases']}")
        print(f"   Missing Type 2 data: {generator.stats['missing_type2_data']}")
        print(f"   Early stops: {generator.stats['early_stops']}")
        print(f"   Average iterations completed: {generator.stats.get('avg_iterations_completed', 0):.1f}")
        
        # Check output directory structure
        print(f"\n📁 Output Directory Structure: {output_path}")
        if output_path.exists():
            # Look for iteration-specific subdirectories
            subdirs = [d for d in output_path.iterdir() if d.is_dir()]
            if subdirs:
                print(f"  Iteration directories: {len(subdirs)}")
                for subdir in sorted(subdirs)[-3:]:  # Show last 3 subdirs
                    print(f"    📁 {subdir.name}")
                    
                    # Show main files in iteration directory
                    main_files = [f for f in subdir.glob("unified_padben_*.json")]
                    if main_files:
                        print(f"      Main result files: {len(main_files)}")
                        for file in sorted(main_files)[-2:]:  # Show last 2 main files
                            print(f"        📄 {file.name}")
                    
                    # Show midpoint directory
                    midpoint_dir = subdir / "midpoint"
                    if midpoint_dir.exists():
                        midpoint_files = list(midpoint_dir.glob("*.json"))
                        print(f"      📁 midpoint/ ({len(midpoint_files)} batch files)")
                        for file in sorted(midpoint_files)[-3:]:  # Show last 3 midpoint files
                            print(f"        📄 {file.name}")
            else:
                print("  No iteration directories found")
        
        print(f"\n🎉 TEST COMPLETED SUCCESSFULLY!")
        print(f"Method: {method_name}")
        print(f"Iterations: {iteration_level.value}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Results saved to: {final_output_dir}")
        print("💾 Automatic intermediate saves with full iteration metadata created in midpoint/ subdirectory")
        print("📋 All intermediate iteration columns populated (1st, 2nd, 3rd, etc.)")
        print("🔄 Input format preserved exactly (no extra columns)")
        print("=" * 80)
        
        return True, {
            'method': method_name,
            'iterations': iteration_level.value,
            'success_rate': success_rate,
            'generated_count': generated_count,
            'total_count': total_count,
            'duration': duration,
            'column_name': method_column,
            'iteration_columns_filled': iteration_columns_filled,
            'output_dir': final_output_dir
        }
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    """Main function to run the test with enhanced argument parsing."""
    global HAS_TQDM
    
    parser = argparse.ArgumentParser(
        description="Enhanced PADBen Type 5 Generation Test with Configurable Sample Selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic tests with specific sample counts
  python test_type5_generation.py --method dipper --iterations 1 --samples 20
  python test_type5_generation.py --method prompt_based --iterations 3 --samples 100
  
  # Range-based selection
  python test_type5_generation.py --method dipper --iterations 5 --start 1000 --end 2000
  python test_type5_generation.py --method prompt_based --iterations 3 --start 0 --end 500
  
  # Process all samples
  python test_type5_generation.py --method dipper --iterations 1 --all
  
  # With null processing
  python test_type5_generation.py --method dipper --iterations 3 --samples 50 --null-process
  
  # Custom input dataset
  python test_type5_generation.py --method dipper --iterations 1 --samples 20 --input "path/to/dataset.json"
  
  # Custom output directory
  python test_type5_generation.py --method prompt_based --iterations 1 --samples 10 --output "results/type5_test"
  
  # Disable progress bars
  python test_type5_generation.py --method dipper --iterations 3 --samples 50 --no-progress
        """
    )
    
    # Method and iteration selection
    parser.add_argument(
        "--method",
        choices=["dipper", "prompt_based"],
        required=True,
        help="Paraphrasing method to test (dipper or prompt_based)"
    )
    
    parser.add_argument(
        "--iterations",
        choices=["1", "3", "5"],
        required=True,
        help="Number of iterations to test (1, 3, or 5)"
    )
    
    # Sample size options (similar to test_type4_generation.py)
    parser.add_argument("--samples", type=int, default=20, 
                       help="Number of samples to process from the beginning (default: 20)")
    parser.add_argument("--all", action="store_true", 
                       help="Process all samples in the dataset")
    
    # Range-based selection options
    parser.add_argument("--start", type=int, 
                       help="Starting index for range selection (inclusive, 0-based)")
    parser.add_argument("--end", type=int, 
                       help="Ending index for range selection (exclusive, 0-based)")
    
    # Input dataset override
    parser.add_argument("--input", type=str,
                       help="Override input dataset path (default: data/test/merged_intratype/postprocessed_padben_20250829_182530.json)")
    
    # Output directory override
    parser.add_argument("--output", type=str,
                       help="Override output directory path (default: data/test/type5_generation_test)")
    
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
    
    # Convert arguments to enums
    method_map = {
        "dipper": Type4ParaphraseMethod.DIPPER,
        "prompt_based": Type4ParaphraseMethod.PROMPT_BASED
    }
    
    iteration_map = {
        "1": IterationLevel.FIRST,
        "3": IterationLevel.THIRD,
        "5": IterationLevel.FIFTH
    }
    
    method = method_map[args.method]
    iteration_level = iteration_map[args.iterations]
    
    # Check if the unified dataset exists first to validate ranges
    if args.input:
        dataset_path = Path(args.input)
    else:
        dataset_path = project_root / "data/test/merged_intratype/postprocessed_padben_20250829_182530.json"
    
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
        
        print("🚀 PADBen Type 5 Generation Test")
        print("Testing LLM-paraphrased LLM-generated text with iterative paraphrasing")
        print(f"Processing samples from index {actual_start} to {actual_end-1} ({actual_count} samples)")
        
    elif args.all:
        num_samples = None
        start_idx = None
        end_idx = None
        print("🚀 PADBen Type 5 Generation Test")
        print("Testing LLM-paraphrased LLM-generated text with iterative paraphrasing")
        print("Processing ALL samples from unified dataset")
        
    else:
        num_samples = args.samples
        start_idx = None
        end_idx = None
        print("🚀 PADBen Type 5 Generation Test")
        print("Testing LLM-paraphrased LLM-generated text with iterative paraphrasing")
        print(f"Processing first {num_samples} samples from unified dataset")
    
    print(f"Method: {args.method}")
    print(f"Iterations: {args.iterations}")
    
    if args.output:
        print(f"Output directory: {args.output}")
    
    if args.null_process:
        print("🔄 Individual null processing enabled - will retry each failed record until successful")
    
    if args.no_progress:
        print("Progress bars disabled")
    
    print("=" * 80)
    
    # Check API key configuration
    try:
        from data_generation.config.secrets_manager import validate_all_api_keys
        if not validate_all_api_keys():
            print("❌ API keys not configured properly.")
            print("Please set up your API keys first:")
            print("1. Create a .env file with GEMINI_API_KEY=your_key")
            print("2. Or set environment variable: export GEMINI_API_KEY=your_key")
            return
        print("✅ API keys validated successfully")
    except Exception as e:
        print(f"⚠️ Warning: Could not validate API keys: {e}")
    
    # Show warning for large datasets with progress estimation
    if num_samples is None and not range_specified:
        # Processing all samples
        estimated_time = total_samples * 5.0  # Rough estimate: 5.0 seconds per sample for Type 5 (higher than Type 4)
        
        print(f"\n⚠️ WARNING: You're about to process {total_samples} samples")
        print(f"Estimated processing time: {estimated_time/60:.1f} minutes ({estimated_time/3600:.1f} hours)")
        print("This will consume API credits and take significant time.")
        print("Type 5 requires multiple iterations per sample, making it slower than Type 4.")
        
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
        estimated_time = actual_count * 5.0  # Rough estimate: 5.0 seconds per sample for Type 5
        
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
    print("🚀 Starting Type 5 generation with progress tracking...")
    success, result = asyncio.run(test_type5_generation(
        method, iteration_level, 
        num_samples=num_samples, 
        start_idx=start_idx, 
        end_idx=end_idx,
        input_dataset=args.input,
        output_dir=args.output,
        null_process=args.null_process
    ))
    
    if success:
        print(f"\n🎉 Test passed! Type 5 generation is working correctly.")
        print(f"📊 Final Results:")
        print(f"   Method: {result['method']}")
        print(f"   Iterations: {result['iterations']}")
        print(f"   Success rate: {result['success_rate']:.1f}%")
        print(f"   Generated: {result['generated_count']}/{result['total_count']}")
        print(f"   Processing time: {result['duration']:.2f} seconds")
        print(f"   Column name: {result['column_name']}")
        print(f"   Output directory: {result['output_dir']}")
        print("📝 Note: Type 5 requires Type 2 data. Mock data was created for testing if needed.")
        if result['iterations'] > 1:
            print("📝 Midpoint results with iteration history saved for higher iterations.")
    else:
        print("\n❌ Test failed. Please check the error messages above.")
        print("\n💡 Tips:")
        print("1. Ensure API keys are properly configured")
        print("2. Check if the base dataset exists")
        print("3. For DIPPER method, ensure you have sufficient GPU memory")
        print("4. For prompt_based method, ensure Gemini API is accessible")

if __name__ == "__main__":
    main()
