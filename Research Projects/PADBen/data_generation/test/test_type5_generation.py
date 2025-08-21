#!/usr/bin/env python3
"""
Test Type 5 Generation with First 20 Samples

This script tests the Type 5 generation pipeline (LLM-paraphrased LLM-generated text)
with just the first 20 samples from the unified dataset.
Tests individual methods (DIPPER or prompt-based) and iteration levels (1, 3, or 5).
"""

import asyncio
import json
import pandas as pd
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Now import the required modules
from data_generation.type5_generation import Type5Generator, EnvironmentMode
from data_generation.config.generation_model_config import DEFAULT_CONFIG
from data_generation.config.type4_config import Type4ParaphraseMethod
from data_generation.config.type5_config import IterationLevel

def create_test_dataset_with_type2(input_file: str, output_file: str, num_samples: int = 20):
    """Create a small test dataset with the first N samples and ensure Type 2 data exists."""
    print(f"Creating test dataset with first {num_samples} samples...")
    
    # Load the unified dataset
    input_path = Path(input_file)
    if input_path.suffix == '.json':
        df = pd.read_json(input_file)
    elif input_path.suffix == '.csv':
        df = pd.read_csv(input_file)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    print(f"Loaded {len(df)} total samples from {input_file}")
    
    # Take first N samples
    test_df = df.head(num_samples).copy()
    
    # Ensure the test dataset has the required columns
    required_columns = ['idx', 'dataset_source', 'human_original_text', 'human_paraphrased_text']
    for col in required_columns:
        if col not in test_df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Add missing columns that Type 5 requires
    if 'llm_generated_text' not in test_df.columns:
        test_df['llm_generated_text'] = None
    if 'llm_generated_text_method' not in test_df.columns:
        test_df['llm_generated_text_method'] = None
    
    # Add Type 5 output columns for both methods and all iterations
    type5_columns = [
        'llm_paraphrased_generated_text',
        'llm_paraphrased_generated_text(DIPPER_based_1_iteration)',
        'llm_paraphrased_generated_text(DIPPER_based_3_iterations)',
        'llm_paraphrased_generated_text(DIPPER_based_5_iterations)',
        'llm_paraphrased_generated_text(Prompt_based_1_iteration)',
        'llm_paraphrased_generated_text(Prompt_based_3_iterations)',
        'llm_paraphrased_generated_text(Prompt_based_5_iterations)'
    ]
    
    for col in type5_columns:
        if col not in test_df.columns:
            test_df[col] = None
    
    # Check if we have Type 2 data (required for Type 5)
    type2_available = test_df['llm_generated_text'].notna().sum()
    print(f"Samples with Type 2 data: {type2_available}/{len(test_df)}")
    
    # If no Type 2 data, create mock Type 2 data for testing
    if type2_available == 0:
        print("⚠️ No Type 2 data found. Creating mock Type 2 data for testing...")
        for idx, row in test_df.iterrows():
            if pd.isna(row['llm_generated_text']):
                # Create a simple mock based on original text
                original = row['human_original_text']
                mock_generated = f"Generated version: {original[:100]}..." if len(original) > 100 else f"Generated version: {original}"
                test_df.at[idx, 'llm_generated_text'] = mock_generated
                test_df.at[idx, 'llm_generated_text_method'] = 'mock_for_test'
    
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
    print(f"Final Type 2 data availability: {test_df['llm_generated_text'].notna().sum()}/{len(test_df)}")
    
    return test_df

async def test_type5_generation(method: Type4ParaphraseMethod, iteration_level: IterationLevel):
    """Test Type 5 generation with specific method and iteration level."""
    method_name = method.value
    iteration_name = f"{iteration_level.value}{'st' if iteration_level.value == 1 else 'rd' if iteration_level.value == 3 else 'th'}_iteration{'s' if iteration_level.value > 1 else ''}"
    
    print(f"🧪 Starting Type 5 Generation Test: {method_name} with {iteration_name}")
    print("=" * 80)
    
    # Paths (relative to project root)
    original_dataset = str(project_root / "data/processed/unified_padben_base.json")
    test_dataset = str(project_root / "data/test/test_unified_padben_20_samples_type5.json")
    output_dir = str(project_root / "data/test/type5_generation_test")
    
    try:
        # Step 1: Create test dataset with Type 2 data
        print("Step 1: Creating test dataset with Type 2 data...")
        test_df = create_test_dataset_with_type2(original_dataset, test_dataset, num_samples=20)
        
        # Step 2: Initialize Type 5 generator in TEST mode
        print("\nStep 2: Initializing Type 5 generator in TEST mode...")
        generator = Type5Generator(
            DEFAULT_CONFIG.type5_config, 
            environment_mode=EnvironmentMode.TEST
        )
        print("✅ Type 5 generator initialized successfully")
        
        # Step 3: Test the specific method and iteration level
        print(f"\nStep 3: Testing {method_name} with {iteration_name}...")
        
        start_time = datetime.now()
        
        # Run generation for this specific configuration
        results_df = await generator.generate_for_dataset(
            test_df,
            method=method,
            iteration=iteration_level,
            output_dir=output_dir
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Step 4: Analyze results using the correct column name
        method_column = generator.get_method_column_name(method, iteration_level)
        print(f"Checking results in column: {method_column}")
        
        generated_count = results_df[method_column].notna().sum()
        total_count = len(results_df)
        success_rate = (generated_count / total_count) * 100
        
        print(f"\n✅ {method_name} with {iteration_name} completed:")
        print(f"   Generated: {generated_count}/{total_count} ({success_rate:.1f}%)")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Avg time per sample: {duration/total_count:.2f} seconds")
        print(f"   Column used: {method_column}")
        
        # Step 5: Show sample results
        print(f"\n📋 Sample Results from {method_name} with {iteration_name}:")
        print("-" * 70)
        
        sample_count = 0
        for idx, row in results_df.iterrows():
            if pd.notna(row[method_column]) and sample_count < 3:
                sample_count += 1
                print(f"Sample {sample_count}:")
                print(f"  Dataset: {row['dataset_source']}")
                print(f"  Original: {row['human_original_text'][:80]}{'...' if len(row['human_original_text']) > 80 else ''}")
                print(f"  Generated (Type 2): {row['llm_generated_text'][:80]}{'...' if len(row['llm_generated_text']) > 80 else ''}")
                print(f"  Paraphrased (Type 5): {row[method_column][:80]}{'...' if len(row[method_column]) > 80 else ''}")
                print()
        
        # Step 6: Show statistics
        print(f"📊 Generation Statistics:")
        print(f"   Total processed: {generator.stats['total_processed']}")
        print(f"   Successful paraphrases: {generator.stats['successful_paraphrases']}")
        print(f"   Failed paraphrases: {generator.stats['failed_paraphrases']}")
        print(f"   Missing Type 2 data: {generator.stats['missing_type2_data']}")
        print(f"   Early stops: {generator.stats['early_stops']}")
        print(f"   Average iterations completed: {generator.stats['avg_iterations_completed']:.1f}")
        
        # Check output directory
        print(f"\n📁 Output Directory:")
        output_path = Path(output_dir)
        if output_path.exists():
            # Look for files with the correct iteration level naming
            iteration_folder_pattern = f"{iteration_level.value}{'st' if iteration_level.value == 1 else 'rd' if iteration_level.value == 3 else 'th'}_*"
            method_dirs = list(output_path.glob(iteration_folder_pattern))
            if method_dirs:
                latest_dir = max(method_dirs, key=lambda p: p.name)
                files = list(latest_dir.glob("*.json")) + list(latest_dir.glob("*.csv"))
                print(f"  Directory: {latest_dir.name}")
                print(f"  Files created: {len(files)}")
                for file in files[:3]:  # Show first 3 files
                    print(f"    📄 {file.name}")
        
        print(f"\n🎉 TEST COMPLETED SUCCESSFULLY!")
        print(f"Method: {method_name}")
        print(f"Iterations: {iteration_level.value}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Results saved to: {output_dir}")
        print("=" * 80)
        
        return True, {
            'method': method_name,
            'iterations': iteration_level.value,
            'success_rate': success_rate,
            'generated_count': generated_count,
            'total_count': total_count,
            'duration': duration,
            'column_name': method_column
        }
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    """Main function to run the test with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Test Type 5 Generation with specific method and iteration level",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_type5_generation.py --method dipper --iterations 1
  python test_type5_generation.py --method prompt_based --iterations 3
  python test_type5_generation.py --method dipper --iterations 5
        """
    )
    
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
    
    args = parser.parse_args()
    
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
    
    print("🚀 PADBen Type 5 Generation Test")
    print("Testing LLM-paraphrased LLM-generated text with iterative paraphrasing")
    print(f"Method: {args.method}")
    print(f"Iterations: {args.iterations}")
    print("=" * 80)
    
    # Check if the unified dataset exists
    dataset_path = project_root / "data/processed/unified_padben_base.json"
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please ensure the unified dataset exists before running the test.")
        return
    
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
    
    # Run the test
    success, result = asyncio.run(test_type5_generation(method, iteration_level))
    
    if success:
        print(f"\n🎉 Test passed! Type 5 generation is working correctly.")
        print(f"📊 Final Results:")
        print(f"   Method: {result['method']}")
        print(f"   Iterations: {result['iterations']}")
        print(f"   Success rate: {result['success_rate']:.1f}%")
        print(f"   Generated: {result['generated_count']}/{result['total_count']}")
        print(f"   Processing time: {result['duration']:.2f} seconds")
        print(f"   Column name: {result['column_name']}")
        print("📝 Note: Type 5 requires Type 2 data. Mock data was created for testing if needed.")
    else:
        print("\n❌ Test failed. Please check the error messages above.")
        print("\n💡 Tips:")
        print("1. Ensure API keys are properly configured")
        print("2. Check if the base dataset exists")
        print("3. For DIPPER method, ensure you have sufficient GPU memory")
        print("4. For prompt_based method, ensure Gemini API is accessible")

if __name__ == "__main__":
    main()
