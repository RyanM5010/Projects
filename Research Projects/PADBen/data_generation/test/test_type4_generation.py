#!/usr/bin/env python3
"""
Test Type 4 Generation with First 20 Samples

This script tests the Type 4 generation pipeline (LLM-paraphrased original text)
with just the first 20 samples from the unified dataset.
Tests both DIPPER and prompt-based paraphrasing methods with individual method selection.
"""

import asyncio
import json
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Now import the required modules
from data_generation.type4_generation import Type4Generator, EnvironmentMode
from data_generation.config.generation_model_config import DEFAULT_CONFIG
from data_generation.config.type4_config import Type4ParaphraseMethod

def create_test_dataset(input_file: str, output_file: str, num_samples: int = 20):
    """Create a small test dataset with the first N samples."""
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
    
    # Add missing columns that Type 4 will fill
    if 'llm_paraphrased_original_text' not in test_df.columns:
        test_df['llm_paraphrased_original_text'] = None
    if 'llm_paraphrased_original_text_dipper' not in test_df.columns:
        test_df['llm_paraphrased_original_text_dipper'] = None
    if 'llm_paraphrased_original_text_prompt' not in test_df.columns:
        test_df['llm_paraphrased_original_text_prompt'] = None
    
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

async def test_single_method(method_enum, method_name, test_df, output_dir):
    """Test a single paraphrasing method."""
    print(f"\n--- Testing {method_name} method ---")
    
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
    
    # Collect results
    generated_count = results_df['llm_paraphrased_original_text(Prompt_based)'].notna().sum() + results_df['llm_paraphrased_original_text(DIPPER_based)'].notna().sum()
    total_count = len(results_df)
    success_rate = (generated_count / total_count) * 100
    
    result_data = {
        'generated_count': generated_count,
        'total_count': total_count,
        'success_rate': success_rate,
        'duration': duration,
        'results_df': results_df
    }
    
    print(f"✅ {method_name} completed:")
    print(f"   Generated: {generated_count}/{total_count} ({success_rate:.1f}%)")
    print(f"   Duration: {duration:.2f} seconds")
    
    return result_data

async def test_type4_generation(method_choice=None):
    """Test Type 4 generation with optional method selection."""
    print("🧪 Starting Type 4 Generation Test")
    print("=" * 60)
    
    # Paths (relative to project root)
    original_dataset = str(project_root / "data/processed/unified_padben_base.json")
    test_dataset = str(project_root / "data/test/test_unified_padben_20_samples.json")
    output_dir = str(project_root / "data/test/type4_generation_test")
    
    try:
        # Step 1: Create test dataset
        print("Step 1: Creating test dataset...")
        test_df = create_test_dataset(original_dataset, test_dataset, num_samples=20)
        
        # Step 2: Determine which methods to test
        if method_choice:
            if method_choice == "dipper":
                methods_to_test = [(Type4ParaphraseMethod.DIPPER, "dipper")]
            elif method_choice == "prompt_based":
                methods_to_test = [(Type4ParaphraseMethod.PROMPT_BASED, "prompt_based")]
            else:
                raise ValueError(f"Invalid method choice: {method_choice}")
            print(f"\nStep 2: Testing single method: {method_choice}")
        else:
            methods_to_test = [
                (Type4ParaphraseMethod.DIPPER, "dipper"),
                (Type4ParaphraseMethod.PROMPT_BASED, "prompt_based")
            ]
            print("\nStep 2: Testing both paraphrasing methods...")
        
        results = {}
        
        for method_enum, method_name in methods_to_test:
            result_data = await test_single_method(method_enum, method_name, test_df, output_dir)
            results[method_name] = result_data
        
        # Step 3: Analyze and compare results
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
        
        # Show sample outputs from each method
        print(f"\n📋 Sample Paraphrased Texts:")
        print("-" * 40)
        
        for method_name, result in results.items():
            print(f"\n--- {method_name.title()} Method ---")
            df = result['results_df']
            sample_count = 0
            
            for idx, row in df.iterrows():
                if pd.notna(row['llm_paraphrased_original_text']) and sample_count < 2:
                    sample_count += 1
                    print(f"Sample {sample_count}:")
                    print(f"  Dataset: {row['dataset_source']}")
                    print(f"  Original: {row['human_original_text'][:100]}{'...' if len(row['human_original_text']) > 100 else ''}")
                    print(f"  Paraphrased: {row['llm_paraphrased_original_text'][:100]}{'...' if len(row['llm_paraphrased_original_text']) > 100 else ''}")
                    print()
        
        # Check output directories
        print(f"📁 Output Directories:")
        output_path = Path(output_dir)
        if output_path.exists():
            for method_name in [m[1] for m in methods_to_test]:
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
    """Main function to run the test with method selection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="PADBen Type 4 Generation Test")
    parser.add_argument(
        "--method",
        choices=["dipper", "prompt_based", "both"],
        default="both",
        help="Method to test: dipper, prompt_based, or both (default: both)"
    )
    
    args = parser.parse_args()
    
    # Determine method choice
    method_choice = None if args.method == "both" else args.method
    
    print("🚀 PADBen Type 4 Generation Test")
    print("Testing LLM-paraphrased original text generation")
    if method_choice:
        print(f"Testing single method: {method_choice}")
    else:
        print("Testing both DIPPER and prompt-based methods")
    print("=" * 60)
    
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
    success = asyncio.run(test_type4_generation(method_choice))
    
    if success:
        if method_choice:
            print(f"\n🎉 {method_choice.title()} method test passed! Type 4 generation is working correctly.")
        else:
            print("\n🎉 All tests passed! Type 4 generation is working correctly.")
    else:
        print("\n❌ Test failed. Please check the error messages above.")

if __name__ == "__main__":
    main()