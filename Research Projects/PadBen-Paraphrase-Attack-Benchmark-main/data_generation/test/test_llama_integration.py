#!/usr/bin/env python3
"""
Test script for Llama-3.1-8B integration in Type 4 generation.

This script demonstrates how to use the new Llama paraphrasing method
alongside the existing DIPPER and prompt-based methods.
"""

import asyncio
import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_llama_integration():
    """Test the Llama integration for Type 4 generation."""
    
    try:
        # Import the required modules
        from data_generation.type4_generation import Type4Generator, Type4ParaphraseMethod
        from data_generation.config.type4_config import DEFAULT_TYPE4_CONFIG
        from data_generation.config.base_model_config import create_llama_paraphrase_config
        
        logger.info("🦙 Testing Llama-3.1-8B Integration for Type 4 Generation")
        logger.info("=" * 60)
        
        # Create sample test data
        test_data = pd.DataFrame({
            'idx': [1, 2, 3],
            'dataset_source': ['mrpc', 'hlpc', 'paws'],
            'human_original_text': [
                "The weather is beautiful today.",
                "Machine learning is transforming the world.",
                "The quick brown fox jumps over the lazy dog."
            ],
            'human_paraphrased_text': [
                "Today's weather is lovely.",
                "AI is revolutionizing the globe.",
                "A fast brown fox leaps above a sleepy dog."
            ]
        })
        
        logger.info(f"Created test dataset with {len(test_data)} samples")
        
        # Configure Llama model
        logger.info("Configuring Llama-3.1-8B model...")
        llama_config = create_llama_paraphrase_config(
            model_id="mradermacher/Llama-3.1-8B-paraphrase-type-generation-apty-sigmoid-GGUF",
            device="auto",
            temperature=0.7
        )
        
        # Update the default config with Llama model
        config = DEFAULT_TYPE4_CONFIG
        config.llama_model = llama_config
        
        # Initialize the generator
        logger.info("Initializing Type 4 generator with Llama support...")
        generator = Type4Generator(config)
        
        # Test each paraphrasing method
        methods_to_test = [
            (Type4ParaphraseMethod.DIPPER, "DIPPER"),
            (Type4ParaphraseMethod.PROMPT_BASED, "Prompt-based"),
            (Type4ParaphraseMethod.LLAMA, "Llama-3.1-8B")
        ]
        
        results = {}
        
        for method, method_name in methods_to_test:
            logger.info(f"\n🔄 Testing {method_name} paraphrasing...")
            
            try:
                # Generate paraphrases using the specific method
                result_df = await generator.generate_for_dataset(
                    test_data.copy(),
                    method=method,
                    output_dir="test_output"
                )
                
                # Count successful paraphrases
                method_column = generator.get_method_column_name(method)
                successful_count = result_df[method_column].notna().sum()
                
                results[method_name] = {
                    'successful': successful_count,
                    'total': len(test_data),
                    'success_rate': (successful_count / len(test_data)) * 100
                }
                
                logger.info(f"✅ {method_name}: {successful_count}/{len(test_data)} successful ({results[method_name]['success_rate']:.1f}%)")
                
                # Show sample results
                if successful_count > 0:
                    sample_result = result_df[result_df[method_column].notna()].iloc[0]
                    logger.info(f"Sample result:")
                    logger.info(f"  Original: {sample_result['human_original_text']}")
                    logger.info(f"  Paraphrased: {sample_result[method_column]}")
                
            except Exception as e:
                logger.error(f"❌ {method_name} failed: {str(e)}")
                results[method_name] = {
                    'successful': 0,
                    'total': len(test_data),
                    'success_rate': 0,
                    'error': str(e)
                }
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 INTEGRATION TEST SUMMARY")
        logger.info("=" * 60)
        
        for method_name, stats in results.items():
            if 'error' in stats:
                logger.info(f"{method_name}: ❌ FAILED - {stats['error']}")
            else:
                logger.info(f"{method_name}: ✅ {stats['successful']}/{stats['total']} ({stats['success_rate']:.1f}%)")
        
        # Check if all methods are available
        available_methods = [name for name, stats in results.items() if 'error' not in stats]
        logger.info(f"\n🎯 Available methods: {', '.join(available_methods)}")
        
        if len(available_methods) == len(methods_to_test):
            logger.info("🎉 All paraphrasing methods are working correctly!")
        else:
            logger.warning(f"⚠️  Only {len(available_methods)}/{len(methods_to_test)} methods are working")
        
        return results
        
    except ImportError as e:
        logger.error(f"❌ Import error: {str(e)}")
        logger.error("Make sure all dependencies are installed:")
        logger.error("  pip install llama-cpp-python transformers torch")
        return None
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        return None

async def test_individual_paraphrasing():
    """Test individual paraphrasing methods."""
    
    try:
        from data_generation.type4_generation import Type4Generator, Type4ParaphraseMethod
        from data_generation.config.type4_config import DEFAULT_TYPE4_CONFIG
        
        logger.info("\n🔬 Testing individual paraphrasing methods...")
        
        # Create a single test sample
        test_row = pd.Series({
            'idx': 1,
            'dataset_source': 'mrpc',
            'human_original_text': 'The weather is beautiful today.',
            'human_paraphrased_text': 'Today\'s weather is lovely.'
        })
        
        generator = Type4Generator(DEFAULT_TYPE4_CONFIG)
        
        # Test each method individually
        for method in [Type4ParaphraseMethod.DIPPER, Type4ParaphraseMethod.PROMPT_BASED, Type4ParaphraseMethod.LLAMA]:
            try:
                logger.info(f"\nTesting {method.value} method...")
                result = await generator.paraphrase_text(test_row, method)
                
                if result.success:
                    logger.info(f"✅ {method.value}: {result.paraphrased_text}")
                else:
                    logger.warning(f"⚠️  {method.value}: Failed - {result.metadata.get('failure_reason', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"❌ {method.value}: Exception - {str(e)}")
        
    except Exception as e:
        logger.error(f"❌ Individual test failed: {str(e)}")

def main():
    """Main test function."""
    logger.info("🚀 Starting Llama-3.1-8B Integration Tests")
    logger.info("=" * 60)
    
    # Run the main integration test
    results = asyncio.run(test_llama_integration())
    
    if results:
        # Run individual method tests
        asyncio.run(test_individual_paraphrasing())
    
    logger.info("\n🏁 Test completed!")

if __name__ == "__main__":
    main()
