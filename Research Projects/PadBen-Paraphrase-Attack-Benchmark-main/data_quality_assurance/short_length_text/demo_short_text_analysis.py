#!/usr/bin/env python3
"""
Demo script for short text analysis functionality.
"""

import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def demo_short_text_analysis():
    """Demonstrate short text analysis functionality."""
    print("=" * 60)
    print("PADBen Short Text Analysis - Demo")
    print("=" * 60)
    
    try:
        # Import the analyzer
        try:
            from .short_text_analyzer import ShortTextAnalyzer
        except ImportError:
            from short_text_analyzer import ShortTextAnalyzer
        
        print("\n1. Initializing Short Text Analyzer (threshold=10 tokens)...")
        analyzer = ShortTextAnalyzer(threshold=10)
        
        print("\n2. Running analysis on PADBen dataset...")
        results = analyzer.analyze_short_texts()
        
        print("\n3. Analysis Results Summary:")
        print("-" * 30)
        
        # Show key statistics
        for text_type, stats in results["short_text_statistics"].items():
            short_pct = stats["short_texts_percentage"]
            if short_pct > 1:  # Only show types with > 1% short texts
                print(f"  {text_type}: {stats['short_texts_count']:,} short texts ({short_pct:.2f}%)")
                print(f"    Min length: {stats['min_length']} tokens")
                print(f"    Examples: {len(stats['short_text_examples'])}")
        
        print(f"\n4. Problematic Records: {len(results['problematic_records'])}")
        
        # Show some examples if available
        if results['problematic_records']:
            print("\n5. Example Problematic Record:")
            example = results['problematic_records'][0]
            print(f"   Record ID: {example['record_idx']}")
            print(f"   Source: {example['dataset_source']}")
            print(f"   Issue: {example['issue_type']}")
            for text_type, details in example['text_details'].items():
                if details['is_short']:
                    print(f"   {text_type}: '{details['text']}' ({details['token_count']} tokens)")
        
        print("\n6. Top Recommendations:")
        for i, rec in enumerate(results['recommendations'][:3], 1):
            print(f"   {i}. {rec}")
        
        print(f"\n✓ Demo completed successfully!")
        print(f"  Found issues in {len(results['problematic_records'])} records")
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running from the correct directory")
        return False
        
    except FileNotFoundError as e:
        print(f"Data file not found: {e}")
        print("Make sure the data file exists at: data/test/final_generated_data.json")
        return False
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return False

def main():
    """Main demo function."""
    success = demo_short_text_analysis()
    
    if success:
        print(f"\n💡 To run full analysis with file output:")
        print(f"   python run_short_text_analysis.py --threshold 10")
        print(f"\n📊 To see detailed statistics:")
        print(f"   python run_short_text_analysis.py --threshold 5 --verbose")
    else:
        print(f"\n❌ Demo failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
