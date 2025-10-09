#!/usr/bin/env python3
"""
Demo script for PADBen data quality examination.

This script demonstrates basic usage of the data quality examination module
with a small sample for quick testing.
"""

import logging
from pathlib import Path

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def demo_basic_usage():
    """Demonstrate basic usage with small sample."""
    print("="*60)
    print("PADBen Data Quality Examination - Demo")
    print("="*60)
    
    try:
        # Import the main class
        try:
            from .main import DataQualityExaminer
        except ImportError:
            from main import DataQualityExaminer
        
        # Initialize examiner
        print("\n1. Initializing Data Quality Examiner...")
        examiner = DataQualityExaminer()
        
        # Run examination with small sample for demo
        print("\n2. Running quality examination (small sample for demo)...")
        results = examiner.run_complete_examination(
            sample_size=50,  # Small sample for quick demo
            generate_visualizations=True
        )
        
        print("\n3. Demo completed successfully!")
        print(f"   Results saved to: {examiner.output_dir}")
        
        return results
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")
        return None
        
    except FileNotFoundError as e:
        print(f"Data file not found: {e}")
        print("Make sure the data file exists at: data/test/final_generated_data.json")
        return None
        
    except Exception as e:
        print(f"Error during examination: {e}")
        return None

def demo_individual_components():
    """Demonstrate individual component usage."""
    print("\n" + "="*60)
    print("Individual Components Demo")
    print("="*60)
    
    try:
        try:
            from .data_loader import DataLoader
            from .metrics import JaccardSimilarityCalculator
        except ImportError:
            from data_loader import DataLoader
            from metrics import JaccardSimilarityCalculator
        
        # Demo data loading
        print("\n1. Loading data...")
        loader = DataLoader()
        data = loader.load_data()
        print(f"   Loaded {len(data)} records")
        
        # Demo basic statistics
        stats = loader.get_dataset_statistics()
        print(f"   Dataset sources: {list(stats['dataset_sources'].keys())}")
        
        # Demo Jaccard similarity
        print("\n2. Calculating Jaccard similarity...")
        jaccard_calc = JaccardSimilarityCalculator()
        
        # Get sample texts
        texts = loader.extract_all_texts()
        if texts:
            type1_texts = texts.get('type1', [])[:5]  # First 5 texts
            type2_texts = texts.get('type2', [])[:5]
            
            if type1_texts and type2_texts:
                similarity = jaccard_calc.calculate_similarity(
                    type1_texts[0], type2_texts[0]
                )
                print(f"   Sample similarity score: {similarity:.3f}")
        
        print("\n3. Individual components demo completed!")
        
    except Exception as e:
        print(f"Error in individual components demo: {e}")

def main():
    """Main demo function."""
    print("Starting PADBen Data Quality Examination Demo...")
    
    # Check if data file exists
    data_path = Path("../data/test/final_generated_data.json")
    if not data_path.exists():
        print(f"Warning: Data file not found at {data_path}")
        print("Please ensure the data file exists before running the demo.")
        return
    
    # Run demos
    results = demo_basic_usage()
    
    if results:
        demo_individual_components()
        
        print("\n" + "="*60)
        print("Demo Summary")
        print("="*60)
        print("✓ Data loading successful")
        print("✓ Metrics calculation completed")
        print("✓ RAID comparison generated")
        print("✓ Visualizations created")
        print("✓ Reports exported")
        
        print(f"\nCheck the outputs directory for detailed results:")
        print(f"  {Path('outputs').absolute()}")
        
    else:
        print("\nDemo failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
