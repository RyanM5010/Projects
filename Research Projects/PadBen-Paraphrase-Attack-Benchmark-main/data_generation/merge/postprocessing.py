#!/usr/bin/env python3
"""
Postprocessing script for PADBen merged dataset.
Removes problematic records and reindexes the dataset from 0 to k.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Set


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a JSON file and return the data.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of records from the JSON file
        
    Raises:
        Exception: If file cannot be loaded
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise Exception(f"Failed to load {file_path}: {e}")


def analyze_data_quality(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze data quality and identify problematic records.
    
    Args:
        data: List of records to analyze
        
    Returns:
        Dictionary with analysis results
    """
    if not data:
        return {"total_records": 0, "issues": []}
    
    issues = []
    null_type2_records = []
    null_type4_records = []
    
    # Check for records with null values in critical columns
    for i, record in enumerate(data):
        idx = record.get("idx")
        
        # Check Type 2 column
        type2_value = record.get("llm_generated_text(type2)")
        if type2_value is None:
            null_type2_records.append(idx)
        
        # Check Type 4 columns (both should not be null)
        type4_prompt = record.get("llm_paraphrased_original_text(type4)-prompt-based")
        type4_dipper = record.get("llm_paraphrased_original_text(type4)-dipper-based")
        
        if type4_prompt is None and type4_dipper is None:
            null_type4_records.append(idx)
    
    return {
        "total_records": len(data),
        "null_type2_records": null_type2_records,
        "null_type4_records": null_type4_records,
        "null_type2_count": len(null_type2_records),
        "null_type4_count": len(null_type4_records)
    }


def remove_problematic_records(data: List[Dict[str, Any]], 
                             specific_idx_to_remove: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """
    Remove records with problematic data.
    
    Args:
        data: List of records
        specific_idx_to_remove: Optional list of specific idx values to remove
        
    Returns:
        Filtered list of records
    """
    if not data:
        return []
    
    # Default problematic idx values
    if specific_idx_to_remove is None:
        specific_idx_to_remove = [4008, 19387]
    
    print(f"🗑️  Removing records with idx: {specific_idx_to_remove}")
    
    filtered_data = []
    removed_count = 0
    
    for record in data:
        idx = record.get("idx")
        
        # Check if this record should be removed
        should_remove = False
        
        # Remove specific idx values
        if idx in specific_idx_to_remove:
            should_remove = True
            print(f"  ❌ Removing idx {idx}: Specified for removal")
        
        # Additional quality checks (optional)
        # Check for records with null Type 2 AND null Type 4
        type2_value = record.get("llm_generated_text(type2)")
        type4_prompt = record.get("llm_paraphrased_original_text(type4)-prompt-based")
        type4_dipper = record.get("llm_paraphrased_original_text(type4)-dipper-based")
        
        if (type2_value is None and 
            type4_prompt is None and 
            type4_dipper is None):
            should_remove = True
            print(f"  ❌ Removing idx {idx}: No Type 2 or Type 4 data")
        
        if not should_remove:
            filtered_data.append(record)
        else:
            removed_count += 1
    
    print(f"✅ Removed {removed_count} problematic records")
    print(f"✅ Kept {len(filtered_data)} records")
    
    return filtered_data


def reindex_records(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove old idx column and reindex records from 0 to k.
    
    Args:
        data: List of records to reindex
        
    Returns:
        List of records with new sequential idx values
    """
    print(f"🔢 Reindexing {len(data)} records...")
    
    reindexed_data = []
    
    for new_idx, record in enumerate(data):
        # Create new record without old idx
        new_record = {key: value for key, value in record.items() if key != "idx"}
        
        # Add new sequential idx at the beginning
        new_record = {"idx": new_idx, **new_record}
        
        reindexed_data.append(new_record)
        
        # Show progress for large datasets
        if (new_idx + 1) % 10000 == 0:
            print(f"  Reindexed {new_idx + 1}/{len(data)} records...")
    
    print(f"✅ Reindexed all records from 0 to {len(data) - 1}")
    
    return reindexed_data


def postprocess_dataset(input_file: str, 
                       output_file: Optional[str] = None,
                       specific_idx_to_remove: Optional[List[int]] = None) -> bool:
    """
    Postprocess PADBen merged dataset by removing problematic records and reindexing.
    
    Args:
        input_file: Path to the input JSON file
        output_file: Optional output file path
        specific_idx_to_remove: Optional list of specific idx values to remove
        
    Returns:
        bool: True if postprocessing successful, False otherwise
    """
    print("🔧 PADBen Dataset Postprocessor")
    print("=" * 80)
    print(f"Input file: {Path(input_file).name}")
    print("Tasks:")
    print("  1. Remove problematic records")
    print("  2. Remove old idx column")
    print("  3. Reindex from 0 to k")
    print("=" * 80)
    
    # Validate input file
    if not Path(input_file).exists():
        print(f"❌ Error: Input file does not exist: {input_file}")
        return False
    
    # Load data
    print("📂 Loading dataset...")
    try:
        data = load_json_file(input_file)
        print(f"✅ Loaded {len(data)} records")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return False
    
    # Analyze data quality
    print("\n🔍 Analyzing data quality...")
    quality_analysis = analyze_data_quality(data)
    
    print(f"📊 Quality Analysis:")
    print(f"  Total records: {quality_analysis['total_records']}")
    print(f"  Records with null Type 2: {quality_analysis['null_type2_count']}")
    print(f"  Records with null Type 4: {quality_analysis['null_type4_count']}")
    
    if quality_analysis['null_type2_records']:
        print(f"  Null Type 2 idx values: {quality_analysis['null_type2_records'][:10]}...")
    if quality_analysis['null_type4_records']:
        print(f"  Null Type 4 idx values: {quality_analysis['null_type4_records'][:10]}...")
    
    # Remove problematic records
    print(f"\n🗑️  Removing problematic records...")
    filtered_data = remove_problematic_records(data, specific_idx_to_remove)
    
    if len(filtered_data) == 0:
        print("❌ Error: No records remaining after filtering!")
        return False
    
    # Reindex records
    print(f"\n🔢 Reindexing records...")
    final_data = reindex_records(filtered_data)
    
    # Show sample of final data
    if final_data:
        print(f"\n📖 Sample of processed data:")
        sample = final_data[0]
        for key, value in sample.items():
            if value is not None:
                value_preview = str(value)[:60] + "..." if len(str(value)) > 60 else str(value)
                print(f"  ✓ {key}: {value_preview}")
            else:
                print(f"  ○ {key}: null")
        
        print(f"\n📊 Final data statistics:")
        print(f"  First idx: {final_data[0]['idx']}")
        print(f"  Last idx: {final_data[-1]['idx']}")
        print(f"  Total records: {len(final_data)}")
    
    # Generate output filename if not provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_path = Path(input_file)
        output_file = input_path.parent / f"postprocessed_padben_{timestamp}.json"
    else:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save processed file
    print(f"\n💾 Saving postprocessed dataset...")
    print(f"Output: {output_file.name}")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        
        file_size = output_file.stat().st_size / (1024 * 1024)
        print(f"✅ Successfully saved!")
        print(f"📁 File size: {file_size:.2f} MB")
        
        # Final summary
        print(f"\n🎉 Postprocessing Summary:")
        print(f"  Input records: {len(data)}")
        print(f"  Removed records: {len(data) - len(final_data)}")
        print(f"  Final records: {len(final_data)}")
        print(f"  Index range: 0 to {len(final_data) - 1}")
        print(f"  Output file: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving postprocessed file: {e}")
        return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Postprocess PADBen merged dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic postprocessing (removes idx 4008 and 19387 by default)
  python postprocessing.py merged_padben_intratype_20250829_181953.json
  
  # Specify output file
  python postprocessing.py input.json -o clean_padben_dataset.json
  
  # Remove specific idx values
  python postprocessing.py input.json --remove-idx 4008 19387 1234
  
  # Full example
  python postprocessing.py \\
    data/test/merged_intratype/merged_padben_intratype_20250829_181953.json \\
    -o data/processed/clean_padben_final.json \\
    --remove-idx 4008 19387
        """
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the merged JSON file to postprocess"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file path (default: auto-generated with timestamp)"
    )
    
    parser.add_argument(
        "--remove-idx",
        type=int,
        nargs="*",
        default=[4008, 19387],
        help="Specific idx values to remove (default: 4008 19387)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    success = postprocess_dataset(
        input_file=args.input_file,
        output_file=args.output,
        specific_idx_to_remove=args.remove_idx
    )
    
    if success:
        print("\n🎉 Dataset postprocessing completed successfully!")
    else:
        print("\n❌ Dataset postprocessing failed!")
