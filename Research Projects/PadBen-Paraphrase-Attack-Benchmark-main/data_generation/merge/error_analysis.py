#!/usr/bin/env python3
"""
Error handling script for PADBen files.
Checks for null values in specific columns based on text generation type.
Generates metadata summary and detailed null records list.
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


def get_column_mapping() -> Dict[str, List[str]]:
    """
    Get mapping of text generation types to their corresponding column names.
    
    Returns:
        Dict mapping type names to list of possible column names
    """
    return {
        "type2": [
            "llm_generated_text(sentence_completion)",
            "llm_generated_text(question_answer)",
            "llm_generated_text",
            "llm_generated_text(type2)"
        ],
        "type4": [
            "llm_paraphrased_original_text(Prompt_based)",
            "llm_paraphrased_original_text(prompt_based)",
            "llm_paraphrased_original_text_prompt_based",
            "llm_paraphrased_original_text(type4)-prompt-based",
            "llm_paraphrased_original_text"
        ],
        "type5": [
            "llm_paraphrased_generated_text(1st)",
            "llm_paraphrased_generated_text_1st",
            "llm_paraphrased_generated_text(3rd)",
            "llm_paraphrased_generated_text_3rd",
            "llm_paraphrased_generated_text(type5)-1st",
            "llm_paraphrased_generated_text(type5)-3rd",
            "llm_paraphrased_generated_text"
        ]
    }


def find_target_column(df: pd.DataFrame, generation_type: str) -> Optional[str]:
    """
    Find the actual column name in the dataframe for the given generation type.
    
    Args:
        df: Input dataframe
        generation_type: Type of text generation (type2, type4, type5)
        
    Returns:
        Column name if found, None otherwise
    """
    column_mapping = get_column_mapping()
    possible_columns = column_mapping.get(generation_type, [])
    
    for col in possible_columns:
        if col in df.columns:
            return col
    
    return None


def analyze_null_values(
    input_file: str,
    generation_type: str,
    output_dir: Optional[str] = None
) -> bool:
    """
    Analyze null values in the specified column for the given generation type.
    
    Args:
        input_file: Path to the input JSON file
        generation_type: Type of text generation to check
        output_dir: Optional output directory. If None, uses input file directory
        
    Returns:
        bool: True if analysis successful, False otherwise
    """
    
    print("🔍 PADBen Null Value Analyzer")
    print("=" * 80)
    print(f"Input file: {Path(input_file).name}")
    print(f"Generation type: {generation_type}")
    print("=" * 80)
    
    # Validate input file exists
    if not Path(input_file).exists():
        print(f"❌ Error: Input file does not exist: {input_file}")
        return False
    
    # Load the file
    print("📂 Loading file...")
    try:
        df = pd.read_json(input_file)
        print(f"✅ Loaded {len(df)} records")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return False
    
    # Find the target column
    target_column = find_target_column(df, generation_type)
    if target_column is None:
        print(f"❌ Error: No suitable column found for {generation_type}")
        print("Available columns:")
        for col in df.columns:
            print(f"  - {col}")
        return False
    
    print(f"🎯 Target column: '{target_column}'")
    
    # Analyze null values
    print(f"\n📊 Analyzing null values...")
    
    # Check if column exists and analyze
    if target_column not in df.columns:
        print(f"❌ Error: Column '{target_column}' not found in file")
        return False
    
    # Count null values (including None, NaN, empty strings)
    null_mask = df[target_column].isnull() | (df[target_column] == "") | (df[target_column] == "null")
    null_count = null_mask.sum()
    total_count = len(df)
    non_null_count = total_count - null_count
    null_percentage = (null_count / total_count) * 100
    
    print(f"📈 Null Value Analysis:")
    print(f"  Total records: {total_count}")
    print(f"  Non-null records: {non_null_count}")
    print(f"  Null records: {null_count}")
    print(f"  Null percentage: {null_percentage:.2f}%")
    
    # Get null records (complete records, not just idx)
    null_records_df = df[null_mask].copy()
    null_idx_list = null_records_df['idx'].tolist() if 'idx' in null_records_df.columns else []
    
    print(f"📋 Null Record Details:")
    if null_idx_list:
        print(f"  First 10 null idx values: {null_idx_list[:10]}")
        print(f"  Last 10 null idx values: {null_idx_list[-10:]}")
        
        # Check for patterns in null idx values
        if len(null_idx_list) > 1:
            null_idx_sorted = sorted(null_idx_list)
            min_idx = min(null_idx_sorted)
            max_idx = max(null_idx_sorted)
            print(f"  Null idx range: {min_idx} - {max_idx}")
            
            # Check for consecutive nulls
            consecutive_groups = []
            current_group = [null_idx_sorted[0]]
            
            for i in range(1, len(null_idx_sorted)):
                if null_idx_sorted[i] == null_idx_sorted[i-1] + 1:
                    current_group.append(null_idx_sorted[i])
                else:
                    if len(current_group) > 1:
                        consecutive_groups.append(current_group)
                    current_group = [null_idx_sorted[i]]
            
            if len(current_group) > 1:
                consecutive_groups.append(current_group)
            
            if consecutive_groups:
                print(f"  Found {len(consecutive_groups)} consecutive null groups")
                for i, group in enumerate(consecutive_groups[:5]):  # Show first 5 groups
                    print(f"    Group {i+1}: {group[0]}-{group[-1]} ({len(group)} records)")
    else:
        print("  No null records found!")
    
    # Prepare output directory
    if output_dir is None:
        output_dir = Path(input_file).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create metadata summary
    metadata = {
        "analysis_info": {
            "input_file": str(Path(input_file).name),
            "full_input_path": str(input_file),
            "generation_type": generation_type,
            "target_column": target_column,
            "analysis_timestamp": datetime.now().isoformat(),
            "script_version": "1.0"
        },
        "statistics": {
            "total_records": int(total_count),
            "non_null_records": int(non_null_count),
            "null_records": int(null_count),
            "null_percentage": round(null_percentage, 2)
        },
        "null_idx_info": {
            "total_null_idx": len(null_idx_list),
            "min_null_idx": int(min(null_idx_list)) if null_idx_list else None,
            "max_null_idx": int(max(null_idx_list)) if null_idx_list else None,
            "first_10_null_idx": null_idx_list[:10],
            "last_10_null_idx": null_idx_list[-10:] if len(null_idx_list) > 10 else null_idx_list,
            "null_idx_list": null_idx_list  # Complete list of null idx values
        },
        "consecutive_groups": []
    }
    
    # Add consecutive groups info to metadata
    if 'consecutive_groups' in locals() and consecutive_groups:
        for group in consecutive_groups:
            metadata["consecutive_groups"].append({
                "start_idx": int(group[0]),
                "end_idx": int(group[-1]),
                "count": len(group),
                "range": f"{group[0]}-{group[-1]}"
            })
    
    # Save metadata file
    metadata_filename = f"null_analysis_metadata_{generation_type}_{timestamp}.json"
    metadata_path = output_dir / metadata_filename
    
    print(f"\n💾 Saving analysis results...")
    print(f"Metadata file: {metadata_filename}")
    
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"✅ Metadata saved successfully")
        
        metadata_size = metadata_path.stat().st_size / 1024  # KB
        print(f"📁 Metadata file size: {metadata_size:.2f} KB")
        
    except Exception as e:
        print(f"❌ Error saving metadata: {e}")
        return False
    
    # Save null records file (complete records in same format as original)
    null_records_filename = f"null_records_{generation_type}_{timestamp}.json"
    null_records_path = output_dir / null_records_filename
    
    print(f"Null records file: {null_records_filename}")
    
    try:
        if len(null_records_df) > 0:
            # Convert to JSON in the same format as the original file
            # This creates a file like the original merged file but only with null records
            null_records_df.to_json(null_records_path, orient='records', indent=2)
            print(f"✅ Null records saved successfully")
            print(f"📊 Saved {len(null_records_df)} complete null records")
            
            null_records_size = null_records_path.stat().st_size / (1024 * 1024)  # MB
            print(f"📁 Null records file size: {null_records_size:.2f} MB")
        else:
            # Create empty file with proper structure if no nulls found
            empty_data = []
            with open(null_records_path, 'w', encoding='utf-8') as f:
                json.dump(empty_data, f, indent=2, ensure_ascii=False)
            print(f"✅ Empty null records file created (no null values found)")
        
    except Exception as e:
        print(f"❌ Error saving null records: {e}")
        return False
    
    # Final summary
    print(f"\n📊 Final Summary:")
    print(f"  Input file: {Path(input_file).name}")
    print(f"  Generation type: {generation_type}")
    print(f"  Target column: {target_column}")
    print(f"  Total records: {total_count}")
    print(f"  Null records: {null_count} ({null_percentage:.2f}%)")
    print(f"  Output directory: {output_dir}")
    print(f"  Metadata file: {metadata_filename}")
    print(f"  Null records file: {null_records_filename}")
    
    if null_count == 0:
        print("🎉 No null values found! Data quality is excellent.")
    elif null_percentage < 5:
        print("✅ Low null percentage - data quality is good.")
    elif null_percentage < 20:
        print("⚠️  Moderate null percentage - consider data quality review.")
    else:
        print("❌ High null percentage - data quality issues detected.")
    
    return True


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze null values in PADBen JSON files for different generation types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check type2 null values (sentence completion)
  python error_handling.py --input file.json --type type2
  
  # Check type4 null values (prompt-based paraphrasing)
  python error_handling.py --input file.json --type type4
  
  # Check type5 null values (paraphrased generated text)
  python error_handling.py --input file.json --type type5
  
  # Specify custom output directory
  python error_handling.py --input file.json --type type4 -o analysis_results/
  
  # Full example
  python error_handling.py \\
    --input data/merged/merged_type4_file.json \\
    --type type4 \\
    -o data/analysis/

Output Files:
  1. Metadata file: Summary statistics and null idx list
  2. Null records file: Complete records (same format as input) but only for null idx values

Supported Types:
  type2: Checks LLM generated text columns
  type4: Checks LLM paraphrased original text columns  
  type5: Checks LLM paraphrased generated text columns
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to the input JSON file to analyze"
    )
    
    parser.add_argument(
        "--type", "-t",
        choices=['type2', 'type4', 'type5'],
        required=True,
        help="Type of text generation to check for null values"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for analysis results (default: same as input file)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    success = analyze_null_values(
        input_file=args.input,
        generation_type=args.type,
        output_dir=args.output
    )
    
    if success:
        print("\n🎉 Null value analysis completed successfully!")
    else:
        print("\n❌ Null value analysis failed!")
