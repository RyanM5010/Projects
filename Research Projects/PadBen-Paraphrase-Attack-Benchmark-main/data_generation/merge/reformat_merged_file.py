#!/usr/bin/env python3
"""
Reformat merged PADBen file to standardized column naming convention.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


def reformat_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reformat a single record to the new standardized column names.
    
    Args:
        record: Original record dictionary
        
    Returns:
        Dict with standardized column names
    """
    # Create new formatted record with standardized structure
    formatted = {
        "idx": record.get("idx"),
        "dataset_source": record.get("dataset_source"),
        "human_original_text(type1)": record.get("human_original_text"),
        "llm_generated_text(type2)": None,
        "human_paraphrased_text(type3)": record.get("human_paraphrased_text"),
        "llm_paraphrased_original_text(type4)-prompt-based": None,
        "llm_paraphrased_original_text(type4)-dipper-based": None,
        "llm_paraphrased_generated_text(type5)-1st": None,
        "llm_paraphrased_generated_text(type5)-3rd": None,
    }
    
    # Handle Type 2 LLM generated text
    # Priority: specific method columns > generic column
    if "llm_generated_text(sentence_completion)" in record and record["llm_generated_text(sentence_completion)"] is not None:
        formatted["llm_generated_text(type2)"] = record["llm_generated_text(sentence_completion)"]
    elif "llm_generated_text(question_answer)" in record and record["llm_generated_text(question_answer)"] is not None:
        formatted["llm_generated_text(type2)"] = record["llm_generated_text(question_answer)"]
    elif "llm_generated_text" in record and record["llm_generated_text"] is not None:
        formatted["llm_generated_text(type2)"] = record["llm_generated_text"]
    
    # Handle Type 4 LLM paraphrased original text
    # Check for existing method-specific columns
    if "llm_paraphrased_original_text_prompt_based" in record:
        formatted["llm_paraphrased_original_text(type4)-prompt-based"] = record["llm_paraphrased_original_text_prompt_based"]
    elif "llm_paraphrased_original_text(prompt_based)" in record:
        formatted["llm_paraphrased_original_text(type4)-prompt-based"] = record["llm_paraphrased_original_text(prompt_based)"]
    elif "llm_paraphrased_original_text(Prompt_based)" in record:
        formatted["llm_paraphrased_original_text(type4)-prompt-based"] = record["llm_paraphrased_original_text(Prompt_based)"]
    
    if "llm_paraphrased_original_text_dipper_based" in record:
        formatted["llm_paraphrased_original_text(type4)-dipper-based"] = record["llm_paraphrased_original_text_dipper_based"]
    elif "llm_paraphrased_original_text(dipper_based)" in record:
        formatted["llm_paraphrased_original_text(type4)-dipper-based"] = record["llm_paraphrased_original_text(dipper_based)"]
    elif "llm_paraphrased_original_text(DIPPER_based)" in record:
        formatted["llm_paraphrased_original_text(type4)-dipper-based"] = record["llm_paraphrased_original_text(DIPPER_based)"]
    
    # If there's a generic Type 4 column, try to determine method or default to prompt-based
    if "llm_paraphrased_original_text" in record and record["llm_paraphrased_original_text"] is not None:
        # Check if we already have method-specific data
        if formatted["llm_paraphrased_original_text(type4)-prompt-based"] is None and formatted["llm_paraphrased_original_text(type4)-dipper-based"] is None:
            # Default to prompt-based if no method specified
            method = record.get("paraphrase_method", "prompt_based")
            if "dipper" in str(method).lower():
                formatted["llm_paraphrased_original_text(type4)-dipper-based"] = record["llm_paraphrased_original_text"]
            else:
                formatted["llm_paraphrased_original_text(type4)-prompt-based"] = record["llm_paraphrased_original_text"]
    
    # Handle Type 5 LLM paraphrased generated text
    # Check for iteration-specific columns
    if "llm_paraphrased_generated_text_1st" in record:
        formatted["llm_paraphrased_generated_text(type5)-1st"] = record["llm_paraphrased_generated_text_1st"]
    elif "llm_paraphrased_generated_text(1st)" in record:
        formatted["llm_paraphrased_generated_text(type5)-1st"] = record["llm_paraphrased_generated_text(1st)"]
    elif "llm_paraphrased_generated_text_1" in record:
        formatted["llm_paraphrased_generated_text(type5)-1st"] = record["llm_paraphrased_generated_text_1"]
    
    if "llm_paraphrased_generated_text_3rd" in record:
        formatted["llm_paraphrased_generated_text(type5)-3rd"] = record["llm_paraphrased_generated_text_3rd"]
    elif "llm_paraphrased_generated_text(3rd)" in record:
        formatted["llm_paraphrased_generated_text(type5)-3rd"] = record["llm_paraphrased_generated_text(3rd)"]
    elif "llm_paraphrased_generated_text_3" in record:
        formatted["llm_paraphrased_generated_text(type5)-3rd"] = record["llm_paraphrased_generated_text_3"]
    
    # If there's a generic Type 5 column, try to determine iteration
    if "llm_paraphrased_generated_text" in record and record["llm_paraphrased_generated_text"] is not None:
        # Check if we already have iteration-specific data
        if formatted["llm_paraphrased_generated_text(type5)-1st"] is None and formatted["llm_paraphrased_generated_text(type5)-3rd"] is None:
            # Try to determine iteration level
            iteration = record.get("iteration_level", "1")
            if "3" in str(iteration):
                formatted["llm_paraphrased_generated_text(type5)-3rd"] = record["llm_paraphrased_generated_text"]
            else:
                formatted["llm_paraphrased_generated_text(type5)-1st"] = record["llm_paraphrased_generated_text"]
    
    return formatted


def show_original_structure(data: list) -> None:
    """Show the original file structure for analysis."""
    if not data:
        return
    
    print("📋 Original file structure analysis:")
    sample_record = data[0]
    
    print(f"Sample record keys ({len(sample_record)} total):")
    for key in sample_record.keys():
        value = sample_record[key]
        if value is not None:
            value_preview = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
            print(f"  ✓ {key}: {value_preview}")
        else:
            print(f"  ○ {key}: null")
    
    # Count non-null values for each column
    print(f"\nData availability across {len(data)} records:")
    for key in sample_record.keys():
        non_null_count = sum(1 for record in data if record.get(key) is not None)
        percentage = (non_null_count / len(data)) * 100
        print(f"  {key}: {non_null_count}/{len(data)} ({percentage:.1f}%)")


def reformat_merged_file(input_file: str, output_file: Optional[str] = None) -> bool:
    """
    Reformat a merged PADBen file to standardized column naming convention.
    
    Args:
        input_file: Path to the input JSON file to reformat
        output_file: Optional custom output path. If None, generates timestamp-based name
        
    Returns:
        bool: True if reformatting successful, False otherwise
    """
    
    print("🎨 PADBen File Reformatter")
    print("=" * 80)
    print(f"Input file: {Path(input_file).name}")
    print("Target: Standardized column naming convention")
    print("=" * 80)
    
    # Validate input file exists
    if not Path(input_file).exists():
        print(f"❌ Error: Input file does not exist: {input_file}")
        return False
    
    # Load the file
    print("📂 Loading file...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} records")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return False
    
    # Show original structure
    show_original_structure(data)
    
    # Reformat all records
    print(f"\n🔄 Reformatting records...")
    reformatted_data = []
    
    for i, record in enumerate(data):
        try:
            formatted_record = reformat_record(record)
            reformatted_data.append(formatted_record)
            
            # Show progress for large files
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(data)} records...")
                
        except Exception as e:
            print(f"⚠️  Error reformatting record {i} (idx: {record.get('idx', 'unknown')}): {e}")
            continue
    
    print(f"✅ Reformatted {len(reformatted_data)} records")
    
    # Show sample of reformatted data
    if reformatted_data:
        print(f"\n📖 Sample reformatted record:")
        sample = reformatted_data[0]
        for key, value in sample.items():
            if value is not None:
                value_preview = str(value)[:60] + "..." if len(str(value)) > 60 else str(value)
                print(f"  ✓ {key}: {value_preview}")
            else:
                print(f"  ○ {key}: null")
        
        # Show data availability in reformatted structure
        print(f"\n📊 Reformatted data availability:")
        for key in sample.keys():
            non_null_count = sum(1 for record in reformatted_data if record.get(key) is not None)
            percentage = (non_null_count / len(reformatted_data)) * 100
            print(f"  {key}: {non_null_count}/{len(reformatted_data)} ({percentage:.1f}%)")
    
    # Generate output filename if not provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_path = Path(input_file)
        output_file = input_path.parent / f"reformatted_padben_type2_{timestamp}.json"
    else:
        output_file = Path(output_file)
        # Create output directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save reformatted file
    print(f"\n💾 Saving reformatted file...")
    print(f"Output: {output_file.name}")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(reformatted_data, f, indent=2, ensure_ascii=False)
        
        file_size = output_file.stat().st_size / (1024 * 1024)
        print(f"✅ Successfully saved!")
        print(f"📁 File size: {file_size:.2f} MB")
        
        # Final summary
        print(f"\n📊 Final Summary:")
        print(f"  Input records: {len(data)}")
        print(f"  Output records: {len(reformatted_data)}")
        print(f"  Format: Standardized column names")
        print(f"  Output file: {output_file}")
        
        # Show what was removed/transformed
        if data:
            original_keys = set(data[0].keys())
            new_keys = set(reformatted_data[0].keys())
            
            print(f"\n🔄 Transformation Summary:")
            print(f"  Original columns: {len(original_keys)}")
            print(f"  New columns: {len(new_keys)}")
            
            # Show removed columns
            removed_keys = original_keys - new_keys
            if removed_keys:
                print(f"  Removed columns: {len(removed_keys)}")
                for key in sorted(removed_keys):
                    print(f"    - {key}")
            
            # Show key transformations
            print(f"  Key transformations:")
            print(f"    - llm_generated_text(*) → llm_generated_text(type2)")
            print(f"    - human_original_text → human_original_text(type1)")
            print(f"    - human_paraphrased_text → human_paraphrased_text(type3)")
            print(f"    - Added type4 and type5 placeholder columns")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving reformatted file: {e}")
        return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Reformat merged PADBen file to standardized column naming convention",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic reformat
  python reformat_merged_file.py input.json
  
  # Specify output file
  python reformat_merged_file.py input.json -o reformatted_output.json
  
  # Full example
  python reformat_merged_file.py \\
    data/test/merged_file.json \\
    -o data/processed/reformatted_file.json
        """
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the JSON file to reformat"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file path (default: auto-generated with timestamp)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    success = reformat_merged_file(
        input_file=args.input_file,
        output_file=args.output
    )
    
    if success:
        print("\n🎉 File reformatting completed successfully!")
    else:
        print("\n❌ File reformatting failed!")
