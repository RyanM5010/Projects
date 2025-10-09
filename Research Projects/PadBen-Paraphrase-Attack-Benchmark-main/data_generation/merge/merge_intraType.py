#!/usr/bin/env python3
"""
Flexible intra-type merge script for combining multiple reformatted PADBen JSON files.
Merges files with different columns filled while preserving all data.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict


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


def analyze_file_structure(data: List[Dict[str, Any]], file_name: str) -> Dict[str, Any]:
    """
    Analyze the structure and data availability of a file.
    
    Args:
        data: List of records
        file_name: Name of the file for reporting
        
    Returns:
        Dictionary with analysis results
    """
    if not data:
        return {"file_name": file_name, "record_count": 0, "columns": {}}
    
    sample_record = data[0]
    column_stats = {}
    
    # Analyze each column
    for column in sample_record.keys():
        non_null_count = sum(1 for record in data if record.get(column) is not None)
        column_stats[column] = {
            "non_null_count": non_null_count,
            "percentage": (non_null_count / len(data)) * 100,
            "has_data": non_null_count > 0
        }
    
    return {
        "file_name": file_name,
        "record_count": len(data),
        "columns": column_stats
    }


def merge_records(records: List[Dict[str, Any]], idx: int) -> Dict[str, Any]:
    """
    Merge multiple records with the same idx, combining non-null values.
    
    Args:
        records: List of records with the same idx
        idx: The idx value being merged
        
    Returns:
        Merged record with combined data
    """
    if not records:
        return {}
    
    # Start with the first record as base
    merged = records[0].copy()
    
    # Merge data from other records
    for record in records[1:]:
        for key, value in record.items():
            if value is not None and merged.get(key) is None:
                # Fill null values with non-null values from other records
                merged[key] = value
            elif value is not None and merged.get(key) is not None:
                # Handle conflicts (both records have non-null values)
                if merged[key] != value:
                    print(f"⚠️  Conflict at idx {idx}, column '{key}': "
                          f"'{merged[key]}' vs '{value}' - keeping first value")
    
    return merged


def merge_files(file_paths: List[str], output_file: Optional[str] = None) -> bool:
    """
    Merge multiple reformatted PADBen JSON files.
    
    Args:
        file_paths: List of paths to JSON files to merge
        output_file: Optional output file path
        
    Returns:
        bool: True if merge successful, False otherwise
    """
    print("🔗 PADBen Intra-Type Merger")
    print("=" * 80)
    print(f"Files to merge: {len(file_paths)}")
    for i, path in enumerate(file_paths, 1):
        print(f"  {i}. {Path(path).name}")
    print("=" * 80)
    
    # Load all files
    all_data = {}
    file_analyses = []
    
    print("\n📂 Loading files...")
    for file_path in file_paths:
        try:
            data = load_json_file(file_path)
            file_name = Path(file_path).name
            all_data[file_name] = data
            
            # Analyze file structure
            analysis = analyze_file_structure(data, file_name)
            file_analyses.append(analysis)
            
            print(f"✅ Loaded {file_name}: {len(data)} records")
            
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return False
    
    # Show file analysis
    print(f"\n📊 File Analysis:")
    all_columns = set()
    for analysis in file_analyses:
        all_columns.update(analysis["columns"].keys())
        print(f"\n  📁 {analysis['file_name']}:")
        print(f"    Records: {analysis['record_count']}")
        print(f"    Columns with data:")
        
        for col, stats in analysis["columns"].items():
            if stats["has_data"]:
                print(f"      ✓ {col}: {stats['non_null_count']} ({stats['percentage']:.1f}%)")
    
    print(f"\n🔍 Column Coverage Across All Files:")
    for column in sorted(all_columns):
        files_with_data = []
        for analysis in file_analyses:
            if column in analysis["columns"] and analysis["columns"][column]["has_data"]:
                files_with_data.append(analysis["file_name"])
        
        if files_with_data:
            print(f"  ✓ {column}: {len(files_with_data)} file(s) - {', '.join(files_with_data)}")
        else:
            print(f"  ○ {column}: No data in any file")
    
    # Organize records by idx for merging
    print(f"\n🔄 Organizing records by idx...")
    records_by_idx = defaultdict(list)
    
    for file_name, data in all_data.items():
        for record in data:
            idx = record.get("idx")
            if idx is not None:
                records_by_idx[idx].append(record)
            else:
                print(f"⚠️  Record without idx in {file_name}: {record}")
    
    print(f"✅ Found {len(records_by_idx)} unique idx values")
    
    # Check for consistency
    total_records = sum(len(data) for data in all_data.values())
    expected_unique_idx = len(records_by_idx)
    
    print(f"📈 Merge Statistics:")
    print(f"  Total input records: {total_records}")
    print(f"  Unique idx values: {expected_unique_idx}")
    
    # Identify records that appear in multiple files
    multi_file_records = sum(1 for records in records_by_idx.values() if len(records) > 1)
    single_file_records = len(records_by_idx) - multi_file_records
    
    print(f"  Records in multiple files: {multi_file_records}")
    print(f"  Records in single file: {single_file_records}")
    
    # Merge records
    print(f"\n🔗 Merging records...")
    merged_data = []
    conflicts_count = 0
    
    for idx in sorted(records_by_idx.keys()):
        records = records_by_idx[idx]
        
        if len(records) == 1:
            # Single record, no merging needed
            merged_data.append(records[0])
        else:
            # Multiple records, merge them
            merged_record = merge_records(records, idx)
            merged_data.append(merged_record)
            
            # Count potential conflicts (records with overlapping non-null data)
            for i, record1 in enumerate(records):
                for j, record2 in enumerate(records[i+1:], i+1):
                    for key in record1.keys():
                        if (record1.get(key) is not None and 
                            record2.get(key) is not None and 
                            record1[key] != record2[key]):
                            conflicts_count += 1
    
    print(f"✅ Merged {len(merged_data)} records")
    if conflicts_count > 0:
        print(f"⚠️  Detected {conflicts_count} data conflicts (kept first values)")
    
    # Analyze merged data
    print(f"\n📊 Merged Data Analysis:")
    merged_analysis = analyze_file_structure(merged_data, "merged_data")
    
    for col, stats in merged_analysis["columns"].items():
        if stats["has_data"]:
            print(f"  ✓ {col}: {stats['non_null_count']} ({stats['percentage']:.1f}%)")
        else:
            print(f"  ○ {col}: No data")
    
    # Generate output filename if not provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"data/test/merged_intratype/merged_padben_intratype_{timestamp}.json"
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save merged file
    print(f"\n💾 Saving merged file...")
    print(f"Output: {output_path.name}")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        
        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"✅ Successfully saved!")
        print(f"📁 File size: {file_size:.2f} MB")
        
        # Final summary
        print(f"\n🎉 Merge Summary:")
        print(f"  Input files: {len(file_paths)}")
        print(f"  Total input records: {total_records}")
        print(f"  Output records: {len(merged_data)}")
        print(f"  Data conflicts resolved: {conflicts_count}")
        print(f"  Output file: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving merged file: {e}")
        return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge multiple reformatted PADBen JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge Type 2 and Type 4 files
  python merge_intraType.py \\
    data/test/merged_type2/reformatted_padben_type2_20250829_180512.json \\
    data/test/merged_type4/reformatted_padben_type4_20250829_181330.json
  
  # Specify output file
  python merge_intraType.py file1.json file2.json -o merged_output.json
  
  # Merge multiple files (extensible for Type 5)
  python merge_intraType.py \\
    reformatted_type2.json \\
    reformatted_type4.json \\
    reformatted_type5.json \\
    -o complete_padben_dataset.json
        """
    )
    
    parser.add_argument(
        "files",
        nargs="+",
        help="JSON files to merge (must be reformatted with standardized columns)"
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
    
    success = merge_files(
        file_paths=args.files,
        output_file=args.output
    )
    
    if success:
        print("\n🎉 Intra-type merge completed successfully!")
    else:
        print("\n❌ Intra-type merge failed!")
