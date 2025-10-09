#!/usr/bin/env python3
"""
Merge multiple PADBen JSON files based on 'idx' field.
For overlapping entries, randomly select one or prioritize specific files.
IMPORTANT: idx values are NOT necessarily continuous!
Supports merging any number of files for any type (type2, type4, type5, etc.)
"""

import argparse
import json
import random
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any


def merge_padben_files(
    input_files: List[str], 
    output_path: Optional[str] = None,
    random_seed: int = 42,
    merge_type: str = "auto",
    null_fill_merge: bool = False
) -> bool:
    """
    Merge multiple PADBen JSON files based on 'idx' field.
    
    Args:
        input_files: List of paths to JSON files to merge
        output_path: Optional custom output path. If None, generates timestamp-based name
        random_seed: Random seed for reproducible overlap resolution
        merge_type: Type of merge (type2, type4, type5, or auto-detect)
        null_fill_merge: If True, prioritize later files over earlier files for overlaps
        
    Returns:
        bool: True if merge successful, False otherwise
    """
    
    if len(input_files) < 2:
        print("❌ Error: At least 2 files are required for merging")
        return False
    
    print("🚀 PADBen Multi-File Merger")
    print("=" * 80)
    print(f"Merge type: {merge_type}")
    print(f"Null-fill merge: {'Yes (prioritize later files)' if null_fill_merge else 'No (random selection)'}")
    print(f"Input files: {len(input_files)}")
    for i, file_path in enumerate(input_files, 1):
        print(f"  File {i}: {Path(file_path).name}")
    print(f"Random seed: {random_seed}")
    print("⚠️  Note: idx values may NOT be continuous!")
    if null_fill_merge:
        print("🔄 Null-fill mode: Later files will override earlier files for overlapping idx values")
    print("=" * 80)
    
    # Validate all input files exist
    for i, file_path in enumerate(input_files, 1):
        if not Path(file_path).exists():
            print(f"❌ Error: File {i} does not exist: {file_path}")
            return False
    
    # Load all files
    print("📂 Loading files...")
    dataframes = []
    total_records = 0
    
    for i, file_path in enumerate(input_files, 1):
        try:
            print(f"Loading file {i}/{len(input_files)}: {Path(file_path).name}")
            df = pd.read_json(file_path)
            dataframes.append(df)
            total_records += len(df)
            print(f"✅ File {i} loaded: {len(df)} records")
            
        except Exception as e:
            print(f"❌ Error loading file {i}: {e}")
            return False
    
    print(f"📊 Total records across all files: {total_records}")
    
    # Validate that 'idx' column exists in all files
    for i, df in enumerate(dataframes, 1):
        if 'idx' not in df.columns:
            print(f"❌ Error: 'idx' column not found in file {i}")
            return False
    
    # Analyze each file
    print(f"\n📊 Individual File Analysis:")
    all_idx_sets = []
    
    for i, df in enumerate(dataframes, 1):
        duplicates = df['idx'].duplicated().sum()
        if duplicates > 0:
            print(f"⚠️  Warning: File {i} has {duplicates} duplicate idx values")
        
        idx_set = set(df['idx'].values)
        all_idx_sets.append(idx_set)
        
        print(f"File {i}: {len(df)} records")
        print(f"  Unique idx values: {df['idx'].nunique()}")
        print(f"  idx range: {df['idx'].min()} - {df['idx'].max()}")
    
    # Global overlap analysis
    print(f"\n🔍 Global Overlap Analysis:")
    all_idx_union = set.union(*all_idx_sets)
    total_unique_idx = len(all_idx_union)
    total_raw_records = sum(len(idx_set) for idx_set in all_idx_sets)
    total_overlaps = total_raw_records - total_unique_idx
    
    print(f"  Total unique idx values across all files: {total_unique_idx}")
    print(f"  Total raw records: {total_raw_records}")
    print(f"  Total overlapping records: {total_overlaps}")
    
    # Detailed overlap analysis
    if len(input_files) <= 5:  # Only show detailed analysis for reasonable number of files
        print(f"\n🔎 Detailed Overlap Analysis:")
        for i in range(len(all_idx_sets)):
            for j in range(i + 1, len(all_idx_sets)):
                overlap = all_idx_sets[i].intersection(all_idx_sets[j])
                if overlap:
                    sample_overlaps = sorted(list(overlap))[:5]
                    print(f"  File {i+1} ∩ File {j+1}: {len(overlap)} overlaps")
                    print(f"    Examples: {sample_overlaps}{'...' if len(overlap) > 5 else ''}")
    
    # Merge process using idx-based logic
    print(f"\n🔄 Merging files based on idx values...")
    
    # Convert all dataframes to dictionaries keyed by idx
    print("Creating idx-based dictionaries...")
    file_dicts = []
    
    for i, df in enumerate(dataframes, 1):
        file_dict = {}
        duplicates_found = 0
        
        for _, row in df.iterrows():
            idx_val = row['idx']
            if idx_val in file_dict:
                print(f"⚠️  Warning: Duplicate idx {idx_val} in file {i}, keeping first occurrence")
                duplicates_found += 1
            else:
                row_dict = row.to_dict()
                # Add source file info for tracking
                row_dict['_source_file'] = i
                file_dict[idx_val] = row_dict
        
        file_dicts.append(file_dict)
        print(f"Dict {i} created: {len(file_dict)} unique idx values")
        if duplicates_found > 0:
            print(f"  Removed {duplicates_found} duplicates from file {i}")
    
    # Build merged dictionary
    merged_dict = {}
    file_selection_stats = {i: 0 for i in range(1, len(input_files) + 1)}
    overlap_resolution_log = {}
    
    # Set seed for reproducible random selection
    random.seed(random_seed)
    
    # Process each unique idx value
    for idx_val in all_idx_union:
        # Find which files contain this idx
        files_with_idx = []
        records_for_idx = []
        
        for i, file_dict in enumerate(file_dicts, 1):
            if idx_val in file_dict:
                files_with_idx.append(i)
                records_for_idx.append(file_dict[idx_val])
        
        if len(files_with_idx) == 1:
            # No overlap - use the only available record
            selected_record = records_for_idx[0]
            selected_file = files_with_idx[0]
            file_selection_stats[selected_file] += 1
        else:
            # Overlap - select based on strategy
            if null_fill_merge:
                # For null-fill merge: ALWAYS prefer the LAST file (highest file number)
                # This ensures that filled records (typically in later files) override original records
                max_file_index = max(files_with_idx)
                selected_index = files_with_idx.index(max_file_index)
                selected_record = records_for_idx[selected_index]
                selected_file = max_file_index
            else:
                # Random selection (original behavior)
                selected_index = random.randint(0, len(records_for_idx) - 1)
                selected_record = records_for_idx[selected_index]
                selected_file = files_with_idx[selected_index]
            
            file_selection_stats[selected_file] += 1
            
            # Log overlap resolution
            overlap_resolution_log[idx_val] = {
                'available_files': files_with_idx,
                'selected_file': selected_file,
                'strategy': 'prefer_last_file' if null_fill_merge else 'random'
            }
        
        # Remove source file tracking before final storage
        final_record = {k: v for k, v in selected_record.items() if k != '_source_file'}
        merged_dict[idx_val] = final_record
    
    print(f"✅ Merge completed!")
    print(f"📈 Final dataset size: {len(merged_dict)} records")
    
    # Show file selection statistics
    print(f"\n🎲 File Selection Statistics:")
    for i, count in file_selection_stats.items():
        percentage = (count / len(merged_dict)) * 100
        print(f"  File {i}: {count} records ({percentage:.1f}%)")
    
    if overlap_resolution_log:
        print(f"  Overlaps resolved: {len(overlap_resolution_log)}")
        
        # Show some examples of overlap resolution
        sample_overlaps = list(overlap_resolution_log.items())[:5]
        print(f"  Example overlap resolutions:")
        for idx_val, resolution in sample_overlaps:
            strategy_info = f" [{resolution['strategy']}]" if null_fill_merge else ""
            print(f"    idx {idx_val}: files {resolution['available_files']} → selected file {resolution['selected_file']}{strategy_info}")
    
    # Convert back to DataFrame and sort by idx
    print(f"\n📊 Creating final dataset...")
    merged_records = list(merged_dict.values())
    merged_df = pd.DataFrame(merged_records)
    
    # Sort by idx for consistency
    merged_df = merged_df.sort_values('idx').reset_index(drop=True)
    
    print(f"📊 Final idx range: {merged_df['idx'].min()}-{merged_df['idx'].max()}")
    print(f"🔍 Unique idx values: {merged_df['idx'].nunique()}")
    
    # Verify no duplicate idx in final result
    final_duplicates = merged_df['idx'].duplicated().sum()
    if final_duplicates > 0:
        print(f"❌ Error: Final dataset has {final_duplicates} duplicate idx values!")
        return False
    else:
        print(f"✅ Verified: No duplicate idx values in final dataset")
    
    # Generate output filename if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(input_files[0]).parent
        
        # Auto-detect merge type from first file if not specified
        if merge_type == "auto":
            first_file_name = Path(input_files[0]).name
            if "type2" in first_file_name:
                detected_type = "type2"
            elif "type4" in first_file_name:
                detected_type = "type4"
            elif "type5" in first_file_name:
                detected_type = "type5"
            else:
                detected_type = "unknown"
            merge_type = detected_type
        
        output_filename = f"merged_padben_{merge_type}{'_null_filled' if null_fill_merge else ''}_{timestamp}.json"
        output_path = output_dir / output_filename
    else:
        output_path = Path(output_path)
        
        # Check if output_path is a directory or file
        if output_path.is_dir() or (not output_path.suffix and not output_path.exists()):
            # It's a directory, create a filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if merge_type == "auto":
                first_file_name = Path(input_files[0]).name
                if "type2" in first_file_name:
                    detected_type = "type2"
                elif "type4" in first_file_name:
                    detected_type = "type4"
                elif "type5" in first_file_name:
                    detected_type = "type5"
                else:
                    detected_type = "unknown"
                merge_type = detected_type
            
            output_filename = f"merged_padben_{merge_type}{'_null_filled' if null_fill_merge else ''}_{timestamp}.json"
            output_path = output_path / output_filename
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save merged file
    print(f"\n💾 Saving merged file...")
    print(f"Output: {output_path.name}")
    print(f"Full path: {output_path}")
    
    try:
        merged_df.to_json(output_path, orient='records', indent=2)
        print(f"✅ Successfully saved merged file!")
        
        # Get file size
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        print(f"📁 File size: {file_size:.2f} MB")
        
        # Final summary statistics
        print(f"\n📊 Final Summary:")
        print(f"  Input files: {len(input_files)}")
        print(f"  Total input records: {total_records}")
        print(f"  Final merged records: {len(merged_df)}")
        print(f"  Unique idx values: {merged_df['idx'].nunique()}")
        print(f"  idx range: {merged_df['idx'].min()} to {merged_df['idx'].max()}")
        print(f"  Overlaps resolved: {len(overlap_resolution_log)}")
        print(f"  Merge type: {merge_type}")
        print(f"  Null-fill merge: {'Yes' if null_fill_merge else 'No'}")
        print(f"  Output file: {output_path}")
        print(f"  File size: {file_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving merged file: {e}")
        return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge multiple PADBen JSON files based on 'idx' field",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic merge of 2 files
  python merge_script.py --files file1.json file2.json
  
  # Merge with null-fill priority (later files override earlier files)
  python merge_script.py --files original.json filled.json --null_fill_merge
  
  # Merge multiple files (type5 with multiple chunks)
  python merge_script.py --files chunk1.json chunk2.json chunk3.json chunk4.json
  
  # Specify output directory and merge type
  python merge_script.py --files file1.json file2.json \\
    -o data/merged/ --type type4 --null_fill_merge
  
  # Use different random seed
  python merge_script.py --files file1.json file2.json --seed 123
  
  # Full example with type5 chunks
  python merge_script.py \\
    --files data/type5/chunk_0_3333.json \\
            data/type5/chunk_3333_6666.json \\
            data/type5/chunk_6666_10000.json \\
            data/type5/chunk_10000_end.json \\
    --type type5 \\
    -o data/merged/merged_type5_complete.json \\
    --seed 42
        """
    )
    
    parser.add_argument(
        "--files",
        nargs='+',
        required=True,
        help="Paths to JSON files to merge (space-separated, minimum 2 files)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file path or directory (default: auto-generated with timestamp)"
    )
    
    parser.add_argument(
        "--type",
        choices=['type2', 'type4', 'type5', 'auto'],
        default='auto',
        help="Type of merge for output naming (default: auto-detect from first file)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible overlap resolution (default: 42)"
    )
    
    parser.add_argument(
        "--null_fill_merge",
        action='store_true',
        help="Prioritize later files over earlier files for overlapping idx values (useful for merging original + filled records)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Validate minimum number of files
    if len(args.files) < 2:
        print("❌ Error: At least 2 files are required for merging")
        print("Use --help for usage examples")
        exit(1)
    
    success = merge_padben_files(
        input_files=args.files,
        output_path=args.output,
        random_seed=args.seed,
        merge_type=args.type,
        null_fill_merge=args.null_fill_merge
    )
    
    if success:
        print("\n🎉 Merge completed successfully!")
    else:
        print("\n❌ Merge failed!")