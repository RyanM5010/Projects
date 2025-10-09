#!/usr/bin/env python3
"""
Remove null type5 records and unwanted columns from PADBen JSON files.

This script:
1. Removes records where type5-1st OR type5-3rd columns are null
2. Removes unwanted columns (type4-dipper-based and type5-5th)
3. Reindexes the remaining records starting from 0
4. Saves the cleaned data to a new file

Usage:
    python remove_type5_null.py --input input_file.json --output output_file.json
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


def clean_type5_data(
    input_file: str,
    output_file: str = None,
    verbose: bool = True
) -> bool:
    """
    Clean PADBen type5 data by removing null records and unwanted columns.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file (optional, auto-generated if None)
        verbose: Whether to print detailed progress information
        
    Returns:
        bool: True if successful, False otherwise
    """
    
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"❌ Error: Input file does not exist: {input_file}")
        return False
    
    if verbose:
        print("🧹 PADBen Type5 Data Cleaner")
        print("=" * 50)
        print(f"Input file: {input_path.name}")
        print(f"Full path: {input_path}")
    
    try:
        # Load the JSON file
        if verbose:
            print("\n📂 Loading JSON file...")
        
        df = pd.read_json(input_path)
        original_count = len(df)
        
        if verbose:
            print(f"✅ Loaded {original_count:,} records")
            
            # Show original column structure
            print(f"\n📊 Original columns:")
            for col in df.columns:
                print(f"  - {col}")
        
        # Analyze null counts before cleaning
        type5_1st_col = 'llm_paraphrased_generated_text(type5)-1st'
        type5_3rd_col = 'llm_paraphrased_generated_text(type5)-3rd'
        
        if verbose:
            print(f"\n🔍 Null analysis before cleaning:")
            if type5_1st_col in df.columns:
                null_1st = df[type5_1st_col].isna().sum()
                print(f"  {type5_1st_col}: {null_1st:,} nulls ({null_1st/len(df)*100:.1f}%)")
            
            if type5_3rd_col in df.columns:
                null_3rd = df[type5_3rd_col].isna().sum()
                print(f"  {type5_3rd_col}: {null_3rd:,} nulls ({null_3rd/len(df)*100:.1f}%)")
            
            # Count records that will be removed
            records_to_remove = df[
                (df[type5_1st_col].isna()) | (df[type5_3rd_col].isna())
            ] if type5_1st_col in df.columns and type5_3rd_col in df.columns else pd.DataFrame()
            
            print(f"  Records to remove (1st OR 3rd null): {len(records_to_remove):,}")
        
        # Step 1: Remove records where type5-1st OR type5-3rd are null
        if verbose:
            print(f"\n🗑️  Step 1: Removing records with null type5 values...")
        
        # Keep only records where BOTH type5-1st AND type5-3rd are NOT null
        if type5_1st_col in df.columns and type5_3rd_col in df.columns:
            df_cleaned = df[
                (df[type5_1st_col].notna()) & (df[type5_3rd_col].notna())
            ].copy()
        else:
            print(f"⚠️  Warning: Expected type5 columns not found in data")
            df_cleaned = df.copy()
        
        records_removed = original_count - len(df_cleaned)
        
        if verbose:
            print(f"✅ Removed {records_removed:,} records")
            print(f"✅ Remaining records: {len(df_cleaned):,}")
        
        # Step 2: Remove unwanted columns
        columns_to_remove = [
            'llm_paraphrased_original_text(type4)-dipper-based',
            'llm_paraphrased_generated_text(type5)-5th'
        ]
        
        if verbose:
            print(f"\n🗑️  Step 2: Removing unwanted columns...")
        
        columns_removed = []
        for col in columns_to_remove:
            if col in df_cleaned.columns:
                df_cleaned = df_cleaned.drop(columns=[col])
                columns_removed.append(col)
                if verbose:
                    print(f"✅ Removed column: {col}")
            else:
                if verbose:
                    print(f"⚠️  Column not found (skipped): {col}")
        
        # Step 3: Reindex from 0
        if verbose:
            print(f"\n🔢 Step 3: Reindexing records from 0...")
        
        # Reset the 'idx' column to start from 0
        df_cleaned['idx'] = range(len(df_cleaned))
        
        if verbose:
            print(f"✅ Reindexed {len(df_cleaned):,} records (0 to {len(df_cleaned)-1})")
            
            # Show final column structure
            print(f"\n📊 Final columns:")
            for col in df_cleaned.columns:
                print(f"  - {col}")
        
        # Generate output filename if not provided
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{input_path.stem}_cleaned_{timestamp}.json"
            output_path = input_path.parent / output_filename
        else:
            output_path = Path(output_file)
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save cleaned data
        if verbose:
            print(f"\n💾 Saving cleaned data...")
            print(f"Output: {output_path.name}")
            print(f"Full path: {output_path}")
        
        # Save as JSON with proper formatting
        df_cleaned.to_json(output_path, orient='records', indent=2)
        
        # Get file size
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        
        if verbose:
            print(f"✅ Successfully saved cleaned file!")
            print(f"📁 File size: {file_size:.2f} MB")
            
            # Final summary
            print(f"\n📊 Cleaning Summary:")
            print(f"  Original records: {original_count:,}")
            print(f"  Records removed: {records_removed:,} ({records_removed/original_count*100:.1f}%)")
            print(f"  Final records: {len(df_cleaned):,}")
            print(f"  Columns removed: {len(columns_removed)}")
            print(f"  Reindexed: 0 to {len(df_cleaned)-1}")
            print(f"  Output file: {output_path}")
            print(f"  File size: {file_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return False


def analyze_file_before_cleaning(input_file: str) -> Dict[str, Any]:
    """
    Analyze the input file to show what will be cleaned.
    
    Args:
        input_file: Path to input JSON file
        
    Returns:
        dict: Analysis results
    """
    try:
        df = pd.read_json(input_file)
        
        analysis = {
            'total_records': len(df),
            'columns': list(df.columns),
            'type5_analysis': {}
        }
        
        # Analyze type5 columns
        type5_columns = [
            'llm_paraphrased_generated_text(type5)-1st',
            'llm_paraphrased_generated_text(type5)-3rd',
            'llm_paraphrased_generated_text(type5)-5th'
        ]
        
        for col in type5_columns:
            if col in df.columns:
                null_count = df[col].isna().sum()
                non_null_count = df[col].notna().sum()
                analysis['type5_analysis'][col] = {
                    'null_count': null_count,
                    'non_null_count': non_null_count,
                    'null_percentage': (null_count / len(df)) * 100
                }
        
        # Count records that will be removed
        type5_1st = 'llm_paraphrased_generated_text(type5)-1st'
        type5_3rd = 'llm_paraphrased_generated_text(type5)-3rd'
        
        if type5_1st in df.columns and type5_3rd in df.columns:
            records_to_remove = len(df[(df[type5_1st].isna()) | (df[type5_3rd].isna())])
            analysis['records_to_remove'] = records_to_remove
            analysis['records_to_keep'] = len(df) - records_to_remove
        
        return analysis
        
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")
        return {}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean PADBen type5 data by removing null records and unwanted columns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic cleaning with auto-generated output filename
  python remove_type5_null.py --input merged_data.json
  
  # Specify custom output file
  python remove_type5_null.py --input merged_data.json --output cleaned_data.json
  
  # Analyze file without cleaning (dry run)
  python remove_type5_null.py --input merged_data.json --analyze_only
  
  # Clean with minimal output
  python remove_type5_null.py --input merged_data.json --quiet

Cleaning operations:
  1. Remove records where type5-1st OR type5-3rd are null
  2. Remove columns: type4-dipper-based, type5-5th
  3. Reindex records starting from 0
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input JSON file to clean"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Path to output JSON file (default: auto-generated with timestamp)"
    )
    
    parser.add_argument(
        "--analyze_only",
        action='store_true',
        help="Only analyze the file without performing cleaning"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action='store_true',
        help="Minimize output messages"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Validate input file
    if not Path(args.input).exists():
        print(f"❌ Error: Input file does not exist: {args.input}")
        exit(1)
    
    if args.analyze_only:
        # Perform analysis only
        print("🔍 Analyzing file (no changes will be made)...")
        analysis = analyze_file_before_cleaning(args.input)
        
        if analysis:
            print(f"\n📊 File Analysis Results:")
            print(f"  Total records: {analysis['total_records']:,}")
            print(f"  Total columns: {len(analysis['columns'])}")
            
            if 'type5_analysis' in analysis:
                print(f"\n🔍 Type5 Column Analysis:")
                for col, stats in analysis['type5_analysis'].items():
                    print(f"  {col}:")
                    print(f"    Non-null: {stats['non_null_count']:,} ({100-stats['null_percentage']:.1f}%)")
                    print(f"    Null: {stats['null_count']:,} ({stats['null_percentage']:.1f}%)")
            
            if 'records_to_remove' in analysis:
                print(f"\n🗑️  Cleaning Preview:")
                print(f"  Records to remove: {analysis['records_to_remove']:,}")
                print(f"  Records to keep: {analysis['records_to_keep']:,}")
                print(f"  Removal percentage: {(analysis['records_to_remove']/analysis['total_records'])*100:.1f}%")
        
        print(f"\n✅ Analysis complete (no files modified)")
    else:
        # Perform cleaning
        verbose = not args.quiet
        success = clean_type5_data(
            input_file=args.input,
            output_file=args.output,
            verbose=verbose
        )
        
        if success:
            if not args.quiet:
                print("\n🎉 Cleaning completed successfully!")
        else:
            print("\n❌ Cleaning failed!")
            exit(1)
