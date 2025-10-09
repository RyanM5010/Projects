"""
Short Text Analyzer for PADBen Dataset Quality Examination.

This module identifies and analyzes abnormally short texts that may indicate
data quality issues or processing errors.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
from datetime import datetime
from collections import Counter

try:
    from .config import TEXT_TYPES, OUTPUT_DIR
    from .data_loader import DataLoader
except ImportError:
    from config import TEXT_TYPES, OUTPUT_DIR
    from data_loader import DataLoader

logger = logging.getLogger(__name__)


class ShortTextAnalyzer:
    """Analyzes abnormally short texts in the PADBen dataset."""
    
    def __init__(self, threshold: int = 10, data_loader: Optional[DataLoader] = None) -> None:
        """
        Initialize short text analyzer.
        
        Args:
            threshold: Maximum token length to consider as "short" (default: 10).
            data_loader: Optional DataLoader instance. If None, creates new one.
        """
        self.threshold = threshold
        self.data_loader = data_loader or DataLoader()
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.short_texts_analysis: Dict[str, Any] = {}
        self.problematic_records: List[Dict[str, Any]] = []
        
    def analyze_short_texts(self) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of short texts.
        
        Returns:
            Dictionary containing analysis results.
        """
        logger.info(f"Starting short text analysis with threshold: {self.threshold} tokens")
        
        # Load data
        data = self.data_loader.load_data()
        logger.info(f"Analyzing {len(data)} records")
        
        # Initialize analysis structure
        analysis = {
            "analysis_metadata": {
                "timestamp": datetime.now().isoformat(),
                "threshold": self.threshold,
                "total_records": len(data),
                "text_types": list(TEXT_TYPES.keys())
            },
            "short_text_statistics": {},
            "problematic_records": [],
            "pattern_analysis": {},
            "recommendations": []
        }
        
        # Analyze each text type
        for text_type, field_name in TEXT_TYPES.items():
            logger.info(f"Analyzing {text_type}...")
            type_analysis = self._analyze_text_type(data, text_type, field_name)
            analysis["short_text_statistics"][text_type] = type_analysis
            
        # Find records with multiple short texts
        analysis["problematic_records"] = self._find_problematic_records(data)
        
        # Perform pattern analysis
        analysis["pattern_analysis"] = self._analyze_patterns(data)
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        self.short_texts_analysis = analysis
        logger.info("Short text analysis completed")
        
        return analysis
    
    def _analyze_text_type(self, data: List[Dict], text_type: str, field_name: str) -> Dict[str, Any]:
        """
        Analyze short texts for a specific text type.
        
        Args:
            data: List of data records.
            text_type: Type identifier (e.g., 'type1').
            field_name: Field name in the data.
            
        Returns:
            Analysis results for this text type.
        """
        short_texts = []
        all_lengths = []
        empty_count = 0
        
        for i, record in enumerate(data):
            text = record.get(field_name, "")
            if not text or not str(text).strip():
                empty_count += 1
                continue
                
            text_str = str(text).strip()
            tokens = text_str.split()
            length = len(tokens)
            all_lengths.append(length)
            
            if length <= self.threshold:
                short_texts.append({
                    "record_idx": record.get("idx", i),
                    "dataset_source": record.get("dataset_source", "unknown"),
                    "text": text_str,
                    "token_count": length,
                    "char_count": len(text_str),
                    "tokens": tokens
                })
        
        # Calculate statistics
        total_texts = len(data) - empty_count
        short_count = len(short_texts)
        short_percentage = (short_count / total_texts * 100) if total_texts > 0 else 0
        
        # Length distribution
        length_distribution = Counter(all_lengths)
        
        return {
            "total_texts": total_texts,
            "empty_texts": empty_count,
            "short_texts_count": short_count,
            "short_texts_percentage": short_percentage,
            "average_length": sum(all_lengths) / len(all_lengths) if all_lengths else 0,
            "min_length": min(all_lengths) if all_lengths else 0,
            "max_length": max(all_lengths) if all_lengths else 0,
            "length_distribution": dict(sorted(length_distribution.items())),
            "short_text_examples": short_texts[:50],  # Limit examples for readability
            "all_short_texts": short_texts  # Full list for detailed analysis
        }
    
    def _find_problematic_records(self, data: List[Dict]) -> List[Dict[str, Any]]:
        """
        Find records that have multiple short texts across different types.
        
        Args:
            data: List of data records.
            
        Returns:
            List of problematic records.
        """
        problematic = []
        
        for i, record in enumerate(data):
            short_types = []
            record_analysis = {
                "record_idx": record.get("idx", i),
                "dataset_source": record.get("dataset_source", "unknown"),
                "short_text_types": [],
                "text_details": {}
            }
            
            for text_type, field_name in TEXT_TYPES.items():
                text = record.get(field_name, "")
                if text and str(text).strip():
                    text_str = str(text).strip()
                    tokens = text_str.split()
                    length = len(tokens)
                    
                    record_analysis["text_details"][text_type] = {
                        "text": text_str,
                        "token_count": length,
                        "char_count": len(text_str),
                        "is_short": length <= self.threshold
                    }
                    
                    if length <= self.threshold:
                        short_types.append(text_type)
            
            # Consider problematic if multiple types are short or any type is extremely short (<= 3)
            has_multiple_short = len(short_types) >= 2
            has_extremely_short = any(
                details.get("token_count", float('inf')) <= 3 
                for details in record_analysis["text_details"].values()
            )
            
            if has_multiple_short or has_extremely_short:
                record_analysis["short_text_types"] = short_types
                record_analysis["issue_type"] = "multiple_short" if has_multiple_short else "extremely_short"
                problematic.append(record_analysis)
        
        return problematic
    
    def _analyze_patterns(self, data: List[Dict]) -> Dict[str, Any]:
        """
        Analyze patterns in short texts to identify common issues.
        
        Args:
            data: List of data records.
            
        Returns:
            Pattern analysis results.
        """
        patterns = {
            "common_short_phrases": Counter(),
            "single_word_texts": Counter(),
            "dataset_source_distribution": Counter(),
            "length_patterns": {
                "1_token": [],
                "2_tokens": [],
                "3_tokens": [],
                "4_5_tokens": [],
                "6_10_tokens": []
            }
        }
        
        for record in data:
            for text_type, field_name in TEXT_TYPES.items():
                text = record.get(field_name, "")
                if not text or not str(text).strip():
                    continue
                    
                text_str = str(text).strip()
                tokens = text_str.split()
                length = len(tokens)
                
                if length <= self.threshold:
                    # Track common phrases
                    patterns["common_short_phrases"][text_str] += 1
                    
                    # Track single words
                    if length == 1:
                        patterns["single_word_texts"][tokens[0]] += 1
                    
                    # Track dataset sources
                    source = record.get("dataset_source", "unknown")
                    patterns["dataset_source_distribution"][source] += 1
                    
                    # Categorize by length
                    example = {
                        "text": text_str,
                        "type": text_type,
                        "source": source,
                        "idx": record.get("idx", "unknown")
                    }
                    
                    if length == 1:
                        patterns["length_patterns"]["1_token"].append(example)
                    elif length == 2:
                        patterns["length_patterns"]["2_tokens"].append(example)
                    elif length == 3:
                        patterns["length_patterns"]["3_tokens"].append(example)
                    elif length <= 5:
                        patterns["length_patterns"]["4_5_tokens"].append(example)
                    else:
                        patterns["length_patterns"]["6_10_tokens"].append(example)
        
        # Convert Counters to regular dicts and limit examples
        for key in ["common_short_phrases", "single_word_texts", "dataset_source_distribution"]:
            patterns[key] = dict(patterns[key].most_common(20))
        
        # Limit examples in length patterns
        for length_key in patterns["length_patterns"]:
            patterns["length_patterns"][length_key] = patterns["length_patterns"][length_key][:20]
        
        return patterns
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on the analysis.
        
        Args:
            analysis: Analysis results.
            
        Returns:
            List of recommendations.
        """
        recommendations = []
        
        # Check overall short text percentages
        high_short_percentage_types = []
        for text_type, stats in analysis["short_text_statistics"].items():
            if stats["short_texts_percentage"] > 5:  # More than 5% short texts
                high_short_percentage_types.append((text_type, stats["short_texts_percentage"]))
        
        if high_short_percentage_types:
            recommendations.append(
                f"High percentage of short texts detected in: {', '.join([f'{t}({p:.1f}%)' for t, p in high_short_percentage_types])}. "
                "Consider reviewing data preprocessing pipeline."
            )
        
        # Check for problematic records
        problematic_count = len(analysis["problematic_records"])
        if problematic_count > 100:
            recommendations.append(
                f"Found {problematic_count} records with multiple short texts or extremely short texts. "
                "This suggests systematic data quality issues."
            )
        
        # Check for common patterns
        patterns = analysis["pattern_analysis"]
        if patterns["single_word_texts"]:
            top_single_word = list(patterns["single_word_texts"].keys())[0]
            count = patterns["single_word_texts"][top_single_word]
            recommendations.append(
                f"Most common single-word text: '{top_single_word}' ({count} occurrences). "
                "Check if this indicates truncation or processing errors."
            )
        
        # Dataset-specific recommendations
        if patterns["dataset_source_distribution"]:
            worst_source = max(patterns["dataset_source_distribution"], 
                             key=patterns["dataset_source_distribution"].get)
            worst_count = patterns["dataset_source_distribution"][worst_source]
            recommendations.append(
                f"Dataset '{worst_source}' has the most short texts ({worst_count}). "
                "Review data extraction/processing for this source."
            )
        
        if not recommendations:
            recommendations.append("No significant short text issues detected. Data quality appears good.")
        
        return recommendations
    
    def save_analysis_results(self, filename: Optional[str] = None) -> Path:
        """
        Save analysis results to JSON file.
        
        Args:
            filename: Optional custom filename.
            
        Returns:
            Path to saved file.
        """
        if not self.short_texts_analysis:
            raise ValueError("No analysis results to save. Run analyze_short_texts() first.")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"short_text_analysis_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.short_texts_analysis, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Short text analysis saved to: {output_path}")
        return output_path
    
    def export_for_human_evaluation(self, filename: Optional[str] = None, 
                                   max_examples: int = 200) -> Path:
        """
        Export problematic texts in a format suitable for human evaluation.
        
        Args:
            filename: Optional custom filename.
            max_examples: Maximum number of examples to include.
            
        Returns:
            Path to exported file.
        """
        if not self.short_texts_analysis:
            raise ValueError("No analysis results available. Run analyze_short_texts() first.")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"short_texts_human_eval_{timestamp}.json"
        
        # Prepare human evaluation data
        eval_data = {
            "evaluation_instructions": {
                "purpose": "Human evaluation of abnormally short texts in PADBen dataset",
                "threshold": self.threshold,
                "task": "Review each text and categorize the issue type",
                "categories": [
                    "truncation_error",
                    "preprocessing_issue", 
                    "valid_short_text",
                    "empty_or_noise",
                    "encoding_problem",
                    "other"
                ]
            },
            "summary": {
                "total_problematic_records": len(self.short_texts_analysis.get("problematic_records", [])),
                "examples_included": min(max_examples, len(self.short_texts_analysis.get("problematic_records", []))),
                "threshold_used": self.threshold
            },
            "examples": []
        }
        
        # Add examples for human evaluation
        problematic_records = self.short_texts_analysis.get("problematic_records", [])[:max_examples]
        
        for i, record in enumerate(problematic_records):
            eval_example = {
                "example_id": i + 1,
                "record_idx": record["record_idx"],
                "dataset_source": record["dataset_source"],
                "issue_type": record["issue_type"],
                "texts": record["text_details"],
                "evaluation": {
                    "category": "",  # To be filled by human evaluator
                    "notes": "",     # To be filled by human evaluator
                    "action_needed": ""  # To be filled by human evaluator
                }
            }
            eval_data["examples"].append(eval_example)
        
        # Save evaluation file
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Human evaluation file saved to: {output_path}")
        logger.info(f"Included {len(eval_data['examples'])} examples for evaluation")
        
        return output_path
    
    def generate_summary_report(self) -> str:
        """
        Generate a human-readable summary report.
        
        Returns:
            Summary report as string.
        """
        if not self.short_texts_analysis:
            return "No analysis results available. Run analyze_short_texts() first."
        
        analysis = self.short_texts_analysis
        
        report = []
        report.append("=" * 80)
        report.append("PADBen Short Text Analysis Report")
        report.append("=" * 80)
        report.append(f"Analysis Date: {analysis['analysis_metadata']['timestamp']}")
        report.append(f"Threshold: {analysis['analysis_metadata']['threshold']} tokens")
        report.append(f"Total Records: {analysis['analysis_metadata']['total_records']:,}")
        report.append("")
        
        # Summary by text type
        report.append("SHORT TEXT STATISTICS BY TYPE:")
        report.append("-" * 40)
        
        for text_type, stats in analysis["short_text_statistics"].items():
            report.append(f"{text_type.upper()}:")
            report.append(f"  Total texts: {stats['total_texts']:,}")
            report.append(f"  Short texts: {stats['short_texts_count']:,} ({stats['short_texts_percentage']:.2f}%)")
            report.append(f"  Length range: {stats['min_length']}-{stats['max_length']} tokens")
            report.append(f"  Average length: {stats['average_length']:.1f} tokens")
            report.append("")
        
        # Problematic records
        report.append("PROBLEMATIC RECORDS:")
        report.append("-" * 20)
        report.append(f"Records with multiple short texts: {len(analysis['problematic_records'])}")
        report.append("")
        
        # Patterns
        patterns = analysis["pattern_analysis"]
        report.append("COMMON PATTERNS:")
        report.append("-" * 15)
        
        if patterns["single_word_texts"]:
            report.append("Most common single words:")
            for word, count in list(patterns["single_word_texts"].items())[:5]:
                report.append(f"  '{word}': {count} times")
            report.append("")
        
        if patterns["dataset_source_distribution"]:
            report.append("Short texts by dataset source:")
            for source, count in patterns["dataset_source_distribution"].items():
                report.append(f"  {source}: {count} short texts")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 15)
        for i, rec in enumerate(analysis["recommendations"], 1):
            report.append(f"{i}. {rec}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Main function to run short text analysis."""
    analyzer = ShortTextAnalyzer(threshold=10)
    
    # Run analysis
    results = analyzer.analyze_short_texts()
    
    # Save results
    analysis_file = analyzer.save_analysis_results()
    eval_file = analyzer.export_for_human_evaluation()
    
    # Print summary
    print(analyzer.generate_summary_report())
    
    print(f"\nFiles generated:")
    print(f"- Analysis results: {analysis_file}")
    print(f"- Human evaluation: {eval_file}")


if __name__ == "__main__":
    main()
