"""Main module for comprehensive data quality examination of PADBen dataset."""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

try:
    from .config import OUTPUT_DIR, OUTPUT_FILES, LOGGING_CONFIG
    from .data_loader import DataLoader
    from .metrics import MetricsAggregator
    from .comparison import RAIDComparator
    from .visualization import SimilarityVisualizer
except ImportError:
    from config import OUTPUT_DIR, OUTPUT_FILES, LOGGING_CONFIG
    from data_loader import DataLoader
    from metrics import MetricsAggregator
    from comparison import RAIDComparator
    from visualization import SimilarityVisualizer

# Configure logging
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def json_serializer(obj):
    """Custom JSON serializer for numpy types and other non-serializable objects."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    else:
        return str(obj)


class DataQualityExaminer:
    """
    Comprehensive data quality examination for PADBen dataset.
    
    This class orchestrates all quality assessment tasks including:
    - Loading and preprocessing data
    - Calculating similarity, self-BLEU, and perplexity metrics
    - Comparing with RAID benchmark
    - Generating visualizations and reports
    """
    
    def __init__(self, data_path: Optional[Path] = None, 
                 output_dir: Optional[Path] = None,
                 perplexity_model: str = "gpt2-xl") -> None:
        """
        Initialize data quality examiner.
        
        Args:
            data_path: Path to the JSON data file.
            output_dir: Directory for output files.
            perplexity_model: Perplexity model to use ('gpt2-xl' or 'llama3-7b-4bit').
        """
        self.perplexity_model = perplexity_model
        self.data_loader = DataLoader(data_path)
        self.metrics_aggregator = MetricsAggregator(perplexity_model=perplexity_model)
        self.raid_comparator = RAIDComparator()
        self.visualizer = SimilarityVisualizer(output_dir)
        
        self.output_dir = output_dir or OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.padben_metrics: Dict[str, Any] = {}
        self.comparison_results: Dict[str, Any] = {}
        
    def run_complete_examination(self, sample_size: Optional[int] = 1000,
                               generate_visualizations: bool = True) -> Dict[str, Any]:
        """
        Run complete data quality examination.
        
        Args:
            sample_size: Number of texts to sample for computationally expensive metrics.
            generate_visualizations: Whether to generate visualization plots.
            
        Returns:
            Dictionary containing all examination results.
        """
        logger.info("Starting comprehensive data quality examination...")
        start_time = datetime.now()
        
        try:
            # Step 1: Load and validate data
            logger.info("Step 1: Loading and validating data...")
            self._load_and_validate_data()
            
            # Step 2a: Calculate Jaccard similarity matrix
            logger.info("Step 2a: Calculating Jaccard similarity matrix...")
            self._calculate_jaccard_similarity()
            self._save_jaccard_results()
            
            # Step 2b: Calculate self-BLEU scores
            logger.info("Step 2b: Calculating self-BLEU scores...")
            self._calculate_self_bleu(sample_size)
            self._save_self_bleu_results()
            
            # Step 2c: Calculate perplexity scores (GPT-2 XL)
            logger.info("Step 2c: Calculating perplexity scores with GPT-2 XL...")
            self._calculate_perplexity(sample_size)
            self._save_perplexity_results()
            
            # Step 3: Compare with RAID benchmark
            logger.info("Step 3: Comparing with RAID benchmark...")
            self._compare_with_raid()
            
            # Step 4: Generate tables and reports
            logger.info("Step 4: Generating tables and reports...")
            self._generate_tables_and_reports()
            
            # Step 5: Create visualizations
            if generate_visualizations:
                logger.info("Step 5: Creating visualizations...")
                self._create_visualizations()
            
            # Step 6: Generate final summary
            logger.info("Step 6: Generating final summary...")
            final_results = self._generate_final_summary()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Complete examination finished in {duration:.2f} seconds")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error during examination: {e}")
            raise
    
    def _load_and_validate_data(self) -> None:
        """Load data and perform basic validation."""
        # Load data
        data = self.data_loader.load_data()
        logger.info(f"Loaded {len(data)} records")
        
        # Get dataset statistics
        stats = self.data_loader.get_dataset_statistics()
        logger.info(f"Dataset sources: {stats['dataset_sources']}")
        logger.info(f"Text type counts: {stats['text_type_counts']}")
        
        # Check data completeness
        completeness = self.data_loader.validate_data_completeness()
        logger.info("Data completeness by type:")
        for text_type, percentage in completeness.items():
            logger.info(f"  {text_type}: {percentage:.1f}%")
            
        # Store basic stats
        self.dataset_stats = stats
        self.data_completeness = completeness
    
    def _calculate_jaccard_similarity(self) -> None:
        """Calculate Jaccard similarity matrix."""
        # Extract all texts by type
        all_texts = self.data_loader.extract_all_texts()
        logger.info(f"Extracted texts for {len(all_texts)} types")
        
        # Initialize padben_metrics if not exists
        if not hasattr(self, 'padben_metrics'):
            self.padben_metrics = {
                "jaccard_similarity_matrix": {},
                "self_bleu_scores": {},
                "perplexity_scores": {},
                "dataset_statistics": {}
            }
        
        # Calculate Jaccard similarity matrix
        logger.info("Calculating Jaccard similarity matrix...")
        self.padben_metrics["jaccard_similarity_matrix"] = self.metrics_aggregator.jaccard_calc.calculate_similarity_matrix(all_texts)
        
        logger.info("Jaccard similarity calculation completed")
    
    def _calculate_self_bleu(self, sample_size: Optional[int]) -> None:
        """Calculate self-BLEU scores."""
        # Extract all texts by type
        all_texts = self.data_loader.extract_all_texts()
        
        # Calculate self-BLEU scores
        logger.info("Calculating self-BLEU scores...")
        self.padben_metrics["self_bleu_scores"] = self.metrics_aggregator.bleu_calc.calculate_self_bleu_by_type(all_texts, sample_size)
        
        logger.info("Self-BLEU calculation completed")
    
    def _calculate_perplexity(self, sample_size: Optional[int]) -> None:
        """Calculate perplexity scores using GPT-2 XL."""
        # Extract all texts by type
        all_texts = self.data_loader.extract_all_texts()
        
        # Calculate perplexity scores
        logger.info("Calculating perplexity scores with GPT-2 XL...")
        self.padben_metrics["perplexity_scores"] = self.metrics_aggregator.perplexity_calc.calculate_perplexity_by_type(all_texts, sample_size)
        
        # Calculate dataset statistics
        self.padben_metrics["dataset_statistics"] = self.metrics_aggregator._calculate_dataset_stats(all_texts)
        
        logger.info("Perplexity calculation completed")
    
    def _save_jaccard_results(self) -> None:
        """Save Jaccard similarity results to file."""
        try:
            # Save similarity matrix CSV
            similarity_matrix = self.padben_metrics.get("jaccard_similarity_matrix", {})
            if similarity_matrix:
                similarity_df = self._create_similarity_matrix_dataframe(similarity_matrix)
                similarity_path = self.output_dir / "jaccard_similarity_matrix.csv"
                similarity_df.to_csv(similarity_path, index=True)
                logger.info(f"[OK] Jaccard similarity matrix saved to {similarity_path}")
                
                # Save as JSON for intermediate storage
                json_compatible_similarity = {
                    f"{key[0]}_vs_{key[1]}": value 
                    for key, value in similarity_matrix.items()
                }
                
                intermediate_results = {
                    "timestamp": datetime.now().isoformat(),
                    "step": "jaccard_similarity",
                    "jaccard_similarity_matrix": json_compatible_similarity,
                    "dataset_statistics": self.dataset_stats
                }
                
                json_path = self.output_dir / "intermediate_jaccard_results.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(intermediate_results, f, indent=2, ensure_ascii=False, default=json_serializer)
                logger.info(f"[OK] Intermediate Jaccard results saved to {json_path}")
                
        except Exception as e:
            logger.error(f"Error saving Jaccard results: {e}")
    
    def _save_self_bleu_results(self) -> None:
        """Save self-BLEU results to file."""
        try:
            # Save self-BLEU scores
            self_bleu_scores = self.padben_metrics.get("self_bleu_scores", {})
            if self_bleu_scores:
                # Create a simple table
                bleu_data = []
                for text_type, score in self_bleu_scores.items():
                    bleu_data.append({
                        "Text_Type": text_type,
                        "Self_BLEU_Score": f"{score:.4f}"
                    })
                
                bleu_df = pd.DataFrame(bleu_data)
                bleu_path = self.output_dir / "self_bleu_scores.csv"
                bleu_df.to_csv(bleu_path, index=False)
                logger.info(f"[OK] Self-BLEU scores saved to {bleu_path}")
                
                # Save as JSON for intermediate storage
                intermediate_results = {
                    "timestamp": datetime.now().isoformat(),
                    "step": "self_bleu",
                    "self_bleu_scores": self_bleu_scores,
                    "average_self_bleu": np.mean(list(self_bleu_scores.values())) if self_bleu_scores else 0.0
                }
                
                json_path = self.output_dir / "intermediate_self_bleu_results.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(intermediate_results, f, indent=2, ensure_ascii=False, default=json_serializer)
                logger.info(f"[OK] Intermediate self-BLEU results saved to {json_path}")
                
        except Exception as e:
            logger.error(f"Error saving self-BLEU results: {e}")
    
    def _save_perplexity_results(self) -> None:
        """Save perplexity results to file."""
        try:
            # Save perplexity scores
            perplexity_scores = self.padben_metrics.get("perplexity_scores", {})
            if perplexity_scores:
                # Create a simple table
                ppl_data = []
                for text_type, score in perplexity_scores.items():
                    if not np.isnan(score) and not np.isinf(score):
                        ppl_data.append({
                            "Text_Type": text_type,
                            "Perplexity_Score": f"{score:.4f}",
                            "Model": "GPT-2 XL"
                        })
                    else:
                        ppl_data.append({
                            "Text_Type": text_type,
                            "Perplexity_Score": "Invalid",
                            "Model": "GPT-2 XL"
                        })
                
                ppl_df = pd.DataFrame(ppl_data)
                ppl_path = self.output_dir / "perplexity_scores_gpt2xl.csv"
                ppl_df.to_csv(ppl_path, index=False)
                logger.info(f"[OK] Perplexity scores saved to {ppl_path}")
                
                # Calculate valid average
                valid_scores = [score for score in perplexity_scores.values() 
                              if not np.isnan(score) and not np.isinf(score)]
                avg_perplexity = np.mean(valid_scores) if valid_scores else float('inf')
                
                # Save as JSON for intermediate storage
                intermediate_results = {
                    "timestamp": datetime.now().isoformat(),
                    "step": "perplexity",
                    "model": "gpt2-xl",
                    "perplexity_scores": perplexity_scores,
                    "average_perplexity": avg_perplexity,
                    "valid_scores_count": len(valid_scores),
                    "total_scores_count": len(perplexity_scores)
                }
                
                json_path = self.output_dir / "intermediate_perplexity_results.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(intermediate_results, f, indent=2, ensure_ascii=False, default=json_serializer)
                logger.info(f"[OK] Intermediate perplexity results saved to {json_path}")
                
        except Exception as e:
            logger.error(f"Error saving perplexity results: {e}")
    
    def _compare_with_raid(self) -> None:
        """Compare PADBen metrics with RAID benchmark."""
        self.comparison_results = self.raid_comparator.generate_comparison_summary(
            self.padben_metrics
        )
        
        logger.info("RAID comparison completed")
        logger.info(f"Overall assessment: {self.comparison_results['overall_assessment']}")
    
    def _generate_tables_and_reports(self) -> None:
        """Generate CSV tables and detailed reports."""
        # 1. Generate similarity matrix table
        similarity_matrix = self.padben_metrics.get("jaccard_similarity_matrix", {})
        if similarity_matrix:
            similarity_df = self._create_similarity_matrix_dataframe(similarity_matrix)
            similarity_path = self.output_dir / OUTPUT_FILES["similarity_matrix"]
            similarity_df.to_csv(similarity_path, index=True)
            logger.info(f"Similarity matrix saved to {similarity_path}")
        
        # 2. Generate PADBen metrics table
        metrics_df = self.raid_comparator.create_detailed_metrics_table(self.padben_metrics)
        metrics_path = self.output_dir / OUTPUT_FILES["metrics_table"]
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"PADBen metrics table saved to {metrics_path}")
        
        # 3. Generate comparison table with RAID
        comparison_df = self.raid_comparator.create_comparison_table(self.padben_metrics)
        comparison_path = self.output_dir / OUTPUT_FILES["comparison_table"]
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"RAID comparison table saved to {comparison_path}")
        
        # 4. Generate detailed JSON report
        # Convert similarity matrix tuple keys to strings for JSON serialization
        similarity_matrix = self.padben_metrics.get("jaccard_similarity_matrix", {})
        json_compatible_similarity = {
            f"{key[0]}_vs_{key[1]}": value 
            for key, value in similarity_matrix.items()
        }
        
        # Create a JSON-compatible copy of padben_metrics
        json_padben_metrics = self.padben_metrics.copy()
        json_padben_metrics["jaccard_similarity_matrix"] = json_compatible_similarity
        
        detailed_report = {
            "examination_metadata": {
                "timestamp": datetime.now().isoformat(),
                "dataset_statistics": self.dataset_stats,
                "data_completeness": self.data_completeness
            },
            "padben_metrics": json_padben_metrics,
            "raid_comparison": self.comparison_results,
            "summary": {
                "total_records": self.dataset_stats.get("total_records", 0),
                "text_types_analyzed": len(self.padben_metrics.get("self_bleu_scores", {})),
                "overall_assessment": self.comparison_results.get("overall_assessment", ""),
            }
        }
        
        report_path = self.output_dir / OUTPUT_FILES["detailed_report"]
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_report, f, indent=2, ensure_ascii=False, default=json_serializer)
        logger.info(f"Detailed report saved to {report_path}")
    
    def _create_visualizations(self) -> None:
        """Create all visualizations."""
        similarity_matrix = self.padben_metrics.get("jaccard_similarity_matrix", {})
        
        # 1. Similarity heatmap
        if similarity_matrix:
            heatmap_path = self.output_dir / OUTPUT_FILES["similarity_heatmap"]
            self.visualizer.create_similarity_heatmap(similarity_matrix, save_path=heatmap_path)
        
        # 2. Metrics comparison chart
        metrics_comparison_path = self.output_dir / "metrics_comparison.png"
        self.visualizer.create_metrics_comparison_chart(
            self.padben_metrics, 
            self.comparison_results["raid_reference"],
            save_path=metrics_comparison_path
        )
        
        # 3. Text type breakdown chart
        text_type_path = self.output_dir / "text_type_metrics.png"
        self.visualizer.create_text_type_metrics_chart(
            self.padben_metrics,
            save_path=text_type_path
        )
        
        # 4. Comprehensive dashboard
        dashboard_path = self.output_dir / "quality_dashboard.png"
        self.visualizer.create_comprehensive_dashboard(
            similarity_matrix,
            self.padben_metrics,
            self.comparison_results["raid_reference"],
            save_path=dashboard_path
        )
        
        logger.info("All visualizations created")
    
    def _create_similarity_matrix_dataframe(self, similarity_matrix: Dict) -> pd.DataFrame:
        """
        Convert similarity matrix to DataFrame format.
        
        Args:
            similarity_matrix: Dictionary mapping (type1, type2) tuples to scores.
            
        Returns:
            DataFrame with similarity matrix.
        """
        # Get unique text types
        text_types = list(set([t for pair in similarity_matrix.keys() for t in pair]))
        text_types.sort()
        
        # Create matrix data
        matrix_data = {}
        for type1 in text_types:
            row = {}
            for type2 in text_types:
                if (type1, type2) in similarity_matrix:
                    row[type2] = similarity_matrix[(type1, type2)]
                elif (type2, type1) in similarity_matrix:
                    row[type2] = similarity_matrix[(type2, type1)]
                else:
                    row[type2] = 0.0
            matrix_data[type1] = row
        
        return pd.DataFrame(matrix_data).T  # Transpose for proper orientation
    
    def _generate_final_summary(self) -> Dict[str, Any]:
        """
        Generate final summary of all results.
        
        Returns:
            Dictionary containing comprehensive summary.
        """
        summary = {
            "examination_summary": {
                "timestamp": datetime.now().isoformat(),
                "dataset_size": self.dataset_stats.get("total_records", 0),
                "text_types_analyzed": len(self.padben_metrics.get("self_bleu_scores", {})),
                "metrics_calculated": ["jaccard_similarity", "self_bleu", "perplexity"],
                "comparison_baseline": "RAID"
            },
            "key_findings": {
                "average_self_bleu": self._get_average_metric("self_bleu_scores"),
                "average_perplexity": self._get_average_metric("perplexity_scores"),
                "similarity_range": self._get_similarity_range(),
                "raid_comparison": self.comparison_results.get("overall_assessment", "")
            },
            "data_quality_assessment": self._assess_overall_quality(),
            "output_files": {
                "similarity_matrix": str(self.output_dir / OUTPUT_FILES["similarity_matrix"]),
                "metrics_table": str(self.output_dir / OUTPUT_FILES["metrics_table"]),
                "comparison_table": str(self.output_dir / OUTPUT_FILES["comparison_table"]),
                "detailed_report": str(self.output_dir / OUTPUT_FILES["detailed_report"]),
                "similarity_heatmap": str(self.output_dir / OUTPUT_FILES["similarity_heatmap"])
            }
        }
        
        # Print summary to console
        self._print_summary(summary)
        
        return summary
    
    def _get_average_metric(self, metric_key: str) -> float:
        """Get average value for a metric across all text types."""
        metric_dict = self.padben_metrics.get(metric_key, {})
        if not metric_dict:
            return 0.0
            
        if metric_key == "perplexity_scores":
            # Filter out invalid perplexity scores
            valid_scores = [score for score in metric_dict.values() 
                          if not pd.isna(score) and not np.isinf(score)]
            return float(pd.Series(valid_scores).mean()) if valid_scores else 0.0
        else:
            return float(pd.Series(list(metric_dict.values())).mean())
    
    def _get_similarity_range(self) -> Dict[str, float]:
        """Get range of similarity scores."""
        similarity_matrix = self.padben_metrics.get("jaccard_similarity_matrix", {})
        if not similarity_matrix:
            return {"min": 0.0, "max": 0.0}
            
        scores = list(similarity_matrix.values())
        return {
            "min": float(min(scores)),
            "max": float(max(scores))
        }
    
    def _assess_overall_quality(self) -> str:
        """Provide overall quality assessment."""
        assessments = []
        
        # Check data completeness
        avg_completeness = sum(self.data_completeness.values()) / len(self.data_completeness)
        if avg_completeness > 95:
            assessments.append("Excellent data completeness")
        elif avg_completeness > 85:
            assessments.append("Good data completeness")
        else:
            assessments.append("Data completeness could be improved")
            
        # Check diversity (self-BLEU)
        avg_self_bleu = self._get_average_metric("self_bleu_scores")
        if avg_self_bleu < 10:
            assessments.append("High text diversity")
        elif avg_self_bleu < 15:
            assessments.append("Moderate text diversity")
        else:
            assessments.append("Low text diversity")
            
        # Check fluency (perplexity)
        avg_perplexity = self._get_average_metric("perplexity_scores")
        if avg_perplexity < 10:
            assessments.append("Good text fluency")
        elif avg_perplexity < 20:
            assessments.append("Moderate text fluency")
        else:
            assessments.append("Text fluency could be improved")
            
        return ". ".join(assessments) + "."
    
    def _print_summary(self, summary: Dict[str, Any]) -> None:
        """Print formatted summary to console."""
        print("\n" + "="*80)
        print("PADBen DATA QUALITY EXAMINATION SUMMARY")
        print("="*80)
        
        print(f"\nDataset Size: {summary['examination_summary']['dataset_size']:,} records")
        print(f"Text Types Analyzed: {summary['examination_summary']['text_types_analyzed']}")
        print(f"Timestamp: {summary['examination_summary']['timestamp']}")
        
        print("\nKEY FINDINGS:")
        print(f"  • Average Self-BLEU: {summary['key_findings']['average_self_bleu']:.3f}")
        print(f"  • Average Perplexity: {summary['key_findings']['average_perplexity']:.3f}")
        print(f"  • Similarity Range: {summary['key_findings']['similarity_range']['min']:.3f} - {summary['key_findings']['similarity_range']['max']:.3f}")
        
        print(f"\nRAID COMPARISON:")
        print(f"  {summary['key_findings']['raid_comparison']}")
        
        print(f"\nOVERALL ASSESSMENT:")
        print(f"  {summary['data_quality_assessment']}")
        
        print(f"\nOUTPUT FILES:")
        for file_type, path in summary['output_files'].items():
            print(f"  • {file_type.replace('_', ' ').title()}: {path}")
        
        print("\n" + "="*80)


def main() -> None:
    """Main function to run data quality examination."""
    examiner = DataQualityExaminer()
    results = examiner.run_complete_examination(
        sample_size=1000,  # Adjust based on computational resources
        generate_visualizations=True
    )
    
    print("\nData quality examination completed successfully!")
    print(f"Results saved to: {examiner.output_dir}")


if __name__ == "__main__":
    main()
