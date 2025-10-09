"""Comparison utilities for benchmarking against RAID dataset."""

import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np

try:
    from .config import RAID_METRICS
except ImportError:
    from config import RAID_METRICS

logger = logging.getLogger(__name__)


class RAIDComparator:
    """Compares PADBen metrics with RAID benchmark."""
    
    def __init__(self) -> None:
        """Initialize RAID comparator with benchmark metrics."""
        self.raid_metrics = RAID_METRICS
        
    def compare_self_bleu(self, padben_self_bleu: Dict[str, float]) -> Dict[str, Any]:
        """
        Compare PADBen self-BLEU scores with RAID benchmark.
        
        Args:
            padben_self_bleu: Dictionary mapping text types to self-BLEU scores.
            
        Returns:
            Dictionary containing comparison results.
        """
        raid_self_bleu = self.raid_metrics["self_bleu"]
        
        # Calculate overall PADBen self-BLEU (average across all types)
        padben_scores = list(padben_self_bleu.values())
        padben_avg_self_bleu = np.mean(padben_scores) if padben_scores else 0.0
        
        comparison = {
            "padben_self_bleu_by_type": padben_self_bleu,
            "padben_avg_self_bleu": padben_avg_self_bleu,
            "raid_self_bleu": raid_self_bleu,
            "difference": padben_avg_self_bleu - raid_self_bleu,
            "relative_improvement": ((padben_avg_self_bleu - raid_self_bleu) / raid_self_bleu * 100) if raid_self_bleu > 0 else 0,
            "interpretation": self._interpret_self_bleu_comparison(padben_avg_self_bleu, raid_self_bleu)
        }
        
        logger.info(f"PADBen avg self-BLEU: {padben_avg_self_bleu:.3f}, RAID: {raid_self_bleu:.3f}")
        return comparison
    
    def compare_perplexity(self, padben_perplexity: Dict[str, float]) -> Dict[str, Any]:
        """
        Compare PADBen perplexity scores with RAID benchmark.
        
        Args:
            padben_perplexity: Dictionary mapping text types to perplexity scores.
            
        Returns:
            Dictionary containing comparison results.
        """
        raid_ppl_g2x = self.raid_metrics["perplexity_g2x"]
        
        # Calculate overall PADBen perplexity (average across all types)
        padben_scores = [score for score in padben_perplexity.values() 
                        if not np.isnan(score) and not np.isinf(score)]
        padben_avg_perplexity = np.mean(padben_scores) if padben_scores else float('inf')
        
        comparison = {
            "padben_perplexity_by_type": padben_perplexity,
            "padben_avg_perplexity": padben_avg_perplexity,
            "raid_perplexity_g2x": raid_ppl_g2x,
            "difference_vs_g2x": padben_avg_perplexity - raid_ppl_g2x,
            "relative_change_vs_g2x": ((padben_avg_perplexity - raid_ppl_g2x) / raid_ppl_g2x * 100) if raid_ppl_g2x > 0 else 0,
            "interpretation": self._interpret_perplexity_comparison(padben_avg_perplexity, raid_ppl_g2x)
        }
        
        logger.info(f"PADBen avg perplexity: {padben_avg_perplexity:.3f}, RAID G2X: {raid_ppl_g2x:.3f}")
        return comparison
    
    def create_comparison_table(self, padben_metrics: Dict[str, Any]) -> pd.DataFrame:
        """
        Create a comprehensive comparison table between PADBen and RAID.
        
        Args:
            padben_metrics: PADBen metrics dictionary.
            
        Returns:
            DataFrame containing side-by-side comparison.
        """
        # Extract PADBen metrics
        padben_self_bleu = padben_metrics.get("self_bleu_scores", {})
        padben_perplexity = padben_metrics.get("perplexity_scores", {})
        
        # Calculate averages
        padben_avg_self_bleu = np.mean(list(padben_self_bleu.values())) if padben_self_bleu else 0.0
        padben_scores = [score for score in padben_perplexity.values() 
                        if not np.isnan(score) and not np.isinf(score)]
        padben_avg_perplexity = np.mean(padben_scores) if padben_scores else float('inf')
        
        # Create comparison data (only G2X for perplexity since PADBen uses GPT-2)
        comparison_data = {
            "Metric": ["Self-BLEU", "Perplexity (vs G2X)"],
            "PADBen": [
                f"{padben_avg_self_bleu:.3f}",
                f"{padben_avg_perplexity:.3f}"
            ],
            "RAID": [
                f"{self.raid_metrics['self_bleu']:.3f}",
                f"{self.raid_metrics['perplexity_g2x']:.3f}"
            ],
            "Difference": [
                f"{padben_avg_self_bleu - self.raid_metrics['self_bleu']:+.3f}",
                f"{padben_avg_perplexity - self.raid_metrics['perplexity_g2x']:+.3f}"
            ],
            "Relative Change (%)": [
                f"{((padben_avg_self_bleu - self.raid_metrics['self_bleu']) / self.raid_metrics['self_bleu'] * 100):+.2f}",
                f"{((padben_avg_perplexity - self.raid_metrics['perplexity_g2x']) / self.raid_metrics['perplexity_g2x'] * 100):+.2f}"
            ]
        }
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def create_detailed_metrics_table(self, padben_metrics: Dict[str, Any]) -> pd.DataFrame:
        """
        Create a detailed metrics table for PADBen by text type.
        
        Args:
            padben_metrics: PADBen metrics dictionary.
            
        Returns:
            DataFrame with metrics broken down by text type.
        """
        padben_self_bleu = padben_metrics.get("self_bleu_scores", {})
        padben_perplexity = padben_metrics.get("perplexity_scores", {})
        padben_stats = padben_metrics.get("dataset_statistics", {})
        
        # Get all text types
        text_types = set(padben_self_bleu.keys()) | set(padben_perplexity.keys()) | set(padben_stats.keys())
        
        table_data = []
        for text_type in sorted(text_types):
            row = {
                "Text Type": text_type,
                "Count": padben_stats.get(text_type, {}).get("count", 0),
                "Avg Length (tokens)": f"{padben_stats.get(text_type, {}).get('avg_length', 0):.1f}",
                "Self-BLEU": f"{padben_self_bleu.get(text_type, 0):.3f}",
                "Perplexity": f"{padben_perplexity.get(text_type, float('inf')):.3f}"
            }
            table_data.append(row)
        
        # Add overall averages
        avg_self_bleu = np.mean(list(padben_self_bleu.values())) if padben_self_bleu else 0.0
        valid_perplexities = [score for score in padben_perplexity.values() 
                             if not np.isnan(score) and not np.isinf(score)]
        avg_perplexity = np.mean(valid_perplexities) if valid_perplexities else float('inf')
        total_count = sum(padben_stats.get(t, {}).get("count", 0) for t in padben_stats.keys())
        avg_length = np.mean([padben_stats.get(t, {}).get("avg_length", 0) for t in padben_stats.keys()])
        
        table_data.append({
            "Text Type": "OVERALL AVERAGE",
            "Count": total_count,
            "Avg Length (tokens)": f"{avg_length:.1f}",
            "Self-BLEU": f"{avg_self_bleu:.3f}",
            "Perplexity": f"{avg_perplexity:.3f}"
        })
        
        return pd.DataFrame(table_data)
    
    def _interpret_self_bleu_comparison(self, padben_score: float, raid_score: float) -> str:
        """
        Provide interpretation of self-BLEU comparison.
        
        Args:
            padben_score: PADBen self-BLEU score.
            raid_score: RAID self-BLEU score.
            
        Returns:
            Interpretation string.
        """
        difference = padben_score - raid_score
        
        if abs(difference) < 1.0:
            return "Similar diversity levels"
        elif difference > 1.0:
            return "Higher self-BLEU indicates lower diversity (more repetitive)"
        else:
            return "Lower self-BLEU indicates higher diversity (less repetitive)"
    
    def _interpret_perplexity_comparison(self, padben_score: float, raid_g2x: float) -> str:
        """
        Provide interpretation of perplexity comparison.
        
        Args:
            padben_score: PADBen perplexity score.
            raid_g2x: RAID G2X perplexity score.
            
        Returns:
            Interpretation string.
        """
        if np.isinf(padben_score):
            return "Invalid perplexity score"
            
        # Compare with RAID G2X score (similar model to PADBen's GPT-2)
        diff_g2x = padben_score - raid_g2x
        
        if abs(diff_g2x) < 1.0:
            return "Similar fluency to RAID G2X"
        elif diff_g2x > 1.0:
            return "Lower fluency than RAID G2X (higher perplexity)"
        else:
            return "Higher fluency than RAID G2X (lower perplexity)"
    
    def generate_comparison_summary(self, padben_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive comparison summary.
        
        Args:
            padben_metrics: PADBen metrics dictionary.
            
        Returns:
            Dictionary containing comparison summary.
        """
        self_bleu_comparison = self.compare_self_bleu(padben_metrics.get("self_bleu_scores", {}))
        perplexity_comparison = self.compare_perplexity(padben_metrics.get("perplexity_scores", {}))
        
        summary = {
            "self_bleu_comparison": self_bleu_comparison,
            "perplexity_comparison": perplexity_comparison,
            "overall_assessment": self._generate_overall_assessment(self_bleu_comparison, perplexity_comparison),
            "raid_reference": self.raid_metrics
        }
        
        return summary
    
    def _generate_overall_assessment(self, self_bleu_comp: Dict[str, Any], 
                                   perplexity_comp: Dict[str, Any]) -> str:
        """
        Generate overall quality assessment compared to RAID.
        
        Args:
            self_bleu_comp: Self-BLEU comparison results.
            perplexity_comp: Perplexity comparison results.
            
        Returns:
            Overall assessment string.
        """
        assessments = []
        
        # Self-BLEU assessment
        bleu_diff = self_bleu_comp["difference"]
        if abs(bleu_diff) < 1.0:
            assessments.append("PADBen shows similar text diversity to RAID")
        elif bleu_diff > 1.0:
            assessments.append("PADBen shows lower text diversity than RAID")
        else:
            assessments.append("PADBen shows higher text diversity than RAID")
            
        # Perplexity assessment
        ppl_padben = perplexity_comp["padben_avg_perplexity"]
        if not np.isinf(ppl_padben):
            ppl_g2x = perplexity_comp["raid_perplexity_g2x"]
            if abs(ppl_padben - ppl_g2x) < 2.0:
                assessments.append("PADBen shows similar text fluency to RAID")
            elif ppl_padben > ppl_g2x + 2.0:
                assessments.append("PADBen shows lower text fluency than RAID")
            else:
                assessments.append("PADBen shows higher text fluency than RAID")
        else:
            assessments.append("PADBen perplexity could not be reliably calculated")
            
        return ". ".join(assessments) + "."
