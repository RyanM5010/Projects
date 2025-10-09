"""Visualization utilities for data quality examination results."""

import logging
from typing import Dict, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from .config import VISUALIZATION_CONFIG, OUTPUT_DIR
except ImportError:
    from config import VISUALIZATION_CONFIG, OUTPUT_DIR

logger = logging.getLogger(__name__)

# Set style
plt.style.use('default')
sns.set_palette("husl")


class SimilarityVisualizer:
    """Creates visualizations for similarity matrices and quality metrics."""
    
    def __init__(self, output_dir: Optional[Path] = None) -> None:
        """
        Initialize similarity visualizer.
        
        Args:
            output_dir: Directory to save visualizations. Defaults to config OUTPUT_DIR.
        """
        self.output_dir = output_dir or OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = VISUALIZATION_CONFIG
        
    def create_similarity_heatmap(self, similarity_matrix: Dict[Tuple[str, str], float],
                                text_types: Optional[list] = None,
                                save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create a heatmap visualization of the similarity matrix.
        
        Args:
            similarity_matrix: Dictionary mapping (type1, type2) tuples to similarity scores.
            text_types: List of text types to include. If None, infer from matrix keys.
            save_path: Path to save the figure. If None, use default path.
            
        Returns:
            Matplotlib figure object.
        """
        # Extract text types if not provided
        if text_types is None:
            text_types = list(set([t for pair in similarity_matrix.keys() for t in pair]))
            text_types.sort()
            
        # Create matrix array
        n_types = len(text_types)
        matrix_array = np.zeros((n_types, n_types))
        
        for i, type1 in enumerate(text_types):
            for j, type2 in enumerate(text_types):
                if (type1, type2) in similarity_matrix:
                    matrix_array[i, j] = similarity_matrix[(type1, type2)]
                elif (type2, type1) in similarity_matrix:
                    matrix_array[i, j] = similarity_matrix[(type2, type1)]
                else:
                    matrix_array[i, j] = 0.0
                    
        # Create heatmap
        fig, ax = plt.subplots(figsize=self.config["figsize"])
        
        # Clean up text type labels for display
        display_labels = [self._clean_label(t) for t in text_types]
        
        heatmap = sns.heatmap(
            matrix_array,
            annot=self.config["annot"],
            fmt=self.config["fmt"],
            cmap=self.config["cmap"],
            xticklabels=display_labels,
            yticklabels=display_labels,
            cbar_kws=self.config["cbar_kws"],
            ax=ax,
            square=True
        )
        
        # Customize plot
        ax.set_title("Jaccard Similarity Matrix Between Text Types", 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Text Type", fontsize=12, fontweight='bold')
        ax.set_ylabel("Text Type", fontsize=12, fontweight='bold')
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Similarity heatmap saved to {save_path}")
        else:
            default_path = self.output_dir / "similarity_heatmap.png"
            fig.savefig(default_path, dpi=300, bbox_inches='tight')
            logger.info(f"Similarity heatmap saved to {default_path}")
            
        return fig
    
    def create_metrics_comparison_chart(self, padben_metrics: Dict[str, Any],
                                      raid_metrics: Dict[str, Any],
                                      save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create a comparison chart between PADBen and RAID metrics.
        
        Args:
            padben_metrics: PADBen metrics dictionary.
            raid_metrics: RAID metrics dictionary.
            save_path: Path to save the figure.
            
        Returns:
            Matplotlib figure object.
        """
        # Extract self-BLEU scores
        padben_self_bleu = list(padben_metrics.get("self_bleu_scores", {}).values())
        padben_avg_self_bleu = np.mean(padben_self_bleu) if padben_self_bleu else 0.0
        raid_self_bleu = raid_metrics.get("self_bleu", 0.0)
        
        # Extract perplexity scores
        padben_perplexity = [score for score in padben_metrics.get("perplexity_scores", {}).values()
                           if not np.isnan(score) and not np.isinf(score)]
        padben_avg_perplexity = np.mean(padben_perplexity) if padben_perplexity else 0.0
        raid_perplexity_l7b = raid_metrics.get("perplexity_l7b", 0.0)
        raid_perplexity_g2x = raid_metrics.get("perplexity_g2x", 0.0)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Self-BLEU comparison
        datasets = ['PADBen', 'RAID']
        self_bleu_scores = [padben_avg_self_bleu, raid_self_bleu]
        colors = ['#3498db', '#e74c3c']
        
        bars1 = ax1.bar(datasets, self_bleu_scores, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('Self-BLEU Score Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Self-BLEU Score', fontsize=12)
        ax1.set_ylim(0, max(self_bleu_scores) * 1.2)
        
        # Add value labels on bars
        for bar, score in zip(bars1, self_bleu_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Perplexity comparison (only G2X since PADBen uses GPT-2)
        datasets_ppl = ['PADBen', 'RAID (G2X)']
        perplexity_scores = [padben_avg_perplexity, raid_perplexity_g2x]
        colors_ppl = ['#3498db', '#f39c12']
        
        bars2 = ax2.bar(datasets_ppl, perplexity_scores, color=colors_ppl, alpha=0.7, edgecolor='black')
        ax2.set_title('Perplexity Score Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Perplexity Score', fontsize=12)
        ax2.set_ylim(0, max(perplexity_scores) * 1.2)
        
        # Add value labels on bars
        for bar, score in zip(bars2, perplexity_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Rotate x-axis labels for better readability
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics comparison chart saved to {save_path}")
        else:
            default_path = self.output_dir / "metrics_comparison.png"
            fig.savefig(default_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics comparison chart saved to {default_path}")
            
        return fig
    
    def create_text_type_metrics_chart(self, padben_metrics: Dict[str, Any],
                                     save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create a chart showing metrics breakdown by text type.
        
        Args:
            padben_metrics: PADBen metrics dictionary.
            save_path: Path to save the figure.
            
        Returns:
            Matplotlib figure object.
        """
        self_bleu_scores = padben_metrics.get("self_bleu_scores", {})
        perplexity_scores = padben_metrics.get("perplexity_scores", {})
        
        # Get common text types
        text_types = list(set(self_bleu_scores.keys()) & set(perplexity_scores.keys()))
        text_types.sort()
        
        if not text_types:
            logger.warning("No common text types found for visualization")
            return plt.figure()
        
        # Prepare data
        display_labels = [self._clean_label(t) for t in text_types]
        self_bleu_values = [self_bleu_scores.get(t, 0) for t in text_types]
        perplexity_values = [perplexity_scores.get(t, 0) for t in text_types 
                           if not np.isnan(perplexity_scores.get(t, 0)) and not np.isinf(perplexity_scores.get(t, 0))]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Self-BLEU by text type
        bars1 = ax1.bar(display_labels, self_bleu_values, color='skyblue', alpha=0.7, edgecolor='black')
        ax1.set_title('Self-BLEU Scores by Text Type', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Self-BLEU Score', fontsize=12)
        ax1.set_ylim(0, max(self_bleu_values) * 1.2 if self_bleu_values else 1)
        
        # Add value labels
        for bar, score in zip(bars1, self_bleu_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Perplexity by text type (only valid values)
        valid_types = [t for t in text_types 
                      if not np.isnan(perplexity_scores.get(t, float('inf'))) 
                      and not np.isinf(perplexity_scores.get(t, float('inf')))]
        valid_labels = [self._clean_label(t) for t in valid_types]
        valid_perplexity = [perplexity_scores[t] for t in valid_types]
        
        if valid_perplexity:
            bars2 = ax2.bar(valid_labels, valid_perplexity, color='lightcoral', alpha=0.7, edgecolor='black')
            ax2.set_title('Perplexity Scores by Text Type', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Perplexity Score', fontsize=12)
            ax2.set_ylim(0, max(valid_perplexity) * 1.2)
            
            # Add value labels
            for bar, score in zip(bars2, valid_perplexity):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{score:.2f}', ha='center', va='bottom', fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'No valid perplexity scores available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Perplexity Scores by Text Type', fontsize=14, fontweight='bold')
        
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Text type metrics chart saved to {save_path}")
        else:
            default_path = self.output_dir / "text_type_metrics.png"
            fig.savefig(default_path, dpi=300, bbox_inches='tight')
            logger.info(f"Text type metrics chart saved to {default_path}")
            
        return fig
    
    def create_comprehensive_dashboard(self, similarity_matrix: Dict[Tuple[str, str], float],
                                     padben_metrics: Dict[str, Any],
                                     raid_metrics: Dict[str, Any],
                                     save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create a comprehensive dashboard with all visualizations.
        
        Args:
            similarity_matrix: Jaccard similarity matrix.
            padben_metrics: PADBen metrics dictionary.
            raid_metrics: RAID metrics dictionary.
            save_path: Path to save the figure.
            
        Returns:
            Matplotlib figure object.
        """
        # Create large figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # Create grid for subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Similarity heatmap (top-left, spans 2x2)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        text_types = list(set([t for pair in similarity_matrix.keys() for t in pair]))
        text_types.sort()
        
        n_types = len(text_types)
        matrix_array = np.zeros((n_types, n_types))
        
        for i, type1 in enumerate(text_types):
            for j, type2 in enumerate(text_types):
                if (type1, type2) in similarity_matrix:
                    matrix_array[i, j] = similarity_matrix[(type1, type2)]
                elif (type2, type1) in similarity_matrix:
                    matrix_array[i, j] = similarity_matrix[(type2, type1)]
        
        display_labels = [self._clean_label(t) for t in text_types]
        sns.heatmap(matrix_array, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=display_labels, yticklabels=display_labels,
                   ax=ax1, square=True, cbar_kws={"shrink": 0.8})
        ax1.set_title('Jaccard Similarity Matrix', fontsize=14, fontweight='bold')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # 2. Self-BLEU comparison (top-right)
        ax2 = fig.add_subplot(gs[0, 2])
        padben_self_bleu = list(padben_metrics.get("self_bleu_scores", {}).values())
        padben_avg_self_bleu = np.mean(padben_self_bleu) if padben_self_bleu else 0.0
        raid_self_bleu = raid_metrics.get("self_bleu", 0.0)
        
        datasets = ['PADBen', 'RAID']
        scores = [padben_avg_self_bleu, raid_self_bleu]
        bars = ax2.bar(datasets, scores, color=['#3498db', '#e74c3c'], alpha=0.7)
        ax2.set_title('Self-BLEU Comparison', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Self-BLEU Score')
        
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 3. Perplexity comparison (middle-right)
        ax3 = fig.add_subplot(gs[1, 2])
        padben_perplexity = [score for score in padben_metrics.get("perplexity_scores", {}).values()
                           if not np.isnan(score) and not np.isinf(score)]
        padben_avg_perplexity = np.mean(padben_perplexity) if padben_perplexity else 0.0
        
        datasets_ppl = ['PADBen', 'RAID\n(G2X)']
        scores_ppl = [padben_avg_perplexity, raid_metrics.get("perplexity_g2x", 0)]
        bars = ax3.bar(datasets_ppl, scores_ppl, color=['#3498db', '#f39c12'], alpha=0.7)
        ax3.set_title('Perplexity Comparison', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Perplexity Score')
        
        for bar, score in zip(bars, scores_ppl):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{score:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Text type breakdown (bottom row)
        ax4 = fig.add_subplot(gs[2, :])
        self_bleu_by_type = padben_metrics.get("self_bleu_scores", {})
        types = list(self_bleu_by_type.keys())
        types.sort()
        
        if types:
            display_labels = [self._clean_label(t) for t in types]
            scores = [self_bleu_by_type[t] for t in types]
            
            bars = ax4.bar(display_labels, scores, color='lightgreen', alpha=0.7, edgecolor='black')
            ax4.set_title('Self-BLEU Scores by Text Type', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Self-BLEU Score')
            
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        # Add main title
        fig.suptitle('PADBen Dataset Quality Examination Dashboard', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comprehensive dashboard saved to {save_path}")
        else:
            default_path = self.output_dir / "quality_dashboard.png"
            fig.savefig(default_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comprehensive dashboard saved to {default_path}")
            
        return fig
    
    def _clean_label(self, label: str) -> str:
        """
        Clean up text type labels for better display.
        
        Args:
            label: Original label.
            
        Returns:
            Cleaned label.
        """
        # Mapping for cleaner labels
        label_mapping = {
            "type1": "Human Original",
            "type2": "LLM Generated", 
            "type3": "Human Paraphrased",
            "type4": "LLM Para. Original",
            "type5_1st": "LLM Para. Gen. (1st)",
            "type5_3rd": "LLM Para. Gen. (3rd)"
        }
        
        return label_mapping.get(label, label.replace("_", " ").title())
