"""
Data Quality Examination Module for PADBen Dataset.

This module provides comprehensive quality assessment tools including:
- Jaccard similarity calculations
- Self-BLEU score computations
- Perplexity measurements
- Comparison with RAID benchmark
- Visualization of similarity matrices
"""

from .metrics import (
    JaccardSimilarityCalculator,
    SelfBLEUCalculator,
    PerplexityCalculator,
    MetricsAggregator
)
from .comparison import RAIDComparator
from .visualization import SimilarityVisualizer
from .data_loader import DataLoader
from .main import DataQualityExaminer
from .short_text_analyzer import ShortTextAnalyzer

__version__ = "1.0.0"
__author__ = "PADBen Team"

__all__ = [
    "JaccardSimilarityCalculator",
    "SelfBLEUCalculator", 
    "PerplexityCalculator",
    "MetricsAggregator",
    "RAIDComparator",
    "SimilarityVisualizer",
    "DataLoader",
    "DataQualityExaminer",
    "ShortTextAnalyzer"
]
