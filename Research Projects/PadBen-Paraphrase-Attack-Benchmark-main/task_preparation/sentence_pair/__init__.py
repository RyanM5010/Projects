"""
Sentence Pair Task Preparation Module

This module provides functionality for creating sentence pair classification tasks
where models need to determine which sentence in a pair is machine-generated vs human-written.
"""

from .data_models import InputSample, SentencePairSample, ProcessedSample
from .base_processor import AbstractSentencePairProcessor
from .task_processors import (
    Task1SentencePairProcessor,
    Task2SentencePairProcessor,
    Task3SentencePairProcessor,
    Task4SentencePairProcessor,
    Task5SentencePairProcessor
)
from .pipeline import SentencePairTaskPipeline
from .config import SentencePairTaskConfig
from .utils import DataValidator, StatisticsReporter

__version__ = "1.0.0"
__author__ = "PADBen Team"

__all__ = [
    "InputSample",
    "SentencePairSample", 
    "ProcessedSample",
    "AbstractSentencePairProcessor",
    "Task1SentencePairProcessor",
    "Task2SentencePairProcessor",
    "Task3SentencePairProcessor",
    "Task4SentencePairProcessor",
    "Task5SentencePairProcessor",
    "SentencePairTaskPipeline",
    "SentencePairTaskConfig",
    "DataValidator",
    "StatisticsReporter"
]
