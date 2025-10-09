"""
Task Preparation Module

This module provides functionality to convert organized JSON data containing various text types
into task-specific datasets for paraphrase-based LLM detection research.
"""

from .data_models import ProcessedSample, PairSample, SingleSample
from .base_processor import AbstractTaskProcessor
from .task_processors import Task1Processor, Task2Processor, Task3Processor, Task4Processor
from .pipeline import TaskDataPreparationPipeline
from .config import TaskPreparationConfig
from .utils import DataValidator, LabelBalancer, StatisticsReporter

__all__ = [
    "ProcessedSample",
    "PairSample", 
    "SingleSample",
    "AbstractTaskProcessor",
    "Task1Processor",
    "Task2Processor",
    "Task3Processor",
    "Task4Processor",
    "TaskDataPreparationPipeline",
    "TaskPreparationConfig",
    "DataValidator",
    "LabelBalancer",
    "StatisticsReporter",
]
