"""
Data models for sentence pair task preparation pipeline.
"""

from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass
class ProcessedSample(ABC):
    """Base class for processed samples.
    
    Attributes:
        idx: Sample index
        label: Classification label (0 or 1)
    """
    idx: int
    label: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary representation.
        
        Returns:
            Dictionary representation of the sample
        """
        return {"idx": self.idx, "label": self.label}


@dataclass
class SentencePairSample(ProcessedSample):
    """Sample containing a pair of sentences for comparison.
    
    Attributes:
        idx: Sample index
        sentence_pair: Tuple of (sentence1, sentence2)
        label_pair: Tuple of (label1, label2) corresponding to each sentence
        label: Single label for compatibility (derived from label_pair)
    """
    sentence_pair: Tuple[str, str]
    label_pair: Tuple[int, int]
    
    def __post_init__(self):
        """Set the label field for compatibility with ProcessedSample."""
        # Use the first label for compatibility, though both labels are available in label_pair
        self.label = self.label_pair[0]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert sentence pair sample to dictionary representation.
        
        Returns:
            Dictionary representation with all fields
        """
        return {
            "idx": self.idx,
            "sentence_pair": self.sentence_pair,
            "label_pair": self.label_pair
        }


@dataclass
class InputSample:
    """Raw input sample from the organized JSON data.
    
    Attributes:
        idx: Original sample index
        dataset_source: Source dataset name
        type1: Human original text
        type2: LLM generated text
        type3: Human paraphrased text
        type4: LLM paraphrased original text
        type5_1st: LLM paraphrased generated text (1st iteration)
        type5_3rd: LLM paraphrased generated text (3rd iteration)
    """
    idx: int
    dataset_source: str
    type1: str  # human_original_text
    type2: str  # llm_generated_text
    type3: str  # human_paraphrased_text
    type4: str  # llm_paraphrased_original_text
    type5_1st: str  # llm_paraphrased_generated_text(type5)-1st
    type5_3rd: str  # llm_paraphrased_generated_text(type5)-3rd
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InputSample":
        """Create InputSample from dictionary.
        
        Args:
            data: Dictionary containing sample data
            
        Returns:
            InputSample instance
            
        Raises:
            KeyError: If required fields are missing
            ValueError: If data is invalid
        """
        try:
            return cls(
                idx=data["idx"],
                dataset_source=data["dataset_source"],
                type1=data["human_original_text(type1)"],
                type2=data["llm_generated_text(type2)"],
                type3=data["human_paraphrased_text(type3)"],
                type4=data["llm_paraphrased_original_text(type4)-prompt-based"],
                type5_1st=data["llm_paraphrased_generated_text(type5)-1st"],
                type5_3rd=data["llm_paraphrased_generated_text(type5)-3rd"]
            )
        except KeyError as e:
            raise KeyError(f"Missing required field in input data: {e}")
        except Exception as e:
            raise ValueError(f"Invalid input data format: {e}")
