"""
Data models for task preparation pipeline.
"""

from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict


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
class PairSample(ProcessedSample):
    """Sample containing a pair of sentences.
    
    Attributes:
        idx: Sample index
        label: Classification label (0 or 1)
        original_sentence: The original sentence
        paraphrased_sentence: The paraphrased sentence
    """
    original_sentence: str
    paraphrased_sentence: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pair sample to dictionary representation.
        
        Returns:
            Dictionary representation with all fields
        """
        return {
            "idx": self.idx,
            "original_sentence": self.original_sentence,
            "paraphrased_sentence": self.paraphrased_sentence,
            "label": self.label
        }


@dataclass
class SingleSample(ProcessedSample):
    """Sample containing a single sentence.
    
    Attributes:
        idx: Sample index
        label: Classification label (0 or 1)
        sentence: The sentence text
    """
    sentence: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert single sample to dictionary representation.
        
        Returns:
            Dictionary representation with all fields
        """
        return {
            "idx": self.idx,
            "sentence": self.sentence,
            "label": self.label
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
