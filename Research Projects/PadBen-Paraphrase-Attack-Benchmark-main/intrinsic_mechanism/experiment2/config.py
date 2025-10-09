"""
Configuration module for semantic space experiment.

This module contains all configuration parameters and model specifications
for the semantic space analysis experiment.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import torch
import os


class ExperimentConfig:
    """Configuration class for the semantic space experiment."""
    
    # Data paths (using relative paths from project root)
    DATA_PATH: str = str(Path(__file__).parent.parent / "data" / "test" / "final_generated_data.json")
    OUTPUT_DIR: str = str(Path(__file__).parent / "output")
    
    # Experiment parameters
    MAX_ITERATIONS: int = 5
    NUM_SAMPLES: int = 100
    RANDOM_SEED: int = 42
    
    # Model configurations
    PARAPHRASE_MODEL: str = "Qwen/Qwen2.5-3B-Instruct"  # Using available model
    EMBEDDING_MODEL: str = "baai/bge-m3"  # Via Novita AI API
    
    # Novita AI API configuration (use environment variables)
    NOVITA_API_KEY: str = os.getenv("NOVITA_API_KEY", "")
    NOVITA_BASE_URL: str = os.getenv("NOVITA_BASE_URL", "https://api.novita.ai/openai")
    
    # Generation parameters
    MAX_NEW_TOKENS: int = 150
    TEMPERATURE: float = 0.7
    MAX_LENGTH: int = 512
    
    # Device configuration
    DEVICE: Optional[str] = None  # Auto-detect if None
    USE_FP16: bool = True  # Use half precision if CUDA available
    
    # PCA parameters
    PCA_COMPONENTS: int = 2
    
    # Visualization parameters
    FIGURE_SIZE: tuple = (15, 12)
    DPI: int = 300
    
    @classmethod
    def get_device(cls) -> torch.device:
        """Get the appropriate device for computation."""
        if cls.DEVICE is not None:
            return torch.device(cls.DEVICE)
        
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get model configuration parameters."""
        return {
            "max_new_tokens": cls.MAX_NEW_TOKENS,
            "temperature": cls.TEMPERATURE,
            "max_length": cls.MAX_LENGTH,
            "use_fp16": cls.USE_FP16 and torch.cuda.is_available()
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration parameters."""
        if not cls.NOVITA_API_KEY:
            print("Warning: NOVITA_API_KEY not set. Original sentence analysis will not work.")
            return False
        return True


# Paraphrase prompts for different text types
PARAPHRASE_PROMPTS = {
    "type1": "Please paraphrase the following sentence while maintaining its original meaning:",
    "type2": "Please rephrase the following text in a different way while keeping the same meaning:"
}

# Default prompt type
DEFAULT_PROMPT_TYPE = "type1"
