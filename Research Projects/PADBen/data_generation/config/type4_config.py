"""
Type 4 Generation Configuration for PADBen.

Configuration for LLM-paraphrased original text (Type 4) using two methods:
1. DIPPER paraphraser (HuggingFace specialized model)
2. Prompt-based paraphrasing (Gemini)
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
from data_generation.config.base_model_config import LLMModelConfig, create_dipper_config, create_gemini_flash_config

class Type4ParaphraseMethod(Enum):
    """Available paraphrasing methods for Type 4."""
    DIPPER = "dipper"
    PROMPT_BASED = "prompt_based"

@dataclass
class Type4GenerationConfig:
    """Configuration for Type 4 (LLM-paraphrased original text) generation."""
    
    # Primary method: DIPPER paraphraser
    primary_model: LLMModelConfig = field(default_factory=lambda: create_dipper_config(
        device="auto"
    ))
    
    # Fallback method: Gemini prompt-based paraphrasing
    fallback_model: LLMModelConfig = field(default_factory=lambda: create_gemini_flash_config(
        temperature=0.8,
        max_tokens=250,
        top_p=0.9
    ))
    
    # Available paraphrasing methods
    paraphrase_methods: List[Type4ParaphraseMethod] = field(default_factory=lambda: [
        Type4ParaphraseMethod.DIPPER,
        Type4ParaphraseMethod.PROMPT_BASED
    ])
    
    # DIPPER-specific settings (direct input, no prompt needed)
    dipper_settings: dict = field(default_factory=lambda: {
        "max_length": 300,
        "num_beams": 4,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.9,
        "repetition_penalty": 1.1
    })
    
    # Prompt-based paraphrasing template (for Gemini fallback)
    prompt_based_template: str = """Paraphrase the following text while maintaining its meaning and naturalness:

Text: "{text}"

Requirements:
- Maintain the core meaning and information
- Use different words and sentence structures where possible
- Make the text sound natural and fluent
- Keep approximately the same length ({target_length} characters)
- Ensure the result flows naturally
- Do not exceed {max_length} characters

OUTPUT FORMAT:
Return ONLY the paraphrased text without any additional commentary, explanations, or formatting.
Do NOT include labels like "Paraphrased text:" or "Result:" in your response.
Do NOT add quotation marks around your response.
Do NOT provide multiple versions or alternatives.
Do NOT include explanations of what you changed.
Do NOT add meta-commentary about the paraphrasing process.
Do NOT mention this is an iteration or reference previous versions.

EXAMPLE FORMAT:
[Your paraphrased text here]

"""
    
    # Length and quality settings
    length_tolerance: float = 0.3  # ±30% tolerance for paraphrasing
    max_length_multiplier: float = 1.3
    min_length_threshold: int = 15
    
    # Target datasets
    target_datasets: List[str] = field(default_factory=lambda: ["mrpc", "hlpc", "paws"])
    
    # Processing settings
    batch_size: int = 8
    max_concurrent_requests: int = 4
    save_intermediate: bool = True
    
    # HuggingFace specific settings
    model_cache_dir: Optional[str] = "./models/cache"
    use_gpu: bool = True
    memory_efficient: bool = True
    
    # Method selection strategy
    default_method: Type4ParaphraseMethod = Type4ParaphraseMethod.DIPPER
    fallback_method: Type4ParaphraseMethod = Type4ParaphraseMethod.PROMPT_BASED

# Default Type 4 configuration
DEFAULT_TYPE4_CONFIG = Type4GenerationConfig()

def create_memory_efficient_type4_config() -> Type4GenerationConfig:
    """Create memory-efficient Type 4 configuration for limited resources."""
    config = Type4GenerationConfig()
    
    # Use 8-bit quantization for DIPPER
    config.primary_model = create_dipper_config(
        device="auto"
    )
    
    config.memory_efficient = True
    config.batch_size = 4  # Smaller batch size
    config.max_concurrent_requests = 2
    
    return config

def validate_type4_config(config: Type4GenerationConfig) -> bool:
    """Validate Type 4 configuration."""
    from data_generation.config.base_model_config import validate_provider_config
    
    # Check DIPPER model configuration
    if config.primary_model.provider != "huggingface":
        print("Type 4 primary model must use HuggingFace provider for DIPPER")
        return False
    
    if "dipper" not in config.primary_model.model_id.lower():
        print("Type 4 primary model should be a DIPPER model")
        return False
    
    # Check fallback model configuration
    if config.fallback_model.provider != "gemini":
        print("Type 4 fallback model must use Gemini provider")
        return False
    
    # Validate HuggingFace dependencies
    try:
        import transformers
        import torch
    except ImportError as e:
        print(f"Missing HuggingFace dependencies for Type 4: {e}")
        print("Install with: pip install transformers torch accelerate")
        return False
    
    # Validate provider configurations
    if not validate_provider_config("huggingface"):
        return False
    if not validate_provider_config("gemini"):
        return False
    
    # Check prompt template placeholders
    required_placeholders = ["text", "target_length", "max_length"]
    for placeholder in required_placeholders:
        if f"{{{placeholder}}}" not in config.prompt_based_template:
            print(f"Missing placeholder '{placeholder}' in prompt_based_template")
            return False
    
    return True 