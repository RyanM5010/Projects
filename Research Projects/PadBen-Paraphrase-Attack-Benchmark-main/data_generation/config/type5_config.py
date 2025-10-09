"""
Type 5 Generation Configuration for PADBen.

Configuration for LLM-paraphrased LLM-generated text (Type 5).
Uses same methods as Type 4 but with iterative paraphrasing:
- 1 iteration: Paraphrase once
- 3 iterations: Paraphrase 3 times sequentially  
- 5 iterations: Paraphrase 5 times sequentially
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum
from data_generation.config.base_model_config import LLMModelConfig
from data_generation.config.type4_config import Type4ParaphraseMethod, create_dipper_config, create_gemini_flash_config

class IterationLevel(Enum):
    """Supported iteration levels for Type 5 paraphrasing."""
    FIRST = 1    # Paraphrase 1 time
    THIRD = 3    # Paraphrase 3 times sequentially
    FIFTH = 5    # Paraphrase 5 times sequentially

@dataclass
class Type5GenerationConfig:
    """Configuration for Type 5 (LLM-paraphrased LLM-generated text) generation."""
    
    # Same models as Type 4 but with iterative processing
    primary_model: LLMModelConfig = field(default_factory=lambda: create_dipper_config(
        device="auto"
    ))
    
    fallback_model: LLMModelConfig = field(default_factory=lambda: create_gemini_flash_config(
        temperature=0.8,
        max_tokens=250,
        top_p=0.9
    ))
    
    # Available paraphrasing methods (same as Type 4)
    paraphrase_methods: List[Type4ParaphraseMethod] = field(default_factory=lambda: [
        Type4ParaphraseMethod.DIPPER,
        Type4ParaphraseMethod.PROMPT_BASED
    ])
    
    # DIPPER settings (same as Type 4)
    dipper_settings: dict = field(default_factory=lambda: {
        "max_length": 300,
        "num_beams": 4,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "lex_diversity": 60,
        "order_diversity": 0,
        "sent_interval": 3
    })
    
    # Type 5: Iterative Paraphrasing
    paraphrase_prompt_template: str = """Paraphrase the following text while maintaining its meaning and naturalness:

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
    
    # Retry prompt template for failed attempts
    retry_prompt_template: str = """Paraphrase the following text while maintaining its meaning and naturalness.
Previous attempt failed due to: {failure_reason}

Text: "{text}"

Requirements:
- Maintain the core meaning and information
- Use different words and sentence structures where possible
- Make the text sound natural and fluent
- Keep approximately the same length ({target_length} characters)
- Ensure the result flows naturally
- Do not exceed {max_length} characters
- Address the previous failure by being more careful with the requirements

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
    
    # Iterative paraphrasing settings
    supported_iterations: List[IterationLevel] = field(default_factory=lambda: [
        IterationLevel.FIRST,
        IterationLevel.THIRD,
        IterationLevel.FIFTH
    ])
    
    default_iteration: IterationLevel = IterationLevel.FIRST
    
    # Iteration-specific settings
    iteration_settings: Dict[IterationLevel, Dict] = field(default_factory=lambda: {
        IterationLevel.FIRST: {
            "temperature_increment": 0.0,  # No change for first iteration
            "diversity_boost": False
        },
        IterationLevel.THIRD: {
            "temperature_increment": 0.1,  # Slightly higher temperature for more variation
            "diversity_boost": True
        },
        IterationLevel.FIFTH: {
            "temperature_increment": 0.15,  # Higher temperature for maximum variation
            "diversity_boost": True
        }
    })
    
    # Length and quality settings (more tolerant for iterative processing)
    length_tolerance: float = 0.4  # ±40% tolerance for iterative paraphrasing
    max_length_multiplier: float = 1.5  # Allow more length variation after multiple iterations
    min_length_threshold: int = 15
    
    # Iteration quality control
    max_identical_iterations: int = 2  # Stop if text becomes identical for 2 consecutive iterations
    similarity_threshold: float = 0.95  # Consider texts identical if similarity > 95%
    
    # Retry settings (same as Type 4)
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds between retries
    enable_retry_with_context: bool = True  # Include failure reason in retry prompts
    
    # Target datasets (same as Type 4)
    target_datasets: List[str] = field(default_factory=lambda: ["mrpc", "hlpc", "paws"])
    
    # Processing settings (adjusted for iterative processing)
    batch_size: int = 4  # Smaller batch due to multiple iterations per sample
    max_concurrent_requests: int = 2  # Reduced concurrency for stability
    save_intermediate: bool = True
    save_iteration_history: bool = True  # Save intermediate iteration results
    
    # HuggingFace settings (same as Type 4)
    model_cache_dir: Optional[str] = "./models/cache"
    use_gpu: bool = True
    memory_efficient: bool = True
    
    # Type 5 specific: requires Type 2 data
    require_type2_data: bool = True
    
    # Method selection strategy
    default_method: Type4ParaphraseMethod = Type4ParaphraseMethod.DIPPER
    fallback_method: Type4ParaphraseMethod = Type4ParaphraseMethod.PROMPT_BASED

# Default Type 5 configuration
DEFAULT_TYPE5_CONFIG = Type5GenerationConfig()

def create_iterative_type5_config(iteration_level: IterationLevel) -> Type5GenerationConfig:
    """Create Type 5 configuration for specific iteration level."""
    config = Type5GenerationConfig()
    config.default_iteration = iteration_level
    return config

def create_memory_efficient_type5_config() -> Type5GenerationConfig:
    """Create memory-efficient Type 5 configuration."""
    config = Type5GenerationConfig()
    config.batch_size = 2
    config.max_concurrent_requests = 1
    config.memory_efficient = True
    config.dipper_settings["max_length"] = 200  # Reduce max length for memory efficiency
    return config

def validate_type5_config(config: Type5GenerationConfig) -> bool:
    """Validate Type 5 configuration."""
    from data_generation.config.base_model_config import validate_provider_config
    
    # Validate provider configurations
    if not validate_provider_config("huggingface") or not validate_provider_config("gemini"):
        return False
    
    # Check Type 5 specific requirements
    if not config.require_type2_data:
        print("Type 5 must require Type 2 data to be present")
        return False
    
    # Validate prompt template placeholders
    required_placeholders = ["text", "target_length", "max_length"]
    for placeholder in required_placeholders:
        if f"{{{placeholder}}}" not in config.paraphrase_prompt_template:
            print(f"Missing placeholder '{placeholder}' in paraphrase_prompt_template")
            return False
    
    # Validate retry prompt template placeholders
    retry_placeholders = ["text", "target_length", "max_length", "failure_reason"]
    for placeholder in retry_placeholders:
        if f"{{{placeholder}}}" not in config.retry_prompt_template:
            print(f"Missing placeholder '{placeholder}' in retry_prompt_template")
            return False
    
    # Check that all supported iterations have settings
    for iteration in config.supported_iterations:
        if iteration not in config.iteration_settings:
            print(f"Missing settings for supported iteration level: {iteration.value}")
            return False
    
    # Validate iteration values
    for iteration in config.supported_iterations:
        if iteration.value < 1 or iteration.value > 10:
            print(f"Invalid iteration level: {iteration.value}. Must be between 1-10.")
            return False
    
    # Validate retry settings
    if config.max_retries < 0:
        print("max_retries must be non-negative")
        return False
    
    if config.retry_delay < 0:
        print("retry_delay must be non-negative")
        return False
    
    return True 