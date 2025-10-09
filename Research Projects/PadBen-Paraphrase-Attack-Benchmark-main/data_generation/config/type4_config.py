"""
Type 4 Generation Configuration for PADBen.

Configuration for LLM-paraphrased original text (Type 4) using two methods:
1. DIPPER paraphraser (HuggingFace specialized model)
2. Prompt-based paraphrasing (Gemini)
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
from data_generation.config.base_model_config import LLMModelConfig, create_dipper_config, create_gemini_flash_config, create_llama_paraphrase_config

class Type4ParaphraseMethod(Enum):
    """Available paraphrasing methods for Type 4."""
    DIPPER = "dipper"
    PROMPT_BASED = "prompt_based"
    LLAMA = "llama"

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
    
    # Llama paraphrase model
    llama_model: LLMModelConfig = field(default_factory=lambda: create_llama_paraphrase_config(
        device="auto",
        torch_dtype="float16",
        temperature=0.7
    ))
    
    # Available paraphrasing methods
    paraphrase_methods: List[Type4ParaphraseMethod] = field(default_factory=lambda: [
        Type4ParaphraseMethod.DIPPER,
        Type4ParaphraseMethod.PROMPT_BASED,
        Type4ParaphraseMethod.LLAMA
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
    
    # Llama-specific settings
    llama_settings: dict = field(default_factory=lambda: {
        "max_length": 300,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.1,
        "pad_token_id": None,
        "eos_token_id": None
    })
    
    # Llama paraphrase prompt template
    llama_prompt_template: str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that paraphrases text while maintaining its original meaning. Your task is to rewrite the given text using different words and sentence structures while preserving the core information and meaning.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Please paraphrase the following text:

{text}

Requirements:
- Maintain the core meaning and information
- Use different words and sentence structures where possible
- Make the text sound natural and fluent
- Keep approximately the same length ({target_length} characters)
- Do not exceed {max_length} characters

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
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
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds between retries
    enable_retry_with_context: bool = True  # Include failure reason in retry prompts

    # Enhanced prompt template for retries (for Gemini fallback)
    retry_prompt_template: str = """Paraphrase the following text while maintaining its meaning and naturalness.

PREVIOUS ATTEMPT FAILED: {failure_reason}

Text: "{text}"

Requirements:
- Maintain the core meaning and information
- Use different words and sentence structures where possible
- Make the text sound natural and fluent
- Keep approximately the same length ({target_length} characters)
- Ensure the result flows naturally
- Do not exceed {max_length} characters
- Address the previous failure: {failure_reason}

CRITICAL INSTRUCTIONS:
- Generate a COMPLETE paraphrased text, not empty or null
- Ensure the output is substantive and meaningful
- Do not generate partial responses
- Avoid the issue that caused the previous failure

OUTPUT FORMAT:
Return ONLY the paraphrased text without any additional commentary, explanations, or formatting.
Do NOT include labels like "Paraphrased text:" or "Result:" in your response.
Do NOT add quotation marks around your response.
Do NOT provide multiple versions or alternatives.
Do NOT include explanations of what you changed.
Do NOT add meta-commentary about the paraphrasing process.
Do NOT mention this is a retry or reference previous attempts.

EXAMPLE FORMAT:
[Your paraphrased text here]

"""
    
    # Llama-specific prompt template
    llama_prompt_template: str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a professional text paraphraser. Your task is to rewrite the given text while maintaining its original meaning and naturalness. Focus on using different vocabulary and sentence structures while preserving the core information.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Please paraphrase the following text:

Text: "{text}"

Requirements:
- Maintain the exact same meaning and information
- Use different words and sentence structures
- Keep the text natural and fluent
- Preserve the original length (approximately {target_length} characters)
- Do not exceed {max_length} characters
- Return only the paraphrased text without any additional commentary

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

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
    
    # Check Llama model configuration
    if config.llama_model.provider != "huggingface":
        print("Type 4 Llama model must use HuggingFace provider")
        return False
    
    if "llama" not in config.llama_model.model_id.lower():
        print("Type 4 Llama model should be a Llama model")
        return False
    
    # Validate HuggingFace dependencies
    try:
        import transformers
        import torch
    except ImportError as e:
        print(f"Missing HuggingFace dependencies for Type 4: {e}")
        print("Install with: pip install transformers torch accelerate")
        return False
    
    # Validate GGUF dependencies for Llama model
    try:
        from llama_cpp import Llama
    except ImportError:
        print("Warning: llama-cpp-python not installed. Install with: pip install llama-cpp-python")
        print("This is required for GGUF model support")
    
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
        
        if f"{{{placeholder}}}" not in config.llama_prompt_template:
            print(f"Missing placeholder '{placeholder}' in llama_prompt_template")
            return False
    
    return True 