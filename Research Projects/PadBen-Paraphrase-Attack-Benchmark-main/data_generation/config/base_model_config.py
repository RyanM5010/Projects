"""
Enhanced Base Model Configuration with secure API key handling.
"""

from dataclasses import dataclass
from typing import Optional
import os

# Import the enhanced API key manager
try:
    from data_generation.config.secrets_manager import get_api_key, validate_all_api_keys
except ImportError:
    # Fallback to simple implementation
    def get_api_key(env_var: str, required: bool = True) -> Optional[str]:
        """Fallback API key getter."""
        key = os.getenv(env_var)
        if not key and required:
            print(f"Warning: {env_var} not found in environment variables")
        return key
    
    def validate_all_api_keys() -> bool:
        """Fallback validation."""
        return bool(get_api_key("GEMINI_API_KEY", required=False))

@dataclass
class LLMModelConfig:
    """Configuration for a specific LLM model."""
    name: str
    provider: str  # 'gemini' or 'huggingface'
    model_id: str
    api_key_env: Optional[str] = None  # Environment variable name for API key
    api_base: Optional[str] = None
    
    # Core inference parameters
    temperature: float = 0.7
    max_tokens: int = 150
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.0
    
    # Connection settings
    timeout: int = 60
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # HuggingFace specific settings
    device: Optional[str] = None  # 'cuda', 'cpu', 'auto'
    torch_dtype: Optional[str] = None  # 'float16', 'float32', 'bfloat16'
    trust_remote_code: bool = False
    use_cache: bool = True

def validate_provider_config(provider: str) -> bool:
    """
    Validate that a specific provider's configuration is correct.
    
    Args:
        provider: The provider name ('gemini' or 'huggingface')
        
    Returns:
        True if the provider is properly configured, False otherwise
    """
    if provider.lower() == "gemini":
        api_key = get_api_key("GEMINI_API_KEY", required=False)
        if api_key:
            print("✅ gemini API key found")
            return True
        else:
            print("❌ GEMINI_API_KEY not found in environment")
            return False
    elif provider.lower() == "huggingface":
        # HuggingFace is optional, so always return True
        return True
    else:
        print(f"❌ Unknown provider: {provider}")
        return False

def create_gemini_flash_config(
    temperature: float = 0.7,
    max_tokens: int = 200,
    top_p: float = 0.9
) -> LLMModelConfig:
    """Create Gemini 2.5 Pro configuration for Type 2 generation."""
    return LLMModelConfig(
        name="Gemini 2.5 Pro",
        provider="gemini",
        model_id="gemini-2.5-pro",
        api_key_env="GEMINI_API_KEY",
        api_base="https://generativelanguage.googleapis.com/v1beta",
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        timeout=60,
        retry_attempts=3,
        retry_delay=1.0
    )

def create_dipper_config(
    device: str = "auto",
    torch_dtype: str = "float16",
    temperature: float = 0.7
) -> LLMModelConfig:
    """Create DIPPER paraphraser configuration."""
    return LLMModelConfig(
        name="DIPPER Paraphraser XXL",
        provider="huggingface",
        model_id="kalpeshk2011/dipper-paraphraser-xxl",
        device=device,
        torch_dtype=torch_dtype,
        temperature=temperature,
        max_tokens=250,
        top_p=0.9,
        timeout=120,
        retry_attempts=3,
        retry_delay=2.0,
        trust_remote_code=True,
        use_cache=True
    )

def create_llama_paraphrase_config(
    device: str = "auto",
    torch_dtype: str = "float16",
    temperature: float = 0.7
) -> LLMModelConfig:
    """Create Llama-3.1-8B paraphrase model configuration."""
    return LLMModelConfig(
        name="Llama-3.1-8B Paraphrase",
        provider="huggingface",
        model_id="mradermacher/Llama-3.1-8B-paraphrase-type-generation-apty-sigmoid-GGUF",
        device=device,
        torch_dtype=torch_dtype,
        temperature=temperature,
        max_tokens=300,
        top_p=0.9,
        top_k=40,
        repetition_penalty=1.1,
        timeout=120,
        retry_attempts=3,
        retry_delay=2.0,
        trust_remote_code=True,
        use_cache=True
    )

def get_default_model_configs():
    """Get default model configurations for all types."""
    return {
        "type2": create_gemini_flash_config(),
        "type4": create_dipper_config(),
        "type5": create_dipper_config()
    }

def validate_model_config(config: LLMModelConfig) -> bool:
    """
    Validate a model configuration.
    
    Args:
        config: The model configuration to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not config.name or not config.provider or not config.model_id:
        return False
    
    if config.provider == "gemini":
        return validate_provider_config("gemini")
    elif config.provider == "huggingface":
        return True  # HuggingFace validation happens at runtime
    else:
        return False

def check_provider_availability(provider: str) -> bool:
    """
    Check if a provider's dependencies are available.
    
    Args:
        provider: The provider name
        
    Returns:
        True if available, False otherwise
    """
    if provider.lower() == "gemini":
        try:
            from google import genai
            return True
        except ImportError:
            print(f"❌ {provider} dependencies not available. Install with: pip install google-generativeai")
            return False
    elif provider.lower() == "huggingface":
        try:
            import torch
            import transformers
            return True
        except ImportError:
            print(f"❌ {provider} dependencies not available. Install with: pip install torch transformers")
            return False
    else:
        return False

# Default configurations
DEFAULT_GEMINI_CONFIG = create_gemini_flash_config()
DEFAULT_DIPPER_CONFIG = create_dipper_config()