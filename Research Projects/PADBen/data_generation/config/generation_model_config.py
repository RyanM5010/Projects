"""
Main Generation Configuration for PADBen Text Generation Pipeline.

This module imports and combines configurations for all generation types,
focusing only on the models we actually use and providing clear batch processing
and error handling settings.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
from pathlib import Path

from data_generation.config.base_model_config import LLMModelConfig, validate_provider_config
from data_generation.config.type2_config import Type2GenerationConfig, DEFAULT_TYPE2_CONFIG, validate_type2_config
from data_generation.config.type4_config import Type4GenerationConfig, DEFAULT_TYPE4_CONFIG, validate_type4_config
from data_generation.config.type5_config import Type5GenerationConfig, DEFAULT_TYPE5_CONFIG, validate_type5_config

from data_generation.config.secrets_manager import validate_all_api_keys, setup_api_keys_interactive

class OutputFormat(Enum):
    """Supported output formats for generated data."""
    CSV = "csv"
    JSON = "json"
    BOTH = "both"

class ErrorHandlingStrategy(Enum):
    """Error handling strategies for generation failures."""
    SKIP = "skip"  # Skip failed samples
    RETRY = "retry"  # Retry with fallback models
    FAIL_FAST = "fail_fast"  # Stop on first error

@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing settings."""
    
    # Batch sizes per generation type
    type2_batch_size: int = 5
    type4_batch_size: int = 8
    type5_batch_size: int = 6
    
    # Concurrency limits
    max_concurrent_type2: int = 3
    max_concurrent_type4: int = 4
    max_concurrent_type5: int = 3
    
    # Progress and checkpointing
    checkpoint_frequency: int = 50  # Save every N batches
    show_progress: bool = True
    log_batch_stats: bool = True
    
    # Memory management
    clear_cache_frequency: int = 100  # Clear model cache every N samples
    memory_threshold_mb: int = 8192  # Memory threshold for cleanup

@dataclass
class OutputConfig:
    """Configuration for output format and file handling."""
    
    # Output formats
    primary_format: OutputFormat = OutputFormat.CSV
    secondary_format: OutputFormat = OutputFormat.JSON
    
    # File naming
    timestamp_format: str = "%Y%m%d_%H%M%S"
    include_metadata: bool = True
    compress_large_files: bool = True
    
    # Directory structure
    output_dir: str = "data/generated"
    checkpoint_dir: str = "data/generated/checkpoints"
    metadata_dir: str = "data/generated/metadata"
    
    # File size limits
    max_file_size_mb: int = 100
    split_large_files: bool = True

@dataclass
class ErrorHandlingConfig:
    """Configuration for error handling and recovery."""
    
    # Error handling strategy
    strategy: ErrorHandlingStrategy = ErrorHandlingStrategy.RETRY
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    
    # Failure thresholds
    max_consecutive_failures: int = 10
    max_total_failure_rate: float = 0.2  # 20% max failure rate
    
    # Recovery settings
    save_failed_samples: bool = True
    continue_from_checkpoint: bool = True
    
    # Logging
    log_all_errors: bool = True
    error_log_file: str = "generation_errors.log"

@dataclass
class GenerationConfig:
    """Master configuration for all text generation types."""
    
    # Type-specific configurations
    type2_config: Type2GenerationConfig = field(default_factory=lambda: DEFAULT_TYPE2_CONFIG)
    type4_config: Type4GenerationConfig = field(default_factory=lambda: DEFAULT_TYPE4_CONFIG)
    type5_config: Type5GenerationConfig = field(default_factory=lambda: DEFAULT_TYPE5_CONFIG)
    
    # Processing configurations
    batch_config: BatchProcessingConfig = field(default_factory=BatchProcessingConfig)
    output_config: OutputConfig = field(default_factory=OutputConfig)
    error_config: ErrorHandlingConfig = field(default_factory=ErrorHandlingConfig)
    
    # Global settings
    input_file: str = "data/processed/unified_padben_base.csv"
    log_level: str = "INFO"
    
    # Generation order and dependencies
    generation_order: List[str] = field(default_factory=lambda: ["type2", "type4", "type5"])
    parallel_type4_type5: bool = False  # Whether Type 4 and 5 can run in parallel
    
    # Resource management
    gpu_memory_fraction: float = 0.8
    cpu_count: int = -1  # -1 for auto-detect

# Default configuration instance
DEFAULT_CONFIG = GenerationConfig()

def validate_all_configs(config: GenerationConfig) -> Dict[str, bool]:
    """Validate all generation configurations with enhanced API key checking."""
    results = {
        "type2": validate_type2_config(config.type2_config),
        "type4": validate_type4_config(config.type4_config),
        "type5": validate_type5_config(config.type5_config),
        "api_keys": validate_all_api_keys(),  # Enhanced API key validation
        "directories": validate_directories(config)
    }
    
    overall_valid = all(results.values())
    print(f"Configuration validation results: {results}")
    print(f"Overall valid: {overall_valid}")
    
    # If API keys are missing, offer interactive setup
    if not results["api_keys"]:
        print("\n🔐 API Key Setup Required")
        print("Run the interactive setup to configure your API keys:")
        print("python -c \"from data_generation.config.secrets_manager import setup_api_keys_interactive; setup_api_keys_interactive()\"")
    
    return results

def validate_all_providers() -> bool:
    """Validate all required providers are properly configured."""
    providers = ["gemini", "huggingface"]
    for provider in providers:
        if not validate_provider_config(provider):
            return False
    return True

def validate_directories(config: GenerationConfig) -> bool:
    """Validate that all required directories exist or can be created."""
    directories = [
        config.output_config.output_dir,
        config.output_config.checkpoint_dir,
        config.output_config.metadata_dir
    ]
    
    for dir_path in directories:
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Cannot create directory {dir_path}: {e}")
            return False
    
    return True

def get_config_summary(config: GenerationConfig) -> Dict[str, Any]:
    """Get a comprehensive summary of the current configuration."""
    return {
        "models_used": {
            "type2": {
                "primary": f"{config.type2_config.primary_model.name} ({config.type2_config.primary_model.model_id})",
                "provider": config.type2_config.primary_model.provider
            },
            "type4": {
                "primary": f"{config.type4_config.primary_model.name} ({config.type4_config.primary_model.model_id})",
                "fallback": f"{config.type4_config.fallback_model.name} ({config.type4_config.fallback_model.model_id})"
            },
            "type5": {
                "primary": f"{config.type5_config.primary_model.name} ({config.type5_config.primary_model.model_id})",
                "fallback": f"{config.type5_config.fallback_model.name} ({config.type5_config.fallback_model.model_id})"
            }
        },
        "batch_processing": {
            "type2_batch_size": config.batch_config.type2_batch_size,
            "type4_batch_size": config.batch_config.type4_batch_size,
            "type5_batch_size": config.batch_config.type5_batch_size,
            "checkpoint_frequency": config.batch_config.checkpoint_frequency
        },
        "output_settings": {
            "primary_format": config.output_config.primary_format.value,
            "output_dir": config.output_config.output_dir,
            "include_metadata": config.output_config.include_metadata
        },
        "error_handling": {
            "strategy": config.error_config.strategy.value,
            "max_retries": config.error_config.max_retries,
            "max_failure_rate": config.error_config.max_total_failure_rate
        },
        "global": {
            "input_file": config.input_file,
            "generation_order": config.generation_order,
            "log_level": config.log_level
        }
    }

def create_production_config() -> GenerationConfig:
    """Create a production-ready configuration with conservative settings."""
    config = GenerationConfig()
    
    # Conservative batch sizes
    config.batch_config.type2_batch_size = 3
    config.batch_config.type4_batch_size = 5
    config.batch_config.type5_batch_size = 4
    
    # Conservative concurrency
    config.batch_config.max_concurrent_type2 = 2
    config.batch_config.max_concurrent_type4 = 2
    config.batch_config.max_concurrent_type5 = 2
    
    # More frequent checkpointing
    config.batch_config.checkpoint_frequency = 25
    
    # Robust error handling
    config.error_config.strategy = ErrorHandlingStrategy.RETRY
    config.error_config.max_retries = 5
    config.error_config.max_consecutive_failures = 5
    
    return config

def create_fast_config() -> GenerationConfig:
    """Create a configuration optimized for speed (less robust)."""
    config = GenerationConfig()
    
    # Larger batch sizes
    config.batch_config.type2_batch_size = 10
    config.batch_config.type4_batch_size = 15
    config.batch_config.type5_batch_size = 12
    
    # Higher concurrency
    config.batch_config.max_concurrent_type2 = 5
    config.batch_config.max_concurrent_type4 = 6
    config.batch_config.max_concurrent_type5 = 5
    
    # Less frequent checkpointing
    config.batch_config.checkpoint_frequency = 100
    
    # Fast error handling
    config.error_config.strategy = ErrorHandlingStrategy.SKIP
    config.error_config.max_retries = 1
    
    return config

# Convenience imports for easy access
__all__ = [
    "GenerationConfig",
    "DEFAULT_CONFIG",
    "BatchProcessingConfig",
    "OutputConfig", 
    "ErrorHandlingConfig",
    "OutputFormat",
    "ErrorHandlingStrategy",
    "Type2GenerationConfig",
    "Type4GenerationConfig",
    "Type5GenerationConfig",
    "LLMModelConfig",
    "validate_all_configs",
    "get_config_summary",
    "create_production_config",
    "create_fast_config"
]