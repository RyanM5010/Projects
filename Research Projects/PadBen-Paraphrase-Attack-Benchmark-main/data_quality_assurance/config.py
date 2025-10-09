"""Configuration settings for data quality examination module."""

from typing import Dict, List, Any
from pathlib import Path
import logging

# File paths
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "test" / "final_generated_data.json"
OUTPUT_DIR = BASE_DIR / "data_quality_examine" / "outputs"

# Text type mappings
TEXT_TYPES = {
    "type1": "human_original_text(type1)",
    "type2": "llm_generated_text(type2)", 
    "type3": "human_paraphrased_text(type3)",
    "type4": "llm_paraphrased_original_text(type4)-prompt-based",
    "type5_1st": "llm_paraphrased_generated_text(type5)-1st",
    "type5_3rd": "llm_paraphrased_generated_text(type5)-3rd"
}

# RAID benchmark data (from the document)
RAID_METRICS = {
    "self_bleu": 13.7,
    "perplexity_l7b": 6.61,
    "perplexity_g2x": 23.8,
    "num_generations": 509000,  # 509k
    "avg_tokens": 323.4
}

# Metric calculation settings
JACCARD_CONFIG = {
    "n_gram": 1,  # Use unigrams for Jaccard similarity
    "case_sensitive": False
}

BLEU_CONFIG = {
    "max_n": 4,  # Use up to 4-grams for BLEU
    "smooth": True,
    "weights": [0.25, 0.25, 0.25, 0.25]  # Equal weights for 1-4 grams
}

# Perplexity model configurations
PERPLEXITY_MODELS = {
    "gpt2-xl": {
        "model_name": "gpt2-xl",
        "model_type": "gpt2",
        "max_length": 1024,
        "stride": 512,
        "batch_size": 4,
        "quantization": None,
        "torch_dtype": "float16",
        "description": "GPT-2 XL (1.5B params) - Original RAID benchmark model"
    },
    "llama3-7b-4bit": {
        "model_name": "meta-llama/Meta-Llama-3-8B",  # Using 8B as closest to 7B
        "model_type": "llama",
        "max_length": 512,       # Reduced for memory efficiency
        "stride": 256,           # Adjusted accordingly
        "batch_size": 1,         # Conservative for quantized model
        "quantization": "4bit",
        "load_in_4bit": True,
        "load_in_8bit": False,
        "torch_dtype": "float16",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
        "description": "Llama-3-8B 4-bit quantized - Optimized for RTX 3060 8GB (requires auth)"
    },
    "llama3-8b-full": {
        "model_name": "meta-llama/Meta-Llama-3-8B",
        "model_type": "llama",
        "max_length": 1024,      # Full context for better perplexity
        "stride": 512,
        "batch_size": 1,         # Conservative for full model
        "quantization": None,
        "load_in_4bit": False,
        "load_in_8bit": False,
        "torch_dtype": "float16",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
        "description": "Llama-3-8B full precision - High VRAM requirement (~16GB+), requires auth"
    },
    "llama3-8b-4bit": {
        "model_name": "meta-llama/Meta-Llama-3-8B",  # Using 8B as closest to 7B
        "model_type": "llama",
        "max_length": 512,       # Reduced for memory efficiency
        "stride": 256,           # Adjusted accordingly
        "batch_size": 1,         # Conservative for quantized model
        "quantization": "4bit",
        "load_in_4bit": True,
        "load_in_8bit": False,
        "torch_dtype": "float16",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
        "description": "Llama-3-8B 4-bit quantized - Optimized for RTX 3060 8GB (requires auth)"
    },
    "llama2-7b-full": {
        "model_name": "NousResearch/Llama-2-7b-hf",
        "model_type": "llama",
        "max_length": 1024,      # Full context for better perplexity
        "stride": 512,
        "batch_size": 1,         # Conservative for full model
        "quantization": None,
        "load_in_4bit": False,
        "load_in_8bit": False,
        "torch_dtype": "float16",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
        "description": "Llama-2-7B full precision - High VRAM requirement (~14GB+), ungated"
    },
    "llama2-7b-4bit": {
        "model_name": "NousResearch/Llama-2-7b-hf",  # Alternative ungated Llama-2
        "model_type": "llama",
        "max_length": 512,       # Reduced for memory efficiency
        "stride": 256,           # Adjusted accordingly
        "batch_size": 1,         # Conservative for quantized model
        "quantization": "4bit",
        "load_in_4bit": True,
        "load_in_8bit": False,
        "torch_dtype": "float16",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
        "description": "Llama-2-7B 4-bit quantized - Ungated alternative for RTX 3060 8GB"
    }
}

# Default perplexity configuration (backward compatibility)
PERPLEXITY_CONFIG = PERPLEXITY_MODELS["gpt2-xl"]

# GPU memory configurations
GPU_MEMORY_LIMITS = {
    "rtx_3060_8gb": {
        "total_vram_gb": 8,
        "safe_usage_threshold": 0.85,
        "recommended_models": ["gpt2-xl", "llama3-8b-4bit", "llama2-7b-4bit"]
    },
    "rtx_4070_12gb": {
        "total_vram_gb": 12,
        "safe_usage_threshold": 0.85,
        "recommended_models": ["gpt2-xl", "llama3-8b-4bit", "llama2-7b-4bit", "llama2-7b-full"]
    },
    "rtx_4080_16gb": {
        "total_vram_gb": 16,
        "safe_usage_threshold": 0.85,
        "recommended_models": ["gpt2-xl", "llama3-8b-full", "llama3-8b-4bit", "llama2-7b-full", "llama2-7b-4bit"]
    },
    "a100_40gb": {
        "total_vram_gb": 40,
        "safe_usage_threshold": 0.90,
        "recommended_models": ["gpt2-xl", "llama3-8b-full", "llama2-7b-full", "llama3-8b-4bit", "llama2-7b-4bit"]
    }
}

# Visualization settings
VISUALIZATION_CONFIG = {
    "figsize": (12, 10),
    "cmap": "Blues",
    "annot": True,
    "fmt": ".3f",
    "cbar_kws": {"shrink": 0.8}
}

# Logging configuration
LOGGING_CONFIG = {
    "level": logging.INFO,
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": [
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / "quality_examination.log")
    ]
}

# Output file names
OUTPUT_FILES = {
    "similarity_matrix": "similarity_matrix.csv",
    "metrics_table": "padben_metrics_table.csv",
    "comparison_table": "padben_vs_raid_comparison.csv",
    "similarity_heatmap": "similarity_heatmap.png",
    "detailed_report": "quality_examination_report.json"
}
