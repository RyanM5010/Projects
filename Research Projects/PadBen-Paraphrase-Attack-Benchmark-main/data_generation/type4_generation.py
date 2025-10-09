"""
Type 4 Text Generation for PADBen Benchmark.

This module implements LLM-based paraphrasing of Type 1 human original text using:
1. DIPPER paraphraser (HuggingFace specialized model) - Primary method
2. Prompt-based paraphrasing (Gemini) - Fallback method

Maintains semantic meaning while changing word choice and structure.
"""

import asyncio
import logging
import time
import re
import json
import gc
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
import numpy as np

# HuggingFace libraries for DIPPER and Llama
try:
    import torch
    from transformers import (
        T5Tokenizer,  # Use T5Tokenizer instead of AutoTokenizer
        T5ForConditionalGeneration,  # Use T5ForConditionalGeneration instead of AutoModelForSeq2SeqLM
        pipeline,
        AutoTokenizer,
        AutoModelForCausalLM
    )
    from nltk.tokenize import sent_tokenize  # For sentence tokenization as in official implementation
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers/torch not installed. Install with: pip install transformers torch accelerate")

# GGUF model loading for Llama
try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False
    print("Warning: llama-cpp-python not installed. Install with: pip install llama-cpp-python")

# Make sure NLTK sentence tokenizer is available
try:
    import nltk
    nltk.download('punkt', quiet=True)
except:
    pass

# Gemini for prompt-based fallback - Updated to use new Google GenAI API
try:
    from google import genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    print("Warning: google-generativeai not installed. Install with: pip install google-generativeai")

# Llama GGUF model support
try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False
    print("Warning: llama-cpp-python not installed. Install with: pip install llama-cpp-python")

# Local imports
from data_generation.config.type4_config import (
    Type4GenerationConfig,
    Type4ParaphraseMethod,
    DEFAULT_TYPE4_CONFIG,
    validate_type4_config,
    create_memory_efficient_type4_config
)
from data_generation.config.base_model_config import get_api_key

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnvironmentMode(Enum):
    """Environment modes for directory structure."""
    PRODUCTION = "production"
    TEST = "test"

@dataclass
class ParaphraseContext:
    """Context data for Type 4 paraphrasing."""
    original_text: str
    target_length: int
    max_length: int
    method: Type4ParaphraseMethod
    dataset_source: str

@dataclass
class ParaphraseResult:
    """Result of Type 4 paraphrasing."""
    paraphrased_text: Optional[str]
    method_used: Type4ParaphraseMethod
    success: bool
    metadata: Dict[str, Any]

class DipperParaphraser:
    """DIPPER model handler for paraphrasing using official implementation."""
    
    def __init__(self, model_config, dipper_settings: Dict[str, Any]):
        """Initialize DIPPER paraphraser using official implementation."""
        self.config = model_config
        self.settings = dipper_settings
        self.model = None
        self.tokenizer = None
        self.device = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize DIPPER model and tokenizer using official implementation."""
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers and torch required for DIPPER")
        
        try:
            logger.info(f"Step 1: Starting DIPPER model initialization (Official Implementation)")
            logger.info(f"Config details: device={self.config.device}, torch_dtype={self.config.torch_dtype}, model_id={self.config.model_id}")
            
            # Set device
            if self.config.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = self.config.device
            
            logger.info(f"Step 2: Device set to: {self.device}")
            logger.info(f"Initializing DIPPER model on device: {self.device}")
            
            # Load tokenizer - Official DIPPER implementation uses google/t5-v1_1-xxl tokenizer
            try:
                logger.info(f"Step 3: Loading official DIPPER tokenizer: google/t5-v1_1-xxl")
                self.tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')
                logger.info(f"Step 3 SUCCESS: Official DIPPER tokenizer loaded successfully")
            except Exception as tokenizer_error:
                logger.error(f"Step 3 FAILED: Tokenizer loading failed: {str(tokenizer_error)}")
                logger.error(f"Tokenizer error type: {type(tokenizer_error).__name__}")
                raise
            
            # Convert torch_dtype string to actual torch dtype
            try:
                logger.info(f"Step 4: Converting torch_dtype: {self.config.torch_dtype}")
                torch_dtype = torch.float16  # default
                if self.config.torch_dtype:
                    dtype_map = {
                        'float16': torch.float16,
                        'float32': torch.float32,
                        'bfloat16': torch.bfloat16
                    }
                    torch_dtype = dtype_map.get(self.config.torch_dtype, torch.float16)
                logger.info(f"Step 4 SUCCESS: torch_dtype set to: {torch_dtype}")
            except Exception as dtype_error:
                logger.error(f"Step 4 FAILED: torch_dtype conversion failed: {str(dtype_error)}")
                logger.error(f"torch_dtype error type: {type(dtype_error).__name__}")
                raise
            
            # Load model - Official DIPPER implementation
            try:
                logger.info(f"Step 5: Loading official DIPPER model: {self.config.model_id}")
                self.model = T5ForConditionalGeneration.from_pretrained(
                    self.config.model_id,
                    torch_dtype=torch_dtype,
                    trust_remote_code=self.config.trust_remote_code
                )
                
                # Move to device
                if self.device == "cuda" and torch.cuda.is_available():
                    self.model = self.model.cuda()
                
                self.model.eval()  # Set to evaluation mode
                logger.info(f"Step 5 SUCCESS: Official DIPPER model loaded successfully")
            except Exception as model_error:
                logger.error(f"Step 5 FAILED: Model loading failed: {str(model_error)}")
                logger.error(f"Model error type: {type(model_error).__name__}")
                raise
            
            logger.info(f"SUCCESS: DIPPER model initialized completely using official implementation")
            
        except Exception as e:
            logger.error(f"OVERALL FAILURE: DIPPER model initialization failed")
            logger.error(f"Error message: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def paraphrase_text(self, text: str, max_length: int = None) -> str:
        """Paraphrase text using official DIPPER implementation."""
        try:
            # Official DIPPER parameters
            lex_diversity = self.settings.get("lex_diversity", 60)  # 0-100 in steps of 20
            order_diversity = self.settings.get("order_diversity", 0)  # 0-100 in steps of 20
            sent_interval = self.settings.get("sent_interval", 3)
            max_length = max_length or self.settings.get("max_length", 512)
            
            # Validate diversity parameters
            assert lex_diversity in [0, 20, 40, 60, 80, 100], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
            assert order_diversity in [0, 20, 40, 60, 80, 100], "Order diversity must be one of 0, 20, 40, 60, 80, 100."
            
            # Official DIPPER encoding
            lex_code = int(100 - lex_diversity)
            order_code = int(100 - order_diversity)
            
            # Prepare input text
            input_text = " ".join(text.split())
            sentences = sent_tokenize(input_text)
            output_text = ""
            prefix = ""
            
            # Process in sentence intervals as per official implementation
            for sent_idx in range(0, len(sentences), sent_interval):
                curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])
                final_input_text = f"lexical = {lex_code}, order = {order_code}"
                if prefix:
                    final_input_text += f" {prefix}"
                final_input_text += f" <sent> {curr_sent_window} </sent>"
                
                # Tokenize
                final_input = self.tokenizer([final_input_text], return_tensors="pt")
                
                # Move to device
                if self.device == "cuda":
                    final_input = {k: v.cuda() for k, v in final_input.items()}
                
                # Generate
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **final_input,
                        do_sample=self.settings.get("do_sample", True),
                        top_p=self.settings.get("top_p", 0.75),
                        top_k=self.settings.get("top_k", None),
                        max_length=max_length
                    )
                
                # Decode
                outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                current_output = outputs[0]
                prefix += " " + current_output
                output_text += " " + current_output
            
            # Clean and return
            paraphrased = output_text.strip()
            paraphrased = self._clean_paraphrase(paraphrased, text)
            
            return paraphrased
            
        except Exception as e:
            logger.error(f"Official DIPPER paraphrasing failed: {str(e)}")
            raise
    
    def _clean_paraphrase(self, paraphrased: str, original: str) -> str:
        """Clean and validate paraphrased text."""
        # Remove any unwanted prefixes/suffixes
        cleaned = paraphrased.strip()
        
        # Remove common generation artifacts
        artifacts = ['paraphrase:', 'paraphrased:', 'result:', 'output:']
        for artifact in artifacts:
            if cleaned.lower().startswith(artifact):
                cleaned = cleaned[len(artifact):].strip()
        
        # If paraphrase is identical to original, it might be a model issue
        if cleaned.lower() == original.lower():
            logger.warning("DIPPER produced identical paraphrase to original")
        
        return cleaned
    
    def cleanup(self):
        """Clean up model resources."""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("Cleaned up DIPPER model resources")

class GeminiParaphraser:
    """Gemini client for prompt-based paraphrasing using new Google GenAI API."""
    
    def __init__(self, model_config):
        """Initialize Gemini paraphraser."""
        self.config = model_config
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Gemini client with new Google GenAI API."""
        if not HAS_GEMINI:
            raise ImportError("google-generativeai required for Gemini")
        
        # Use enhanced API key getter
        api_key = get_api_key(self.config.api_key_env, required=True)
        if not api_key:
            raise ValueError(f"API key not found for {self.config.api_key_env}")
        
        # Initialize the new Google GenAI client
        self.client = genai.Client(api_key=api_key)
        
        logger.info(f"Initialized Gemini client: {self.config.model_id}")
    
    async def paraphrase_text(self, text: str, target_length: int, max_length: int, prompt_template: str) -> str:
        """Paraphrase text using Gemini with custom prompt and new Google GenAI API."""
        try:
            # Create paraphrasing prompt
            prompt = prompt_template.format(
                text=text,
                target_length=target_length,
                max_length=max_length
            )
            
            # Generate paraphrase using new API format
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.config.model_id,
                contents=prompt
            )
            
            paraphrased = response.text.strip()
            
            # Clean result
            paraphrased = self._clean_paraphrase(paraphrased)
            
            return paraphrased
            
        except Exception as e:
            logger.error(f"Gemini paraphrasing failed: {str(e)}")
            raise
    
    async def paraphrase_text_with_retry(self, text: str, target_length: int, max_length: int, 
                                       prompt_template: str, failure_reason: str) -> str:
        """Paraphrase text using Gemini with custom prompt including failure reason."""
        try:
            # Create paraphrasing prompt with failure context
            prompt = prompt_template.format(
                text=text,
                target_length=target_length,
                max_length=max_length,
                failure_reason=failure_reason
            )
            
            # Generate paraphrase using new API format
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.config.model_id,
                contents=prompt
            )
            
            paraphrased = response.text.strip()
            
            # Clean result
            paraphrased = self._clean_paraphrase(paraphrased)
            
            return paraphrased
            
        except Exception as e:
            logger.error(f"Gemini paraphrasing with retry failed: {str(e)}")
            raise
    
    def _clean_paraphrase(self, text: str) -> str:
        """Clean generated paraphrase."""
        cleaned = text.strip()
        
        # Remove common prefixes
        prefixes = [
            'paraphrased text:', 'paraphrase:', 'result:', 'output:', 
            'here is the paraphrase:', 'paraphrased version:'
        ]
        
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        return cleaned

class LlamaParaphraser:
    """Llama GGUF model handler for paraphrasing using llama-cpp-python."""
    
    def __init__(self, model_config, llama_settings: Dict[str, Any]):
        """Initialize Llama paraphraser using GGUF model."""
        self.config = model_config
        self.settings = llama_settings
        self.model = None
        self.device = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Llama GGUF model."""
        if not HAS_LLAMA_CPP:
            raise ImportError("llama-cpp-python required for Llama GGUF models")
        
        try:
            logger.info(f"Step 1: Starting Llama GGUF model initialization")
            logger.info(f"Config details: device={self.config.device}, model_id={self.config.model_id}")
            
            # Set device
            if self.config.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = self.config.device
            
            logger.info(f"Step 2: Device set to: {self.device}")
            logger.info(f"Initializing Llama GGUF model on device: {self.device}")
            
            # Load GGUF model
            try:
                logger.info(f"Step 3: Loading Llama GGUF model: {self.config.model_id}")
                
                # Configure model parameters
                model_kwargs = {
                    "model_path": self.config.model_id,  # This will be the local path or HuggingFace model
                    "n_ctx": self.settings.get("max_length", 300),
                    "n_threads": 4,  # Adjust based on CPU cores
                    "verbose": False
                }
                
                # Add GPU support if available
                if self.device == "cuda" and torch.cuda.is_available():
                    model_kwargs["n_gpu_layers"] = -1  # Use all GPU layers
                    logger.info("Step 3: Using GPU acceleration")
                else:
                    logger.info("Step 3: Using CPU inference")
                
                self.model = Llama(**model_kwargs)
                logger.info(f"Step 3 SUCCESS: Llama GGUF model loaded successfully")
                
            except Exception as model_error:
                logger.error(f"Step 3 FAILED: Model loading failed: {str(model_error)}")
                logger.error(f"Model error type: {type(model_error).__name__}")
                raise
            
            logger.info(f"SUCCESS: Llama GGUF model initialized completely")
            
        except Exception as e:
            logger.error(f"OVERALL FAILURE: Llama GGUF model initialization failed")
            logger.error(f"Error message: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def paraphrase_text(self, text: str, prompt_template: str, max_length: int = None) -> str:
        """Paraphrase text using Llama GGUF model."""
        try:
            max_length = max_length or self.settings.get("max_length", 300)
            
            # Format the prompt
            formatted_prompt = prompt_template.format(
                text=text,
                target_length=len(text),
                max_length=max_length
            )
            
            # Generate paraphrase using Llama
            response = self.model(
                formatted_prompt,
                max_tokens=max_length,
                temperature=self.settings.get("temperature", 0.7),
                top_p=self.settings.get("top_p", 0.9),
                top_k=self.settings.get("top_k", 40),
                repeat_penalty=self.settings.get("repetition_penalty", 1.1),
                stop=["<|eot_id|>", "<|end_of_text|>", "\n\n"],
                echo=False
            )
            
            # Extract the generated text
            if response and "choices" in response and len(response["choices"]) > 0:
                paraphrased = response["choices"][0]["text"].strip()
            else:
                paraphrased = ""
            
            # Clean and return
            paraphrased = self._clean_paraphrase(paraphrased, text)
            
            return paraphrased
            
        except Exception as e:
            logger.error(f"Llama GGUF paraphrasing failed: {str(e)}")
            raise
    
    def _clean_paraphrase(self, paraphrased: str, original: str) -> str:
        """Clean and validate paraphrased text."""
        # Remove any unwanted prefixes/suffixes
        cleaned = paraphrased.strip()
        
        # Remove common generation artifacts
        artifacts = [
            'paraphrase:', 'paraphrased:', 'result:', 'output:',
            'here is the paraphrase:', 'paraphrased version:',
            'assistant:', 'user:', 'system:'
        ]
        for artifact in artifacts:
            if cleaned.lower().startswith(artifact):
                cleaned = cleaned[len(artifact):].strip()
        
        # Remove special tokens
        special_tokens = ['<|eot_id|>', '<|end_of_text|>', '<|begin_of_text|>']
        for token in special_tokens:
            cleaned = cleaned.replace(token, '').strip()
        
        # If paraphrase is identical to original, it might be a model issue
        if cleaned.lower() == original.lower():
            logger.warning("Llama produced identical paraphrase to original")
        
        return cleaned
    
    def cleanup(self):
        """Clean up model resources."""
        if self.model is not None:
            del self.model
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("Cleaned up Llama GGUF model resources")

class Type4Generator:
    """Generator for Type 4 (LLM-paraphrased original text)."""
    
    def __init__(self, config: Optional[Type4GenerationConfig] = None, 
                 environment_mode: EnvironmentMode = EnvironmentMode.PRODUCTION):
        """Initialize the Type 4 generator."""
        self.config = config or DEFAULT_TYPE4_CONFIG
        self.environment_mode = environment_mode
        
        # Validate configuration
        if not validate_type4_config(self.config):
            raise ValueError("Invalid Type 4 configuration")
        
        # Initialize paraphrasers
        self.dipper_paraphraser = None
        self.gemini_paraphraser = None
        self.llama_paraphraser = None
        
        # Statistics tracking
        self.stats = {
            "total_processed": 0,
            "successful_paraphrases": 0,
            "failed_paraphrases": 0,
            "length_violations": 0,
            "method_usage": {
                "dipper": 0,
                "prompt_based": 0,
                "llama": 0
            },
            "start_time": None,
            "end_time": None
        }
    
    def get_directory_structure(self, base_output_dir: str, method: Type4ParaphraseMethod, timestamp: str) -> str:
        """
        Get the appropriate directory structure based on environment mode and method.
        
        Returns:
            Final output directory path
        """
        if self.environment_mode == EnvironmentMode.TEST:
            # Test environment: data/test/type4_generation_test/method_based_timestamp/
            method_name = method.value if method else "auto"
            test_output_dir = f"{base_output_dir}/{method_name}_based_{timestamp}"
            return test_output_dir
        else:
            # Production environment: data/generated/timestamp/
            prod_output_dir = f"{base_output_dir}/{timestamp}"
            return prod_output_dir
    
    def get_method_column_name(self, method: Type4ParaphraseMethod) -> str:
        """Get the appropriate column name for the paraphrasing method."""
        method_name_map = {
            Type4ParaphraseMethod.DIPPER: "DIPPER_based",
            Type4ParaphraseMethod.PROMPT_BASED: "Prompt_based",
            Type4ParaphraseMethod.LLAMA: "Llama_based"
        }
        method_name = method_name_map.get(method, method.value)
        return f"llm_paraphrased_original_text({method_name})"
    
    def _initialize_dipper(self):
        """Initialize DIPPER paraphraser (lazy loading)."""
        if self.dipper_paraphraser is None:
            try:
                self.dipper_paraphraser = DipperParaphraser(
                    self.config.primary_model,
                    self.config.dipper_settings
                )
                logger.info("DIPPER paraphraser initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize DIPPER: {str(e)}")
                raise
    
    def _initialize_gemini(self):
        """Initialize Gemini paraphraser (lazy loading)."""
        if self.gemini_paraphraser is None:
            try:
                self.gemini_paraphraser = GeminiParaphraser(self.config.fallback_model)
                logger.info("Gemini paraphraser initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {str(e)}")
                raise
    
    def _initialize_llama(self):
        """Initialize Llama paraphraser (lazy loading)."""
        if self.llama_paraphraser is None:
            try:
                self.llama_paraphraser = LlamaParaphraser(
                    self.config.llama_model,
                    self.config.llama_settings
                )
                logger.info("Llama paraphraser initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Llama: {str(e)}")
                raise
    
    def calculate_length_constraints(self, original_text: str) -> Tuple[int, int]:
        """Calculate target and maximum length for paraphrased text."""
        original_length = len(original_text)
        target_length = original_length
        max_length = int(original_length * self.config.max_length_multiplier)
        return target_length, max_length
    
    def choose_paraphrase_method(self, context: ParaphraseContext) -> Type4ParaphraseMethod:
        """Choose paraphrasing method based on context and availability."""
        # If specific method requested, try to use it
        if context.method != Type4ParaphraseMethod.DIPPER:
            return context.method
        
        # Default to DIPPER for primary paraphrasing
        return Type4ParaphraseMethod.DIPPER
    
    def validate_paraphrase(self, original: str, paraphrased: str, context: ParaphraseContext) -> Tuple[bool, str]:
        """Validate paraphrased text quality."""
        if not paraphrased or len(paraphrased.strip()) < self.config.min_length_threshold:
            return False, "Paraphrased text too short"
        
        if len(paraphrased) > context.max_length:
            return False, f"Paraphrased text too long ({len(paraphrased)} > {context.max_length})"
        
        # Check if text is too similar (simple check)
        if paraphrased.lower().strip() == original.lower().strip():
            return False, "Paraphrased text identical to original"
        
        # Check minimum length requirement
        min_length = context.target_length * (1 - self.config.length_tolerance)
        if len(paraphrased) < min_length:
            return False, f"Paraphrased text below minimum length ({len(paraphrased)} < {min_length})"
        
        return True, "Valid"
    
    async def paraphrase_with_dipper(self, context: ParaphraseContext) -> ParaphraseResult:
        """Paraphrase text using DIPPER model with retry logic."""
        last_failure_reason = None
        
        for attempt in range(self.config.max_retries + 1):  # +1 for initial attempt
            try:
                # Initialize DIPPER if needed
                self._initialize_dipper()
                
                # For DIPPER, we can't modify the input with failure context,
                # but we can adjust parameters slightly for retries
                if attempt > 0:
                    logger.info(f"DIPPER retry attempt {attempt}/{self.config.max_retries}")
                    # Slightly modify generation parameters for retry
                    original_settings = self.dipper_paraphraser.settings.copy()
                    self.dipper_paraphraser.settings["temperature"] = min(1.0, original_settings.get("temperature", 0.8) + 0.1 * attempt)
                    self.dipper_paraphraser.settings["top_p"] = min(0.95, original_settings.get("top_p", 0.9) + 0.05 * attempt)
                
                # Paraphrase using DIPPER
                paraphrased = await asyncio.to_thread(
                    self.dipper_paraphraser.paraphrase_text,
                    context.original_text,
                    context.max_length
                )
                
                # Validate result - check for null/empty first
                if not paraphrased or not paraphrased.strip():
                    failure_reason = "Empty or null paraphrased text"
                    last_failure_reason = failure_reason
                    
                    if attempt < self.config.max_retries:
                        logger.warning(f"DIPPER attempt {attempt + 1} failed: {failure_reason}. Retrying...")
                        await asyncio.sleep(self.config.retry_delay)
                        continue
                    else:
                        metadata = {
                            "method": "dipper",
                            "original_length": len(context.original_text),
                            "paraphrased_length": 0,
                            "validation_status": failure_reason,
                            "model_used": self.config.primary_model.model_id,
                            "dataset_source": context.dataset_source,
                            "retry_count": attempt,
                            "total_attempts": attempt + 1,
                            "failure_reason": failure_reason,
                            "final_attempt": True
                        }
                        return ParaphraseResult(
                            paraphrased_text=None,
                            method_used=Type4ParaphraseMethod.DIPPER,
                            success=False,
                            metadata=metadata
                        )
                
                # Validate result quality
                is_valid, validation_msg = self.validate_paraphrase(
                    context.original_text, paraphrased, context
                )
                
                metadata = {
                    "method": "dipper",
                    "original_length": len(context.original_text),
                    "paraphrased_length": len(paraphrased),
                    "validation_status": validation_msg,
                    "model_used": self.config.primary_model.model_id,
                    "dataset_source": context.dataset_source,
                    "retry_count": attempt,
                    "total_attempts": attempt + 1
                }
                
                if last_failure_reason:
                    metadata["previous_failures"] = last_failure_reason
                
                if is_valid:
                    self.stats["method_usage"]["dipper"] += 1
                    if attempt > 0:
                        metadata["succeeded_on_retry"] = True
                        logger.info(f"DIPPER succeeded on retry attempt {attempt}")
                    return ParaphraseResult(
                        paraphrased_text=paraphrased,
                        method_used=Type4ParaphraseMethod.DIPPER,
                        success=True,
                        metadata=metadata
                    )
                else:
                    # Failed validation
                    last_failure_reason = validation_msg
                    
                    if attempt < self.config.max_retries:
                        logger.warning(f"DIPPER attempt {attempt + 1} failed: {validation_msg}. Retrying...")
                        await asyncio.sleep(self.config.retry_delay)
                        continue
                    else:
                        metadata["failure_reason"] = validation_msg
                        metadata["final_attempt"] = True
                        return ParaphraseResult(
                            paraphrased_text=None,
                            method_used=Type4ParaphraseMethod.DIPPER,
                            success=False,
                            metadata=metadata
                        )
                        
            except Exception as e:
                error_msg = str(e)
                last_failure_reason = error_msg
                
                if attempt < self.config.max_retries:
                    logger.warning(f"DIPPER attempt {attempt + 1} failed with exception: {error_msg}. Retrying...")
                    await asyncio.sleep(self.config.retry_delay)
                    continue
                else:
                    logger.error(f"DIPPER paraphrasing failed after {attempt + 1} attempts: {error_msg}")
                    return ParaphraseResult(
                        paraphrased_text=None,
                        method_used=Type4ParaphraseMethod.DIPPER,
                        success=False,
                        metadata={
                            "failure_reason": error_msg, 
                            "method": "dipper",
                            "retry_count": attempt,
                            "total_attempts": attempt + 1,
                            "final_attempt": True
                        }
                    )
        
        # Should not reach here, but safety fallback
        return ParaphraseResult(
            paraphrased_text=None,
            method_used=Type4ParaphraseMethod.DIPPER,
            success=False,
            metadata={"failure_reason": "Unexpected retry loop exit", "method": "dipper"}
        )

    async def paraphrase_with_prompt(self, context: ParaphraseContext) -> ParaphraseResult:
        """Paraphrase text using Gemini prompt-based method with retry logic."""
        last_failure_reason = None
        
        for attempt in range(self.config.max_retries + 1):  # +1 for initial attempt
            try:
                # Initialize Gemini if needed
                self._initialize_gemini()
                
                # Choose prompt template based on attempt
                if attempt == 0 or not self.config.enable_retry_with_context:
                    prompt_template = self.config.prompt_based_template
                    # Use regular paraphrase method
                    paraphrased = await self.gemini_paraphraser.paraphrase_text(
                        context.original_text,
                        context.target_length,
                        context.max_length,
                        prompt_template
                    )
                else:
                    prompt_template = self.config.retry_prompt_template
                    logger.info(f"Gemini retry attempt {attempt}/{self.config.max_retries} with failure context")
                    # Use retry method with failure reason
                    paraphrased = await self.gemini_paraphraser.paraphrase_text_with_retry(
                        context.original_text,
                        context.target_length,
                        context.max_length,
                        prompt_template,
                        last_failure_reason
                    )
                
                # Validate result - check for null/empty first
                if not paraphrased or not paraphrased.strip():
                    failure_reason = "Empty or null paraphrased text"
                    last_failure_reason = failure_reason
                    
                    if attempt < self.config.max_retries:
                        logger.warning(f"Gemini attempt {attempt + 1} failed: {failure_reason}. Retrying...")
                        await asyncio.sleep(self.config.retry_delay)
                        continue
                    else:
                        metadata = {
                            "method": "prompt_based",
                            "original_length": len(context.original_text),
                            "paraphrased_length": 0,
                            "validation_status": failure_reason,
                            "model_used": self.config.fallback_model.model_id,
                            "dataset_source": context.dataset_source,
                            "retry_count": attempt,
                            "total_attempts": attempt + 1,
                            "failure_reason": failure_reason,
                            "final_attempt": True
                        }
                        return ParaphraseResult(
                            paraphrased_text=None,
                            method_used=Type4ParaphraseMethod.PROMPT_BASED,
                            success=False,
                            metadata=metadata
                        )
                
                # Validate result quality
                is_valid, validation_msg = self.validate_paraphrase(
                    context.original_text, paraphrased, context
                )
                
                metadata = {
                    "method": "prompt_based",
                    "original_length": len(context.original_text),
                    "paraphrased_length": len(paraphrased),
                    "validation_status": validation_msg,
                    "model_used": self.config.fallback_model.model_id,
                    "dataset_source": context.dataset_source,
                    "retry_count": attempt,
                    "total_attempts": attempt + 1
                }
                
                if last_failure_reason:
                    metadata["previous_failures"] = last_failure_reason
                
                if is_valid:
                    self.stats["method_usage"]["prompt_based"] += 1
                    if attempt > 0:
                        metadata["succeeded_on_retry"] = True
                        logger.info(f"Gemini succeeded on retry attempt {attempt}")
                    return ParaphraseResult(
                        paraphrased_text=paraphrased,
                        method_used=Type4ParaphraseMethod.PROMPT_BASED,
                        success=True,
                        metadata=metadata
                    )
                else:
                    # Failed validation
                    last_failure_reason = validation_msg
                    
                    if attempt < self.config.max_retries:
                        logger.warning(f"Gemini attempt {attempt + 1} failed: {validation_msg}. Retrying...")
                        await asyncio.sleep(self.config.retry_delay)
                        continue
                    else:
                        metadata["failure_reason"] = validation_msg
                        metadata["final_attempt"] = True
                        return ParaphraseResult(
                            paraphrased_text=None,
                            method_used=Type4ParaphraseMethod.PROMPT_BASED,
                            success=False,
                            metadata=metadata
                        )
                        
            except Exception as e:
                error_msg = str(e)
                last_failure_reason = error_msg
                
                if attempt < self.config.max_retries:
                    logger.warning(f"Gemini attempt {attempt + 1} failed with exception: {error_msg}. Retrying...")
                    await asyncio.sleep(self.config.retry_delay)
                    continue
                else:
                    logger.error(f"Prompt-based paraphrasing failed after {attempt + 1} attempts: {error_msg}")
                    return ParaphraseResult(
                        paraphrased_text=None,
                        method_used=Type4ParaphraseMethod.PROMPT_BASED,
                        success=False,
                        metadata={
                            "failure_reason": error_msg, 
                            "method": "prompt_based",
                            "retry_count": attempt,
                            "total_attempts": attempt + 1,
                            "final_attempt": True
                        }
                    )
        
        # Should not reach here, but safety fallback
        return ParaphraseResult(
            paraphrased_text=None,
            method_used=Type4ParaphraseMethod.PROMPT_BASED,
            success=False,
            metadata={"failure_reason": "Unexpected retry loop exit", "method": "prompt_based"}
        )
    
    async def paraphrase_with_llama(self, context: ParaphraseContext) -> ParaphraseResult:
        """Paraphrase text using Llama GGUF model with retry logic."""
        last_failure_reason = None
        
        for attempt in range(self.config.max_retries + 1):  # +1 for initial attempt
            try:
                # Initialize Llama if needed
                self._initialize_llama()
                
                # For Llama, we can adjust parameters slightly for retries
                if attempt > 0:
                    logger.info(f"Llama retry attempt {attempt}/{self.config.max_retries}")
                    # Slightly modify generation parameters for retry
                    original_settings = self.llama_paraphraser.settings.copy()
                    self.llama_paraphraser.settings["temperature"] = min(1.0, original_settings.get("temperature", 0.7) + 0.1 * attempt)
                    self.llama_paraphraser.settings["top_p"] = min(0.95, original_settings.get("top_p", 0.9) + 0.05 * attempt)
                
                # Paraphrase using Llama
                paraphrased = await asyncio.to_thread(
                    self.llama_paraphraser.paraphrase_text,
                    context.original_text,
                    self.config.llama_prompt_template,
                    context.max_length
                )
                
                # Validate result - check for null/empty first
                if not paraphrased or not paraphrased.strip():
                    failure_reason = "Empty or null paraphrased text"
                    last_failure_reason = failure_reason
                    
                    if attempt < self.config.max_retries:
                        logger.warning(f"Llama attempt {attempt + 1} failed: {failure_reason}. Retrying...")
                        await asyncio.sleep(self.config.retry_delay)
                        continue
                    else:
                        metadata = {
                            "method": "llama",
                            "original_length": len(context.original_text),
                            "paraphrased_length": 0,
                            "validation_status": failure_reason,
                            "model_used": self.config.llama_model.model_id,
                            "dataset_source": context.dataset_source,
                            "retry_count": attempt,
                            "total_attempts": attempt + 1,
                            "failure_reason": failure_reason,
                            "final_attempt": True
                        }
                        return ParaphraseResult(
                            paraphrased_text=None,
                            method_used=Type4ParaphraseMethod.LLAMA,
                            success=False,
                            metadata=metadata
                        )
                
                # Validate result quality
                is_valid, validation_msg = self.validate_paraphrase(
                    context.original_text, paraphrased, context
                )
                
                metadata = {
                    "method": "llama",
                    "original_length": len(context.original_text),
                    "paraphrased_length": len(paraphrased),
                    "validation_status": validation_msg,
                    "model_used": self.config.llama_model.model_id,
                    "dataset_source": context.dataset_source,
                    "retry_count": attempt,
                    "total_attempts": attempt + 1
                }
                
                if last_failure_reason:
                    metadata["previous_failures"] = last_failure_reason
                
                if is_valid:
                    self.stats["method_usage"]["llama"] += 1
                    if attempt > 0:
                        metadata["succeeded_on_retry"] = True
                        logger.info(f"Llama succeeded on retry attempt {attempt}")
                    return ParaphraseResult(
                        paraphrased_text=paraphrased,
                        method_used=Type4ParaphraseMethod.LLAMA,
                        success=True,
                        metadata=metadata
                    )
                else:
                    # Failed validation
                    last_failure_reason = validation_msg
                    
                    if attempt < self.config.max_retries:
                        logger.warning(f"Llama attempt {attempt + 1} failed: {validation_msg}. Retrying...")
                        await asyncio.sleep(self.config.retry_delay)
                        continue
                    else:
                        metadata["failure_reason"] = validation_msg
                        metadata["final_attempt"] = True
                        return ParaphraseResult(
                            paraphrased_text=None,
                            method_used=Type4ParaphraseMethod.LLAMA,
                            success=False,
                            metadata=metadata
                        )
                        
            except Exception as e:
                error_msg = str(e)
                last_failure_reason = error_msg
                
                if attempt < self.config.max_retries:
                    logger.warning(f"Llama attempt {attempt + 1} failed with exception: {error_msg}. Retrying...")
                    await asyncio.sleep(self.config.retry_delay)
                    continue
                else:
                    logger.error(f"Llama paraphrasing failed after {attempt + 1} attempts: {error_msg}")
                    return ParaphraseResult(
                        paraphrased_text=None,
                        method_used=Type4ParaphraseMethod.LLAMA,
                        success=False,
                        metadata={
                            "failure_reason": error_msg, 
                            "method": "llama",
                            "retry_count": attempt,
                            "total_attempts": attempt + 1,
                            "final_attempt": True
                        }
                    )
        
        # Should not reach here, but safety fallback
        return ParaphraseResult(
            paraphrased_text=None,
            method_used=Type4ParaphraseMethod.LLAMA,
            success=False,
            metadata={"failure_reason": "Unexpected retry loop exit", "method": "llama"}
        )
    
    async def paraphrase_text(self, 
                             row: pd.Series, 
                             method: Type4ParaphraseMethod = None) -> ParaphraseResult:
        """
        Paraphrase Type 1 text using specified method.
        
        Args:
            row: DataFrame row containing original text
            method: Paraphrasing method to use (None for default)
            
        Returns:
            ParaphraseResult with paraphrased text and metadata
        """
        start_time = time.time()
        
        try:
            # Get original text
            original_text = row.get('human_original_text', '')
            if not original_text or pd.isna(original_text):
                raise ValueError("Missing or empty original text")
            
            # Calculate length constraints
            target_length, max_length = self.calculate_length_constraints(original_text)
            
            # Create paraphrase context
            context = ParaphraseContext(
                original_text=original_text,
                target_length=target_length,
                max_length=max_length,
                method=method or self.config.default_method,
                dataset_source=row.get('dataset_source', 'unknown')
            )
            
            # Choose and execute paraphrasing method
            chosen_method = self.choose_paraphrase_method(context)
            context.method = chosen_method
            
            # Try primary method first
            if chosen_method == Type4ParaphraseMethod.DIPPER:
                result = await self.paraphrase_with_dipper(context)
                
                # If DIPPER fails, try fallback
                if not result.success and self.config.fallback_method == Type4ParaphraseMethod.PROMPT_BASED:
                    logger.warning("DIPPER failed, trying Gemini fallback")
                    context.method = Type4ParaphraseMethod.PROMPT_BASED
                    result = await self.paraphrase_with_prompt(context)
                    if result.success:
                        result.metadata["used_fallback"] = True
            
            elif chosen_method == Type4ParaphraseMethod.PROMPT_BASED:
                result = await self.paraphrase_with_prompt(context)
            
            elif chosen_method == Type4ParaphraseMethod.LLAMA:
                result = await self.paraphrase_with_llama(context)
                
                # If Llama fails, try fallback
                if not result.success and self.config.fallback_method == Type4ParaphraseMethod.PROMPT_BASED:
                    logger.warning("Llama failed, trying Gemini fallback")
                    context.method = Type4ParaphraseMethod.PROMPT_BASED
                    result = await self.paraphrase_with_prompt(context)
                    if result.success:
                        result.metadata["used_fallback"] = True
            
            else:
                raise ValueError(f"Unknown paraphrasing method: {chosen_method}")
            
            # Add timing information
            result.metadata["paraphrase_time"] = time.time() - start_time
            
            # Update statistics
            if result.success:
                self.stats["successful_paraphrases"] += 1
            else:
                self.stats["failed_paraphrases"] += 1
                if "length" in result.metadata.get("failure_reason", "").lower():
                    self.stats["length_violations"] += 1
            
            self.stats["total_processed"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Type 4 paraphrasing failed: {str(e)}")
            self.stats["failed_paraphrases"] += 1
            self.stats["total_processed"] += 1
            
            return ParaphraseResult(
                paraphrased_text=None,
                method_used=method or self.config.default_method,
                success=False,
                metadata={
                    "failure_reason": str(e),
                    "paraphrase_time": time.time() - start_time
                }
            )
    
    async def process_batch(self, 
                           batch_data: List[Tuple[int, pd.Series]], 
                           method: Type4ParaphraseMethod = None) -> List[Tuple[int, ParaphraseResult]]:
        """Process a batch of texts for paraphrasing."""
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        async def process_single(idx: int, row: pd.Series) -> Tuple[int, ParaphraseResult]:
            async with semaphore:
                result = await self.paraphrase_text(row, method)
                return idx, result
        
        tasks = [process_single(idx, row) for idx, row in batch_data]
        return await asyncio.gather(*tasks)
    
    def filter_target_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data to target datasets needing Type 4 paraphrasing."""
        # Filter to target datasets
        target_datasets = [ds.lower() for ds in self.config.target_datasets]
        mask = df['dataset_source'].str.lower().isin(target_datasets)
        
        # Filter to samples missing LLM-paraphrased original text
        mask &= (df['llm_paraphrased_original_text'].isna() | (df['llm_paraphrased_original_text'] == ''))
        
        # Ensure we have required original text
        mask &= df['human_original_text'].notna() & (df['human_original_text'] != '')
        
        filtered_df = df[mask].copy()
        logger.info(f"Filtered {len(filtered_df)} samples requiring Type 4 paraphrasing from {len(df)} total")
        
        return filtered_df
    
    async def generate_for_dataset(self, 
                                  df: pd.DataFrame, 
                                  method: Type4ParaphraseMethod = None,
                                  output_dir: str = "data/generated") -> pd.DataFrame:
        """Generate Type 4 paraphrases for the entire dataset."""
        self.stats["start_time"] = datetime.now()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Get appropriate directory structure
        final_output_dir = self.get_directory_structure(output_dir, method or self.config.default_method, timestamp)
        
        # Filter target data
        target_df = self.filter_target_data(df)
        
        if target_df.empty:
            logger.info("No samples require Type 4 paraphrasing")
            return df
        
        method_name = method.value if method else self.config.default_method.value
        logger.info(f"Starting Type 4 paraphrasing for {len(target_df)} samples using method: {method_name}")
        logger.info(f"Environment mode: {self.environment_mode.value}")
        logger.info(f"Final output directory: {final_output_dir}")
        
        # Prepare results
        results_df = df.copy()
        
        # Get the method-specific column name
        method_column = self.get_method_column_name(method or self.config.default_method)
        
        # Initialize columns with proper data types to avoid pandas warnings
        if method_column not in results_df.columns:
            results_df[method_column] = pd.Series(dtype='object')
        
        paraphrase_metadata = []
        
        # Process in batches
        batch_size = self.config.batch_size
        total_batches = (len(target_df) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(target_df))
            
            batch_df = target_df.iloc[start_idx:end_idx]
            batch_data = [(row.name, row) for _, row in batch_df.iterrows()]
            
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_data)} samples)")
            
            # Process batch
            batch_results = await self.process_batch(batch_data, method)
            
            # Update results
            for idx, result in batch_results:
                if result.success and result.paraphrased_text:
                    results_df.at[idx, method_column] = result.paraphrased_text
                
                # Store metadata
                result.metadata['batch_idx'] = batch_idx
                result.metadata['sample_idx'] = idx
                paraphrase_metadata.append(result.metadata)
            
            # Save intermediate results every 10 batches
            if (batch_idx + 1) % 10 == 0:
                await self._save_intermediate_results(results_df, paraphrase_metadata, batch_idx + 1, final_output_dir)
            
            # Brief pause between batches
            await asyncio.sleep(0.1)
        
        self.stats["end_time"] = datetime.now()
        
        # Save final results
        await self._save_final_results(results_df, paraphrase_metadata, method_name, final_output_dir)
        
        # Clean up resources
        self._cleanup_resources()
        
        # Log final statistics
        self._log_final_statistics()
        
        return results_df
    
    async def _save_intermediate_results(self, df: pd.DataFrame, metadata: List[Dict], batch_num: int, output_dir: str):
        """Save intermediate results during processing."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save DataFrame checkpoint
            checkpoint_file = output_path / f"type4_checkpoint_batch_{batch_num}.csv"
            df.to_csv(checkpoint_file, index=False)
            
            # Save metadata checkpoint
            metadata_file = output_path / f"type4_metadata_batch_{batch_num}.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"Saved checkpoint after batch {batch_num}")
        except Exception as e:
            logger.warning(f"Failed to save intermediate results: {str(e)}")
    
    async def _save_final_results(self, df: pd.DataFrame, metadata: List[Dict], method_name: str, output_dir: str):
        """Save final paraphrasing results."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save CSV
            csv_file = output_path / f"unified_padben_with_type4_{method_name}_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            
            # Save JSON
            json_file = output_path / f"unified_padben_with_type4_{method_name}_{timestamp}.json"
            df.to_json(json_file, orient='records', indent=2)
            
            # Save complete metadata
            metadata_file = output_path / f"type4_paraphrasing_metadata_{method_name}_{timestamp}.json"
            full_metadata = {
                "generation_config": asdict(self.config),
                "method_used": method_name,
                "environment_mode": self.environment_mode.value,
                "statistics": self.stats,
                "sample_metadata": metadata
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(full_metadata, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"Saved final results:")
            logger.info(f"  CSV: {csv_file}")
            logger.info(f"  JSON: {json_file}")
            logger.info(f"  Metadata: {metadata_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save final results: {str(e)}")
    
    def _cleanup_resources(self):
        """Clean up model resources."""
        if self.dipper_paraphraser:
            self.dipper_paraphraser.cleanup()
        
        if self.llama_paraphraser:
            self.llama_paraphraser.cleanup()
        
        # Clear any remaining GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        logger.info("Cleaned up Type 4 generation resources")
    
    def _log_final_statistics(self):
        """Log final paraphrasing statistics."""
        stats = self.stats
        total_time = (stats["end_time"] - stats["start_time"]).total_seconds() if stats["end_time"] and stats["start_time"] else 0
        
        logger.info("=" * 60)
        logger.info("TYPE 4 PARAPHRASING COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total processed: {stats['total_processed']}")
        logger.info(f"Successful paraphrases: {stats['successful_paraphrases']}")
        logger.info(f"Failed paraphrases: {stats['failed_paraphrases']}")
        logger.info(f"Length violations: {stats['length_violations']}")
        logger.info(f"Success rate: {stats['successful_paraphrases'] / max(stats['total_processed'], 1) * 100:.1f}%")
        logger.info("Method usage:")
        logger.info(f"  DIPPER: {stats['method_usage']['dipper']}")
        logger.info(f"  Prompt-based: {stats['method_usage']['prompt_based']}")
        logger.info(f"  Llama: {stats['method_usage']['llama']}")
        logger.info(f"Total time: {total_time:.2f} seconds")
        if stats['successful_paraphrases'] > 0:
            logger.info(f"Average time per paraphrase: {total_time / stats['successful_paraphrases']:.2f} seconds")
        logger.info("=" * 60)

    async def retry_single_record_paraphrasing(self, 
                                             row: pd.Series, 
                                             method: Type4ParaphraseMethod,
                                             max_retries: int = 3) -> ParaphraseResult:
        """
        Retry paraphrasing for a single record until successful or max retries reached.
        
        Args:
            row: DataFrame row containing original text
            method: Paraphrasing method to use
            max_retries: Maximum number of retry attempts
            
        Returns:
            ParaphraseResult with paraphrased text and metadata
        """
        sample_idx = row.get('idx', row.name if hasattr(row, 'name') else -1)
        
        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries + 1} for sample {sample_idx}")
                
                result = await self.paraphrase_text(row, method)
                
                if result.success and result.paraphrased_text and result.paraphrased_text.strip():
                    logger.info(f"✅ Successfully paraphrased text for sample {sample_idx} on attempt {attempt + 1}")
                    result.metadata["retry_attempt"] = attempt + 1
                    result.metadata["retry_successful"] = True
                    return result
                else:
                    failure_reason = result.metadata.get("failure_reason", "Unknown failure")
                    logger.warning(f"⚠️ Attempt {attempt + 1} failed for sample {sample_idx}: {failure_reason}")
                    
                    if attempt < max_retries:
                        # Brief delay before retry to avoid rate limiting
                        await asyncio.sleep(1.0)
                    
            except Exception as e:
                logger.error(f"❌ Exception during attempt {attempt + 1} for sample {sample_idx}: {str(e)}")
                if attempt < max_retries:
                    await asyncio.sleep(1.0)
        
        # All attempts failed
        logger.error(f"❌ All {max_retries + 1} attempts failed for sample {sample_idx}")
        return ParaphraseResult(
            paraphrased_text=None,
            method_used=method,
            success=False,
            metadata={
                "failure_reason": f"Failed after {max_retries + 1} attempts",
                "retry_attempt": max_retries + 1,
                "retry_successful": False,
                "sample_idx": sample_idx
            }
        )

    async def process_null_records_individually(self,
                                              df: pd.DataFrame,
                                              target_field: str,
                                              method: Type4ParaphraseMethod,
                                              max_retries: int = 3) -> pd.DataFrame:
        """
        Process null records individually with immediate retry until each record is filled.
        
        This approach ensures that each null record is processed individually and retried
        until successful, rather than processing the entire batch multiple times.
        
        Args:
            df: DataFrame with null records to process
            target_field: Target field name to fill
            method: Paraphrasing method to use
            max_retries: Maximum retry attempts per record
            
        Returns:
            Updated DataFrame with filled records
        """
        logger.info(f"🔄 Starting individual null record processing for {target_field} (max {max_retries} retries per record)")
        
        updated_df = df.copy()
        
        # Find records with null values in the target field
        null_mask = updated_df[target_field].isnull() | (updated_df[target_field] == '')
        null_records = updated_df[null_mask].copy()
        
        if len(null_records) == 0:
            logger.info("✅ No null records found")
            return updated_df
        
        logger.info(f"🎯 Found {len(null_records)} records with null values")
        
        # Process each null record individually
        successful_fills = 0
        failed_fills = 0
        
        # Progress tracking
        try:
            from tqdm import tqdm
            progress_bar = tqdm(total=len(null_records), desc=f"Processing {target_field} null records")
            HAS_TQDM = True
        except ImportError:
            HAS_TQDM = False
        
        for idx, row in null_records.iterrows():
            try:
                # Retry until successful or max attempts reached
                result = await self.retry_single_record_paraphrasing(row, method, max_retries)
                
                if result.success and result.paraphrased_text:
                    # Update the DataFrame with the successful result
                    updated_df.at[idx, target_field] = result.paraphrased_text
                    successful_fills += 1
                    
                    if HAS_TQDM:
                        progress_bar.set_postfix({
                            'Success': successful_fills,
                            'Failed': failed_fills,
                            'Rate': f'{(successful_fills/(successful_fills+failed_fills)*100):.1f}%' if (successful_fills+failed_fills) > 0 else '0%'
                        })
                else:
                    failed_fills += 1
                    logger.error(f"❌ Failed to fill record {row.get('idx', idx)} after all retries")
                
                if HAS_TQDM:
                    progress_bar.update(1)
                    
            except Exception as e:
                failed_fills += 1
                logger.error(f"❌ Exception processing record {row.get('idx', idx)}: {str(e)}")
                if HAS_TQDM:
                    progress_bar.update(1)
        
        if HAS_TQDM:
            progress_bar.close()
        
        logger.info(f"📊 Individual processing completed for {target_field}:")
        logger.info(f"  Successfully filled: {successful_fills}")
        logger.info(f"  Failed to fill: {failed_fills}")
        logger.info(f"  Success rate: {(successful_fills/(successful_fills+failed_fills)*100):.1f}%" if (successful_fills+failed_fills) > 0 else "0%")
        
        return updated_df

def main():
    """Main function with user method selection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Type 4 Text Paraphrasing")
    parser.add_argument(
        "--method", 
        choices=["dipper", "prompt_based", "llama"],
        default="dipper",
        help="Paraphrasing method to use (default: dipper)"
    )
    parser.add_argument(
        "--input", 
        default="data/processed/unified_padben_base.csv",
        help="Input file with original data"
    )
    parser.add_argument(
        "--output-dir",
        default="data/generated",
        help="Output directory for results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--memory-efficient",
        action="store_true",
        help="Use memory-efficient configuration"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode (changes directory structure)"
    )
    
    args = parser.parse_args()
    
    # Convert method string to enum
    method_map = {
        "dipper": Type4ParaphraseMethod.DIPPER,
        "prompt_based": Type4ParaphraseMethod.PROMPT_BASED,
        "llama": Type4ParaphraseMethod.LLAMA
    }
    method = method_map[args.method]
    
    # Determine environment mode
    environment_mode = EnvironmentMode.TEST if args.test_mode else EnvironmentMode.PRODUCTION
    
    async def run_paraphrasing():
        """Run the paraphrasing process."""
        # Load dataset
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return
        
        logger.info(f"Loading dataset from {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} samples")
        
        # Create configuration
        if args.memory_efficient:
            config = create_memory_efficient_type4_config()
        else:
            config = DEFAULT_TYPE4_CONFIG
        
        config.batch_size = args.batch_size
        
        # Validate configuration
        if not validate_type4_config(config):
            logger.error("Configuration validation failed. Please check dependencies and API keys.")
            return
        
        # Initialize generator
        generator = Type4Generator(config, environment_mode)
        
        # Run paraphrasing
        logger.info(f"Starting Type 4 paraphrasing using method: {method.value}")
        results_df = await generator.generate_for_dataset(df, method, args.output_dir)
        
        # Get the method-specific column name for counting
        method_column = generator.get_method_column_name(method)
        paraphrased_count = results_df[method_column].notna().sum()
        total_count = len(results_df)
        
        logger.info("=" * 60)
        logger.info("PARAPHRASING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Method used: {method.value}")
        logger.info(f"Environment mode: {environment_mode.value}")
        logger.info(f"Successfully paraphrased {paraphrased_count}/{total_count} texts")
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info("=" * 60)
    
    # Run the paraphrasing
    asyncio.run(run_paraphrasing())

if __name__ == "__main__":
    main()
