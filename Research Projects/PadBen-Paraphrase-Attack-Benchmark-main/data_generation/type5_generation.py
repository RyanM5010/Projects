"""
Type 5 Text Generation for PADBen Benchmark.

This module implements iterative LLM-based paraphrasing of Type 2 LLM-generated text:
1. DIPPER paraphraser (HuggingFace specialized model) - Primary method  
2. Prompt-based paraphrasing (Gemini) - Fallback method

Features:
- True iterative paraphrasing: 1, 3, or 5 sequential paraphrase operations
- Iteration history tracking and quality control
- Parameter adjustment for diverse iterations
- Full retry logic with individual null record processing
- Support for new reformatted JSON format
"""

import asyncio
import logging
import time
import json
import gc
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
import numpy as np

# HuggingFace libraries for DIPPER - Updated to match Type 4
try:
    import torch
    from transformers import (
        T5Tokenizer,  # Use T5Tokenizer like Type 4
        T5ForConditionalGeneration,  # Use T5ForConditionalGeneration like Type 4
        pipeline
    )
    from nltk.tokenize import sent_tokenize  # For sentence tokenization as in official implementation
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers/torch not installed. Install with: pip install transformers torch accelerate")

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

# Text similarity checking
try:
    from difflib import SequenceMatcher
    HAS_DIFFLIB = True
except ImportError:
    HAS_DIFFLIB = False

# Progress bar support
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Local imports
from data_generation.config.type5_config import (
    Type5GenerationConfig,
    IterationLevel,
    DEFAULT_TYPE5_CONFIG,
    validate_type5_config,
    create_iterative_type5_config
)
from data_generation.config.type4_config import Type4ParaphraseMethod
from data_generation.config.base_model_config import get_api_key

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnvironmentMode(Enum):
    """Environment modes for directory structure."""
    PRODUCTION = "production"
    TEST = "test"

@dataclass
class IterationHistory:
    """Track the history of iterations for a single text."""
    original_text: str
    iterations: List[str]
    iteration_times: List[float]
    similarities: List[float]
    stopped_early: bool = False
    stop_reason: str = ""

@dataclass
class Type5Context:
    """Context data for Type 5 iterative paraphrasing."""
    generated_text: str  # Type 2 LLM-generated text
    original_text: str   # Type 1 human original text (for reference)
    target_length: int
    max_length: int
    method: Type4ParaphraseMethod
    iteration_level: IterationLevel
    dataset_source: str

@dataclass
class Type5Result:
    """Result of Type 5 iterative paraphrasing."""
    final_paraphrased_text: Optional[str]
    method_used: Type4ParaphraseMethod
    iterations_completed: int
    target_iterations: int
    success: bool
    iteration_history: IterationHistory
    metadata: Dict[str, Any]

class IterativeParaphraser:
    """Handles iterative paraphrasing operations."""
    
    def __init__(self, config: Type5GenerationConfig):
        self.config = config
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        if not HAS_DIFFLIB:
            return 0.0  # Fallback if difflib not available
        
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def should_stop_iteration(self, history: IterationHistory, current_iteration: int) -> Tuple[bool, str]:
        """Determine if iteration should stop early."""
        if len(history.iterations) < 2:
            return False, ""
        
        # Check for identical consecutive iterations
        recent_texts = history.iterations[-self.config.max_identical_iterations:]
        if len(recent_texts) >= self.config.max_identical_iterations:
            if all(self.calculate_similarity(recent_texts[0], text) >= self.config.similarity_threshold 
                   for text in recent_texts[1:]):
                return True, f"Identical text for {self.config.max_identical_iterations} consecutive iterations"
        
        # Check if we're converging to the original
        if len(history.iterations) > 1:
            latest_similarity = self.calculate_similarity(history.original_text, history.iterations[-1])
            if latest_similarity >= self.config.similarity_threshold:
                return True, "Converged back to original text"
        
        return False, ""
    
    def adjust_generation_params(self, base_params: Dict, iteration_num: int, iteration_level: IterationLevel) -> Dict:
        """Adjust generation parameters for iterative diversity."""
        adjusted_params = base_params.copy()
        
        iteration_settings = self.config.iteration_settings.get(iteration_level, {})
        
        # Increase temperature for later iterations to encourage diversity
        temp_increment = iteration_settings.get("temperature_increment", 0.0)
        if temp_increment > 0 and iteration_num > 1:
            adjusted_params["temperature"] = min(
                adjusted_params.get("temperature", 0.8) + (temp_increment * (iteration_num - 1) * 0.5),
                1.0
            )
        
        # Boost diversity for later iterations
        if iteration_settings.get("diversity_boost", False) and iteration_num > 1:
            adjusted_params["top_p"] = min(adjusted_params.get("top_p", 0.9) + 0.05, 0.95)
            if "repetition_penalty" in adjusted_params:
                adjusted_params["repetition_penalty"] = min(
                    adjusted_params.get("repetition_penalty", 1.1) + 0.1,
                    1.5
                )
        
        return adjusted_params

class EnhancedDipperParaphraser:
    """Enhanced DIPPER paraphraser with iterative capabilities using official T5 implementation."""
    
    def __init__(self, model_config, dipper_settings, iterative_paraphraser: IterativeParaphraser):
        """Initialize DIPPER paraphraser using official implementation like Type 4."""
        self.config = model_config
        self.settings = dipper_settings
        self.iterative_paraphraser = iterative_paraphraser
        self.model = None
        self.tokenizer = None
        self.device = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize DIPPER model and tokenizer using official implementation (same as Type 4)."""
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
        """Paraphrase text using official DIPPER implementation (same as Type 4)."""
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
    
    def iterative_paraphrase(self, text: str, iterations: int, max_length: int) -> IterationHistory:
        """Perform iterative paraphrasing with DIPPER."""
        history = IterationHistory(
            original_text=text,
            iterations=[],
            iteration_times=[],
            similarities=[]
        )
        
        current_text = text
        
        for i in range(iterations):
            iteration_start = time.time()
            
            try:
                # Adjust parameters for this iteration
                adjusted_settings = self.iterative_paraphraser.adjust_generation_params(
                    self.settings, i + 1, IterationLevel(iterations)
                )
                
                # Temporarily update settings
                original_settings = self.settings.copy()
                self.settings.update(adjusted_settings)
                
                # Paraphrase current text
                paraphrased = self.paraphrase_text(current_text, max_length)
                
                # Restore original settings
                self.settings = original_settings
                
                # Calculate iteration time
                iteration_time = time.time() - iteration_start
                
                # Calculate similarity to previous text
                similarity = self.iterative_paraphraser.calculate_similarity(current_text, paraphrased)
                
                # Record iteration
                history.iterations.append(paraphrased)
                history.iteration_times.append(iteration_time)
                history.similarities.append(similarity)
                
                # Check if we should stop early
                should_stop, stop_reason = self.iterative_paraphraser.should_stop_iteration(history, i + 1)
                if should_stop:
                    history.stopped_early = True
                    history.stop_reason = stop_reason
                    logger.info(f"Stopping DIPPER iteration early at {i + 1}/{iterations}: {stop_reason}")
                    break
                
                # Update current text for next iteration
                current_text = paraphrased
                
                logger.debug(f"DIPPER iteration {i + 1}/{iterations} completed. Similarity: {similarity:.3f}")
                
            except Exception as e:
                logger.error(f"DIPPER iteration {i + 1} failed: {str(e)}")
                # Use previous text if iteration fails
                if history.iterations:
                    current_text = history.iterations[-1]
                break
        
        return history
    
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

class EnhancedGeminiParaphraser:
    """Enhanced Gemini paraphraser with iterative capabilities using new Google GenAI API."""
    
    def __init__(self, model_config, iterative_paraphraser: IterativeParaphraser):
        """Initialize Gemini paraphraser."""
        self.config = model_config
        self.iterative_paraphraser = iterative_paraphraser
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
    
    async def iterative_paraphrase(self, 
                                  text: str, 
                                  iterations: int, 
                                  target_length: int, 
                                  max_length: int, 
                                  prompt_template: str,
                                  failure_reason: str = None) -> IterationHistory:
        """Perform iterative paraphrasing with Gemini using new Google GenAI API."""
        history = IterationHistory(
            original_text=text,
            iterations=[],
            iteration_times=[],
            similarities=[]
        )
        
        current_text = text
        
        for i in range(iterations):
            iteration_start = time.time()
            
            try:
                # Create prompt for current iteration
                if failure_reason and "{failure_reason}" in prompt_template:
                    # This is a retry template with failure reason
                    prompt = prompt_template.format(
                        text=current_text,
                        target_length=target_length,
                        max_length=max_length,
                        failure_reason=failure_reason
                    )
                else:
                    # Regular template without failure reason
                    prompt = prompt_template.format(
                        text=current_text,
                        target_length=target_length,
                        max_length=max_length
                    )
                
                # Adjust generation config for iteration
                base_config = {
                    "temperature": self.config.temperature,
                    "max_output_tokens": min(max_length + 50, self.config.max_tokens),
                    "top_p": self.config.top_p,
                    "top_k": getattr(self.config, 'top_k', 40),
                }
                
                generation_config = self.iterative_paraphraser.adjust_generation_params(
                    base_config, i + 1, IterationLevel(iterations)
                )
                
                # Generate paraphrase using new API format
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.config.model_id,
                    contents=prompt
                )
                
                paraphrased = response.text.strip()
                paraphrased = self._clean_paraphrase(paraphrased)
                
                # Calculate iteration time
                iteration_time = time.time() - iteration_start
                
                # Calculate similarity to previous text
                similarity = self.iterative_paraphraser.calculate_similarity(current_text, paraphrased)
                
                # Record iteration
                history.iterations.append(paraphrased)
                history.iteration_times.append(iteration_time)
                history.similarities.append(similarity)
                
                # Check if we should stop early
                should_stop, stop_reason = self.iterative_paraphraser.should_stop_iteration(history, i + 1)
                if should_stop:
                    history.stopped_early = True
                    history.stop_reason = stop_reason
                    logger.info(f"Stopping Gemini iteration early at {i + 1}/{iterations}: {stop_reason}")
                    break
                
                # Update current text for next iteration
                current_text = paraphrased
                
                logger.debug(f"Gemini iteration {i + 1}/{iterations} completed. Similarity: {similarity:.3f}")
                
            except Exception as e:
                logger.error(f"Gemini iteration {i + 1} failed: {str(e)}")
                # Use previous text if iteration fails
                if history.iterations:
                    current_text = history.iterations[-1]
                break
        
        return history

class Type5Generator:
    """Generator for Type 5 (LLM-paraphrased LLM-generated text) with true iterative paraphrasing."""
    
    def __init__(self, config: Optional[Type5GenerationConfig] = None, 
                 environment_mode: EnvironmentMode = EnvironmentMode.PRODUCTION):
        """Initialize the Type 5 generator."""
        self.config = config or DEFAULT_TYPE5_CONFIG
        self.environment_mode = environment_mode
        
        # Validate configuration
        if not validate_type5_config(self.config):
            raise ValueError("Invalid Type 5 configuration")
        
        # Initialize iterative paraphraser
        self.iterative_paraphraser = IterativeParaphraser(self.config)
        
        # Initialize enhanced paraphrasers (lazy loading)
        self.dipper_paraphraser = None
        self.gemini_paraphraser = None
        
        # Statistics tracking
        self.stats = {
            "total_processed": 0,
            "successful_paraphrases": 0,
            "failed_paraphrases": 0,
            "length_violations": 0,
            "missing_type2_data": 0,
            "early_stops": 0,
            "method_usage": {
                "dipper": 0,
                "prompt_based": 0
            },
            "iteration_usage": {
                "first": 0,
                "third": 0,
                "fifth": 0
            },
            "avg_iterations_completed": 0,
            "start_time": None,
            "end_time": None
        }
    
    def detect_format_and_get_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Detect the format and return appropriate column mappings."""
        if 'human_original_text(type1)' in df.columns:
            # Reformatted format
            return {
                'type1': 'human_original_text(type1)',
                'type2': 'llm_generated_text(type2)',
                'type3': 'human_paraphrased_text(type3)',
                'type5_1st': 'llm_paraphrased_generated_text(type5)-1st',
                'type5_3rd': 'llm_paraphrased_generated_text(type5)-3rd',
                'type5_5th': 'llm_paraphrased_generated_text(type5)-5th'  # Add 5th iteration support
            }
        else:
            # Standard format
            return {
                'type1': 'human_original_text',
                'type2': 'llm_generated_text',
                'type3': 'human_paraphrased_text',
                'type5_1st': 'llm_paraphrased_generated_text(DIPPER_based_1_iteration)',
                'type5_3rd': 'llm_paraphrased_generated_text(DIPPER_based_3_iterations)',
                'type5_5th': 'llm_paraphrased_generated_text(DIPPER_based_5_iterations)'
            }
    
    def get_type5_column_name(self, method: Type4ParaphraseMethod, iteration_level: IterationLevel, df: pd.DataFrame) -> str:
        """Get the correct Type 5 column name based on format and method/iteration."""
        columns = self.detect_format_and_get_columns(df)
        
        if 'human_original_text(type1)' in df.columns:
            # Reformatted format - simplified naming
            if iteration_level == IterationLevel.FIRST:
                return columns['type5_1st']
            elif iteration_level == IterationLevel.THIRD:
                return columns['type5_3rd']
            elif iteration_level == IterationLevel.FIFTH:
                return columns['type5_5th']
        else:
            # Standard format - detailed naming
            method_name_map = {
                Type4ParaphraseMethod.DIPPER: "DIPPER_based",
                Type4ParaphraseMethod.PROMPT_BASED: "Prompt_based"
            }
            iteration_name_map = {
                IterationLevel.FIRST: "1_iteration",
                IterationLevel.THIRD: "3_iterations",
                IterationLevel.FIFTH: "5_iterations"
            }
            method_name = method_name_map.get(method, method.value)
            iteration_name = iteration_name_map.get(iteration_level, f"{iteration_level.value}_iterations")
            return f"llm_paraphrased_generated_text({method_name}_{iteration_name})"
    
    def get_directory_structure(self, base_output_dir: str, iteration_level: IterationLevel, timestamp: str) -> str:
        """
        Get the appropriate directory structure based on environment mode and iteration level.
        
        Returns:
            Final output directory path
        """
        if self.environment_mode == EnvironmentMode.TEST:
            # Test environment: data/test/type5_generation_test/iteration_num_timestamp/
            iteration_names = {
                IterationLevel.FIRST: "1st",
                IterationLevel.THIRD: "3rd", 
                IterationLevel.FIFTH: "5th"
            }
            iteration_name = iteration_names.get(iteration_level, "unknown")
            test_output_dir = f"{base_output_dir}/{iteration_name}_{timestamp}"
            return test_output_dir
        else:
            # Production environment: data/generated/timestamp/
            prod_output_dir = f"{base_output_dir}/{timestamp}"
            return prod_output_dir
    
    def _initialize_dipper(self):
        """Initialize enhanced DIPPER paraphraser (lazy loading)."""
        if self.dipper_paraphraser is None:
            try:
                self.dipper_paraphraser = EnhancedDipperParaphraser(
                    self.config.primary_model,
                    self.config.dipper_settings,
                    self.iterative_paraphraser
                )
                logger.info("Enhanced DIPPER paraphraser initialized for Type 5")
            except Exception as e:
                logger.error(f"Failed to initialize DIPPER for Type 5: {str(e)}")
                raise
    
    def _initialize_gemini(self):
        """Initialize enhanced Gemini paraphraser (lazy loading)."""
        if self.gemini_paraphraser is None:
            try:
                self.gemini_paraphraser = EnhancedGeminiParaphraser(
                    self.config.fallback_model,
                    self.iterative_paraphraser
                )
                logger.info("Enhanced Gemini paraphraser initialized for Type 5")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini for Type 5: {str(e)}")
                raise
    
    def calculate_length_constraints(self, generated_text: str) -> Tuple[int, int]:
        """Calculate target and maximum length for paraphrased text."""
        generated_length = len(generated_text)
        target_length = generated_length
        max_length = int(generated_length * self.config.max_length_multiplier)
        return target_length, max_length
    
    def validate_final_result(self, 
                             original: str, 
                             generated: str, 
                             final_paraphrased: str, 
                             context: Type5Context) -> Tuple[bool, str]:
        """Validate final paraphrased text quality."""
        if not final_paraphrased or len(final_paraphrased.strip()) < self.config.min_length_threshold:
            return False, "Final paraphrased text too short"
        
        if len(final_paraphrased) > context.max_length:
            return False, f"Final paraphrased text too long ({len(final_paraphrased)} > {context.max_length})"
        
        # Check if identical to generated text (should be different)
        if final_paraphrased.lower().strip() == generated.lower().strip():
            return False, "Final paraphrased text identical to generated text"
        
        # Check minimum length requirement
        min_length = context.target_length * (1 - self.config.length_tolerance)
        if len(final_paraphrased) < min_length:
            return False, f"Final paraphrased text below minimum length ({len(final_paraphrased)} < {min_length})"
        
        return True, "Valid"
    
    async def paraphrase_with_dipper(self, context: Type5Context) -> Type5Result:
        """Perform iterative paraphrasing using DIPPER model with retry logic."""
        last_failure_reason = None
        
        for attempt in range(self.config.max_retries + 1):  # +1 for initial attempt
            try:
                # Initialize DIPPER if needed
                self._initialize_dipper()
                
                # For DIPPER, we can adjust parameters slightly for retries
                if attempt > 0:
                    logger.info(f"DIPPER retry attempt {attempt}/{self.config.max_retries}")
                    # Slightly modify generation parameters for retry
                    original_settings = self.dipper_paraphraser.settings.copy()
                    self.dipper_paraphraser.settings["lex_diversity"] = min(100, original_settings.get("lex_diversity", 60) + 20 * attempt)
                    self.dipper_paraphraser.settings["order_diversity"] = min(100, original_settings.get("order_diversity", 0) + 20 * attempt)
                
                # Perform iterative paraphrasing
                history = await asyncio.to_thread(
                    self.dipper_paraphraser.iterative_paraphrase,
                    context.generated_text,
                    context.iteration_level.value,
                    context.max_length
                )
                
                # Get final result
                final_text = history.iterations[-1] if history.iterations else None
                iterations_completed = len(history.iterations)
                
                if not final_text or not final_text.strip():
                    failure_reason = "Empty or null final paraphrased text"
                    last_failure_reason = failure_reason
                    
                    if attempt < self.config.max_retries:
                        logger.warning(f"DIPPER attempt {attempt + 1} failed: {failure_reason}. Retrying...")
                        await asyncio.sleep(self.config.retry_delay)
                        continue
                    else:
                        return Type5Result(
                            final_paraphrased_text=None,
                            method_used=Type4ParaphraseMethod.DIPPER,
                            iterations_completed=0,
                            target_iterations=context.iteration_level.value,
                            success=False,
                            iteration_history=history,
                            metadata={
                                "failure_reason": failure_reason, 
                                "method": "dipper",
                                "retry_count": attempt,
                                "total_attempts": attempt + 1,
                                "final_attempt": True
                            }
                        )
                
                # Validate final result
                is_valid, validation_msg = self.validate_final_result(
                    context.original_text, context.generated_text, final_text, context
                )
                
                metadata = {
                    "method": "dipper",
                    "target_iterations": context.iteration_level.value,
                    "completed_iterations": iterations_completed,
                    "stopped_early": history.stopped_early,
                    "stop_reason": history.stop_reason,
                    "original_length": len(context.original_text),
                    "generated_length": len(context.generated_text),
                    "final_paraphrased_length": len(final_text),
                    "validation_status": validation_msg,
                    "model_used": self.config.primary_model.model_id,
                    "dataset_source": context.dataset_source,
                    "iteration_times": history.iteration_times,
                    "similarities": history.similarities,
                    "avg_similarity": np.mean(history.similarities) if history.similarities else 0,
                    "retry_count": attempt,
                    "total_attempts": attempt + 1
                }
                
                if last_failure_reason:
                    metadata["previous_failures"] = last_failure_reason
                
                if is_valid:
                    self.stats["method_usage"]["dipper"] += 1
                    self.stats["iteration_usage"][context.iteration_level.name.lower()] += 1
                    if history.stopped_early:
                        self.stats["early_stops"] += 1
                    if attempt > 0:
                        metadata["succeeded_on_retry"] = True
                        logger.info(f"DIPPER succeeded on retry attempt {attempt}")
                    
                    return Type5Result(
                        final_paraphrased_text=final_text,
                        method_used=Type4ParaphraseMethod.DIPPER,
                        iterations_completed=iterations_completed,
                        target_iterations=context.iteration_level.value,
                        success=True,
                        iteration_history=history,
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
                        return Type5Result(
                            final_paraphrased_text=None,
                            method_used=Type4ParaphraseMethod.DIPPER,
                            iterations_completed=iterations_completed,
                            target_iterations=context.iteration_level.value,
                            success=False,
                            iteration_history=history,
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
                    logger.error(f"DIPPER iterative paraphrasing failed after {attempt + 1} attempts: {error_msg}")
                    return Type5Result(
                        final_paraphrased_text=None,
                        method_used=Type4ParaphraseMethod.DIPPER,
                        iterations_completed=0,
                        target_iterations=context.iteration_level.value,
                        success=False,
                        iteration_history=IterationHistory(context.generated_text, [], [], []),
                        metadata={
                            "failure_reason": error_msg, 
                            "method": "dipper",
                            "retry_count": attempt,
                            "total_attempts": attempt + 1,
                            "final_attempt": True
                        }
                    )
        
        # Should not reach here, but safety fallback
        return Type5Result(
            final_paraphrased_text=None,
            method_used=Type4ParaphraseMethod.DIPPER,
            iterations_completed=0,
            target_iterations=context.iteration_level.value,
            success=False,
            iteration_history=IterationHistory(context.generated_text, [], [], []),
            metadata={"failure_reason": "Unexpected retry loop exit", "method": "dipper"}
        )
    
    async def paraphrase_with_prompt(self, context: Type5Context) -> Type5Result:
        """Perform iterative paraphrasing using Gemini prompt-based method with retry logic."""
        last_failure_reason = None
        
        for attempt in range(self.config.max_retries + 1):  # +1 for initial attempt
            try:
                # Initialize Gemini if needed
                self._initialize_gemini()
                
                # Choose prompt template and setup failure context
                prompt_template = self.config.paraphrase_prompt_template
                failure_context = None
                
                if attempt > 0 and self.config.enable_retry_with_context and last_failure_reason:
                    # Use retry template with failure context
                    if hasattr(self.config, 'retry_prompt_template') and self.config.retry_prompt_template:
                        prompt_template = self.config.retry_prompt_template
                        failure_context = last_failure_reason
                        logger.info(f"Gemini retry attempt {attempt}/{self.config.max_retries} with failure context: {last_failure_reason}")
                    else:
                        logger.warning("Retry template not available, using regular template")
                
                # Perform iterative paraphrasing with failure context
                history = await self.gemini_paraphraser.iterative_paraphrase(
                    context.generated_text,
                    context.iteration_level.value,
                    context.target_length,
                    context.max_length,
                    prompt_template,
                    failure_reason=failure_context
                )
                
                # Get final result
                final_text = history.iterations[-1] if history.iterations else None
                iterations_completed = len(history.iterations)
                
                if not final_text or not final_text.strip():
                    failure_reason = "Empty or null final paraphrased text"
                    last_failure_reason = failure_reason
                    
                    if attempt < self.config.max_retries:
                        logger.warning(f"Gemini attempt {attempt + 1} failed: {failure_reason}. Retrying...")
                        await asyncio.sleep(self.config.retry_delay)
                        continue
                    else:
                        return Type5Result(
                            final_paraphrased_text=None,
                            method_used=Type4ParaphraseMethod.PROMPT_BASED,
                            iterations_completed=0,
                            target_iterations=context.iteration_level.value,
                            success=False,
                            iteration_history=history,
                            metadata={
                                "failure_reason": failure_reason,
                                "method": "prompt_based",
                                "retry_count": attempt,
                                "total_attempts": attempt + 1,
                                "final_attempt": True
                            }
                        )
                
                # Validate final result
                is_valid, validation_msg = self.validate_final_result(
                    context.original_text, context.generated_text, final_text, context
                )
                
                metadata = {
                    "method": "prompt_based",
                    "target_iterations": context.iteration_level.value,
                    "completed_iterations": iterations_completed,
                    "stopped_early": history.stopped_early,
                    "stop_reason": history.stop_reason,
                    "original_length": len(context.original_text),
                    "generated_length": len(context.generated_text),
                    "final_paraphrased_length": len(final_text),
                    "validation_status": validation_msg,
                    "model_used": self.config.fallback_model.model_id,
                    "dataset_source": context.dataset_source,
                    "iteration_times": history.iteration_times,
                    "similarities": history.similarities,
                    "avg_similarity": np.mean(history.similarities) if history.similarities else 0,
                    "retry_count": attempt,
                    "total_attempts": attempt + 1
                }
                
                if last_failure_reason:
                    metadata["previous_failures"] = last_failure_reason
                    metadata["used_retry_template"] = failure_context is not None
                
                if is_valid:
                    self.stats["method_usage"]["prompt_based"] += 1
                    self.stats["iteration_usage"][context.iteration_level.name.lower()] += 1
                    if history.stopped_early:
                        self.stats["early_stops"] += 1
                    if attempt > 0:
                        metadata["succeeded_on_retry"] = True
                        logger.info(f"Gemini succeeded on retry attempt {attempt}")
                    
                    return Type5Result(
                        final_paraphrased_text=final_text,
                        method_used=Type4ParaphraseMethod.PROMPT_BASED,
                        iterations_completed=iterations_completed,
                        target_iterations=context.iteration_level.value,
                        success=True,
                        iteration_history=history,
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
                        return Type5Result(
                            final_paraphrased_text=None,
                            method_used=Type4ParaphraseMethod.PROMPT_BASED,
                            iterations_completed=iterations_completed,
                            target_iterations=context.iteration_level.value,
                            success=False,
                            iteration_history=history,
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
                    logger.error(f"Prompt-based iterative paraphrasing failed after {attempt + 1} attempts: {error_msg}")
                    return Type5Result(
                        final_paraphrased_text=None,
                        method_used=Type4ParaphraseMethod.PROMPT_BASED,
                        iterations_completed=0,
                        target_iterations=context.iteration_level.value,
                        success=False,
                        iteration_history=IterationHistory(context.generated_text, [], [], []),
                        metadata={
                            "failure_reason": error_msg,
                            "method": "prompt_based",
                            "retry_count": attempt,
                            "total_attempts": attempt + 1,
                            "final_attempt": True
                        }
                    )
        
        # Should not reach here, but safety fallback
        return Type5Result(
            final_paraphrased_text=None,
            method_used=Type4ParaphraseMethod.PROMPT_BASED,
            iterations_completed=0,
            target_iterations=context.iteration_level.value,
            success=False,
            iteration_history=IterationHistory(context.generated_text, [], [], []),
            metadata={"failure_reason": "Unexpected retry loop exit", "method": "prompt_based"}
        )
    
    async def paraphrase_text(self, 
                             row: pd.Series, 
                             method: Type4ParaphraseMethod = None,
                             iteration: IterationLevel = None) -> Type5Result:
        """
        Perform iterative paraphrasing of Type 2 text.
        
        Args:
            row: DataFrame row containing Type 2 generated text
            method: Paraphrasing method to use
            iteration: Number of iterations to perform
            
        Returns:
            Type5Result with final paraphrased text and iteration history
        """
        start_time = time.time()
        
        try:
            # Detect format and get column mappings
            columns = self.detect_format_and_get_columns(pd.DataFrame([row]))
            
            # Get Type 2 generated text
            generated_text = row.get(columns['type2'], '')
            if not generated_text or pd.isna(generated_text):
                self.stats["missing_type2_data"] += 1
                raise ValueError("Missing Type 2 generated text")
            
            # Get original text for reference
            original_text = row.get(columns['type1'], '')
            if not original_text or pd.isna(original_text):
                raise ValueError("Missing original human text for reference")
            
            # Calculate length constraints
            target_length, max_length = self.calculate_length_constraints(generated_text)
            
            # Create Type 5 context
            context = Type5Context(
                generated_text=generated_text,
                original_text=original_text,
                target_length=target_length,
                max_length=max_length,
                method=method or self.config.default_method,
                iteration_level=iteration or self.config.default_iteration,
                dataset_source=row.get('dataset_source', 'unknown')
            )
            
            # Execute iterative paraphrasing
            if context.method == Type4ParaphraseMethod.DIPPER:
                result = await self.paraphrase_with_dipper(context)
                
                # If DIPPER fails, try fallback
                if not result.success and self.config.fallback_method == Type4ParaphraseMethod.PROMPT_BASED:
                    logger.warning("DIPPER failed for Type 5, trying Gemini fallback")
                    context.method = Type4ParaphraseMethod.PROMPT_BASED
                    result = await self.paraphrase_with_prompt(context)
                    if result.success:
                        result.metadata["used_fallback"] = True
            
            else:  # PROMPT_BASED
                result = await self.paraphrase_with_prompt(context)
            
            # Add timing information
            result.metadata["total_paraphrase_time"] = time.time() - start_time
            
            # Update statistics
            if result.success:
                self.stats["successful_paraphrases"] += 1
                # Update average iterations completed
                total_completed = (self.stats["avg_iterations_completed"] * (self.stats["successful_paraphrases"] - 1) + 
                                 result.iterations_completed)
                self.stats["avg_iterations_completed"] = total_completed / self.stats["successful_paraphrases"]
            else:
                self.stats["failed_paraphrases"] += 1
                if "length" in result.metadata.get("failure_reason", "").lower():
                    self.stats["length_violations"] += 1
            
            self.stats["total_processed"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Type 5 iterative paraphrasing failed: {str(e)}")
            self.stats["failed_paraphrases"] += 1
            self.stats["total_processed"] += 1
            
            return Type5Result(
                final_paraphrased_text=None,
                method_used=method or self.config.default_method,
                iterations_completed=0,
                target_iterations=(iteration or self.config.default_iteration).value,
                success=False,
                iteration_history=IterationHistory("", [], [], []),
                metadata={
                    "failure_reason": str(e),
                    "total_paraphrase_time": time.time() - start_time
                }
            )
    
    def populate_all_iteration_columns(self, results_df: pd.DataFrame, idx: int, result: Type5Result, 
                                     method: Type4ParaphraseMethod, target_iteration: IterationLevel):
        """
        Populate all iteration columns (1st, 2nd, 3rd, etc.) based on iteration history.
        
        This ensures that when doing 3rd iteration, we save 1st and 2nd results too.
        """
        if not result.success or not result.iteration_history.iterations:
            return
        
        # Get all possible iteration columns for this method
        iteration_levels = [IterationLevel.FIRST, IterationLevel.THIRD, IterationLevel.FIFTH]
        
        for i, iteration_text in enumerate(result.iteration_history.iterations):
            iteration_num = i + 1
            
            # Find the appropriate iteration level for this iteration number
            if iteration_num == 1:
                iter_level = IterationLevel.FIRST
            elif iteration_num <= 3:
                iter_level = IterationLevel.THIRD if iteration_num == 3 else None
            elif iteration_num <= 5:
                iter_level = IterationLevel.FIFTH if iteration_num == 5 else None
            else:
                continue  # Skip iterations beyond 5
            
            # Only populate if we have a valid iteration level and the column exists
            if iter_level and iter_level.value <= target_iteration.value:
                column_name = self.get_type5_column_name(method, iter_level, results_df)
                if column_name in results_df.columns:
                    # For intermediate iterations (not the final one), we populate with intermediate results
                    if iteration_num < len(result.iteration_history.iterations):
                        # This is an intermediate iteration
                        results_df.at[idx, column_name] = iteration_text
                    elif iter_level == target_iteration:
                        # This is the final target iteration
                        results_df.at[idx, column_name] = result.final_paraphrased_text
    
    async def process_batch(self, 
                           batch_data: List[Tuple[int, pd.Series]], 
                           method: Type4ParaphraseMethod = None,
                           iteration: IterationLevel = None) -> List[Tuple[int, Type5Result]]:
        """Process a batch of texts for Type 5 paraphrasing."""
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        async def process_single(idx: int, row: pd.Series) -> Tuple[int, Type5Result]:
            async with semaphore:
                result = await self.paraphrase_text(row, method, iteration)
                return idx, result
        
        tasks = [process_single(idx, row) for idx, row in batch_data]
        return await asyncio.gather(*tasks)
    
    def filter_target_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data for Type 5 generation with format detection."""
        # Detect format and get column mappings
        columns = self.detect_format_and_get_columns(df)
        
        # Filter to target datasets
        target_datasets = [ds.lower() for ds in self.config.target_datasets]
        mask = df['dataset_source'].str.lower().isin(target_datasets)
        
        # Ensure we have required Type 2 generated text
        mask &= df[columns['type2']].notna() & (df[columns['type2']] != '')
        
        # Ensure we have required original text
        mask &= df[columns['type1']].notna() & (df[columns['type1']] != '')
        
        filtered_df = df[mask].copy()
        logger.info(f"Filtered {len(filtered_df)} samples requiring Type 5 paraphrasing from {len(df)} total")
        
        return filtered_df
    
    async def retry_single_record_paraphrasing(self, 
                                             row: pd.Series, 
                                             method: Type4ParaphraseMethod,
                                             iteration: IterationLevel,
                                             max_retries: int = 3) -> Type5Result:
        """
        Retry paraphrasing for a single record until successful or max retries reached.
        
        Args:
            row: DataFrame row containing Type 2 generated text
            method: Paraphrasing method to use
            iteration: Number of iterations to perform
            max_retries: Maximum number of retry attempts
            
        Returns:
            Type5Result with paraphrased text and metadata
        """
        sample_idx = row.get('idx', row.name if hasattr(row, 'name') else -1)
        
        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries + 1} for sample {sample_idx}")
                
                result = await self.paraphrase_text(row, method, iteration)
                
                if result.success and result.final_paraphrased_text and result.final_paraphrased_text.strip():
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
        return Type5Result(
            final_paraphrased_text=None,
            method_used=method,
            iterations_completed=0,
            target_iterations=iteration.value,
            success=False,
            iteration_history=IterationHistory("", [], [], []),
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
                                              iteration: IterationLevel,
                                              max_retries: int = 3) -> pd.DataFrame:
        """
        Process null records individually with immediate retry until each record is filled.
        
        This approach ensures that each null record is processed individually and retried
        until successful, rather than processing the entire batch multiple times.
        
        Args:
            df: DataFrame with null records to process
            target_field: Target field name to fill
            method: Paraphrasing method to use
            iteration: Number of iterations to perform
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
        if HAS_TQDM:
            progress_bar = tqdm(total=len(null_records), desc=f"Processing {target_field} null records")
        
        for idx, row in null_records.iterrows():
            try:
                # Retry until successful or max attempts reached
                result = await self.retry_single_record_paraphrasing(row, method, iteration, max_retries)
                
                if result.success and result.final_paraphrased_text:
                    # Update the DataFrame with the successful result
                    updated_df.at[idx, target_field] = result.final_paraphrased_text
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
    
    async def generate_for_dataset(self, 
                                  df: pd.DataFrame, 
                                  method: Type4ParaphraseMethod = None,
                                  iteration: IterationLevel = None,
                                  output_dir: str = "data/generated") -> pd.DataFrame:
        """Generate Type 5 paraphrases for the entire dataset with proper format preservation."""
        self.stats["start_time"] = datetime.now()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        method_name = method.value if method else self.config.default_method.value
        iteration_level = iteration or self.config.default_iteration
        
        # Create proper directory structure: output_dir/iteration_timestamp/
        iteration_names = {
            IterationLevel.FIRST: "1st",
            IterationLevel.THIRD: "3rd", 
            IterationLevel.FIFTH: "5th"
        }
        iteration_name = iteration_names.get(iteration_level, "unknown")
        final_output_dir = f"{output_dir}/{iteration_name}_{timestamp}"
        
        # Create midpoint subdirectory
        midpoint_dir = f"{final_output_dir}/midpoint"
        
        # Filter target data
        target_df = self.filter_target_data(df)
        
        if target_df.empty:
            logger.info("No samples require Type 5 paraphrasing")
            return df
        
        logger.info(f"Starting Type 5 paraphrasing for {len(target_df)} samples")
        logger.info(f"Method: {method_name}, Iterations: {iteration_level.value}")
        logger.info(f"Environment mode: {self.environment_mode.value}")
        logger.info(f"Final output directory: {final_output_dir}")
        logger.info(f"Midpoint directory: {midpoint_dir}")
        
        # Prepare results - PRESERVE EXACT INPUT FORMAT
        results_df = df.copy()
        
        # Initialize ALL iteration columns that might be needed
        iteration_levels_to_init = [IterationLevel.FIRST]
        if iteration_level.value >= 3:
            iteration_levels_to_init.append(IterationLevel.THIRD)
        if iteration_level.value >= 5:
            iteration_levels_to_init.append(IterationLevel.FIFTH)
        
        for iter_level in iteration_levels_to_init:
            column_name = self.get_type5_column_name(method or self.config.default_method, iter_level, results_df)
            if column_name not in results_df.columns:
                results_df[column_name] = pd.Series(dtype='object')
        
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
            batch_results = await self.process_batch(batch_data, method, iteration)
            
            # Update results with ALL iterations
            for idx, result in batch_results:
                if result.success:
                    # Populate all iteration columns based on iteration history
                    self.populate_all_iteration_columns(results_df, idx, result, 
                                                       method or self.config.default_method, iteration_level)
                
                # Store metadata with full iteration details
                result.metadata['batch_idx'] = batch_idx
                result.metadata['sample_idx'] = idx
                paraphrase_metadata.append(result.metadata)
            
            # Save midpoint results every 5 batches or if it's the last batch
            if (batch_idx + 1) % 5 == 0 or batch_idx == total_batches - 1:
                await self._save_midpoint_results(results_df, batch_results, batch_idx + 1, 
                                                midpoint_dir, method_name, iteration_name, timestamp)
            
            # Brief pause between batches
            await asyncio.sleep(0.1)
        
        self.stats["end_time"] = datetime.now()
        
        # Save final results
        await self._save_final_results(results_df, paraphrase_metadata, method_name, 
                                     iteration_level.name, final_output_dir)
        
        # Clean up resources
        self._cleanup_resources()
        
        # Log final statistics
        self._log_final_statistics()
        
        return results_df
    
    async def _save_midpoint_results(self, df: pd.DataFrame, batch_results: List[Tuple[int, Type5Result]], 
                                   batch_num: int, midpoint_dir: str, method_name: str, 
                                   iteration_name: str, timestamp: str):
        """Save midpoint results with full iteration metadata."""
        try:
            output_path = Path(midpoint_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Prepare detailed midpoint data
            midpoint_records = []
            
            for idx, result in batch_results:
                if result.success and result.iteration_history.iterations:
                    # Get the row data
                    row = df.loc[idx]
                    
                    # Create base record preserving original format
                    record = {
                        'idx': row['idx'],
                        'dataset_source': row['dataset_source'],
                        'batch_number': batch_num,
                        'sample_index_in_batch': idx,
                        'method_used': method_name,
                        'target_iterations': result.target_iterations,
                        'completed_iterations': result.iterations_completed,
                        'success': result.success
                    }
                    
                    # Add original texts for reference
                    columns = self.detect_format_and_get_columns(df)
                    record['original_text'] = row[columns['type1']]
                    record['generated_text'] = row[columns['type2']]
                    record['final_paraphrased_text'] = result.final_paraphrased_text
                    
                    # Add ALL iteration details
                    for i, iteration_text in enumerate(result.iteration_history.iterations, 1):
                        record[f'iteration_{i}_text'] = iteration_text
                        if i-1 < len(result.iteration_history.iteration_times):
                            record[f'iteration_{i}_time_seconds'] = result.iteration_history.iteration_times[i-1]
                        if i-1 < len(result.iteration_history.similarities):
                            record[f'iteration_{i}_similarity'] = result.iteration_history.similarities[i-1]
                    
                    # Add metadata
                    record.update(result.metadata)
                    
                    midpoint_records.append(record)
            
            if midpoint_records:
                # Save as JSON
                midpoint_file = output_path / f"batch_{batch_num:03d}_type5_{method_name}_{iteration_name}_{timestamp}.json"
                with open(midpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(midpoint_records, f, indent=2, default=str, ensure_ascii=False)
                
                logger.info(f"✅ Saved midpoint results for batch {batch_num}: {midpoint_file}")
                logger.info(f"   Records: {len(midpoint_records)}")
                
        except Exception as e:
            logger.warning(f"Failed to save midpoint results for batch {batch_num}: {str(e)}")
    
    async def _save_final_results(self, df: pd.DataFrame, metadata: List[Dict], method_name: str, 
                                iteration_name: str, output_dir: str):
        """Save final Type 5 paraphrasing results with proper format."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # IMPORTANT: Remove any extra columns that were added during processing
            # to preserve the exact input format
            columns_to_keep = df.columns.tolist()
            
            # Remove temporary columns if they exist
            temp_columns = ['llm_generated_text', 'human_original_text']
            for temp_col in temp_columns:
                if temp_col in columns_to_keep:
                    # Only remove if it's a duplicate (i.e., we have the original format version)
                    original_format_exists = any(
                        col for col in columns_to_keep 
                        if 'type1' in col or 'type2' in col
                    )
                    if original_format_exists:
                        columns_to_keep.remove(temp_col)
            
            # Create clean output dataframe
            clean_df = df[columns_to_keep].copy()
            
            # Save CSV
            csv_file = output_path / f"unified_padben_with_type5_{method_name}_{iteration_name}_{timestamp}.csv"
            clean_df.to_csv(csv_file, index=False)
            
            # Save JSON
            json_file = output_path / f"unified_padben_with_type5_{method_name}_{iteration_name}_{timestamp}.json"
            clean_df.to_json(json_file, orient='records', indent=2)
            
            # Save complete metadata
            metadata_file = output_path / f"type5_paraphrasing_metadata_{method_name}_{iteration_name}_{timestamp}.json"
            full_metadata = {
                "generation_config": asdict(self.config),
                "method_used": method_name,
                "iteration_used": iteration_name,
                "environment_mode": self.environment_mode.value,
                "output_format": "preserved_input_format",
                "iteration_columns_populated": "all_intermediate_iterations_saved",
                "statistics": self.stats,
                "sample_metadata": metadata
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(full_metadata, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"Saved final Type 5 results:")
            logger.info(f"  CSV: {csv_file}")
            logger.info(f"  JSON: {json_file}")
            logger.info(f"  Metadata: {metadata_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save final results: {str(e)}")
    
    def _cleanup_resources(self):
        """Clean up model resources."""
        if self.dipper_paraphraser:
            self.dipper_paraphraser.cleanup()
        
        # Clear any remaining GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        logger.info("Cleaned up Type 5 generation resources")
    
    def _log_final_statistics(self):
        """Log final Type 5 paraphrasing statistics."""
        stats = self.stats
        total_time = (stats["end_time"] - stats["start_time"]).total_seconds() if stats["end_time"] and stats["start_time"] else 0
        
        logger.info("=" * 60)
        logger.info("TYPE 5 ITERATIVE PARAPHRASING COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total processed: {stats['total_processed']}")
        logger.info(f"Successful paraphrases: {stats['successful_paraphrases']}")
        logger.info(f"Failed paraphrases: {stats['failed_paraphrases']}")
        logger.info(f"Missing Type 2 data: {stats['missing_type2_data']}")
        logger.info(f"Length violations: {stats['length_violations']}")
        logger.info(f"Early stops: {stats['early_stops']}")
        logger.info(f"Success rate: {stats['successful_paraphrases'] / max(stats['total_processed'], 1) * 100:.1f}%")
        logger.info("Method usage:")
        logger.info(f"  DIPPER: {stats['method_usage']['dipper']}")
        logger.info(f"  Prompt-based: {stats['method_usage']['prompt_based']}")
        logger.info("Iteration usage:")
        logger.info(f"  1 iteration: {stats['iteration_usage']['first']}")
        logger.info(f"  3 iterations: {stats['iteration_usage']['third']}")
        logger.info(f"  5 iterations: {stats['iteration_usage']['fifth']}")
        logger.info(f"Average iterations completed: {stats['avg_iterations_completed']:.1f}")
        logger.info(f"Total time: {total_time:.2f} seconds")
        if stats['successful_paraphrases'] > 0:
            logger.info(f"Average time per paraphrase: {total_time / stats['successful_paraphrases']:.2f} seconds")
        logger.info("=" * 60)

def main():
    """Main function with user method and iteration selection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Type 5 Iterative Text Paraphrasing")
    parser.add_argument(
        "--method", 
        choices=["dipper", "prompt_based"],
        default="dipper",
        help="Paraphrasing method to use (default: dipper)"
    )
    parser.add_argument(
        "--iterations",
        choices=["1", "3", "5"],
        default="1",
        help="Number of sequential iterations to perform (default: 1)"
    )
    parser.add_argument(
        "--input", 
        default="data/generated/unified_padben_with_type2_latest.csv",
        help="Input file with Type 2 data"
    )
    parser.add_argument(
        "--output-dir",
        default="data/generated",
        help="Output directory for results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for processing (default: 4 due to iterative processing)"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode (changes directory structure)"
    )
    
    args = parser.parse_args()
    
    # Convert method and iteration strings to enums
    method_map = {
        "dipper": Type4ParaphraseMethod.DIPPER,
        "prompt_based": Type4ParaphraseMethod.PROMPT_BASED
    }
    
    iteration_map = {
        "1": IterationLevel.FIRST,
        "3": IterationLevel.THIRD,
        "5": IterationLevel.FIFTH
    }
    
    method = method_map[args.method]
    iterations = iteration_map[args.iterations]
    
    # Determine environment mode
    environment_mode = EnvironmentMode.TEST if args.test_mode else EnvironmentMode.PRODUCTION
    
    async def run_type5_generation():
        """Run the Type 5 generation process."""
        # Load dataset
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return
        
        logger.info(f"Loading dataset from {input_path}")
        if input_path.suffix == '.json':
            df = pd.read_json(input_path)
        else:
            df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} samples")
        
        # Create configuration
        config = create_iterative_type5_config(iterations)
        config.batch_size = args.batch_size
        
        # Validate configuration
        if not validate_type5_config(config):
            logger.error("Configuration validation failed. Please check dependencies and API keys.")
            return
        
        # Initialize generator
        generator = Type5Generator(config, environment_mode)
        
        # Run paraphrasing
        logger.info(f"Starting Type 5 paraphrasing using method: {method.value}, iterations: {iterations.value}")
        results_df = await generator.generate_for_dataset(df, method, iterations, args.output_dir)
        
        # Get the method-specific column name for counting
        method_column = generator.get_type5_column_name(method, iterations, results_df)
        paraphrased_count = results_df[method_column].notna().sum()
        total_count = len(results_df)
        
        logger.info("=" * 60)
        logger.info("TYPE 5 GENERATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Method used: {method.value}")
        logger.info(f"Iterations used: {iterations.value}")
        logger.info(f"Environment mode: {environment_mode.value}")
        logger.info(f"Successfully paraphrased {paraphrased_count}/{total_count} AI-generated texts")
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info("=" * 60)
    
    # Run the generation
    asyncio.run(run_type5_generation())

if __name__ == "__main__":
    main()
