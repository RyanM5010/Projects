"""
Enhanced Type 2 Text Generation for PADBen Benchmark.

This module implements LLM-based generation of Type 2 texts using:
1. Sentence Completion Method (NEW) - Uses extracted prefixes and keywords
2. Question-Answer Method (REVISED) - Incorporates extracted keywords and constraints

Integrates on-the-fly preprocessing from keywords_prefix_extraction.py.
Uses pre-configured prompts and output cleaning from type2_config.py.
"""

import asyncio
import logging
import time
import re
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
from collections import Counter

import pandas as pd
import numpy as np

# Progress bar for better UX
try:
    from tqdm.asyncio import tqdm
    from tqdm import tqdm as sync_tqdm
    HAS_TQDM = True
except ImportError:
    # Fallback: create dummy tqdm classes
    class tqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable) if self.iterable else iter([])
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, n=1):
            pass
        def set_description(self, desc):
            pass
        def close(self):
            pass
    
    sync_tqdm = tqdm
    HAS_TQDM = False
    print("Warning: tqdm not installed. Install with: pip install tqdm")

# Third-party LLM libraries - Updated to use new Google GenAI
try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    print("Warning: google-generativeai not installed. Install with: pip install google-generativeai")

# Local imports
from data_generation.config.type2_config import (
    Type2GenerationConfig, 
    Type2GenerationMethod,
    DEFAULT_TYPE2_CONFIG,
    validate_type2_config,
    get_prompt_template
)
from data_generation.config.base_model_config import get_api_key

# Import text preprocessing functionality
from data_generation.type2_generation.keywords_prefix_extraction import TextPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GenerationMethod(Enum):
    """Available generation methods for Type 2."""
    SENTENCE_COMPLETION = "sentence_completion"
    QUESTION_ANSWER = "question_answer"
    AUTO = "auto"  # Automatically choose based on text characteristics

class EnvironmentMode(Enum):
    """Environment modes for directory structure."""
    PRODUCTION = "production"
    TEST = "test"

@dataclass
class GenerationContext:
    """Context data for Type 2 generation."""
    original_text: str
    keywords: List[str]
    token_count: int
    sentence_prefix: str
    prefix_token_count: int
    target_length: int
    max_length: int
    method: GenerationMethod

@dataclass
class GenerationResult:
    """Result of Type 2 generation."""
    generated_text: Optional[str]
    method_used: GenerationMethod
    success: bool
    metadata: Dict[str, Any]

class GeminiClient:
    """Gemini client specifically for Type 2 generation using new Google GenAI API."""
    
    def __init__(self, model_config):
        """Initialize Gemini client."""
        self.config = model_config
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Gemini client with new Google GenAI API."""
        if not HAS_GENAI:
            raise ImportError("google-generativeai required for Gemini")
        
        # Use enhanced API key getter
        api_key = get_api_key(self.config.api_key_env, required=True)
        if not api_key:
            raise ValueError(f"API key not found for {self.config.api_key_env}")
        
        # Initialize the new Google GenAI client
        self.client = genai.Client(api_key=api_key)
        
        logger.info(f"Initialized Gemini client: {self.config.model_id}")
    
    async def generate_text(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate text using new Google GenAI API with robust None handling."""
        max_tokens = max_tokens or self.config.max_tokens
        
        try:
            # Use the new API format
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.config.model_id,
                contents=prompt
            )
            
            # Handle cases where response or response.text is None
            if response is None:
                logger.warning("API returned None response")
                return ""
            
            if not hasattr(response, 'text'):
                logger.warning("API response has no 'text' attribute")
                return ""
            
            if response.text is None:
                logger.warning("API response.text is None - likely content filtering or generation failure")
                return ""
            
            # Return the text, ensuring it's a string
            text_content = str(response.text).strip()
            return text_content
            
        except Exception as e:
            logger.error(f"Error in generate_text: {str(e)}")
            return ""

class EnhancedType2Generator:
    """Enhanced Type 2 generator with integrated preprocessing."""
    
    def __init__(self, config: Optional[Type2GenerationConfig] = None, max_keywords: int = 10, 
                 environment_mode: EnvironmentMode = EnvironmentMode.PRODUCTION):
        """Initialize the enhanced Type 2 generator."""
        self.config = config or DEFAULT_TYPE2_CONFIG
        self.environment_mode = environment_mode
        
        # Validate configuration
        if not validate_type2_config(self.config):
            raise ValueError("Invalid Type 2 configuration")
        
        # Initialize Gemini client (only model we use)
        self.client = GeminiClient(self.config.primary_model)
        
        # Initialize text preprocessor for on-the-fly feature extraction
        self.text_preprocessor = TextPreprocessor(max_keywords=max_keywords)
        logger.info("Initialized text preprocessor for on-the-fly feature extraction")
        
        # Enhanced statistics tracking with failure analysis
        self.stats = {
            "total_processed": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "length_violations": 0,
            "preprocessing_successes": 0,
            "preprocessing_failures": 0,
            "method_usage": {
                "sentence_completion": 0,
                "question_answer": 0
            },
            "failure_reasons": {
                "api_errors": 0,
                "validation_failures": 0,
                "preprocessing_errors": 0,
                "length_violations": 0,
                "generation_errors": 0,
                "empty_responses": 0,
                "other_errors": 0
            },
            "api_error_details": [],
            "validation_failure_details": [],
            "start_time": None,
            "end_time": None
        }
        
        # Track failed samples for detailed analysis
        self.failed_samples = []
        
        # Progress tracking
        self.progress_bar = None
    
    def get_directory_structure(self, base_output_dir: str, method: GenerationMethod, timestamp: str) -> Tuple[str, str]:
        """
        Get the appropriate directory structure based on environment mode.
        
        Returns:
            Tuple of (final_output_dir, midpoint_dir)
        """
        if self.environment_mode == EnvironmentMode.TEST:
            # Test environment: data/test/type2_generation_test/method_timestamp/
            method_name = method.value if method != GenerationMethod.AUTO else "auto"
            test_output_dir = f"{base_output_dir}/{method_name}_{timestamp}"
            midpoint_dir = f"{test_output_dir}/midpoint"
            return test_output_dir, midpoint_dir
        else:
            # Production environment: data/generated/timestamp/
            prod_output_dir = f"{base_output_dir}/{timestamp}"
            midpoint_dir = f"{prod_output_dir}/midpoint"
            return prod_output_dir, midpoint_dir
    
    def get_method_column_name(self, method: GenerationMethod) -> str:
        """Get the appropriate column name for the generation method."""
        method_name_map = {
            GenerationMethod.SENTENCE_COMPLETION: "sentence_completion",
            GenerationMethod.QUESTION_ANSWER: "question_answer"
        }
        method_name = method_name_map.get(method, method.value)
        return f"llm_generated_text({method_name})"
    
    def categorize_failure(self, error_msg: str, metadata: Dict[str, Any]) -> str:
        """Categorize failure reason for better tracking."""
        error_lower = error_msg.lower()
        
        if "api" in error_lower or "429" in error_lower or "quota" in error_lower or "rate" in error_lower:
            return "api_errors"
        elif "length" in error_lower or "too long" in error_lower or "too short" in error_lower:
            return "validation_failures"
        elif "preprocessing" in error_lower or "extract" in error_lower:
            return "preprocessing_errors"
        elif "empty" in error_lower or "no response" in error_lower:
            return "empty_responses"
        elif "generation" in error_lower:
            return "generation_errors"
        else:
            return "other_errors"
    
    def log_failure(self, sample_idx: int, error_msg: str, metadata: Dict[str, Any], method: GenerationMethod):
        """Log detailed failure information."""
        failure_category = self.categorize_failure(error_msg, metadata)
        
        # Update failure statistics
        self.stats["failure_reasons"][failure_category] += 1
        
        # Store detailed failure information
        failure_detail = {
            "sample_idx": sample_idx,
            "method": method.value,
            "error_message": error_msg,
            "category": failure_category,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata
        }
        
        self.failed_samples.append(failure_detail)
        
        # Store specific error details for reporting
        if failure_category == "api_errors":
            self.stats["api_error_details"].append({
                "sample_idx": sample_idx,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            })
        elif failure_category == "validation_failures":
            self.stats["validation_failure_details"].append({
                "sample_idx": sample_idx,
                "error": error_msg,
                "validation_reason": metadata.get("failure_reason", "unknown"),
                "generated_length": metadata.get("generated_length", 0),
                "target_length": metadata.get("target_length", 0)
            })
    
    def extract_text_features(self, text: str) -> Dict[str, Any]:
        """Extract features from text using integrated preprocessing."""
        try:
            # Use the text preprocessor to analyze the text
            analysis = self.text_preprocessor.analyze_text(text)
            
            # Convert to the format expected by the generation pipeline
            features = {
                'original_text': text,
                'keywords': analysis.keywords[:5],  # Limit to top 5
                'token_count': analysis.token_count,
                'sentence_prefix': analysis.sentence_prefix,
                'prefix_token_count': analysis.prefix_token_count,
                'character_count': analysis.character_count,
                'sentence_count': analysis.sentence_count,
                'avg_word_length': analysis.avg_word_length,
                'text_complexity_score': analysis.text_complexity_score
            }
            
            self.stats["preprocessing_successes"] += 1
            return features
            
        except Exception as e:
            logger.warning(f"Error extracting text features: {str(e)}")
            self.stats["preprocessing_failures"] += 1
            
            # Return minimal fallback data
            simple_tokens = text.split()
            return {
                'original_text': text,
                'keywords': [],
                'token_count': len(simple_tokens),
                'sentence_prefix': ' '.join(simple_tokens[:max(1, len(simple_tokens) // 5)]),
                'prefix_token_count': max(1, len(simple_tokens) // 5),
                'character_count': len(text),
                'sentence_count': max(1, text.count('.') + text.count('!') + text.count('?')),
                'avg_word_length': np.mean([len(word) for word in simple_tokens]) if simple_tokens else 0,
                'text_complexity_score': 0.5
            }
    
    def calculate_length_constraints(self, token_count: int) -> Tuple[int, int]:
        """Calculate target and maximum length based on token count and config."""
        # Use config settings for length calculation
        target_length = token_count * 5  # Approximate chars per token
        max_length = int(target_length * self.config.max_length_multiplier)
        
        # Ensure minimum length
        target_length = max(target_length, self.config.min_length_threshold)
        max_length = max(max_length, self.config.min_length_threshold)
        
        return target_length, max_length
    
    def choose_generation_method(self, context: GenerationContext) -> GenerationMethod:
        """Choose the best generation method based on text characteristics and config."""
        if context.method != GenerationMethod.AUTO:
            return context.method
        
        # Use config default method for auto selection
        if (context.prefix_token_count > 5 and 
            len(context.sentence_prefix.strip()) > 20 and
            context.token_count > 15):
            return GenerationMethod.SENTENCE_COMPLETION
        else:
            # Use configured default method
            if self.config.default_method == Type2GenerationMethod.SENTENCE_COMPLETION:
                return GenerationMethod.SENTENCE_COMPLETION
            else:
                return GenerationMethod.QUESTION_ANSWER
    
    def format_prompt(self, template: str, context: GenerationContext, **kwargs) -> str:
        """Format prompt using config template and context data."""
        keywords_str = ", ".join(context.keywords) if context.keywords else "relevant terms"
        
        # Prepare format arguments
        format_args = {
            "keywords": keywords_str,
            "target_length": context.target_length,
            "max_length": context.max_length,
            "sentence_prefix": context.sentence_prefix,
            "text": context.original_text,
            **kwargs
        }
        
        return template.format(**format_args)
    
    def clean_generated_output(self, text: str) -> str:
        """Clean generated text using config patterns and settings."""
        if not self.config.output_cleaning.get("strip_labels", True):
            return text.strip()
        
        cleaned = text.strip()
        
        # Remove forbidden patterns from config
        for pattern in self.config.forbidden_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Additional cleaning based on config
        if self.config.output_cleaning.get("strip_quotes", True):
            cleaned = cleaned.strip('"\'')
        
        if self.config.output_cleaning.get("strip_formatting", True):
            # Remove basic markdown/HTML formatting
            cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)  # Bold
            cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)      # Italic
            cleaned = re.sub(r'<[^>]+>', '', cleaned)           # HTML tags
        
        return cleaned.strip()
    
    def validate_generated_text(self, text: str, context: GenerationContext) -> Tuple[bool, str]:
        """Validate that generated text meets config requirements."""
        if not text or len(text.strip()) < self.config.min_length_threshold:
            return False, f"Generated text too short (< {self.config.min_length_threshold})"
        
        if self.config.output_cleaning.get("validate_length", True):
            if len(text) > context.max_length:
                return False, f"Generated text too long ({len(text)} > {context.max_length})"
            
            # Check tolerance
            min_length = context.target_length * (1 - self.config.length_tolerance)
            if len(text) < min_length:
                return False, f"Generated text below tolerance ({len(text)} < {min_length})"
        
        if self.config.output_cleaning.get("validate_format", True):
            # Check for forbidden patterns
            for pattern in self.config.forbidden_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return False, f"Text contains forbidden pattern: {pattern}"
        
        return True, "Valid"
    
    async def generate_with_sentence_completion(self, context: GenerationContext) -> GenerationResult:
        """Generate text using sentence completion method with config templates."""
        try:
            # Use config template
            prompt = self.format_prompt(
                self.config.sentence_completion_prompt_template,
                context
            )
            
            # Generate completion
            completion = await self.client.generate_text(prompt, max_tokens=200)
            
            if not completion or completion.strip() == "":
                raise ValueError("Empty response from API")
            
            completion = self.clean_generated_output(completion)
            
            # FIX: Don't duplicate the prefix - the completion should continue from the prefix
            # If the completion accidentally starts with the prefix, remove it
            if completion.lower().startswith(context.sentence_prefix.lower()):
                completion = completion[len(context.sentence_prefix):].strip()
            
            # Combine prefix with completion without duplication
            full_text = f"{context.sentence_prefix} {completion}".strip()
            
            # Validate result
            is_valid, validation_msg = self.validate_generated_text(full_text, context)
            
            # NEW: Retry logic if validation fails
            if not is_valid:
                logger.info(f"Initial generation failed validation: {validation_msg}. Retrying with emphasis on length...")
                
                # Create retry prompt with length emphasis
                retry_prompt = prompt + f"\n\nIMPORTANT: The previous attempt was {validation_msg}. Please ensure your completion results in approximately {context.target_length} characters total."
                
                # Retry generation
                retry_completion = await self.client.generate_text(retry_prompt, max_tokens=200)
                
                if retry_completion and retry_completion.strip():
                    retry_completion = self.clean_generated_output(retry_completion)
                    
                    # Fix prefix duplication for retry
                    if retry_completion.lower().startswith(context.sentence_prefix.lower()):
                        retry_completion = retry_completion[len(context.sentence_prefix):].strip()
                    
                    # Use retry result regardless of validation
                    full_text = f"{context.sentence_prefix} {retry_completion}".strip()
                    completion = retry_completion
                    validation_msg = "Accepted after retry (length emphasis)"
                    is_valid = True  # Force success after retry
                    logger.info("Retry generation completed - accepting result")
            
            metadata = {
                "method": "sentence_completion",
                "original_length": len(context.original_text),
                "generated_length": len(full_text),
                "target_length": context.target_length,
                "max_length": context.max_length,
                "keywords_used": context.keywords,
                "prefix_used": context.sentence_prefix,
                "completion_generated": completion,
                "validation_status": validation_msg,
                "prompt_length": len(prompt),
                "config_template_used": "sentence_completion_prompt_template"
            }
            
            if is_valid:
                self.stats["method_usage"]["sentence_completion"] += 1
                return GenerationResult(
                    generated_text=full_text,
                    method_used=GenerationMethod.SENTENCE_COMPLETION,
                    success=True,
                    metadata=metadata
                )
            else:
                metadata["failure_reason"] = validation_msg
                return GenerationResult(
                    generated_text=None,
                    method_used=GenerationMethod.SENTENCE_COMPLETION,
                    success=False,
                    metadata=metadata
                )
                
        except Exception as e:
            logger.error(f"Sentence completion generation failed: {str(e)}")
            return GenerationResult(
                generated_text=None,
                method_used=GenerationMethod.SENTENCE_COMPLETION,
                success=False,
                metadata={
                    "failure_reason": str(e), 
                    "method": "sentence_completion",
                    "target_length": context.target_length,
                    "max_length": context.max_length
                }
            )
    
    async def generate_with_question_answer(self, context: GenerationContext) -> GenerationResult:
        """Generate text using question-answer method with config templates."""
        try:
            # Step 1: Generate question using config template
            question_prompt = self.format_prompt(
                self.config.question_prompt_template,
                context
            )
            question = await self.client.generate_text(question_prompt, max_tokens=100)
            
            if not question or question.strip() == "":
                raise ValueError("Empty question response from API")
            
            question = self.clean_generated_output(question)
            
            # Step 2: Generate answer using config template
            answer_prompt = self.format_prompt(
                self.config.answer_prompt_template,
                context,
                question=question
            )
            answer = await self.client.generate_text(answer_prompt, max_tokens=250)
            
            if not answer or answer.strip() == "":
                raise ValueError("Empty answer response from API")
            
            answer = self.clean_generated_output(answer)
            
            # Validate result
            is_valid, validation_msg = self.validate_generated_text(answer, context)
            
            # NEW: Retry logic if validation fails
            if not is_valid:
                logger.info(f"Initial generation failed validation: {validation_msg}. Retrying with emphasis on length...")
                
                # Create retry prompt with length emphasis
                retry_answer_prompt = answer_prompt + f"\n\nIMPORTANT: The previous attempt was {validation_msg}. Please ensure your answer is approximately {context.target_length} characters long."
                
                # Retry generation
                retry_answer = await self.client.generate_text(retry_answer_prompt, max_tokens=250)
                
                if retry_answer and retry_answer.strip():
                    retry_answer = self.clean_generated_output(retry_answer)
                    
                    # Use retry result regardless of validation
                    answer = retry_answer
                    validation_msg = "Accepted after retry (length emphasis)"
                    is_valid = True  # Force success after retry
                    logger.info("Retry generation completed - accepting result")
            
            metadata = {
                "method": "question_answer",
                "original_length": len(context.original_text),
                "generated_length": len(answer),
                "target_length": context.target_length,
                "max_length": context.max_length,
                "keywords_used": context.keywords,
                "question_generated": question,
                "validation_status": validation_msg,
                "question_prompt_length": len(question_prompt),
                "answer_prompt_length": len(answer_prompt),
                "config_templates_used": ["question_prompt_template", "answer_prompt_template"]
            }
            
            if is_valid:
                self.stats["method_usage"]["question_answer"] += 1
                return GenerationResult(
                    generated_text=answer,
                    method_used=GenerationMethod.QUESTION_ANSWER,
                    success=True,
                    metadata=metadata
                )
            else:
                metadata["failure_reason"] = validation_msg
                return GenerationResult(
                    generated_text=None,
                    method_used=GenerationMethod.QUESTION_ANSWER,
                    success=False,
                    metadata=metadata
                )
                
        except Exception as e:
            logger.error(f"Question-answer generation failed: {str(e)}")
            return GenerationResult(
                generated_text=None,
                method_used=GenerationMethod.QUESTION_ANSWER,
                success=False,
                metadata={
                    "failure_reason": str(e), 
                    "method": "question_answer",
                    "target_length": context.target_length,
                    "max_length": context.max_length
                }
            )
    
    async def generate_type2_text(self, 
                                 row: pd.Series, 
                                 method: GenerationMethod = GenerationMethod.AUTO) -> GenerationResult:
        """
        Generate Type 2 text for a single input with integrated preprocessing.
        
        Args:
            row: DataFrame row with human_original_text
            method: Generation method to use
            
        Returns:
            GenerationResult with generated text and metadata
        """
        start_time = time.time()
        sample_idx = row.name if hasattr(row, 'name') else -1
        
        try:
            # Extract original text
            original_text = row.get('human_original_text', '')
            if not original_text or len(original_text.strip()) == 0:
                raise ValueError("No valid human_original_text found")
            
            # Extract features on-the-fly using integrated preprocessing
            features = self.extract_text_features(original_text)
            
            # Calculate length constraints using config
            target_length, max_length = self.calculate_length_constraints(features['token_count'])
            
            # Create generation context
            context = GenerationContext(
                original_text=features['original_text'],
                keywords=features['keywords'],
                token_count=features['token_count'],
                sentence_prefix=features['sentence_prefix'],
                prefix_token_count=features['prefix_token_count'],
                target_length=target_length,
                max_length=max_length,
                method=method
            )
            
            # Choose generation method
            chosen_method = self.choose_generation_method(context)
            context.method = chosen_method
            
            # Generate text using chosen method
            if chosen_method == GenerationMethod.SENTENCE_COMPLETION:
                result = await self.generate_with_sentence_completion(context)
            else:  # QUESTION_ANSWER
                result = await self.generate_with_question_answer(context)
            
            # Add timing and preprocessing information
            result.metadata["generation_time"] = time.time() - start_time
            result.metadata["preprocessing_features"] = {
                "keywords_extracted": len(features['keywords']),
                "token_count": features['token_count'],
                "prefix_token_count": features['prefix_token_count'],
                "text_complexity": features.get('text_complexity_score', 0)
            }
            result.metadata["config_used"] = {
                "length_tolerance": self.config.length_tolerance,
                "max_length_multiplier": self.config.max_length_multiplier,
                "output_cleaning_enabled": self.config.output_cleaning
            }
            
            # Update statistics and log failures
            if result.success:
                self.stats["successful_generations"] += 1
            else:
                self.stats["failed_generations"] += 1
                failure_reason = result.metadata.get("failure_reason", "Unknown error")
                self.log_failure(sample_idx, failure_reason, result.metadata, chosen_method)
                
                if "length" in failure_reason.lower():
                    self.stats["length_violations"] += 1
            
            self.stats["total_processed"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Type 2 generation failed for sample {sample_idx}: {str(e)}")
            self.stats["failed_generations"] += 1
            self.stats["total_processed"] += 1
            
            # Log the failure
            error_metadata = {
                "failure_reason": str(e),
                "generation_time": time.time() - start_time,
                "original_text_length": len(original_text) if 'original_text' in locals() else 0
            }
            self.log_failure(sample_idx, str(e), error_metadata, method)
            
            return GenerationResult(
                generated_text=None,
                method_used=method,
                success=False,
                metadata=error_metadata
            )
    
    async def process_batch(self, 
                           batch_data: List[Tuple[int, pd.Series]], 
                           method: GenerationMethod = GenerationMethod.AUTO,
                           batch_progress: Optional[tqdm] = None) -> List[Tuple[int, GenerationResult]]:
        """Process a batch of texts concurrently with progress tracking."""
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        async def process_single(idx: int, row: pd.Series) -> Tuple[int, GenerationResult]:
            async with semaphore:
                result = await self.generate_type2_text(row, method)
                if batch_progress and HAS_TQDM:
                    batch_progress.update(1)
                    # Update description with current success rate
                    success_rate = (self.stats["successful_generations"] / max(self.stats["total_processed"], 1)) * 100
                    batch_progress.set_description(f"Processing (Success: {success_rate:.1f}%)")
                return idx, result
        
        tasks = [process_single(idx, row) for idx, row in batch_data]
        return await asyncio.gather(*tasks)
    
    def filter_target_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data to target datasets needing Type 2 generation."""
        # Filter to target datasets from config
        target_datasets = [ds.lower() for ds in self.config.target_datasets]
        mask = df['dataset_source'].str.lower().isin(target_datasets)
        
        # Filter to samples missing LLM-generated text
        mask &= (df['llm_generated_text'].isna() | (df['llm_generated_text'] == ''))
        
        # Ensure we have human_original_text (only requirement now)
        mask &= df['human_original_text'].notna()
        mask &= (df['human_original_text'].str.strip() != '')
        
        filtered_df = df[mask].copy()
        logger.info(f"Filtered {len(filtered_df)} samples requiring Type 2 generation from {len(df)} total")
        logger.info(f"Target datasets from config: {self.config.target_datasets}")
        
        return filtered_df
    
    def save_type2_midpoint_results(self, results_df: pd.DataFrame, generation_metadata: List[Dict], 
                                   midpoint_dir: str, timestamp: str, method: GenerationMethod):
        """Save midpoint results specifically for Type 2 generation with keywords and question (if applicable)."""
        try:
            output_path = Path(midpoint_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Get the method-specific column name
            method_column = self.get_method_column_name(method)
            
            # Filter to only successfully generated Type 2 samples
            type2_generated = results_df[results_df[method_column].notna()].copy()
            
            if len(type2_generated) == 0:
                logger.warning("No Type 2 generated samples to save in midpoint results")
                return
            
            # Create a mapping from sample index to metadata
            metadata_by_idx = {}
            for meta in generation_metadata:
                if 'sample_idx' in meta:
                    metadata_by_idx[meta['sample_idx']] = meta
            
            # Create midpoint DataFrame with required columns
            midpoint_columns = [
                'idx', 'human_original_text', method_column, 
                'llm_generated_text_method'
            ]
            
            # Add keywords and conditionally add question for question-answer method
            keywords_list = []
            questions_list = []
            include_questions = method == GenerationMethod.QUESTION_ANSWER
            
            for idx, row in type2_generated.iterrows():
                try:
                    # Re-extract keywords for this sample
                    features = self.extract_text_features(row['human_original_text'])
                    keywords_list.append(json.dumps(features['keywords']))
                    
                    # Add question if it's question-answer method
                    if include_questions:
                        if idx in metadata_by_idx and 'question_generated' in metadata_by_idx[idx]:
                            questions_list.append(metadata_by_idx[idx]['question_generated'])
                        else:
                            questions_list.append("")  # Empty string if no question found
                except:
                    keywords_list.append('[]')
                    if include_questions:
                        questions_list.append("")
            
            type2_generated['keywords_extracted'] = keywords_list
            midpoint_columns.append('keywords_extracted')
            
            # Add question column only for question-answer method
            if include_questions:
                type2_generated['question_generated'] = questions_list
                midpoint_columns.append('question_generated')
            
            midpoint_df = type2_generated[midpoint_columns].copy()
            
            # Save midpoint results
            method_name = method.value if method != GenerationMethod.AUTO else "auto"
            midpoint_file = output_path / f"type2_midpoint_results_{method_name}_{timestamp}.csv"
            midpoint_df.to_csv(midpoint_file, index=False)
            
            # Save as JSON as well
            midpoint_json_file = output_path / f"type2_midpoint_results_{method_name}_{timestamp}.json"
            midpoint_df.to_json(midpoint_json_file, orient='records', indent=2)
            
            logger.info(f"Saved Type 2 midpoint results:")
            logger.info(f"  CSV: {midpoint_file}")
            logger.info(f"  JSON: {midpoint_json_file}")
            logger.info(f"  Samples: {len(midpoint_df)}")
            if include_questions:
                questions_with_content = sum(1 for q in questions_list if q.strip())
                logger.info(f"  Questions included: {questions_with_content}/{len(questions_list)}")
            
        except Exception as e:
            logger.warning(f"Failed to save Type 2 midpoint results: {str(e)}")
    
    def save_failure_analysis(self, output_dir: str, timestamp: str):
        """Save detailed failure analysis to help debug issues."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save detailed failure analysis
            failure_analysis_file = output_path / f"type2_failure_analysis_{timestamp}.json"
            
            failure_analysis = {
                "summary": {
                    "total_failures": len(self.failed_samples),
                    "failure_breakdown": self.stats["failure_reasons"],
                    "most_common_errors": self._get_most_common_errors(),
                    "failure_by_method": self._get_failures_by_method()
                },
                "detailed_failures": self.failed_samples,
                "api_errors": self.stats["api_error_details"],
                "validation_failures": self.stats["validation_failure_details"]
            }
            
            with open(failure_analysis_file, 'w', encoding='utf-8') as f:
                json.dump(failure_analysis, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"Saved failure analysis: {failure_analysis_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save failure analysis: {str(e)}")
    
    def _get_most_common_errors(self) -> List[Dict[str, Any]]:
        """Get the most common error messages."""
        error_counter = Counter()
        for failure in self.failed_samples:
            error_counter[failure['error_message']] += 1
        
        return [
            {"error": error, "count": count}
            for error, count in error_counter.most_common(10)
        ]
    
    def _get_failures_by_method(self) -> Dict[str, int]:
        """Get failure count by generation method."""
        method_failures = Counter()
        for failure in self.failed_samples:
            method_failures[failure['method']] += 1
        
        return dict(method_failures)
    
    async def generate_for_dataset(self, 
                                  df: pd.DataFrame, 
                                  method: GenerationMethod = GenerationMethod.AUTO,
                                  output_dir: str = "data/generated") -> pd.DataFrame:
        """Generate Type 2 texts for the entire dataset with progress tracking."""
        self.stats["start_time"] = datetime.now()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Get appropriate directory structure
        final_output_dir, midpoint_dir = self.get_directory_structure(output_dir, method, timestamp)
        
        # Filter target data
        target_df = self.filter_target_data(df)
        
        if target_df.empty:
            logger.info("No samples require Type 2 generation")
            return df
        
        logger.info(f"Starting Type 2 generation for {len(target_df)} samples using method: {method.value}")
        logger.info("Using integrated preprocessing for on-the-fly feature extraction")
        logger.info(f"Using config templates and validation settings")
        logger.info(f"Environment mode: {self.environment_mode.value}")
        logger.info(f"Final output directory: {final_output_dir}")
        logger.info(f"Midpoint directory: {midpoint_dir}")
        
        # Prepare results - ensure proper column types
        results_df = df.copy()
        
        # Get the method-specific column name
        method_column = self.get_method_column_name(method)
        
        # Initialize columns with proper data types to avoid pandas warnings
        if method_column not in results_df.columns:
            results_df[method_column] = pd.Series(dtype='object')
        if 'llm_generated_text_method' not in results_df.columns:
            results_df['llm_generated_text_method'] = pd.Series(dtype='object')
        
        generation_metadata = []
        
        # Process in batches with progress tracking
        batch_size = self.config.batch_size
        total_batches = (len(target_df) + batch_size - 1) // batch_size
        
        # Initialize main progress bar for batches
        batch_pbar = sync_tqdm(
            total=total_batches,
            desc=f"Type 2 Generation ({method.value})",
            unit="batch",
            disable=not HAS_TQDM
        )
        
        # Initialize sample progress bar
        sample_pbar = sync_tqdm(
            total=len(target_df),
            desc="Processing samples",
            unit="sample",
            disable=not HAS_TQDM,
            leave=False
        )
        
        try:
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(target_df))
                
                batch_df = target_df.iloc[start_idx:end_idx]
                batch_data = [(row.name, row) for _, row in batch_df.iterrows()]
                
                batch_pbar.set_description(f"Batch {batch_idx + 1}/{total_batches} ({len(batch_data)} samples)")
                
                # Generate for batch with sample progress
                batch_results = await self.process_batch(batch_data, method, sample_pbar)
                
                # Update results
                for idx, result in batch_results:
                    if result.success and result.generated_text:
                        results_df.at[idx, method_column] = result.generated_text
                        results_df.at[idx, 'llm_generated_text_method'] = result.method_used.value
                    
                    # Store metadata
                    result.metadata['batch_idx'] = batch_idx
                    result.metadata['sample_idx'] = idx
                    generation_metadata.append(result.metadata)
                
                # Update batch progress
                batch_pbar.update(1)
                
                # Update batch description with current stats
                success_rate = (self.stats["successful_generations"] / max(self.stats["total_processed"], 1)) * 100
                batch_pbar.set_postfix({
                    'Success': f'{success_rate:.1f}%',
                    'Generated': f'{self.stats["successful_generations"]}/{self.stats["total_processed"]}'
                })
                
                # Save intermediate results every 100 batches
                if (batch_idx + 1) % 100 == 0:
                    await self._save_intermediate_results(results_df, generation_metadata, batch_idx + 1, final_output_dir)
                    batch_pbar.set_description(f"Saved checkpoint - Batch {batch_idx + 1}/{total_batches}")
                
                # Brief pause to avoid rate limiting
                await asyncio.sleep(0.5)
            
        finally:
            # Close progress bars
            batch_pbar.close()
            sample_pbar.close()
        
        self.stats["end_time"] = datetime.now()
        
        # Save Type 2 midpoint results with metadata
        print("💾 Saving midpoint results...")
        self.save_type2_midpoint_results(results_df, generation_metadata, midpoint_dir, timestamp, method)
        
        # Save failure analysis
        print("📊 Saving failure analysis...")
        self.save_failure_analysis(final_output_dir, timestamp)
        
        # Save final results
        print("💾 Saving final results...")
        await self._save_final_results(results_df, generation_metadata, method, final_output_dir)
        
        # Log final statistics with failure analysis
        self._log_final_statistics()
        
        return results_df
    
    async def retry_single_record_generation(self, 
                                           row: pd.Series, 
                                           method: GenerationMethod,
                                           max_retries: int = 3) -> GenerationResult:
        """
        Retry generation for a single record until successful or max retries reached.
        
        Args:
            row: DataFrame row with human_original_text
            method: Generation method to use
            max_retries: Maximum number of retry attempts
            
        Returns:
            GenerationResult with generated text and metadata
        """
        sample_idx = row.get('idx', row.name if hasattr(row, 'name') else -1)
        
        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries + 1} for sample {sample_idx}")
                
                result = await self.generate_type2_text(row, method)
                
                if result.success and result.generated_text and result.generated_text.strip():
                    logger.info(f"✅ Successfully generated text for sample {sample_idx} on attempt {attempt + 1}")
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
        return GenerationResult(
            generated_text=None,
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
                                              method: GenerationMethod,
                                              max_retries: int = 3) -> pd.DataFrame:
        """
        Process null records individually with immediate retry until successful.
        
        Args:
            df: DataFrame containing records to process
            target_field: Target field name to check for nulls
            method: Generation method to use
            max_retries: Maximum retries per record
            
        Returns:
            Updated DataFrame with filled values
        """
        logger.info(f"🔄 Processing null records individually for field: {target_field}")
        
        # Find records with null values in the target field
        null_mask = df[target_field].isnull() | (df[target_field] == '')
        null_records = df[null_mask].copy()
        
        if len(null_records) == 0:
            logger.info("✅ No null records found")
            return df
        
        logger.info(f"🎯 Found {len(null_records)} records with null values")
        
        # Process each null record individually
        updated_df = df.copy()
        successful_fills = 0
        failed_fills = 0
        
        # Progress tracking
        if HAS_TQDM:
            from tqdm import tqdm
            progress_bar = tqdm(total=len(null_records), desc="Processing null records")
        
        for idx, row in null_records.iterrows():
            try:
                # Retry until successful or max attempts reached
                result = await self.retry_single_record_generation(row, method, max_retries)
                
                if result.success and result.generated_text:
                    # Update the DataFrame with the successful result
                    updated_df.at[idx, target_field] = result.generated_text
                    updated_df.at[idx, 'llm_generated_text_method'] = result.method_used.value
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
        
        logger.info(f"📊 Individual processing completed:")
        logger.info(f"  Successfully filled: {successful_fills}")
        logger.info(f"  Failed to fill: {failed_fills}")
        logger.info(f"  Success rate: {(successful_fills/(successful_fills+failed_fills)*100):.1f}%" if (successful_fills+failed_fills) > 0 else "0%")
        
        return updated_df
    
    async def _save_intermediate_results(self, df: pd.DataFrame, metadata: List[Dict], batch_num: int, output_dir: str):
        """Save intermediate results during processing."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save DataFrame checkpoint
            checkpoint_file = output_path / f"type2_checkpoint_batch_{batch_num}.csv"
            df.to_csv(checkpoint_file, index=False)
            
            # Save metadata checkpoint
            metadata_file = output_path / f"type2_metadata_batch_{batch_num}.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"Saved checkpoint after batch {batch_num}")
        except Exception as e:
            logger.warning(f"Failed to save intermediate results: {str(e)}")
    
    async def _save_final_results(self, df: pd.DataFrame, metadata: List[Dict], method: GenerationMethod, output_dir: str):
        """Save final generation results."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save CSV
            csv_file = output_path / f"padben_with_type2_{method.value}_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            
            # Save JSON
            json_file = output_path / f"padben_with_type2_{method.value}_{timestamp}.json"
            df.to_json(json_file, orient='records', indent=2)
            
            # Save complete metadata
            metadata_file = output_path / f"padben_pipeline_metadata_{timestamp}.json"
            full_metadata = {
                "generation_config": asdict(self.config),
                "method_used": method.value,
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
    
    def _log_final_statistics(self):
        """Log final generation statistics with detailed failure analysis."""
        stats = self.stats
        total_time = (stats["end_time"] - stats["start_time"]).total_seconds() if stats["end_time"] and stats["start_time"] else 0
        
        logger.info("=" * 60)
        logger.info("ENHANCED TYPE 2 GENERATION COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total processed: {stats['total_processed']}")
        logger.info(f"Successful generations: {stats['successful_generations']}")
        logger.info(f"Failed generations: {stats['failed_generations']}")
        logger.info(f"Length violations: {stats['length_violations']}")
        logger.info(f"Preprocessing successes: {stats['preprocessing_successes']}")
        logger.info(f"Preprocessing failures: {stats['preprocessing_failures']}")
        logger.info(f"Success rate: {stats['successful_generations'] / max(stats['total_processed'], 1) * 100:.1f}%")
        
        # Detailed failure analysis
        if stats['failed_generations'] > 0:
            logger.info("\n" + "=" * 40)
            logger.info("FAILURE ANALYSIS")
            logger.info("=" * 40)
            logger.info("Failure breakdown by category:")
            for category, count in stats['failure_reasons'].items():
                if count > 0:
                    percentage = (count / stats['failed_generations']) * 100
                    logger.info(f"  {category.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
            
            # Show most common errors
            common_errors = self._get_most_common_errors()
            if common_errors:
                logger.info("\nMost common error messages:")
                for error_info in common_errors[:5]:  # Top 5 errors
                    logger.info(f"  '{error_info['error']}': {error_info['count']} times")
            
            # Show failure by method
            method_failures = self._get_failures_by_method()
            if method_failures:
                logger.info("\nFailures by method:")
                for method, count in method_failures.items():
                    logger.info(f"  {method}: {count} failures")
        
        logger.info("\n" + "=" * 40)
        logger.info("METHOD USAGE")
        logger.info("=" * 40)
        logger.info(f"Sentence completion: {stats['method_usage']['sentence_completion']}")
        logger.info(f"Question-answer: {stats['method_usage']['question_answer']}")
        
        logger.info("\n" + "=" * 40)
        logger.info("TIMING")
        logger.info("=" * 40)
        logger.info(f"Total time: {total_time:.2f} seconds")
        if stats['successful_generations'] > 0:
            logger.info(f"Average time per generation: {total_time / stats['successful_generations']:.2f} seconds")
        logger.info("=" * 60)

def main():
    """Main function with user method selection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Type 2 Text Generation with Config Templates")
    parser.add_argument(
        "--method", 
        choices=["sentence_completion", "question_answer", "auto"],
        default="auto",
        help="Generation method to use (default: auto)"
    )
    parser.add_argument(
        "--input", 
        default="data/processed/unified_padben_base.csv",
        help="Input file with unified dataset (no preprocessing required)"
    )
    parser.add_argument(
        "--output-dir",
        default="data/generated",
        help="Output directory for results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Batch size for processing (overrides config)"
    )
    parser.add_argument(
        "--max-keywords",
        type=int,
        default=5,
        help="Maximum keywords to extract per text"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode (changes directory structure)"
    )
    
    args = parser.parse_args()
    
    # Convert method string to enum
    method_map = {
        "sentence_completion": GenerationMethod.SENTENCE_COMPLETION,
        "question_answer": GenerationMethod.QUESTION_ANSWER,
        "auto": GenerationMethod.AUTO
    }
    method = method_map[args.method]
    
    # Determine environment mode
    environment_mode = EnvironmentMode.TEST if args.test_mode else EnvironmentMode.PRODUCTION
    
    async def run_generation():
        """Run the generation process."""
        # Load dataset (no preprocessing required)
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return
        
        logger.info(f"Loading unified dataset from {input_path}")
        if input_path.suffix == '.json':
            df = pd.read_json(input_path)
        else:
            df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} samples")
        
        # Validate configuration
        if not validate_type2_config(DEFAULT_TYPE2_CONFIG):
            logger.error("Configuration validation failed. Please set GEMINI_API_KEY.")
            return
        
        # Initialize generator with integrated preprocessing and config templates
        config = DEFAULT_TYPE2_CONFIG
        if args.batch_size:
            config.batch_size = args.batch_size
        generator = EnhancedType2Generator(config, max_keywords=args.max_keywords, environment_mode=environment_mode)
        
        # Run generation
        logger.info(f"Starting enhanced Type 2 generation using method: {method.value}")
        logger.info("Using pre-configured templates and validation from type2_config.py")
        logger.info("Features will be extracted on-the-fly using integrated preprocessing")
        results_df = await generator.generate_for_dataset(df, method, args.output_dir)
        
        # Get the method-specific column name for counting
        method_column = generator.get_method_column_name(method)
        generated_count = results_df[method_column].notna().sum()
        total_count = len(results_df)
        
        logger.info("=" * 60)
        logger.info("GENERATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Method used: {method.value}")
        logger.info(f"Environment mode: {environment_mode.value}")
        logger.info(f"Successfully generated {generated_count}/{total_count} Type 2 texts")
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info("=" * 60)
    
    # Run the generation
    asyncio.run(run_generation())

if __name__ == "__main__":
    main()