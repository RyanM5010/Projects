"""
Comprehensive Error Handling and Null Value Generation for PADBen.

This module provides functionality to generate missing values for both Type 2 and Type 4
datasets, ensuring all required columns are filled with appropriate text content.

Features:
- Supports both Type 2 (LLM-generated text) and Type 4 (LLM-paraphrased original text)
- Automatic retry mechanism for failed generations
- Configurable generation parameters
- Progress tracking and logging with tqdm progress bars
- Batch processing for efficiency
"""

# Add project root to Python path for imports
import sys
import os
from pathlib import Path

# Get the project root directory (PADBen folder)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import re
from datetime import datetime

# Progress bar for better UX
try:
    from tqdm.asyncio import tqdm as async_tqdm
    from tqdm import tqdm
    HAS_TQDM = True
    logger = logging.getLogger(__name__)
    logger.info("✅ tqdm available for progress bars")
except ImportError:
    # Fallback: create dummy tqdm classes
    class tqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable
            self.total = kwargs.get('total', 0)
            self.desc = kwargs.get('desc', '')
            self.current = 0
            
        def __iter__(self):
            return iter(self.iterable) if self.iterable else iter([])
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
        
        def update(self, n=1):
            self.current += n
            if self.total > 0:
                print(f"\r{self.desc}: {self.current}/{self.total} ({self.current/self.total*100:.1f}%)", end='', flush=True)
        
        def set_description(self, desc):
            self.desc = desc
            
        def set_postfix(self, **kwargs):
            pass
            
        def close(self):
            if self.total > 0:
                print()  # New line after completion
    
    async_tqdm = tqdm
    HAS_TQDM = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ tqdm not installed. Install with: pip install tqdm")

# Import configurations
from data_generation.config.type2_config import (
    DEFAULT_TYPE2_CONFIG, 
    Type2GenerationConfig, 
    Type2GenerationMethod,
    get_prompt_template
)
from data_generation.config.type4_config import (
    DEFAULT_TYPE4_CONFIG, 
    Type4GenerationConfig, 
    Type4ParaphraseMethod
)
from data_generation.config.base_model_config import LLMModelConfig
from data_generation.config.secrets_manager import get_api_key

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetType(Enum):
    """Supported dataset types for error handling."""
    TYPE2 = "type2"
    TYPE4 = "type4"

@dataclass
class GenerationResult:
    """Result of a text generation attempt."""
    success: bool
    text: Optional[str] = None
    method: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = 0
    generation_time: float = 0.0

@dataclass
class ErrorHandlingConfig:
    """Configuration for error handling and null value generation."""
    
    # Retry settings
    max_retries: int = 5
    retry_delay: float = 2.0
    exponential_backoff: bool = True
    
    # Quality validation
    min_text_length: int = 10
    max_text_length: int = 500
    validate_content: bool = True
    
    # Processing settings
    batch_size: int = 10
    max_concurrent: int = 3
    save_progress: bool = True
    progress_interval: int = 50
    
    # Output settings
    output_dir: str = "data/processed/error_handling"
    timestamp_suffix: bool = True
    
    # Progress bar settings
    show_progress: bool = True
    progress_leave: bool = True
    
    # Type-specific configurations
    type2_config: Type2GenerationConfig = field(default_factory=lambda: DEFAULT_TYPE2_CONFIG)
    type4_config: Type4GenerationConfig = field(default_factory=lambda: DEFAULT_TYPE4_CONFIG)

class TextGenerator:
    """Base class for text generation with retry mechanism."""
    
    def __init__(self, config: ErrorHandlingConfig):
        self.config = config
        self.stats = {
            'total_processed': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'retry_attempts': 0
        }
        logger.debug(f"Initialized {self.__class__.__name__} with config: {config}")
    
    async def generate_with_retry(
        self, 
        record: Dict[str, Any], 
        column_name: str,
        generation_method: str,
        progress_callback: Optional[callable] = None
    ) -> GenerationResult:
        """Generate text with automatic retry mechanism."""
        
        logger.debug(f"Starting generation for column '{column_name}' using method '{generation_method}'")
        
        for attempt in range(self.config.max_retries + 1):
            try:
                logger.debug(f"Generation attempt {attempt + 1}/{self.config.max_retries + 1} for column '{column_name}'")
                start_time = time.time()
                
                # Generate text based on method
                generated_text = await self._generate_text(record, column_name, generation_method)
                
                generation_time = time.time() - start_time
                logger.debug(f"Generated text in {generation_time:.2f}s: '{generated_text[:100]}...'")
                
                # Validate generated text
                if self._validate_text(generated_text, record):
                    self.stats['successful_generations'] += 1
                    logger.info(f"✅ Successfully generated text for column '{column_name}' on attempt {attempt + 1}")
                    
                    # Call progress callback if provided
                    if progress_callback:
                        progress_callback('success')
                    
                    return GenerationResult(
                        success=True,
                        text=generated_text,
                        method=generation_method,
                        retry_count=attempt,
                        generation_time=generation_time
                    )
                else:
                    raise ValueError(f"Generated text failed validation: {generated_text[:100]}...")
                    
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"⚠️ Generation attempt {attempt + 1} failed for {column_name}: {error_msg}")
                self.stats['retry_attempts'] += 1
                
                if attempt < self.config.max_retries:
                    # Calculate delay with exponential backoff
                    delay = self.config.retry_delay
                    if self.config.exponential_backoff:
                        delay *= (2 ** attempt)
                    
                    logger.info(f"🔄 Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
                else:
                    self.stats['failed_generations'] += 1
                    logger.error(f"❌ Failed to generate text for column '{column_name}' after {self.config.max_retries + 1} attempts")
                    
                    # Call progress callback if provided
                    if progress_callback:
                        progress_callback('failed')
                    
                    return GenerationResult(
                        success=False,
                        error=error_msg,
                        retry_count=attempt
                    )
        
        return GenerationResult(success=False, error="Max retries exceeded")
    
    async def _generate_text(
        self, 
        record: Dict[str, Any], 
        column_name: str, 
        method: str
    ) -> str:
        """Override in subclasses for specific generation logic."""
        raise NotImplementedError
    
    def _validate_text(self, text: Optional[str], record: Dict[str, Any]) -> bool:
        """Validate generated text quality."""
        if not text:
            logger.debug("Validation failed: Empty text")
            return False
        
        # Length validation
        text_length = len(text.strip())
        if text_length < self.config.min_text_length:
            logger.debug(f"Validation failed: Text too short ({text_length} < {self.config.min_text_length})")
            return False
        
        if text_length > self.config.max_text_length:
            logger.debug(f"Validation failed: Text too long ({text_length} > {self.config.max_text_length})")
            return False
        
        # Content validation
        if self.config.validate_content:
            # Check for common generation artifacts
            artifacts = [
                r'^(Answer|Question|Completion|Result):\s*',
                r'^\"|\"$',  # Surrounding quotes
                r'Here is the|Here\'s the',
                r'The answer is|The question is',
                r'^I (can\'t|cannot|am unable)',  # Refusal patterns
                r'As an AI|I\'m an AI',  # AI self-reference
            ]
            
            for pattern in artifacts:
                if re.search(pattern, text, re.IGNORECASE):
                    logger.debug(f"Validation failed: Text contains artifact pattern: {pattern}")
                    return False
        
        logger.debug(f"✅ Text validation passed: {text_length} chars")
        return True

class Type2Generator(TextGenerator):
    """Generator for Type 2 (LLM-generated text) columns - SENTENCE COMPLETION ONLY."""
    
    def __init__(self, config: ErrorHandlingConfig):
        logger.info("🔧 Initializing Type2Generator (Sentence Completion Only)...")
        super().__init__(config)
        self.type2_config = config.type2_config
        self.model = None
        self._setup_gemini_client()
    
    def _setup_gemini_client(self):
        """Initialize Gemini client using the same approach as type2_generation.py."""
        logger.info("🔑 Setting up Gemini client for Type 2 generation...")
        
        try:
            # Check if google genai is available (same as type2_generation.py)
            logger.debug("Attempting to import google.genai...")
            try:
                from google import genai
                logger.info("✅ Successfully imported google.genai")
            except ImportError as import_err:
                logger.error(f"❌ Failed to import google.genai: {import_err}")
                raise ImportError("google-generativeai package not found. Install with: pip install google-generativeai")
            
            # Get API key
            logger.debug("Retrieving GEMINI_API_KEY...")
            api_key = get_api_key("GEMINI_API_KEY")
            if not api_key:
                logger.error("❌ GEMINI_API_KEY not found in environment variables")
                raise ValueError("GEMINI_API_KEY not found in environment")
            
            logger.info("✅ GEMINI_API_KEY found successfully")
            
            # Initialize client (same as type2_generation.py approach)
            logger.debug(f"Initializing Gemini client with model: {self.type2_config.primary_model.model_id}")
            
            # Create client using the same pattern as type2_generation.py
            self.client = genai.Client(api_key=api_key)
            
            logger.info(f"✅ Gemini client initialized successfully with model: {self.type2_config.primary_model.model_id}")
            
        except ImportError as e:
            logger.error(f"❌ Import error during Gemini client setup: {e}")
            raise ImportError("google-generativeai package not found. Install with: pip install google-generativeai")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini client: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")
            raise
    
    async def _generate_text(
        self, 
        record: Dict[str, Any], 
        column_name: str, 
        method: str
    ) -> str:
        """Generate Type 2 text using ONLY sentence completion method."""
        
        logger.debug(f"🎯 Generating Type 2 text using sentence completion method")
        
        # Only use sentence completion method
        return await self._generate_sentence_completion(record)
    
    async def _generate_sentence_completion(self, record: Dict[str, Any]) -> str:
        """Generate text using sentence completion method."""
        
        logger.debug("📝 Starting sentence completion generation...")
        
        # Extract sentence prefix from human original text
        human_text = record.get("human_original_text") or record.get("human_original_text(type1)", "")
        if not human_text:
            logger.error("❌ No human original text found for sentence completion")
            raise ValueError("No human original text found for sentence completion")
        
        logger.debug(f"📖 Source text: '{human_text[:100]}...'")
        
        # Create sentence prefix (first part of the sentence)
        words = human_text.split()
        prefix_length = min(len(words) // 3, 10)  # Use first third or max 10 words
        sentence_prefix = " ".join(words[:prefix_length])
        
        logger.debug(f"🔧 Created sentence prefix: '{sentence_prefix}'")
        
        # Prepare prompt
        prompt = self.type2_config.sentence_completion_prompt_template.format(
            sentence_prefix=sentence_prefix,
            keywords="",  # Could be enhanced with keyword extraction
            target_length=len(human_text),
            max_length=int(len(human_text) * 1.2)
        )
        
        logger.debug(f"📝 Generated prompt ({len(prompt)} chars)")
        
        # Generate completion
        response = await self._call_gemini_async(prompt)
        
        # Clean and validate response
        completion = self._clean_response(response)
        
        # Combine prefix with completion to create full sentence
        full_completion = f"{sentence_prefix} {completion}".strip()
        
        logger.debug(f"✅ Generated completion: '{full_completion[:100]}...'")
        return full_completion
    
    async def _call_gemini_async(self, prompt: str) -> str:
        """Make async call to Gemini API using the same approach as type2_generation.py."""
        logger.debug(f"🌐 Making Gemini API call with prompt length: {len(prompt)}")
        
        try:
            if not self.client:
                logger.error("❌ Gemini client not initialized")
                raise ValueError("Gemini client not initialized")
            
            # Use the same async pattern as type2_generation.py
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.type2_config.primary_model.model_id,
                contents=prompt
            )
            
            if not hasattr(response, 'text') or not response.text:
                logger.error("❌ Empty or invalid response from Gemini API")
                raise ValueError("Empty response from Gemini API")
            
            logger.debug(f"✅ Received response from Gemini API ({len(response.text)} chars)")
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"❌ Gemini API call failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            raise
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the generated response."""
        logger.debug(f"🧹 Cleaning response: '{response[:100]}...'")
        
        if not response:
            logger.error("❌ Empty response to clean")
            raise ValueError("Empty response from model")
        
        # Remove common artifacts
        cleaned = response.strip()
        
        # Remove labels and prefixes
        for pattern in self.type2_config.forbidden_patterns:
            old_cleaned = cleaned
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()
            if old_cleaned != cleaned:
                logger.debug(f"🧹 Removed pattern '{pattern}'")
        
        # Remove surrounding quotes
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1].strip()
            logger.debug("🧹 Removed surrounding quotes")
        
        # Don't add punctuation here since we'll combine with prefix
        logger.debug(f"✅ Cleaned response: '{cleaned[:100]}...'")
        return cleaned

class Type4Generator(TextGenerator):
    """Generator for Type 4 (LLM-paraphrased original text) columns."""
    
    def __init__(self, config: ErrorHandlingConfig):
        logger.info("🔧 Initializing Type4Generator...")
        super().__init__(config)
        self.type4_config = config.type4_config
        self._setup_models()
    
    def _setup_models(self):
        """Initialize models for Type 4 generation."""
        logger.info("🔑 Setting up models for Type 4 generation...")
        
        # Setup Gemini for prompt-based paraphrasing
        logger.debug("Setting up Gemini for prompt-based paraphrasing...")
        try:
            from google import genai
            api_key = get_api_key("GEMINI_API_KEY")
            if api_key:
                self.gemini_client = genai.Client(api_key=api_key)
                logger.info("✅ Gemini model initialized for Type 4")
            else:
                self.gemini_client = None
                logger.warning("⚠️ Gemini API key not found, prompt-based paraphrasing unavailable")
        except ImportError:
            self.gemini_client = None
            logger.warning("⚠️ google-generativeai not installed, prompt-based paraphrasing unavailable")
        except Exception as e:
            self.gemini_client = None
            logger.error(f"❌ Failed to setup Gemini for Type 4: {e}")
        
        # Setup DIPPER for model-based paraphrasing
        logger.debug("Setting up DIPPER for model-based paraphrasing...")
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            model_name = self.type4_config.primary_model.model_id
            logger.debug(f"Loading DIPPER model: {model_name}")
            
            self.dipper_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.dipper_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            # Move to GPU if available
            if torch.cuda.is_available() and self.type4_config.use_gpu:
                self.dipper_model = self.dipper_model.cuda()
                logger.info("✅ DIPPER model loaded on GPU")
            else:
                logger.info("✅ DIPPER model loaded on CPU")
                
        except ImportError as e:
            self.dipper_tokenizer = None
            self.dipper_model = None
            logger.warning(f"⚠️ transformers/torch not installed, DIPPER paraphrasing unavailable: {e}")
        except Exception as e:
            self.dipper_tokenizer = None
            self.dipper_model = None
            logger.warning(f"⚠️ Failed to load DIPPER model: {e}")
    
    async def _generate_text(
        self, 
        record: Dict[str, Any], 
        column_name: str, 
        method: str
    ) -> str:
        """Generate Type 4 paraphrased text using specified method."""
        
        logger.debug(f"🎯 Generating Type 4 text using method: {method}")
        
        human_text = record.get("human_original_text") or record.get("human_original_text(type1)", "")
        if not human_text:
            logger.error("❌ No human original text found for paraphrasing")
            raise ValueError("No human original text found for paraphrasing")
        
        logger.debug(f"📖 Source text for paraphrasing: '{human_text[:100]}...'")
        
        # Determine paraphrasing method
        if "dipper" in method.lower():
            logger.debug("Using DIPPER paraphrasing method")
            return await self._generate_dipper_paraphrase(human_text)
        elif "prompt" in method.lower():
            logger.debug("Using prompt-based paraphrasing method")
            return await self._generate_prompt_paraphrase(human_text)
        else:
            # Try DIPPER first, fall back to prompt-based
            logger.debug("Auto-selecting paraphrasing method (DIPPER preferred)")
            try:
                return await self._generate_dipper_paraphrase(human_text)
            except Exception as e:
                logger.warning(f"⚠️ DIPPER paraphrasing failed, trying prompt-based: {e}")
                return await self._generate_prompt_paraphrase(human_text)
    
    async def _generate_dipper_paraphrase(self, text: str) -> str:
        """Generate paraphrase using DIPPER model."""
        logger.debug("🤖 Starting DIPPER paraphrasing...")
        
        if not self.dipper_model or not self.dipper_tokenizer:
            logger.error("❌ DIPPER model not available")
            raise ValueError("DIPPER model not available")
        
        try:
            # Prepare input
            input_text = f"paraphrase: {text}"
            logger.debug(f"📝 DIPPER input: '{input_text[:100]}...'")
            
            # Tokenize
            inputs = self.dipper_tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # Move to same device as model
            if next(self.dipper_model.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}
                logger.debug("🔄 Moved inputs to GPU")
            
            # Generate
            logger.debug("🔄 Generating paraphrase with DIPPER...")
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(
                None,
                lambda: self.dipper_model.generate(
                    **inputs,
                    **self.type4_config.dipper_settings
                )
            )
            
            # Decode
            paraphrase = self.dipper_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean output
            paraphrase = paraphrase.strip()
            if paraphrase.lower().startswith("paraphrase:"):
                paraphrase = paraphrase[11:].strip()
                logger.debug("🧹 Removed 'paraphrase:' prefix")
            
            logger.debug(f"✅ DIPPER paraphrase generated: '{paraphrase[:100]}...'")
            return paraphrase
            
        except Exception as e:
            logger.error(f"❌ DIPPER paraphrasing failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            raise
    
    async def _generate_prompt_paraphrase(self, text: str) -> str:
        """Generate paraphrase using prompt-based method with Gemini."""
        logger.debug("📝 Starting prompt-based paraphrasing...")
        
        if not self.gemini_client:
            logger.error("❌ Gemini model not available for prompt-based paraphrasing")
            raise ValueError("Gemini model not available for prompt-based paraphrasing")
        
        try:
            # Prepare prompt
            prompt = self.type4_config.prompt_based_template.format(
                text=text,
                target_length=len(text),
                max_length=int(len(text) * 1.3)
            )
            
            logger.debug(f"📝 Prompt-based paraphrasing prompt generated ({len(prompt)} chars)")
            
            # Generate
            response = await asyncio.to_thread(
                self.gemini_client.models.generate_content,
                model=self.type4_config.fallback_model.model_id,
                contents=prompt
            )
            
            paraphrase = response.text.strip()
            
            # Clean response
            artifacts = [
                r'^Paraphrased text:\s*',
                r'^Result:\s*',
                r'^\"|\"$',  # Surrounding quotes
            ]
            
            for pattern in artifacts:
                old_paraphrase = paraphrase
                paraphrase = re.sub(pattern, '', paraphrase, flags=re.IGNORECASE).strip()
                if old_paraphrase != paraphrase:
                    logger.debug(f"🧹 Removed pattern '{pattern}'")
            
            logger.debug(f"✅ Prompt-based paraphrase generated: '{paraphrase[:100]}...'")
            return paraphrase
            
        except Exception as e:
            logger.error(f"❌ Prompt-based paraphrasing failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            raise

class NullValueHandler:
    """Main handler for processing null values in datasets with tqdm progress bars."""
    
    def __init__(self, config: Optional[ErrorHandlingConfig] = None):
        logger.info("🚀 Initializing NullValueHandler...")
        self.config = config or ErrorHandlingConfig()
        
        logger.debug(f"Configuration: {self.config}")
        
        # Initialize generators
        logger.info("🔧 Initializing generators...")
        try:
            self.type2_generator = Type2Generator(self.config)
            logger.info("✅ Type2Generator initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Type2Generator: {e}")
            self.type2_generator = None
        
        try:
            self.type4_generator = Type4Generator(self.config)
            logger.info("✅ Type4Generator initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Type4Generator: {e}")
            self.type4_generator = None
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Output directory created: {self.output_dir}")
        
        # Initialize progress tracking
        self.progress = {
            'total_records': 0,
            'processed_records': 0,
            'successful_fills': 0,
            'failed_fills': 0,
            'start_time': None,
            'columns_processed': {}
        }
        
        # Progress bars
        self.main_pbar = None
        self.batch_pbar = None
        
        logger.info("✅ NullValueHandler initialization complete")
    
    def identify_null_columns(self, data: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Identify which columns need to be filled for each dataset type."""
        
        logger.info("🔍 Identifying null columns...")
        
        # Define column mappings for each type - SIMPLIFIED FOR SENTENCE COMPLETION ONLY
        type2_columns = [
            "llm_generated_text(sentence_completion)",  # PRIMARY TARGET
        ]
        
        type4_columns = [
            "llm_paraphrased_original_text(Prompt_based)",
        ]
        
        # Check which columns exist and have null values
        null_columns = {"type2": [], "type4": []}
        
        if data:
            sample_record = data[0]
            logger.debug(f"Sample record keys: {list(sample_record.keys())}")
            
            # Check Type 2 columns (only sentence completion)
            logger.debug("Checking Type 2 columns...")
            for col in type2_columns:
                if col in sample_record:
                    null_count = sum(1 for record in data if record.get(col) is None)
                    if null_count > 0:
                        null_columns["type2"].append(col)
                        logger.info(f"📊 Found {null_count} null values in Type 2 column: {col}")
            
            # Check Type 4 columns
            logger.debug("Checking Type 4 columns...")
            for col in type4_columns:
                if col in sample_record:
                    null_count = sum(1 for record in data if record.get(col) is None)
                    if null_count > 0:
                        null_columns["type4"].append(col)
                        logger.info(f"📊 Found {null_count} null values in Type 4 column: {col}")
        
        logger.info(f"🔍 Null column analysis complete:")
        logger.info(f"   Type 2 columns with nulls: {null_columns['type2']}")
        logger.info(f"   Type 4 columns with nulls: {null_columns['type4']}")
        
        return null_columns
    
    async def process_null_records(
        self, 
        input_file: Union[str, Path], 
        dataset_type: Optional[DatasetType] = None
    ) -> Dict[str, Any]:
        """Process null records and generate missing values with progress bars."""
        
        logger.info(f"🚀 Starting null value processing for {input_file}")
        self.progress['start_time'] = time.time()
        
        # Load data
        logger.info(f"📂 Loading data from {input_file}")
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"✅ Successfully loaded {len(data)} records")
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            raise
        
        self.progress['total_records'] = len(data)
        
        # Identify null columns
        null_columns = self.identify_null_columns(data)
        
        # Auto-detect dataset type if not provided
        if dataset_type is None:
            if null_columns["type2"] and null_columns["type4"]:
                logger.info("🔍 Detected mixed dataset (both Type 2 and Type 4 nulls)")
                dataset_type = DatasetType.TYPE2  # Process Type 2 first
            elif null_columns["type2"]:
                dataset_type = DatasetType.TYPE2
                logger.info("🔍 Detected Type 2 dataset")
            elif null_columns["type4"]:
                dataset_type = DatasetType.TYPE4
                logger.info("🔍 Detected Type 4 dataset")
            else:
                logger.warning("⚠️ No null columns detected")
                return {"status": "no_nulls_found", "data": data}
        
        logger.info(f"🎯 Processing dataset type: {dataset_type.value}")
        
        # Check generator availability
        if dataset_type == DatasetType.TYPE2 and not self.type2_generator:
            logger.error("❌ Type2Generator not available for processing")
            raise ValueError("Type2Generator initialization failed")
        elif dataset_type == DatasetType.TYPE4 and not self.type4_generator:
            logger.error("❌ Type4Generator not available for processing")
            raise ValueError("Type4Generator initialization failed")
        
        # Process records with progress bars
        processed_data = await self._process_records_batch_with_progress(data, null_columns, dataset_type)
        
        # Save results
        output_file = self._generate_output_filename(input_file, dataset_type)
        await self._save_results(processed_data, output_file)
        
        # Generate summary report
        summary = self._generate_summary_report()
        
        logger.info("✅ Null value processing completed successfully")
        return {
            "status": "completed",
            "output_file": str(output_file),
            "summary": summary,
            "data": processed_data
        }
    
    async def _process_records_batch_with_progress(
        self, 
        data: List[Dict[str, Any]], 
        null_columns: Dict[str, List[str]], 
        dataset_type: DatasetType
    ) -> List[Dict[str, Any]]:
        """Process records in batches with comprehensive progress tracking."""
        
        logger.info(f"📦 Starting batch processing with progress bars...")
        processed_data = []
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        # Create batches
        batches = [
            data[i:i + self.config.batch_size] 
            for i in range(0, len(data), self.config.batch_size)
        ]
        
        # Count total null values to fill
        total_nulls = 0
        for batch in batches:
            for record in batch:
                if dataset_type == DatasetType.TYPE2:
                    for col in null_columns.get("type2", []):
                        if record.get(col) is None:
                            total_nulls += 1
                elif dataset_type == DatasetType.TYPE4:
                    for col in null_columns.get("type4", []):
                        if record.get(col) is None:
                            total_nulls += 1
        
        logger.info(f"📦 Processing {len(batches)} batches of size {self.config.batch_size}")
        logger.info(f"🎯 Total null values to fill: {total_nulls}")
        
        # Initialize progress bars
        if self.config.show_progress and HAS_TQDM:
            # Main progress bar for overall progress
            self.main_pbar = tqdm(
                total=total_nulls,
                desc=f"🔄 Generating {dataset_type.value.upper()} content",
                unit="nulls",
                leave=self.config.progress_leave,
                colour='green'
            )
            
            # Batch progress bar
            self.batch_pbar = tqdm(
                total=len(batches),
                desc="📦 Processing batches",
                unit="batch",
                leave=False,
                position=1,
                colour='blue'
            )
        
        try:
            for batch_idx, batch in enumerate(batches):
                logger.debug(f"🔄 Processing batch {batch_idx + 1}/{len(batches)}")
                
                if self.batch_pbar:
                    self.batch_pbar.set_description(f"📦 Batch {batch_idx + 1}/{len(batches)}")
                
                # Process batch with individual progress tracking
                batch_tasks = [
                    self._process_single_record_with_progress(
                        record, null_columns, dataset_type, semaphore
                    )
                    for record in batch
                ]
                
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Collect results
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"❌ Batch processing error: {result}")
                        # Add original record on error
                        processed_data.append(batch[batch_results.index(result)])
                    else:
                        processed_data.append(result)
                
                # Update progress
                self.progress['processed_records'] += len(batch)
                
                # Update batch progress bar
                if self.batch_pbar:
                    success_rate = (self.progress['successful_fills'] / 
                                  max(self.progress['successful_fills'] + self.progress['failed_fills'], 1)) * 100
                    self.batch_pbar.set_postfix({
                        'Success Rate': f'{success_rate:.1f}%',
                        'Filled': f"{self.progress['successful_fills']}/{total_nulls}"
                    })
                    self.batch_pbar.update(1)
                
                # Save intermediate results
                if self.config.save_progress and (batch_idx + 1) % 10 == 0:
                    temp_file = self.output_dir / f"temp_progress_batch_{batch_idx + 1}.json"
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump(processed_data, f, indent=2, ensure_ascii=False)
                    logger.debug(f"💾 Saved intermediate progress: {temp_file}")
                
                # Brief pause to avoid overwhelming the API
                await asyncio.sleep(0.1)
                
        finally:
            # Close progress bars
            if self.batch_pbar:
                self.batch_pbar.close()
            if self.main_pbar:
                # Update final status
                final_success_rate = (self.progress['successful_fills'] / 
                                    max(self.progress['successful_fills'] + self.progress['failed_fills'], 1)) * 100
                self.main_pbar.set_description(f"✅ Completed - {final_success_rate:.1f}% success rate")
                self.main_pbar.close()
        
        logger.info("✅ Batch processing completed")
        return processed_data
    
    async def _process_single_record_with_progress(
        self, 
        record: Dict[str, Any], 
        null_columns: Dict[str, List[str]], 
        dataset_type: DatasetType,
        semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        """Process a single record to fill null values with progress callback."""
        
        async with semaphore:
            processed_record = record.copy()
            record_id = record.get('idx', 'unknown')
            
            logger.debug(f"🔄 Processing record {record_id}")
            
            # Progress callback function
            def update_progress(status):
                if self.main_pbar:
                    if status == 'success':
                        self.main_pbar.set_postfix({'Status': '✅ Success', 'Record': str(record_id)})
                    elif status == 'failed':
                        self.main_pbar.set_postfix({'Status': '❌ Failed', 'Record': str(record_id)})
                    self.main_pbar.update(1)
            
            # Process Type 2 columns (ONLY sentence completion)
            if dataset_type == DatasetType.TYPE2 or dataset_type is None:
                for column in null_columns.get("type2", []):
                    if record.get(column) is None:
                        logger.debug(f"🔧 Generating content for Type 2 column: {column}")
                        
                        # Update progress bar description
                        if self.main_pbar:
                            self.main_pbar.set_description(f"🔄 Generating {column} for record {record_id}")
                        
                        result = await self.type2_generator.generate_with_retry(
                            record, column, "sentence_completion", progress_callback=update_progress
                        )
                        
                        if result.success:
                            # CLEAN OUTPUT: Only fill the target column, no extra metadata
                            processed_record[column] = result.text
                            self.progress['successful_fills'] += 1
                            logger.debug(f"✅ Successfully filled {column} for record {record_id}")
                        else:
                            logger.error(f"❌ Failed to generate {column} for record {record_id}: {result.error}")
                            self.progress['failed_fills'] += 1
                        
                        # Track column processing
                        if column not in self.progress['columns_processed']:
                            self.progress['columns_processed'][column] = {'success': 0, 'failed': 0}
                        
                        if result.success:
                            self.progress['columns_processed'][column]['success'] += 1
                        else:
                            self.progress['columns_processed'][column]['failed'] += 1
            
            # Process Type 4 columns
            if dataset_type == DatasetType.TYPE4 or dataset_type is None:
                for column in null_columns.get("type4", []):
                    if record.get(column) is None:
                        logger.debug(f"🔧 Generating content for Type 4 column: {column}")
                        
                        # Update progress bar description
                        if self.main_pbar:
                            self.main_pbar.set_description(f"🔄 Generating {column} for record {record_id}")
                        
                        result = await self.type4_generator.generate_with_retry(
                            record, column, self._get_paraphrase_method(column), progress_callback=update_progress
                        )
                        
                        if result.success:
                            processed_record[column] = result.text
                            self.progress['successful_fills'] += 1
                            logger.debug(f"✅ Successfully filled {column} for record {record_id}")
                        else:
                            logger.error(f"❌ Failed to generate {column} for record {record_id}: {result.error}")
                            self.progress['failed_fills'] += 1
                        
                        # Track column processing
                        if column not in self.progress['columns_processed']:
                            self.progress['columns_processed'][column] = {'success': 0, 'failed': 0}
                        
                        if result.success:
                            self.progress['columns_processed'][column]['success'] += 1
                        else:
                            self.progress['columns_processed'][column]['failed'] += 1
            
            return processed_record
    
    def _get_paraphrase_method(self, column_name: str) -> str:
        """Determine paraphrase method based on column name."""
        if "dipper" in column_name.lower():
            return "dipper_based"
        elif "prompt" in column_name.lower():
            return "prompt_based"
        else:
            return "prompt_based"  # Default for Type 4
    
    def _generate_output_filename(self, input_file: Union[str, Path], dataset_type: DatasetType) -> Path:
        """Generate output filename based on input and configuration."""
        input_path = Path(input_file)
        base_name = input_path.stem
        
        # Remove existing timestamp if present
        base_name = re.sub(r'_\d{8}_\d{6}$', '', base_name)
        
        suffix = ""
        if self.config.timestamp_suffix:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = f"_{timestamp}"
        
        output_name = f"{base_name}_filled_{dataset_type.value}{suffix}.json"
        return self.output_dir / output_name
    
    async def _save_results(self, data: List[Dict[str, Any]], output_file: Path):
        """Save processed results to file."""
        try:
            logger.info(f"💾 Saving results to {output_file}")
            
            # Show saving progress if tqdm is available
            if HAS_TQDM:
                with tqdm(total=1, desc="💾 Saving results", unit="file", colour='yellow') as save_pbar:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    save_pbar.update(1)
                    save_pbar.set_description("✅ Results saved")
            else:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Results saved successfully: {output_file}")
            
            # Clean up temporary files
            temp_files = list(self.output_dir.glob("temp_progress_batch_*.json"))
            if temp_files:
                if HAS_TQDM:
                    with tqdm(temp_files, desc="🗑️ Cleaning temp files", unit="file", leave=False, colour='red') as cleanup_pbar:
                        for temp_file in cleanup_pbar:
                            temp_file.unlink()
                            cleanup_pbar.set_postfix({'Removed': temp_file.name})
                else:
                    for temp_file in temp_files:
                        temp_file.unlink()
                        logger.debug(f"🗑️ Cleaned up temporary file: {temp_file}")
                
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
            raise
    
    def _log_progress(self):
        """Log current progress."""
        elapsed_time = time.time() - self.progress['start_time']
        processed = self.progress['processed_records']
        total = self.progress['total_records']
        
        if processed > 0:
            rate = processed / elapsed_time
            eta = (total - processed) / rate if rate > 0 else 0
            
            logger.info(
                f"📈 Progress: {processed}/{total} ({processed/total*100:.1f}%) | "
                f"Rate: {rate:.1f} records/sec | "
                f"ETA: {eta/60:.1f} minutes | "
                f"Success: {self.progress['successful_fills']} | "
                f"Failed: {self.progress['failed_fills']}"
            )
    
    def _generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of processing results."""
        elapsed_time = time.time() - self.progress['start_time']
        
        return {
            "processing_summary": {
                "total_records": self.progress['total_records'],
                "processed_records": self.progress['processed_records'],
                "successful_fills": self.progress['successful_fills'],
                "failed_fills": self.progress['failed_fills'],
                "success_rate": (
                    self.progress['successful_fills'] / 
                    (self.progress['successful_fills'] + self.progress['failed_fills'])
                    if (self.progress['successful_fills'] + self.progress['failed_fills']) > 0 
                    else 0
                ),
                "processing_time_seconds": elapsed_time,
                "processing_rate": self.progress['processed_records'] / elapsed_time if elapsed_time > 0 else 0
            },
            "column_breakdown": self.progress['columns_processed'],
            "generator_stats": {
                "type2": self.type2_generator.stats if self.type2_generator else {},
                "type4": self.type4_generator.stats if self.type4_generator else {}
            },
            "configuration": {
                "max_retries": self.config.max_retries,
                "batch_size": self.config.batch_size,
                "max_concurrent": self.config.max_concurrent,
                "min_text_length": self.config.min_text_length,
                "max_text_length": self.config.max_text_length,
                "tqdm_available": HAS_TQDM
            }
        }

# Convenience functions for easy usage

async def process_null_records_file(
    input_file: Union[str, Path],
    dataset_type: Optional[str] = None,
    config: Optional[ErrorHandlingConfig] = None
) -> Dict[str, Any]:
    """
    Process a file containing null records and generate missing values.
    
    Args:
        input_file: Path to the JSON file containing null records
        dataset_type: Type of dataset ("type2", "type4", or None for auto-detection)
        config: Custom configuration (uses default if None)
    
    Returns:
        Dictionary containing processing results and summary
    """
    logger.info(f"🚀 Processing null records file: {input_file}")
    
    # Convert string dataset_type to enum
    if dataset_type:
        dataset_type = DatasetType(dataset_type.lower())
    
    handler = NullValueHandler(config)
    return await handler.process_null_records(input_file, dataset_type)

async def process_null_records_file_with_sampling(
    input_file: Union[str, Path],
    dataset_type: Optional[str] = None,
    config: Optional[ErrorHandlingConfig] = None,
    sample_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Process a file containing null records with optional sampling.
    
    Args:
        input_file: Path to the JSON file containing null records
        dataset_type: Type of dataset ("type2", "type4", or None for auto-detection)
        config: Custom configuration (uses default if None)
        sample_size: Number of samples to process (processes all if None)
    
    Returns:
        Dictionary containing processing results and summary
    """
    logger.info(f"🚀 Processing null records file with sampling: {input_file}")
    
    # Convert string dataset_type to enum
    if dataset_type:
        dataset_type = DatasetType(dataset_type.lower())
    
    # Load and potentially sample the data
    logger.info(f"📂 Loading data from {input_file}")
    
    # Show loading progress if tqdm is available
    if HAS_TQDM:
        with tqdm(total=1, desc="📂 Loading data", unit="file", colour='cyan') as load_pbar:
            with open(input_file, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
            load_pbar.update(1)
            load_pbar.set_description("✅ Data loaded")
    else:
        with open(input_file, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
    
    original_count = len(all_data)
    logger.info(f"📊 Loaded {original_count} total records")
    
    # Apply sampling if specified
    if sample_size is not None and sample_size < len(all_data):
        import random
        
        logger.info(f"🎯 Sampling {sample_size} records from {original_count} total records")
        
        # Show sampling progress
        if HAS_TQDM:
            with tqdm(total=1, desc="🎯 Sampling data", unit="operation", colour='orange') as sample_pbar:
                # Use random sampling but ensure reproducibility with a seed
                random.seed(42)  # Fixed seed for reproducible sampling
                sampled_data = random.sample(all_data, sample_size)
                
                # Sort by idx if available to maintain some order
                if sampled_data and 'idx' in sampled_data[0]:
                    sampled_data.sort(key=lambda x: x.get('idx', 0))
                
                sample_pbar.update(1)
                sample_pbar.set_description("✅ Sampling complete")
        else:
            random.seed(42)
            sampled_data = random.sample(all_data, sample_size)
            if sampled_data and 'idx' in sampled_data[0]:
                sampled_data.sort(key=lambda x: x.get('idx', 0))
        
        data_to_process = sampled_data
        logger.info(f"✅ Selected {len(data_to_process)} records for processing")
    else:
        data_to_process = all_data
        logger.info(f"📊 Processing all {len(data_to_process)} records")
    
    # Create a temporary file with sampled data for processing
    temp_file = None
    try:
        if sample_size is not None and sample_size < original_count:
            # Create temporary file with sampled data
            temp_dir = Path(config.output_dir if config else "data/processed/error_handling")
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_file = temp_dir / f"temp_sampled_data_{sample_size}_{timestamp}.json"
            
            # Show temp file creation progress
            if HAS_TQDM:
                with tqdm(total=1, desc="📝 Creating temp file", unit="file", colour='purple') as temp_pbar:
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump(data_to_process, f, indent=2, ensure_ascii=False)
                    temp_pbar.update(1)
                    temp_pbar.set_description("✅ Temp file created")
            else:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(data_to_process, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📝 Created temporary sampled data file: {temp_file}")
            
            # Process the temporary file
            handler = NullValueHandler(config)
            result = await handler.process_null_records(temp_file, dataset_type)
            
        else:
            # Process the original file directly
            handler = NullValueHandler(config)
            result = await handler.process_null_records(input_file, dataset_type)
        
        # Add sampling information to the result
        result["sampling_info"] = {
            "original_record_count": original_count,
            "processed_record_count": len(data_to_process),
            "sampling_applied": sample_size is not None and sample_size < original_count,
            "sample_size": sample_size
        }
        
        return result
        
    finally:
        # Clean up temporary file
        if temp_file and temp_file.exists():
            temp_file.unlink()
            logger.info(f"🗑️ Cleaned up temporary file: {temp_file}")

async def process_both_null_files(
    type2_file: Union[str, Path],
    type4_file: Union[str, Path],
    config: Optional[ErrorHandlingConfig] = None
) -> Dict[str, Any]:
    """
    Process both Type 2 and Type 4 null record files.
    
    Args:
        type2_file: Path to Type 2 null records file
        type4_file: Path to Type 4 null records file
        config: Custom configuration
    
    Returns:
        Dictionary containing results for both types
    """
    handler = NullValueHandler(config)
    
    # Process both files concurrently
    type2_task = handler.process_null_records(type2_file, DatasetType.TYPE2)
    type4_task = handler.process_null_records(type4_file, DatasetType.TYPE4)
    
    type2_result, type4_result = await asyncio.gather(type2_task, type4_task)
    
    return {
        "type2_result": type2_result,
        "type4_result": type4_result,
        "combined_summary": {
            "total_records_processed": (
                type2_result["summary"]["processing_summary"]["total_records"] +
                type4_result["summary"]["processing_summary"]["total_records"]
            ),
            "total_successful_fills": (
                type2_result["summary"]["processing_summary"]["successful_fills"] +
                type4_result["summary"]["processing_summary"]["successful_fills"]
            ),
            "total_failed_fills": (
                type2_result["summary"]["processing_summary"]["failed_fills"] +
                type4_result["summary"]["processing_summary"]["failed_fills"]
            )
        }
    }

def create_custom_config(
    max_retries: int = 5,
    batch_size: int = 10,
    max_concurrent: int = 3,
    min_text_length: int = 10,
    max_text_length: int = 500,
    output_dir: str = "data/processed/error_handling",
    show_progress: bool = True
) -> ErrorHandlingConfig:
    """
    Create a custom error handling configuration.
    
    Args:
        max_retries: Maximum number of retry attempts
        batch_size: Number of records to process in each batch
        max_concurrent: Maximum concurrent operations
        min_text_length: Minimum acceptable text length
        max_text_length: Maximum acceptable text length
        output_dir: Output directory for results
        show_progress: Whether to show tqdm progress bars
    
    Returns:
        ErrorHandlingConfig instance
    """
    return ErrorHandlingConfig(
        max_retries=max_retries,
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        min_text_length=min_text_length,
        max_text_length=max_text_length,
        output_dir=output_dir,
        timestamp_suffix=True, # Default to True for new_code
        show_progress=show_progress
    )

# Example usage
if __name__ == "__main__":
    import argparse
    
    async def main():
        """Example usage of the error handling system with argument parsing."""
        
        # Set up argument parser
        parser = argparse.ArgumentParser(
            description="Generate missing values for null columns in PADBen datasets (SENTENCE COMPLETION ONLY) with Progress Bars",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Process Type 2 null records (sentence completion only)
  python error_handling.py data/test/null_records_type2.json --type type2
  
  # Process Type 4 null records with 100 samples
  python error_handling.py data/test/null_records_type4.json --type type4 --sample 100
  
  # Auto-detect type with custom config and output path
  python error_handling.py data/test/null_records.json --retries 10 --batch-size 20 --output /custom/path
  
  # Process with minimal logging and no progress bars
  python error_handling.py data/test/null_records.json --sample 50 --quiet --no-progress
            """
        )
        
        # Positional arguments
        parser.add_argument(
            "input_file",
            help="Path to the JSON file containing null records"
        )
        
        # Optional arguments
        parser.add_argument(
            "--type", "--dataset-type",
            choices=["type2", "type4"],
            help="Dataset type (auto-detected if not specified)"
        )
        
        parser.add_argument(
            "--sample", "-s",
            type=int,
            help="Number of samples to process (processes all if not specified)"
        )
        
        # Configuration options
        parser.add_argument(
            "--retries", "-r",
            type=int,
            default=5,
            help="Maximum number of retry attempts (default: 5)"
        )
        
        parser.add_argument(
            "--batch-size", "-b",
            type=int,
            default=10,
            help="Number of records to process in each batch (default: 10)"
        )
        
        parser.add_argument(
            "--max-concurrent", "-c",
            type=int,
            default=3,
            help="Maximum concurrent operations (default: 3)"
        )
        
        parser.add_argument(
            "--min-length",
            type=int,
            default=10,
            help="Minimum acceptable text length (default: 10)"
        )
        
        parser.add_argument(
            "--max-length",
            type=int,
            default=500,
            help="Maximum acceptable text length (default: 500)"
        )
        
        parser.add_argument(
            "--output", "--output-dir", "-o",
            default="data/processed/error_handling",
            help="Output directory for results (default: data/processed/error_handling)"
        )
        
        parser.add_argument(
            "--no-timestamp",
            action="store_true",
            help="Don't add timestamp to output filename"
        )
        
        parser.add_argument(
            "--no-progress",
            action="store_true",
            help="Disable progress bars"
        )
        
        parser.add_argument(
            "--quiet", "-q",
            action="store_true",
            help="Reduce logging output"
        )
        
        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Increase logging output"
        )
        
        # Parse arguments
        args = parser.parse_args()
        
        # Set logging level based on verbosity
        if args.quiet:
            logging.getLogger().setLevel(logging.WARNING)
            logger.info("🔇 Quiet mode enabled")
        elif args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.info("🔊 Verbose mode enabled")
        else:
            logging.getLogger().setLevel(logging.INFO)
        
        # Validate input file
        input_path = Path(args.input_file)
        if not input_path.exists():
            logger.error(f"❌ Input file not found: {input_path}")
            sys.exit(1)
        
        logger.info(f"✅ Input file validated: {input_path}")
        
        # Create custom config
        config = ErrorHandlingConfig(
            max_retries=args.retries,
            batch_size=args.batch_size,
            max_concurrent=args.max_concurrent,
            min_text_length=args.min_length,
            max_text_length=args.max_length,
            output_dir=args.output,
            timestamp_suffix=not args.no_timestamp,
            show_progress=not args.no_progress and HAS_TQDM
        )
        
        logger.info(f"🚀 Starting processing with configuration:")
        logger.info(f"   Input file: {input_path}")
        logger.info(f"   Dataset type: {args.type or 'auto-detect'}")
        logger.info(f"   Sample size: {args.sample or 'all records'}")
        logger.info(f"   Max retries: {args.retries}")
        logger.info(f"   Batch size: {args.batch_size}")
        logger.info(f"   Max concurrent: {args.max_concurrent}")
        logger.info(f"   Output directory: {args.output}")
        logger.info(f"   Progress bars: {'✅ Enabled' if config.show_progress else '❌ Disabled'}")
        logger.info("   🎯 SENTENCE COMPLETION ONLY for Type 2")
        
        try:
            # Process the file with sampling
            result = await process_null_records_file_with_sampling(
                input_file=input_path,
                dataset_type=args.type,
                config=config,
                sample_size=args.sample
            )
            
            # Print results
            print("\n" + "="*60)
            print("PROCESSING SUMMARY (SENTENCE COMPLETION ONLY)")
            print("="*60)
            summary = result["summary"]["processing_summary"]
            print(f"Total Records: {summary['total_records']}")
            print(f"Processed Records: {summary['processed_records']}")
            print(f"Successful Fills: {summary['successful_fills']}")
            print(f"Failed Fills: {summary['failed_fills']}")
            print(f"Success Rate: {summary['success_rate']:.2%}")
            print(f"Processing Time: {summary['processing_time_seconds']:.1f} seconds")
            print(f"Processing Rate: {summary['processing_rate']:.1f} records/sec")
            print(f"Output File: {result['output_file']}")
            print(f"TQDM Available: {'✅ Yes' if HAS_TQDM else '❌ No'}")
            
            # Print sampling info if applicable
            if "sampling_info" in result:
                sampling = result["sampling_info"]
                if sampling["sampling_applied"]:
                    print(f"\nSampling Applied: {sampling['sample_size']} out of {sampling['original_record_count']} records")
            
            # Print column breakdown if verbose
            if args.verbose and result["summary"]["column_breakdown"]:
                print("\n" + "-"*40)
                print("COLUMN BREAKDOWN")
                print("-"*40)
                for column, stats in result["summary"]["column_breakdown"].items():
                    success_rate = stats['success'] / (stats['success'] + stats['failed']) if (stats['success'] + stats['failed']) > 0 else 0
                    print(f"{column}: {stats['success']} success, {stats['failed']} failed ({success_rate:.1%})")
            
            logger.info("✅ Processing completed successfully")
            
        except KeyboardInterrupt:
            logger.warning("⚠️ Processing interrupted by user")
            sys.exit(130)
        except Exception as e:
            logger.error(f"❌ Processing failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    
    # Run the async main function
    asyncio.run(main())
