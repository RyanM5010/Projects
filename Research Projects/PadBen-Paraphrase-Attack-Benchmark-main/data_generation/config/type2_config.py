"""
Type 2 Generation Configuration for PADBen.

Configuration for LLM-generated text (Type 2) using two methods:
1. Sentence completion method
2. Question-answer method

Only uses gemini-2.5-pro as specified.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from enum import Enum
from data_generation.config.base_model_config import LLMModelConfig, create_gemini_flash_config

# Set up logger for this module
logger = logging.getLogger(__name__)

class Type2GenerationMethod(Enum):
    """Available generation methods for Type 2."""
    SENTENCE_COMPLETION = "sentence_completion"
    QUESTION_ANSWER = "question_answer"

@dataclass
class Type2GenerationConfig:
    """Enhanced configuration with strict output formatting."""
    
    # Primary model: Only Gemini 2.5 Pro
    primary_model: LLMModelConfig = field(default_factory=lambda: create_gemini_flash_config(
        temperature=0.7,
        max_tokens=200,
        top_p=0.9
    ))
    
    # Generation methods configuration
    generation_methods: List[Type2GenerationMethod] = field(default_factory=lambda: [
        Type2GenerationMethod.SENTENCE_COMPLETION,
        Type2GenerationMethod.QUESTION_ANSWER
    ])
    
    # Enhanced prompts with clear output structure
    sentence_completion_prompt_template: str = """Complete the following sentence or passage in a natural, coherent way:

Sentence prefix: "{sentence_prefix}"

Requirements:
- Continue the text naturally and coherently
- Aim for approximately {target_length} characters total
- Use relevant keywords if provided: {keywords}
- Write in a natural, fluent style
- Do not exceed {max_length} characters

OUTPUT FORMAT:
Return ONLY the completed text without any additional commentary, explanations, or formatting.
Do NOT include labels like "Completion:" or "Answer:" in your response.
Do NOT add quotation marks around your response.
Do NOT provide multiple versions or alternatives.

EXAMPLE FORMAT:
[Your completed text here]

Completion:"""

    # Question-answer method prompts (FIXED: Added missing placeholders)
    question_prompt_template: str = """Based on the following text, generate a clear, specific question that would naturally lead to an answer similar to the given text.

Text: "{text}"

Requirements:
- Generate ONE concise question only
- The question should be answerable based on the content of the text
- Make it sound natural and conversational
- Avoid meta-references (don't mention "the text" or "the passage")
- Consider these keywords if relevant: {keywords}
- Aim for approximately {target_length} characters for the question
- Do not exceed {max_length} characters

OUTPUT FORMAT:
Return ONLY the question without any additional commentary, explanations, or formatting.
Do NOT include labels like "Question:" or "Q:" in your response.
Do NOT add quotation marks around your response.
Do NOT provide multiple questions or alternatives.
End with a question mark.

EXAMPLE FORMAT:
What is the main cause of climate change?

Question:"""

    answer_prompt_template: str = """Answer the following question in a natural, informative way:

Question: {question}

Requirements:
- Provide a clear, concise answer
- Keep the response to approximately {target_length} characters
- Write in a natural, conversational tone
- Be informative but not verbose
- Use relevant keywords if provided: {keywords}
- Do not exceed {max_length} characters

OUTPUT FORMAT:
Return ONLY the answer without any additional commentary, explanations, or formatting.
Do NOT include labels like "Answer:" or "A:" in your response.
Do NOT add quotation marks around your response.
Do NOT provide multiple answers or alternatives.
Do NOT include meta-commentary about the answer.

EXAMPLE FORMAT:
Climate change is primarily caused by greenhouse gas emissions from human activities.

Answer:"""
    
    # Prompt placeholders configuration
    prompt_placeholders: Dict[str, Any] = field(default_factory=lambda: {
        "keywords": "",  # Comma-separated keywords to include
        "target_length": 150,  # Target character length
        "max_length": 200,  # Maximum character length
        "sentence_prefix": "",  # For sentence completion method
    })
    
    # Length control settings
    length_tolerance: float = 0.2  # ±20% tolerance
    max_length_multiplier: float = 1.2  # Allow 20% longer than original
    min_length_threshold: int = 20
    
    # Target datasets
    target_datasets: List[str] = field(default_factory=lambda: ["mrpc", "hlpc", "paws"])
    
    # Processing settings
    batch_size: int = 5
    max_concurrent_requests: int = 3
    save_intermediate: bool = True
    
    # Method selection strategy
    default_method: Type2GenerationMethod = Type2GenerationMethod.QUESTION_ANSWER
    fallback_method: Type2GenerationMethod = Type2GenerationMethod.SENTENCE_COMPLETION

    # Post-processing settings for output cleaning
    output_cleaning: Dict[str, bool] = field(default_factory=lambda: {
        "strip_labels": True,  # Remove common labels like "Answer:", "Question:"
        "strip_quotes": True,  # Remove surrounding quotes
        "strip_formatting": True,  # Remove markdown or HTML formatting
        "validate_length": True,  # Enforce length constraints
        "validate_format": True  # Check output format compliance
    })
    
    # Format validation patterns
    forbidden_patterns: List[str] = field(default_factory=lambda: [
        r"^(Answer|Question|Completion|Result):\s*",  # Labels at start
        r"^\"|\"$",  # Surrounding quotes
        r"Here is the|Here's the",  # Meta-commentary
        r"The answer is|The question is",  # Meta-references
    ])

# Default Type 2 configuration
DEFAULT_TYPE2_CONFIG = Type2GenerationConfig()

def get_prompt_template(config: Type2GenerationConfig, method: Type2GenerationMethod) -> str:
    """Get the appropriate prompt template for a generation method."""
    if method == Type2GenerationMethod.SENTENCE_COMPLETION:
        return config.sentence_completion_prompt_template
    elif method == Type2GenerationMethod.QUESTION_ANSWER:
        return config.question_prompt_template
    else:
        raise ValueError(f"Unknown generation method: {method}")

def validate_type2_config(config: Type2GenerationConfig) -> bool:
    """
    Validate Type 2 configuration.
    
    Returns True if configuration is valid, False otherwise.
    """
    logger.info("Validating Type 2 configuration...")
    
    try:
        # Validate model configuration
        if not hasattr(config, 'primary_model'):
            logger.error("Missing 'primary_model' attribute in Type2GenerationConfig")
            return False
        
        if config.primary_model.provider != "gemini":
            logger.error(f"Type 2 must use Gemini provider, got: {config.primary_model.provider}")
            return False
        
        if config.primary_model.model_id != "gemini-2.5-pro":
            logger.error(f"Type 2 must use gemini-2.5-pro model, got: {config.primary_model.model_id}")
            return False
        
        # Validate provider configuration
        from data_generation.config.base_model_config import validate_provider_config
        if not validate_provider_config("gemini"):
            logger.error("Gemini provider configuration validation failed")
            return False
        
        # Validate prompt templates
        required_placeholders = ["target_length", "max_length", "keywords"]
        templates = [
            ("sentence_completion_prompt_template", "Sentence Completion"),
            ("question_prompt_template", "Question Generation"),
            ("answer_prompt_template", "Answer Generation")
        ]
        
        for template_attr, template_name in templates:
            if not hasattr(config, template_attr):
                logger.error(f"Missing template: {template_attr}")
                return False
            
            template = getattr(config, template_attr)
            for placeholder in required_placeholders:
                if f"{{{placeholder}}}" not in template:
                    logger.error(f"Missing placeholder '{placeholder}' in {template_attr}")
                    return False
        
        # Validate other required attributes
        required_attrs = ["generation_methods", "prompt_placeholders", "default_method", "fallback_method"]
        for attr_name in required_attrs:
            if not hasattr(config, attr_name):
                logger.error(f"Missing attribute: {attr_name}")
                return False
        
        logger.info("Type 2 configuration validation successful")
        return True
        
    except Exception as e:
        logger.error(f"Type 2 configuration validation failed: {str(e)}")
        return False

def validate_type2_config_detailed() -> Tuple[bool, List[Tuple[str, bool, str]]]:
    """
    Validate Type 2 configuration and return detailed results.
    
    Returns:
        Tuple of (is_valid, validation_steps)
        where validation_steps is a list of (step_name, passed, message)
    """
    config = DEFAULT_TYPE2_CONFIG
    return validate_type2_config(config), [] 