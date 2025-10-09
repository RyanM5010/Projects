"""Core metrics implementation for data quality examination."""

import logging
import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from collections import Counter
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm

try:
    from .config import JACCARD_CONFIG, BLEU_CONFIG, PERPLEXITY_CONFIG, PERPLEXITY_MODELS
except ImportError:
    from config import JACCARD_CONFIG, BLEU_CONFIG, PERPLEXITY_CONFIG, PERPLEXITY_MODELS

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = logging.getLogger(__name__)


class JaccardSimilarityCalculator:
    """Calculates Jaccard similarity between text pairs."""
    
    def __init__(self, n_gram: int = 1, case_sensitive: bool = False) -> None:
        """
        Initialize Jaccard similarity calculator.
        
        Args:
            n_gram: N-gram size for similarity calculation.
            case_sensitive: Whether to consider case in similarity calculation.
        """
        self.n_gram = n_gram
        self.case_sensitive = case_sensitive
        
    def _get_ngrams(self, text: str) -> set:
        """
        Extract n-grams from text.
        
        Args:
            text: Input text.
            
        Returns:
            Set of n-grams.
        """
        if not self.case_sensitive:
            text = text.lower()
            
        tokens = word_tokenize(text)
        if self.n_gram == 1:
            return set(tokens)
        
        ngrams = []
        for i in range(len(tokens) - self.n_gram + 1):
            ngram = tuple(tokens[i:i + self.n_gram])
            ngrams.append(ngram)
        return set(ngrams)
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate Jaccard similarity between two texts.
        
        Args:
            text1: First text.
            text2: Second text.
            
        Returns:
            Jaccard similarity score (0-1).
        """
        ngrams1 = self._get_ngrams(text1)
        ngrams2 = self._get_ngrams(text2)
        
        if not ngrams1 and not ngrams2:
            return 1.0
        if not ngrams1 or not ngrams2:
            return 0.0
            
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_similarity_matrix(self, text_groups: Dict[str, List[str]]) -> Dict[Tuple[str, str], float]:
        """
        Calculate pairwise Jaccard similarity matrix for all text type pairs.
        
        Args:
            text_groups: Dictionary mapping text types to lists of texts.
            
        Returns:
            Dictionary mapping (type1, type2) tuples to average similarity scores.
        """
        similarity_matrix = {}
        text_types = list(text_groups.keys())
        
        logger.info("Calculating Jaccard similarity matrix...")
        
        for i, type1 in enumerate(text_types):
            for j, type2 in enumerate(text_types):
                if i <= j:  # Calculate upper triangle and diagonal
                    similarities = []
                    texts1 = text_groups[type1]
                    texts2 = text_groups[type2]
                    
                    # Calculate similarities for corresponding pairs
                    min_len = min(len(texts1), len(texts2))
                    for k in range(min_len):
                        sim = self.calculate_similarity(texts1[k], texts2[k])
                        similarities.append(sim)
                    
                    avg_similarity = np.mean(similarities) if similarities else 0.0
                    similarity_matrix[(type1, type2)] = avg_similarity
                    
                    # Mirror for lower triangle
                    if i != j:
                        similarity_matrix[(type2, type1)] = avg_similarity
                        
        logger.info(f"Calculated similarity matrix for {len(text_types)} text types")
        return similarity_matrix


class SelfBLEUCalculator:
    """Calculates self-BLEU scores for text collections."""
    
    def __init__(self, max_n: int = 4, smooth: bool = True, weights: Optional[List[float]] = None) -> None:
        """
        Initialize self-BLEU calculator.
        
        Args:
            max_n: Maximum n-gram order for BLEU calculation.
            smooth: Whether to apply smoothing.
            weights: Weights for different n-gram orders.
        """
        self.max_n = max_n
        self.smooth = smooth
        self.weights = weights or [1.0/max_n] * max_n
        self.smoothing_function = SmoothingFunction().method1 if smooth else None
        
    def calculate_self_bleu(self, texts: List[str], sample_size: Optional[int] = None) -> float:
        """
        Calculate self-BLEU score for a collection of texts.
        
        Args:
            texts: List of texts to evaluate.
            sample_size: Number of texts to sample for efficiency. If None, use all texts.
            
        Returns:
            Average self-BLEU score.
        """
        if len(texts) < 2:
            logger.warning("Need at least 2 texts for self-BLEU calculation")
            return 0.0
            
        # Sample texts if specified
        if sample_size and len(texts) > sample_size:
            import random
            texts = random.sample(texts, sample_size)
            
        bleu_scores = []
        
        logger.info(f"Calculating self-BLEU for {len(texts)} texts...")
        
        for i, candidate in enumerate(tqdm(texts, desc="Self-BLEU calculation")):
            references = [texts[j] for j in range(len(texts)) if j != i]
            
            # Tokenize candidate and references
            candidate_tokens = word_tokenize(candidate.lower())
            reference_tokens = [word_tokenize(ref.lower()) for ref in references]
            
            # Calculate BLEU score
            try:
                bleu_score = sentence_bleu(
                    reference_tokens,
                    candidate_tokens,
                    weights=self.weights,
                    smoothing_function=self.smoothing_function
                )
                bleu_scores.append(bleu_score)
            except Exception as e:
                logger.warning(f"Error calculating BLEU for text {i}: {e}")
                continue
                
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
        logger.info(f"Average self-BLEU score: {avg_bleu:.4f}")
        return avg_bleu
    
    def calculate_self_bleu_by_type(self, text_groups: Dict[str, List[str]], 
                                   sample_size: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate self-BLEU scores for each text type.
        
        Args:
            text_groups: Dictionary mapping text types to lists of texts.
            sample_size: Number of texts to sample per type for efficiency.
            
        Returns:
            Dictionary mapping text types to self-BLEU scores.
        """
        self_bleu_scores = {}
        
        for text_type, texts in text_groups.items():
            logger.info(f"Calculating self-BLEU for {text_type}")
            self_bleu_scores[text_type] = self.calculate_self_bleu(texts, sample_size)
            
        return self_bleu_scores


class EnhancedPerplexityCalculator:
    """Enhanced perplexity calculator supporting both GPT-2 and Llama models with quantization."""
    
    def __init__(self, model_config: str = "gpt2-xl", **kwargs) -> None:
        """
        Initialize enhanced perplexity calculator.
        
        Args:
            model_config: Model configuration key from PERPLEXITY_MODELS
            **kwargs: Override configuration parameters
        """
        # PERPLEXITY_MODELS is already imported at module level
        
        if model_config not in PERPLEXITY_MODELS:
            raise ValueError(f"Unknown model config: {model_config}. Available: {list(PERPLEXITY_MODELS.keys())}")
        
        # Load configuration
        self.config = PERPLEXITY_MODELS[model_config].copy()
        self.config.update(kwargs)  # Allow parameter overrides
        
        self.model_name = self.config["model_name"]
        self.model_type = self.config["model_type"]
        self.max_length = self.config["max_length"]
        self.stride = self.config["stride"]
        self.batch_size = self.config["batch_size"]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading {self.config['description']}...")
        
        # Load model and tokenizer based on type
        if self.model_type == "gpt2":
            self._load_gpt2_model()
        elif self.model_type == "llama":
            self._load_llama_model()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _load_gpt2_model(self) -> None:
        """Load GPT-2 model and tokenizer."""
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        
        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _load_llama_model(self) -> None:
        """Load Llama model with quantization support."""
        from transformers import (
            AutoTokenizer, 
            AutoModelForCausalLM,
            BitsAndBytesConfig
        )
        from huggingface_hub.errors import GatedRepoError
        import os
        
        # Check for authentication token
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        
        try:
            # Set up quantization config
            quantization_config = None
            if self.config.get("quantization") == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif self.config.get("quantization") == "8bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            # Load tokenizer with token if available
            tokenizer_kwargs = {
                "trust_remote_code": self.config.get("trust_remote_code", True)
            }
            if token:
                tokenizer_kwargs["token"] = token
                
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                **tokenizer_kwargs
            )
            
            # Load model with quantization
            model_kwargs = {
                "torch_dtype": getattr(torch, self.config.get("torch_dtype", "float16")),
                "low_cpu_mem_usage": self.config.get("low_cpu_mem_usage", True),
                "trust_remote_code": self.config.get("trust_remote_code", True)
            }
            
            if token:
                model_kwargs["token"] = token
                
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = self.config.get("device_map", "auto")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                **model_kwargs
            )
            
            if not quantization_config:
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            # Set pad token for Llama
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.model.config.pad_token_id is None:
                self.model.config.pad_token_id = self.tokenizer.eos_token_id
                
        except GatedRepoError as e:
            logger.error(f"Authentication required for {self.model_name}")
            logger.error("Please authenticate with Hugging Face:")
            logger.error("1. Run: python -m huggingface_hub.commands.huggingface_cli login")
            logger.error("2. Or set HF_TOKEN environment variable")
            logger.error("3. Get your token from: https://huggingface.co/settings/tokens")
            raise RuntimeError(f"Authentication required for gated model {self.model_name}. "
                             f"Please authenticate with 'huggingface-cli login' or set HF_TOKEN environment variable.") from e
    
    def calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity for a single text (same implementation as before)."""
        encodings = self.tokenizer(text, return_tensors="pt")
        
        # Handle device placement for quantized models
        if hasattr(self.model, 'device'):
            input_ids = encodings.input_ids.to(self.model.device)
        else:
            input_ids = encodings.input_ids.to(self.device)
        
        seq_len = input_ids.size(1)
        nlls = []
        prev_end_loc = 0
        
        with torch.no_grad():
            for begin_loc in range(0, seq_len, self.stride):
                end_loc = min(begin_loc + self.max_length, seq_len)
                trg_len = end_loc - prev_end_loc
                
                input_ids_chunk = input_ids[:, begin_loc:end_loc]
                target_ids = input_ids_chunk.clone()
                target_ids[:, :-trg_len] = -100
                
                outputs = self.model(input_ids_chunk, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len
                nlls.append(neg_log_likelihood)
                
                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break
        
        ppl = torch.exp(torch.stack(nlls).sum() / end_loc).item()
        return ppl
    
    def calculate_average_perplexity(self, texts: List[str], 
                                   sample_size: Optional[int] = None) -> float:
        """
        Calculate average perplexity for a collection of texts.
        
        Args:
            texts: List of texts to evaluate.
            sample_size: Number of texts to sample for efficiency.
            
        Returns:
            Average perplexity score.
        """
        if sample_size and len(texts) > sample_size:
            import random
            texts = random.sample(texts, sample_size)
            
        perplexities = []
        
        logger.info(f"Calculating perplexity for {len(texts)} texts...")
        
        for text in tqdm(texts, desc="Perplexity calculation"):
            try:
                ppl = self.calculate_perplexity(text)
                if not math.isnan(ppl) and not math.isinf(ppl):
                    perplexities.append(ppl)
            except Exception as e:
                logger.warning(f"Error calculating perplexity: {e}")
                continue
                
        avg_perplexity = np.mean(perplexities) if perplexities else float('inf')
        logger.info(f"Average perplexity: {avg_perplexity:.4f}")
        return avg_perplexity
    
    def calculate_perplexity_by_type(self, text_groups: Dict[str, List[str]], 
                                   sample_size: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate perplexity scores for each text type.
        
        Args:
            text_groups: Dictionary mapping text types to lists of texts.
            sample_size: Number of texts to sample per type for efficiency.
            
        Returns:
            Dictionary mapping text types to perplexity scores.
        """
        perplexity_scores = {}
        
        for text_type, texts in text_groups.items():
            logger.info(f"Calculating perplexity for {text_type}")
            perplexity_scores[text_type] = self.calculate_average_perplexity(texts, sample_size)
            
        return perplexity_scores


class MetricsAggregator:
    """Aggregates and manages all quality metrics."""
    
    def __init__(self, perplexity_model: str = "gpt2-xl") -> None:
        """Initialize metrics aggregator.
        
        Args:
            perplexity_model: Perplexity model to use ('gpt2-xl' or 'llama3-7b-4bit').
        """
        self.jaccard_calc = JaccardSimilarityCalculator(**JACCARD_CONFIG)
        self.bleu_calc = SelfBLEUCalculator(**BLEU_CONFIG)
        self.perplexity_calc = EnhancedPerplexityCalculator(model_config=perplexity_model)
        
    def calculate_all_metrics(self, text_groups: Dict[str, List[str]], 
                            sample_size: Optional[int] = 1000) -> Dict[str, Any]:
        """
        Calculate all quality metrics for the dataset.
        
        Args:
            text_groups: Dictionary mapping text types to lists of texts.
            sample_size: Number of texts to sample for computationally expensive metrics.
            
        Returns:
            Dictionary containing all calculated metrics.
        """
        logger.info("Starting comprehensive metrics calculation...")
        
        metrics = {
            "jaccard_similarity_matrix": {},
            "self_bleu_scores": {},
            "perplexity_scores": {},
            "dataset_statistics": {}
        }
        
        # Calculate Jaccard similarity matrix
        logger.info("Calculating Jaccard similarity matrix...")
        metrics["jaccard_similarity_matrix"] = self.jaccard_calc.calculate_similarity_matrix(text_groups)
        
        # Calculate self-BLEU scores
        logger.info("Calculating self-BLEU scores...")
        metrics["self_bleu_scores"] = self.bleu_calc.calculate_self_bleu_by_type(text_groups, sample_size)
        
        # Calculate perplexity scores
        logger.info("Calculating perplexity scores...")
        metrics["perplexity_scores"] = self.perplexity_calc.calculate_perplexity_by_type(text_groups, sample_size)
        
        # Calculate dataset statistics
        metrics["dataset_statistics"] = self._calculate_dataset_stats(text_groups)
        
        logger.info("All metrics calculation completed!")
        return metrics
    
    def _calculate_dataset_stats(self, text_groups: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Calculate basic dataset statistics.
        
        Args:
            text_groups: Dictionary mapping text types to lists of texts.
            
        Returns:
            Dictionary containing dataset statistics.
        """
        stats = {}
        
        for text_type, texts in text_groups.items():
            if texts:
                lengths = [len(text.split()) for text in texts]
                stats[text_type] = {
                    "count": len(texts),
                    "avg_length": np.mean(lengths),
                    "std_length": np.std(lengths),
                    "min_length": np.min(lengths),
                    "max_length": np.max(lengths)
                }
            else:
                stats[text_type] = {
                    "count": 0,
                    "avg_length": 0,
                    "std_length": 0,
                    "min_length": 0,
                    "max_length": 0
                }
                
        return stats
