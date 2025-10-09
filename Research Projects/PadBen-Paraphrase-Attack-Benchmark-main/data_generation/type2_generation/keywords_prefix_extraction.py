"""
Text Preprocessing for Type 1 Human Text - PADBen Type 2 Generation.

This module analyzes Type 1 human text to extract:
1. Keywords (named entities, key nouns/verbs, excluding stop words)
2. Token length calculation
3. Sentence prefix extraction (first 20% of tokens)

These features support both sentence completion and question-answer generation methods.
"""

import logging
import json
import re
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import Counter

import pandas as pd
import numpy as np

# NLP libraries
try:
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    print("Warning: spaCy not installed. Install with: pip install spacy")
    print("Also run: python -m spacy download en_core_web_sm")

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    HAS_NLTK = True
    # Download required NLTK data at import time
    print("Downloading required NLTK resources...")
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        print("✅ NLTK resources downloaded successfully")
    except Exception as e:
        print(f"⚠️ Warning: Some NLTK resources could not be downloaded: {e}")
except ImportError:
    HAS_NLTK = False
    print("Warning: NLTK not installed. Install with: pip install nltk")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TextAnalysis:
    """Container for text analysis results."""
    
    # Original text information
    original_text: str
    token_count: int
    character_count: int
    
    # Keywords extraction
    keywords: List[str]
    named_entities: List[Dict[str, str]]
    key_nouns: List[str]
    key_verbs: List[str]
    
    # Sentence prefix
    sentence_prefix: str
    prefix_token_count: int
    prefix_percentage: float
    
    # Additional features
    sentence_count: int
    avg_word_length: float
    text_complexity_score: float

class TextPreprocessor:
    """Text preprocessing class for Type 1 human text analysis."""
    
    def __init__(self, use_spacy: bool = True, max_keywords: int = 10):
        """
        Initialize the text preprocessor.
        
        Args:
            use_spacy: Whether to use spaCy for NLP (recommended)
            max_keywords: Maximum number of keywords to extract
        """
        self.use_spacy = use_spacy and HAS_SPACY
        self.max_keywords = max_keywords
        self.nlp = None
        self.stop_words = set()
        
        self._initialize_nlp()
    
    def _initialize_nlp(self):
        """Initialize NLP libraries and models."""
        if self.use_spacy:
            try:
                # Load spaCy model
                self.nlp = spacy.load("en_core_web_sm")
                self.stop_words = STOP_WORDS
                logger.info("Initialized spaCy with en_core_web_sm model")
            except OSError:
                logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.use_spacy = False
        
        if not self.use_spacy and HAS_NLTK:
            try:
                
                self.stop_words = set(stopwords.words('english'))
                logger.info("Initialized NLTK for text processing")
            except Exception as e:
                logger.warning(f"NLTK initialization failed: {e}")
    
    def extract_keywords_spacy(self, text: str) -> Tuple[List[str], List[Dict[str, str]], List[str], List[str]]:
        """Extract keywords using spaCy."""
        doc = self.nlp(text)
        
        # Extract named entities
        named_entities = []
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT', 'PRODUCT', 'WORK_OF_ART']:
                named_entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_)
                })
        
        # Extract key nouns and verbs
        key_nouns = []
        key_verbs = []
        
        for token in doc:
            if (not token.is_stop and 
                not token.is_punct and 
                not token.is_space and 
                len(token.text) > 2 and
                token.text.lower() not in self.stop_words):
                
                if token.pos_ in ['NOUN', 'PROPN']:
                    key_nouns.append(token.lemma_.lower())
                elif token.pos_ in ['VERB']:
                    key_verbs.append(token.lemma_.lower())
        
        # Count frequency and get top words
        noun_counts = Counter(key_nouns)
        verb_counts = Counter(key_verbs)
        
        top_nouns = [noun for noun, _ in noun_counts.most_common(6)]
        top_verbs = [verb for verb, _ in verb_counts.most_common(4)]
        
        # Combine all keywords
        entity_texts = [ent['text'].lower() for ent in named_entities]
        all_keywords = entity_texts + top_nouns + top_verbs
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in all_keywords:
            if keyword not in seen and len(keyword) > 1:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords[:self.max_keywords], named_entities, top_nouns, top_verbs
    
    def extract_keywords_nltk(self, text: str) -> Tuple[List[str], List[Dict[str, str]], List[str], List[str]]:
        """Extract keywords using NLTK (fallback method)."""
        # Basic tokenization and POS tagging
        tokens = word_tokenize(text.lower())
        
        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            if (token.isalpha() and 
                len(token) > 2 and 
                token not in self.stop_words):
                filtered_tokens.append(token)
        
        # Simple frequency-based keyword extraction
        token_counts = Counter(filtered_tokens)
        top_keywords = [word for word, _ in token_counts.most_common(self.max_keywords)]
        
        # Basic noun/verb extraction (simplified)
        try:
            pos_tags = nltk.pos_tag(filtered_tokens)
            nouns = [word for word, pos in pos_tags if pos.startswith('NN')]
            verbs = [word for word, pos in pos_tags if pos.startswith('VB')]
        except:
            nouns = filtered_tokens[:5]  # Fallback
            verbs = filtered_tokens[:3]  # Fallback
        
        # No named entity recognition with basic NLTK
        named_entities = []
        
        return top_keywords, named_entities, nouns[:6], verbs[:4]
    
    def calculate_token_count(self, text: str) -> int:
        """Calculate the number of tokens in text."""
        if self.use_spacy:
            doc = self.nlp(text)
            return len([token for token in doc if not token.is_space])
        else:
            # Simple whitespace tokenization
            return len(text.split())
    
    def extract_sentence_prefix(self, text: str, percentage: float = 0.2) -> Tuple[str, int]:
        """
        Extract sentence prefix (first 20% of tokens).
        
        Args:
            text: Input text
            percentage: Percentage of tokens to include in prefix (default 0.2 = 20%)
            
        Returns:
            Tuple of (prefix_text, prefix_token_count)
        """
        if self.use_spacy:
            doc = self.nlp(text)
            tokens = [token.text for token in doc if not token.is_space]
        else:
            tokens = text.split()
        
        total_tokens = len(tokens)
        prefix_length = max(1, int(total_tokens * percentage))
        
        prefix_tokens = tokens[:prefix_length]
        prefix_text = ' '.join(prefix_tokens)
        
        return prefix_text, prefix_length
    
    def calculate_text_complexity(self, text: str) -> float:
        """Calculate a simple text complexity score."""
        # Simple complexity metrics
        sentences = text.count('.') + text.count('!') + text.count('?')
        sentences = max(1, sentences)  # Avoid division by zero
        
        words = len(text.split())
        avg_sentence_length = words / sentences
        
        # Average word length
        word_lengths = [len(word.strip('.,!?;:')) for word in text.split()]
        avg_word_length = np.mean(word_lengths) if word_lengths else 0
        
        # Simple complexity score (normalized)
        complexity = (avg_sentence_length * 0.6 + avg_word_length * 0.4) / 10
        return min(1.0, complexity)
    
    def analyze_text(self, text: str) -> TextAnalysis:
        """
        Perform comprehensive text analysis.
        
        Args:
            text: Input text to analyze
            
        Returns:
            TextAnalysis object with all extracted features
        """
        # Basic metrics
        token_count = self.calculate_token_count(text)
        character_count = len(text)
        sentence_count = max(1, text.count('.') + text.count('!') + text.count('?'))
        
        # Keywords extraction
        if self.use_spacy:
            keywords, named_entities, key_nouns, key_verbs = self.extract_keywords_spacy(text)
        else:
            keywords, named_entities, key_nouns, key_verbs = self.extract_keywords_nltk(text)
        
        # Sentence prefix
        prefix_text, prefix_token_count = self.extract_sentence_prefix(text)
        prefix_percentage = prefix_token_count / token_count if token_count > 0 else 0
        
        # Additional metrics
        word_lengths = [len(word.strip('.,!?;:')) for word in text.split()]
        avg_word_length = np.mean(word_lengths) if word_lengths else 0
        text_complexity = self.calculate_text_complexity(text)
        
        return TextAnalysis(
            original_text=text,
            token_count=token_count,
            character_count=character_count,
            keywords=keywords,
            named_entities=named_entities,
            key_nouns=key_nouns,
            key_verbs=key_verbs,
            sentence_prefix=prefix_text,
            prefix_token_count=prefix_token_count,
            prefix_percentage=prefix_percentage,
            sentence_count=sentence_count,
            avg_word_length=avg_word_length,
            text_complexity_score=text_complexity
        )

class DatasetPreprocessor:
    """Process the entire PADBen dataset for Type 2 generation."""
    
    def __init__(self, 
                 input_file: str = "data/processed/unified_padben_base.csv",
                 output_dir: str = "data/processed/type2_preprocessing",
                 max_keywords: int = 10):
        """
        Initialize dataset preprocessor.
        
        Args:
            input_file: Path to the unified PADBen dataset
            output_dir: Directory to save preprocessing results
            max_keywords: Maximum keywords per text
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.max_keywords = max_keywords
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize text preprocessor
        self.text_processor = TextPreprocessor(max_keywords=max_keywords)
        
        # Statistics tracking
        self.stats = {
            "total_processed": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "avg_keywords_per_text": 0,
            "avg_token_count": 0,
            "processing_time": 0
        }
    
    def load_dataset(self) -> pd.DataFrame:
        """Load the unified PADBen dataset."""
        if not self.input_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.input_file}")
        
        if self.input_file.suffix == '.csv':
            df = pd.read_csv(self.input_file)
        elif self.input_file.suffix == '.json':
            df = pd.read_json(self.input_file)
        else:
            raise ValueError(f"Unsupported file format: {self.input_file.suffix}")
        
        logger.info(f"Loaded dataset with {len(df)} samples")
        return df
    
    def process_dataset(self, batch_size: int = 100) -> pd.DataFrame:
        """
        Process the entire dataset to extract features for Type 2 generation.
        
        Args:
            batch_size: Number of samples to process in each batch
            
        Returns:
            DataFrame with added preprocessing columns
        """
        import time
        start_time = time.time()
        
        logger.info("Starting dataset preprocessing for Type 2 generation")
        
        # Load dataset
        df = self.load_dataset()
        
        # Initialize new columns
        preprocessing_columns = [
            'keywords', 'named_entities', 'key_nouns', 'key_verbs',
            'token_count', 'character_count', 'sentence_prefix', 
            'prefix_token_count', 'sentence_count', 'avg_word_length',
            'text_complexity_score'
        ]
        
        for col in preprocessing_columns:
            df[col] = None
        
        # Process in batches
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(df))
            
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches} (samples {start_idx+1}-{end_idx})")
            
            for idx in range(start_idx, end_idx):
                try:
                    # Get human original text
                    text = df.loc[idx, 'human_original_text']
                    
                    if pd.isna(text) or not isinstance(text, str) or len(text.strip()) == 0:
                        logger.warning(f"Skipping empty text at index {idx}")
                        self.stats["failed_analyses"] += 1
                        continue
                    
                    # Analyze text
                    analysis = self.text_processor.analyze_text(text)
                    
                    # Store results
                    df.loc[idx, 'keywords'] = json.dumps(analysis.keywords)
                    df.loc[idx, 'named_entities'] = json.dumps(analysis.named_entities)
                    df.loc[idx, 'key_nouns'] = json.dumps(analysis.key_nouns)
                    df.loc[idx, 'key_verbs'] = json.dumps(analysis.key_verbs)
                    df.loc[idx, 'token_count'] = analysis.token_count
                    df.loc[idx, 'character_count'] = analysis.character_count
                    df.loc[idx, 'sentence_prefix'] = analysis.sentence_prefix
                    df.loc[idx, 'prefix_token_count'] = analysis.prefix_token_count
                    df.loc[idx, 'sentence_count'] = analysis.sentence_count
                    df.loc[idx, 'avg_word_length'] = analysis.avg_word_length
                    df.loc[idx, 'text_complexity_score'] = analysis.text_complexity_score
                    
                    self.stats["successful_analyses"] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing text at index {idx}: {str(e)}")
                    self.stats["failed_analyses"] += 1
                
                self.stats["total_processed"] += 1
            
            # Save intermediate results every 10 batches
            if (batch_idx + 1) % 10 == 0:
                self._save_intermediate_results(df, batch_idx + 1)
        
        # Calculate final statistics
        processed_df = df[df['keywords'].notna()]
        if len(processed_df) > 0:
            avg_keywords = processed_df['keywords'].apply(lambda x: len(json.loads(x)) if x else 0).mean()
            avg_tokens = processed_df['token_count'].mean()
            
            self.stats["avg_keywords_per_text"] = avg_keywords
            self.stats["avg_token_count"] = avg_tokens
        
        self.stats["processing_time"] = time.time() - start_time
        
        # Save final results
        self._save_final_results(df)
        self._save_statistics()
        
        logger.info("Dataset preprocessing completed")
        self._log_final_statistics()
        
        return df
    
    def _save_intermediate_results(self, df: pd.DataFrame, batch_num: int):
        """Save intermediate processing results."""
        try:
            checkpoint_file = self.output_dir / f"preprocessing_checkpoint_batch_{batch_num}.csv"
            df.to_csv(checkpoint_file, index=False)
            logger.info(f"Saved checkpoint after batch {batch_num}")
        except Exception as e:
            logger.warning(f"Failed to save intermediate results: {str(e)}")
    
    def _save_final_results(self, df: pd.DataFrame):
        """Save final preprocessing results."""
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        # Save CSV
        csv_file = self.output_dir / f"unified_padben_preprocessed_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        # Save JSON
        json_file = self.output_dir / f"unified_padben_preprocessed_{timestamp}.json"
        df.to_json(json_file, orient='records', indent=2)
        
        # Save a clean version for Type 2 generation (only needed columns)
        type2_columns = [
            'idx', 'dataset_source', 'human_original_text', 
            'keywords', 'token_count', 'sentence_prefix', 'prefix_token_count'
        ]
        
        type2_df = df[type2_columns].copy()
        type2_file = self.output_dir / f"type2_generation_ready_{timestamp}.csv"
        type2_df.to_csv(type2_file, index=False)
        
        logger.info(f"Saved final results:")
        logger.info(f"  Full dataset: {csv_file}")
        logger.info(f"  JSON format: {json_file}")
        logger.info(f"  Type 2 ready: {type2_file}")
    
    def _save_statistics(self):
        """Save processing statistics."""
        stats_file = self.output_dir / "preprocessing_statistics.json"
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, default=str)
        
        logger.info(f"Saved statistics to {stats_file}")
    
    def _log_final_statistics(self):
        """Log final processing statistics."""
        stats = self.stats
        success_rate = (stats["successful_analyses"] / max(stats["total_processed"], 1)) * 100
        
        logger.info("=" * 60)
        logger.info("PREPROCESSING COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total processed: {stats['total_processed']}")
        logger.info(f"Successful analyses: {stats['successful_analyses']}")
        logger.info(f"Failed analyses: {stats['failed_analyses']}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info(f"Average keywords per text: {stats['avg_keywords_per_text']:.1f}")
        logger.info(f"Average token count: {stats['avg_token_count']:.1f}")
        logger.info(f"Processing time: {stats['processing_time']:.2f} seconds")
        logger.info("=" * 60)

def analyze_single_text(text: str, max_keywords: int = 10) -> Dict[str, Any]:
    """
    Analyze a single text and return results as dictionary.
    
    Args:
        text: Text to analyze
        max_keywords: Maximum number of keywords to extract
        
    Returns:
        Dictionary with analysis results
    """
    processor = TextPreprocessor(max_keywords=max_keywords)
    analysis = processor.analyze_text(text)
    return asdict(analysis)

def main():
    """Main function to run dataset preprocessing."""
    # Configuration
    input_file = "data/processed/unified_padben_base.csv"
    output_dir = "data/processed/type2_preprocessing"
    batch_size = 100
    max_keywords = 10
    
    # Initialize and run preprocessor
    preprocessor = DatasetPreprocessor(
        input_file=input_file,
        output_dir=output_dir,
        max_keywords=max_keywords
    )
    
    # Process dataset
    try:
        df = preprocessor.process_dataset(batch_size=batch_size)
        logger.info(f"Successfully processed {len(df)} samples")
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
