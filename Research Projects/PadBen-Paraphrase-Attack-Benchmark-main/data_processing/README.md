# 📊 PADBen Data Processing Module

The PADBen Data Processing module provides a comprehensive framework for loading, analyzing, and standardizing multiple benchmark datasets (MRPC, HLPC, PAWS) into a unified format suitable for the PADBen benchmark evaluation.

## 🎯 Overview

This module is designed to handle diverse datasets with different formats and structures, standardizing them into a common schema with 5 text types:

- **Type 1**: Human original text
- **Type 2**: LLM-generated text  
- **Type 3**: Human-paraphrased human original text
- **Type 4**: LLM-paraphrased human original text
- **Type 5**: LLM-paraphrased LLM-generated text

## 📁 Module Structure

```
data_processing/
├── config.py                    # Configuration management
├── dataset_loaders.py          # Specialized dataset loaders
├── unified_data_processor.py   # Main unified processing pipeline
├── mrpc_analysis_process.py    # MRPC-specific processing
├── hlpc_analysis_process.py    # HLPC-specific processing
├── paws_analysis_process.py    # PAWS-specific processing
└── README.md                   # This documentation
```

## 🔧 Core Components

### 1. Configuration Management (`config.py`)

Centralized configuration for the data processing pipeline:

```python
from data_processing.config import ProcessingConfig, DatasetConfig

# Default configuration
config = ProcessingConfig()

# Custom configuration
config = ProcessingConfig(
    output_dir="./data/processed",
    similarity_threshold=0.95,
    min_text_length=10,
    max_text_length=1000
)
```

**Key Features:**
- Dataset-specific configurations
- Text preprocessing settings
- Duplicate removal parameters
- Output format options

### 2. Dataset Loaders (`dataset_loaders.py`)

Specialized loaders for each dataset type:

#### MRPC Loader
```python
from data_processing.dataset_loaders import MRPCLoader

loader = MRPCLoader(data_path="./data/mrpc/mrpc_paraphrases.csv")
df = loader.load_and_validate()
text_types = loader.get_text_types()  # Returns Type 1 and Type 3
```

#### PAWS Loader
```python
from data_processing.dataset_loaders import PAWSLoader

loader = PAWSLoader(subset="labeled_final")
df = loader.load_and_validate()
text_types = loader.get_text_types()  # Returns Type 1 and Type 3
```

#### HLPC Loader
```python
from data_processing.dataset_loaders import HLPCLoader

loader = HLPCLoader(data_path="./data/HLPC-data")
df = loader.load_and_validate()
text_types = loader.get_text_types()  # Returns all 5 types
```


### 3. Unified Data Processor (`unified_data_processor.py`)

Main processing pipeline that combines all datasets:

```python
from data_processing.unified_data_processor import UnifiedDataProcessor

# Initialize processor
processor = UnifiedDataProcessor(output_dir="./data/processed")

# Process all datasets
unified_df = processor.process_all_datasets(remove_duplicates=True)

# Analyze the unified dataset
analysis = processor.analyze_unified_data(unified_df)

# Save in multiple formats
processor.save_unified_data(unified_df, "unified_padben_base")
```

## 📊 Dataset-Specific Processing

### MRPC (Microsoft Research Paraphrase Corpus)

**Source**: Microsoft Research  
**Text Types**: 1, 3 (Human original, Human paraphrased)  
**Size**: ~4,000 paraphrase pairs  

```python
from data_processing.mrpc_analysis_process import MRPCDataLoader

loader = MRPCDataLoader()
dataset = loader.load_dataset()
paraphrase_df = loader.get_paraphrase_pairs()
```

**Features:**
- Loads from HuggingFace datasets
- Filters for paraphrase pairs only (label=1)
- Combines train/validation/test splits
- Provides detailed statistics and analysis

### HLPC (Human & LLM Paraphrase Collection)

**Source**: Multiple datasets (MRPC, XSum, QQP, Multi-PIT)  
**Text Types**: 1, 2, 3, 4, 5 (All types available)  
**Size**: ~50,000 samples  

```python
from data_processing.hlpc_analysis_process import HLPCDataLoader

loader = HLPCDataLoader(data_dir="./data/HLPC-data")
combined_df = loader.load_and_combine_datasets()
clean_df = loader.clean_and_deduplicate()
```

**Features:**
- Multi-source data combination
- Advanced deduplication using TF-IDF similarity
- Quality filtering and validation
- Comprehensive data analysis

### PAWS (Paraphrase Adversaries from Word Scrambling)

**Source**: Google Research  
**Text Types**: 1, 3 (Human original, Human paraphrased)  
**Size**: ~50,000 paraphrase pairs  

```python
from data_processing.paws_analysis_process import PAWSDataLoader

loader = PAWSDataLoader(subset="labeled_final")
dataset = loader.load_dataset()
paraphrase_df = loader.get_paraphrase_pairs()
```

**Features:**
- Wikipedia and Quora paraphrase pairs
- Challenging paraphrase identification examples
- Multiple subset support
- Quality analysis and statistics

## 🚀 Usage Examples

### Basic Usage

```python
from data_processing.unified_data_processor import UnifiedDataProcessor

# Initialize processor
processor = UnifiedDataProcessor()

# Process all available datasets
unified_data = processor.process_all_datasets()

print(f"Total samples: {len(unified_data)}")
print(f"Dataset composition:")
print(unified_data['dataset_source'].value_counts())
```

### Individual Dataset Processing

```python
# Process MRPC only
from data_processing.mrpc_analysis_process import MRPCDataLoader

mrpc_loader = MRPCDataLoader()
mrpc_data = mrpc_loader.load_dataset()
mrpc_paraphrases = mrpc_loader.get_paraphrase_pairs()

# Process HLPC only
from data_processing.hlpc_analysis_process import HLPCDataLoader

hlpc_loader = HLPCDataLoader()
hlpc_data = hlpc_loader.load_and_combine_datasets()
hlpc_clean = hlpc_loader.clean_and_deduplicate()
```

### Advanced Configuration

```python
from data_processing.config import ProcessingConfig, DatasetConfig

# Custom configuration
config = ProcessingConfig(
    output_dir="./custom_output",
    similarity_threshold=0.9,
    min_text_length=5,
    max_text_length=2000,
    datasets={
        "MRPC": DatasetConfig(
            name="MRPC",
            path="./data/mrpc/mrpc_paraphrases.csv",
            available_types=[1, 3],
            enabled=True
        ),
        "HLPC": DatasetConfig(
            name="HLPC",
            path="./data/HLPC-data",
            available_types=[1, 2, 3, 4, 5],
            enabled=True
        )
    }
)
```

## 📈 Data Analysis Features

### Unified Dataset Analysis

```python
# Analyze unified dataset
analysis = processor.analyze_unified_data(unified_df)

print("Dataset Composition:")
for dataset, info in analysis["datasets"].items():
    print(f"  {dataset}: {info['count']} samples ({info['percentage']:.1f}%)")

print("\nText Statistics:")
for text_type, stats in analysis["text_statistics"].items():
    print(f"  {text_type}: {stats['count']} samples, avg length: {stats['mean_length']:.1f}")
```

### Quality Metrics

- **Duplicate Detection**: TF-IDF based similarity analysis
- **Text Length Analysis**: Mean, median, min, max, standard deviation
- **Missing Data Analysis**: Percentage of missing values per text type
- **Dataset Composition**: Sample counts and percentages per source

## 🔧 Configuration Options

### ProcessingConfig Parameters

```python
@dataclass
class ProcessingConfig:
    # Output settings
    output_dir: str = "./data/processed"
    save_formats: List[str] = ["csv", "json"]
    
    # Duplicate removal settings
    similarity_threshold: float = 0.95
    similarity_method: str = "cosine"
    
    # Text preprocessing settings
    min_text_length: int = 10
    max_text_length: int = 1000
    remove_empty: bool = True
    
    # Dataset configurations
    datasets: Dict[str, DatasetConfig] = {...}
```

### DatasetConfig Parameters

```python
@dataclass
class DatasetConfig:
    name: str
    path: Optional[str] = None
    available_types: List[int] = field(default_factory=list)
    missing_types: List[int] = field(default_factory=list)
    enabled: bool = True
```

## 📊 Output Formats

### Standardized Schema

All datasets are converted to a unified schema:

```python
standard_columns = [
    'idx',                                    # Global index
    'dataset_source',                         # Source dataset name
    'human_original_text',                   # Type 1
    'llm_generated_text',                     # Type 2
    'human_paraphrased_text',                # Type 3
    'llm_paraphrased_original_text',         # Type 4
    'llm_paraphrased_generated_text'         # Type 5
]
```

### Output Files

- **CSV**: `unified_padben_base.csv` - Main dataset
- **JSON**: `unified_padben_base.json` - JSON format
- **Analysis**: `unified_padben_base_analysis.json` - Detailed statistics

## 🧪 Testing and Validation

### Data Quality Checks

```python
# Validate data integrity
def validate_data_quality(df):
    # Check for required columns
    required_cols = ['idx', 'dataset_source', 'human_original_text']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for empty dataset
    if df.empty:
        raise ValueError("Dataset is empty")
    
    # Check for duplicate indices
    if df['idx'].duplicated().any():
        raise ValueError("Duplicate indices found")
    
    return True
```

### Dataset-Specific Validation

Each loader includes validation methods:

```python
# MRPC validation
mrpc_loader.validate_dataset()

# HLPC validation  
hlpc_loader.validate_combined_dataset()

# PAWS validation
paws_loader.validate_dataset()
```

## 🔄 Processing Pipeline

### 1. Data Loading
- Load individual datasets using specialized loaders
- Validate data integrity and format
- Handle missing or corrupted data gracefully

### 2. Data Standardization
- Convert to unified schema
- Map dataset-specific columns to standard format
- Handle missing text types appropriately

### 3. Data Cleaning
- Remove duplicates using similarity analysis
- Filter by text length requirements
- Remove empty or invalid samples

### 4. Data Analysis
- Generate comprehensive statistics
- Analyze dataset composition
- Identify data quality issues

### 5. Data Export
- Save in multiple formats (CSV, JSON)
- Export analysis results
- Generate processing reports

## 🚨 Error Handling

### Common Issues and Solutions

1. **Missing Dataset Files**
   ```python
   # Check if dataset exists before processing
   if not Path("./data/mrpc/mrpc_paraphrases.csv").exists():
       logger.warning("MRPC data not found, skipping...")
   ```

2. **Empty Datasets**
   ```python
   # Handle empty datasets gracefully
   if df.empty:
       logger.warning("Dataset is empty, returning empty DataFrame")
       return pd.DataFrame(columns=standard_columns)
   ```

3. **Memory Issues**
   ```python
   # Process datasets in chunks for large files
   chunk_size = 10000
   for chunk in pd.read_csv(file_path, chunksize=chunk_size):
       process_chunk(chunk)
   ```

## 📝 Logging and Monitoring

### Logging Configuration

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
```

### Progress Tracking

```python
# Track processing progress
logger.info(f"Processing {dataset_name}...")
logger.info(f"Loaded {len(df)} samples")
logger.info(f"Removed {duplicates_removed} duplicates")
logger.info(f"Final dataset size: {len(clean_df)} samples")
```

## 🔮 Future Enhancements

### Planned Features

2. **Advanced Deduplication**: Semantic similarity using embeddings
3. **Quality Metrics**: Automated quality assessment
4. **Parallel Processing**: Multi-threaded data processing
5. **Caching**: Intelligent caching for faster reprocessing
6. **Data Augmentation**: Synthetic data generation capabilities

### Extension Points

- **Custom Loaders**: Add support for new datasets
- **Custom Filters**: Implement dataset-specific filtering
- **Custom Analysis**: Add domain-specific analysis methods
- **Custom Export**: Support additional output formats

## 📚 Dependencies

### Core Dependencies

```bash
pip install pandas numpy scikit-learn
pip install datasets transformers
pip install pathlib typing
```

### Optional Dependencies

```bash
# For advanced text processing
pip install nltk spacy

# For parallel processing
pip install multiprocessing

# For visualization
pip install matplotlib seaborn
```

## 🤝 Contributing

### Adding New Datasets

1. Create a new loader class in `dataset_loaders.py`
2. Implement `load_and_validate()` and `get_text_types()` methods
3. Add dataset configuration to `config.py`
4. Update `unified_data_processor.py` to include new dataset
5. Add tests and documentation

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Add comprehensive docstrings
- Include error handling and logging
- Write unit tests for new functionality

## 📄 License

This module is part of the PADBen benchmark project. See the main project license for details.

## 🆘 Support

For issues and questions:

1. Check the logs for error messages
2. Verify dataset file paths and formats
3. Ensure all dependencies are installed
4. Check available memory for large datasets
5. Review configuration parameters

---

**Note**: This module is designed to work with the PADBen benchmark framework. Ensure compatibility with the main project requirements and data generation pipeline.
