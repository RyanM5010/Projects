# Task Preparation Module

This module converts organized JSON data containing various text types into task-specific datasets for paraphrase-based LLM detection research.

## 🎯 Overview

The module provides **three distinct methods** for preparing classification tasks:

### 1. Single-Sentence Exhaustive Method (`single_sentence/exhaustive_method/`)
- **Approach**: Uses ALL instances from both relevant text types exhaustively
- **Dataset Size**: 2x original size (e.g., 16k + 16k = 32k samples)
- **Label Distribution**: Always balanced (50-50)
- **Use Case**: Maximum data utilization, comprehensive evaluation

### 2. Single-Sentence Sampling Method (`single_sentence/sampling_method/`)
- **Approach**: Randomly samples ONE instance per original sample
- **Dataset Size**: Same as original (e.g., 16k samples)
- **Label Distribution**: Configurable via sampling probabilities (30-70, 50-50, 80-20)
- **Use Case**: Eliminates semantic repetition, tests distribution effects

### 3. Sentence-Pair Method (`sentence_pair/`)
- **Approach**: Creates sentence pairs for classification tasks
- **Dataset Size**: Same as original (e.g., 16k pairs)
- **Label Distribution**: Configurable
- **Use Case**: Pairwise comparison tasks, relationship detection

## 📁 Directory Structure

```
task_preparation/
├── __init__.py                    # Module initialization
├── main.py                        # Main entry point
├── README.md                      # This documentation
├── single_sentence/               # Single-sentence classification tasks
│   ├── exhaustive_method/         # Method 1: Use all instances exhaustively
│   │   ├── __init__.py
│   │   ├── main.py                # Entry point for exhaustive method
│   │   ├── config.py              # Configuration settings
│   │   ├── pipeline.py            # Processing pipeline
│   │   ├── task_processors.py     # Task-specific processors
│   │   ├── base_processor.py      # Base processor class
│   │   ├── data_models.py         # Data models and schemas
│   │   ├── utils.py               # Utility functions
│   │   └── README.md              # Method-specific documentation
│   └── sampling_method/           # Method 2: Random sampling approach
│       ├── __init__.py
│       ├── main.py                # Entry point for sampling method
│       ├── config.py              # Configuration settings
│       ├── pipeline.py            # Processing pipeline
│       ├── task_processors.py     # Task-specific processors
│       ├── base_processor.py      # Base processor class
│       ├── data_models.py         # Data models and schemas
│       ├── utils.py               # Utility functions
│       └── README.md              # Method-specific documentation
└── sentence_pair/                 # Sentence-pair classification tasks
    ├── __init__.py
    ├── main.py                    # Entry point for sentence-pair method
    ├── config.py                  # Configuration settings
    ├── pipeline.py                # Processing pipeline
    ├── task_processors.py         # Task-specific processors
    ├── base_processor.py          # Base processor class
    ├── data_models.py             # Data models and schemas
    ├── utils.py                   # Utility functions
    └── README.md                  # Method-specific documentation
```

## 🚀 Quick Start

### Choose Your Method

#### 1. Single-Sentence Exhaustive Method
```bash
# Get help for exhaustive method
python -m task_preparation.single_sentence.exhaustive_method.main --help

# Run exhaustive method
python -m task_preparation.single_sentence.exhaustive_method.main \
    --input data/merged_data.json \
    --output results/exhaustive/ \
    --task TASK1
```

#### 2. Single-Sentence Sampling Method
```bash
# Get help for sampling method
python -m task_preparation.single_sentence.sampling_method.main --help

# Run sampling method with 50-50 distribution
python -m task_preparation.single_sentence.sampling_method.main \
    --input data/merged_data.json \
    --output results/sampling/ \
    --task TASK1 \
    --label-ratio 0.5
```

#### 3. Sentence-Pair Method
```bash
# Get help for sentence-pair method
python -m task_preparation.sentence_pair.main --help

# Run sentence-pair method
python -m task_preparation.sentence_pair.main \
    --input data/merged_data.json \
    --output results/sentence_pair/ \
    --task TASK1
```

### Using the Main Entry Point
```bash
# Show general help
python -m task_preparation.main --help

# Get help for specific method
python -m task_preparation.main --help-method exhaustive
python -m task_preparation.main --help-method sampling
python -m task_preparation.main --help-method sentence_pair
```

## 📊 Input Data Types

The module processes JSON data containing:
- **Type1**: Human original text
- **Type2**: LLM generated text  
- **Type3**: Human paraphrased text
- **Type4**: LLM paraphrased original text
- **Type5**: LLM paraphrased generated text (1st and 3rd iterations)

## 🎯 Available Tasks

All methods support these tasks:

| Task | Description | Input Types | Labels |
|------|-------------|-------------|---------|
| **Task1** | Paraphrase Source Attribution without Context | Type3 vs Type4 | 0=Human, 1=LLM |
| **Task2** | General Text Authorship Detection | Type1 vs Type2 | 0=Human, 1=LLM |
| **Task3** | AI Text Laundering Detection | Type1 vs Type5 | 0=Original, 1=Laundered |
| **Task4** | Iterative Paraphrase Depth Detection | Type5-1st vs Type5-3rd | 0=1st iter, 1=3rd iter |
| **Task5** | Original vs Deep Paraphrase Attack Detection | Type1 vs Type5-3rd | 0=Original, 1=Deep Attack |

## 🔧 Method Comparison

### Exhaustive Method
- **Pros**: Maximum data utilization, balanced distribution, comprehensive evaluation
- **Cons**: Semantic repetition, larger dataset size, potential overfitting
- **Best for**: Research requiring maximum data, balanced evaluation

### Sampling Method
- **Pros**: Eliminates semantic repetition, configurable distribution, smaller dataset
- **Cons**: Less data utilization, potential sampling bias
- **Best for**: Testing distribution effects, realistic scenarios

### Sentence-Pair Method
- **Pros**: Pairwise comparison, relationship detection, flexible pairing
- **Cons**: Different task format, requires pair construction
- **Best for**: Comparative analysis, relationship-based detection

## 📈 Usage Examples

### Example 1: Exhaustive Method for Task1
```python
# This will create a dataset with ALL Type3 and Type4 instances
# Result: 32k samples (16k Type3 + 16k Type4) with balanced labels
python -m task_preparation.single_sentence.exhaustive_method.main \
    --input data/merged_data.json \
    --output results/exhaustive_task1/ \
    --task TASK1 \
    --seed 42
```

### Example 2: Sampling Method with Custom Distribution
```python
# This will create a dataset with 30% Type4 and 70% Type3
# Result: 16k samples with 30-70 label distribution
python -m task_preparation.single_sentence.sampling_method.main \
    --input data/merged_data.json \
    --output results/sampling_task1_30_70/ \
    --task TASK1 \
    --label-ratio 0.3 \
    --seed 42
```

### Example 3: Sentence-Pair Method for Task2
```python
# This will create sentence pairs (Type1, Type2)
# Result: 16k pairs for pairwise classification
python -m task_preparation.sentence_pair.main \
    --input data/merged_data.json \
    --output results/sentence_pair_task2/ \
    --task TASK2 \
    --seed 42
```

## ⚙️ Configuration

Each method has its own configuration file with specific settings:

- **Exhaustive Method**: `single_sentence/exhaustive_method/config.py`
- **Sampling Method**: `single_sentence/sampling_method/config.py`
- **Sentence-Pair Method**: `sentence_pair/config.py`

Key configuration options:
- Input/output paths
- Task selection
- Random seed
- Label distribution (sampling method)
- Validation settings
- Logging configuration

## 📊 Output Format

All methods generate:
- **Training set**: `train.json` - Main training data
- **Validation set**: `val.json` - Validation data
- **Test set**: `test.json` - Test data
- **Metadata**: `metadata.json` - Dataset statistics and configuration
- **Logs**: Processing logs and statistics

## 🔍 Troubleshooting

### Common Issues

1. **Input Data Format**
   ```bash
   # Ensure your JSON file has the correct structure
   # Required fields: human_original_text, llm_generated_text, etc.
   ```

2. **Task Selection**
   ```bash
   # Available tasks: TASK1, TASK2, TASK3, TASK4, TASK5
   # Check task requirements in the respective README files
   ```

3. **Output Directory**
   ```bash
   # Ensure output directory exists and is writable
   mkdir -p results/your_method/
   ```

### Debug Mode
```bash
# Enable debug logging for any method
python -m task_preparation.single_sentence.exhaustive_method.main \
    --input data/merged_data.json \
    --output results/ \
    --task TASK1 \
    --log-level DEBUG
```

## 📚 Dependencies

### Core Dependencies
```bash
pip install pandas numpy json
pip install scikit-learn
pip install pathlib typing
```

### Optional Dependencies
```bash
# For advanced logging
pip install logging

# For data validation
pip install numpy pandas
```

## 🤝 Contributing

### Adding New Tasks
1. Add task processor in `task_processors.py`
2. Update configuration in `config.py`
3. Add task documentation
4. Test with sample data

### Adding New Methods
1. Create new method directory
2. Implement base processor
3. Add configuration
4. Update main entry point
5. Add documentation

## 📄 License

This module is part of the PADBen benchmark project. See the main project license for details.

## 🆘 Support

For issues and questions:

1. Check the method-specific README files
2. Verify input data format
3. Check configuration settings
4. Review processing logs
5. Test with sample data

---

**Note**: This module is designed to work with the PADBen benchmark framework. Ensure compatibility with the main project requirements and data processing pipeline.