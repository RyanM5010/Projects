# PADBen Data Generation Pipeline

This directory contains the complete text generation pipeline for the PADBen benchmark dataset. The system generates three types of LLM-based text data to create a comprehensive benchmark for paraphrase detection and text similarity tasks.

## 🏗️ Architecture Overview

The PADBen generation pipeline consists of three main text generation types:

- **Type 2**: LLM-generated text (sentence completion & question-answer)
- **Type 4**: LLM-paraphrased original text (DIPPER, prompt-based, & Llama-3.1-8B)
- **Type 5**: LLM-paraphrased generated text (DIPPER & prompt-based, 1/3/5 iterations)

## 📁 Directory Structure

```
data_generation/
├── orchestrator.py              # Main pipeline coordinator
├── type4_generation.py          # Type 4 text generation
├── type5_generation.py         # Type 5 text generation
├── config/                     # Configuration management
│   ├── base_model_config.py    # Base model configurations
│   ├── generation_model_config.py  # Main generation config
│   ├── type2_config.py         # Type 2 specific config
│   ├── type4_config.py         # Type 4 specific config
│   ├── type5_config.py         # Type 5 specific config
│   ├── logging_config.py       # Logging configuration
│   └── secrets_manager.py     # API key management
├── type2_generation/           # Type 2 generation modules
│   ├── type2_generation.py     # Main Type 2 generator
│   └── keywords_prefix_extraction.py  # Text preprocessing
├── merge/                      # Data merging utilities
│   ├── merge_script.py         # Multi-file merger
│   ├── merge_intraType.py      # Intra-type merging
│   ├── postprocessing.py       # Data postprocessing
│   ├── error_analysis.py       # Error analysis tools
│   ├── error_handling.py       # Error handling utilities
│   ├── reformat_merged_file.py # File reformatting
│   └── remove_type5_null.py    # Type 5 null removal
└── test/                       # Test modules
    ├── test_type2_generation.py
    ├── test_type4_generation.py
    └── test_type5_generation.py
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install required dependencies
pip install pandas numpy torch transformers accelerate
pip install google-generativeai nltk tqdm
pip install fastapi uvicorn  # For API endpoints

# Install Llama GGUF support
pip install llama-cpp-python

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

### Llama-3.1-8B Model Setup

The system supports the [Llama-3.1-8B paraphrase model](https://huggingface.co/mradermacher/Llama-3.1-8B-paraphrase-type-generation-apty-sigmoid-GGUF) for efficient paraphrasing:

```bash
# Download the GGUF model (choose appropriate quantization)
# Q4_K_M is recommended for good quality/size balance
wget https://huggingface.co/mradermacher/Llama-3.1-8B-paraphrase-type-generation-apty-sigmoid-GGUF/resolve/main/Llama-3.1-8B-paraphrase-type-generation-apty-sigmoid.Q4_K_M.gguf

# Or use the model directly from HuggingFace (will be downloaded automatically)
```

### 2. Configuration Setup

```python
from data_generation.config.generation_model_config import DEFAULT_CONFIG
from data_generation.config.secrets_manager import setup_api_keys_interactive

# Set up API keys interactively
setup_api_keys_interactive()

# Use default configuration
config = DEFAULT_CONFIG
```

### 3. Basic Usage

```python
import asyncio
from data_generation.orchestrator import PADBenOrchestrator, GenerationOptions

async def main():
    # Initialize orchestrator
    orchestrator = PADBenOrchestrator()
    
    # Configure generation options
    options = GenerationOptions(
        type2=True,           # Generate Type 2 text
        type4=True,           # Generate Type 4 paraphrases
        type5=True,           # Generate Type 5 paraphrases
        type2_method="auto",   # Auto-select Type 2 method
        type5_iterations=[1, 3, 5]  # Type 5 iteration levels
    )
    
    # Run the pipeline
    final_df, status = await orchestrator.run_generation_pipeline(
        input_file="data/input.csv",
        options=options,
        output_dir="data/generated"
    )
    
    print(f"Generated {len(final_df)} samples")
    print(f"Status: {status}")

# Run the pipeline
asyncio.run(main())
```

## 🔧 Generation Types

### Type 2: LLM-Generated Text

Generates original text using LLMs with two methods:

- **Sentence Completion**: Uses extracted prefixes and keywords
- **Question-Answer**: Incorporates extracted keywords and constraints

```python
from data_generation.type2_generation.type2_generation import EnhancedType2Generator

generator = EnhancedType2Generator(config.type2_config)
result_df = await generator.generate_for_dataset(
    df, 
    method=Type2Method.AUTO,
    output_dir="output/"
)
```

### Type 4: LLM-Paraphrased Original Text

Paraphrases human original text using three methods:

- **DIPPER**: Specialized HuggingFace paraphrasing model
- **Prompt-based**: Gemini-based paraphrasing  
- **Llama-3.1-8B**: GGUF-optimized Llama model for efficient inference

```python
from data_generation.type4_generation import Type4Generator

generator = Type4Generator(config.type4_config)

# Using DIPPER method
result_df = await generator.generate_for_dataset(
    df,
    method=Type4ParaphraseMethod.DIPPER,
    output_dir="output/"
)

# Using Llama-3.1-8B method
result_df = await generator.generate_for_dataset(
    df,
    method=Type4ParaphraseMethod.LLAMA,
    output_dir="output/"
)
```

### Type 5: LLM-Paraphrased Generated Text

Iteratively paraphrases Type 2 generated text:

- **1, 3, or 5 iterations** of paraphrasing
- **DIPPER and prompt-based** methods
- **Iteration history tracking**

```python
from data_generation.type5_generation import Type5Generator

generator = Type5Generator(config.type5_config)
result_df = await generator.generate_for_dataset(
    df,
    method=Type4ParaphraseMethod.DIPPER,
    iteration=IterationLevel(3),
    output_dir="output/"
)
```

## 🦙 Llama-3.1-8B Integration

The system now includes comprehensive support for the [Llama-3.1-8B paraphrase model](https://huggingface.co/mradermacher/Llama-3.1-8B-paraphrase-type-generation-apty-sigmoid-GGUF) as a third paraphrasing method for Type 4 generation, alongside DIPPER and prompt-based methods.

### 🎯 Overview

The Llama-3.1-8B model has been successfully integrated as a third paraphrasing method for Type 4 generation, providing users with more options for generating high-quality paraphrases with different model architectures and capabilities.

### 🔧 Key Features Implemented

#### 1. GGUF Model Support
- **Quantization Levels**: Support for Q2_K, Q3_K_S, Q4_K_S, Q5_K_S, Q6_K, Q8_0
- **Memory Efficiency**: Lower memory footprint compared to full precision models
- **GPU Acceleration**: Optional GPU support with `n_gpu_layers`
- **CPU Optimization**: Multi-threading support with `n_threads`

#### 2. Llama-Specific Prompting
- **Chat Format**: Uses Llama-3.1 chat template with system/user/assistant roles
- **Specialized Instructions**: Optimized prompts for paraphrase generation
- **Context Management**: Proper handling of conversation tokens

#### 3. Integration with Existing Pipeline
- **Seamless Integration**: Works alongside DIPPER and prompt-based methods
- **Fallback Support**: Can fall back to other methods if Llama fails
- **Consistent API**: Same interface as other paraphrasing methods
- **Statistics Tracking**: Full integration with existing monitoring

#### 4. Configuration Management
- **Flexible Settings**: Configurable temperature, top_p, top_k, repetition_penalty
- **Device Selection**: Automatic or manual device selection (CPU/GPU)
- **Memory Management**: Configurable context length and batch processing
- **Error Handling**: Comprehensive retry logic and error reporting

### 📁 Files Modified

#### Configuration Files
- **`config/type4_config.py`**: Added `LLAMA` to `Type4ParaphraseMethod` enum, `llama_model` configuration, `llama_settings`, and `llama_prompt_template`
- **`config/base_model_config.py`**: Added `create_llama_paraphrase_config()` function with optimal settings

#### Core Generation Files
- **`type4_generation.py`**: Added `LlamaParaphraser` class, integrated into `Type4Generator`, added `paraphrase_with_llama()` method
- **`orchestrator.py`**: Added `type4_method_llama` option, Llama status tracking, updated CLI arguments

#### Test Files
- **`test/test_llama_integration.py`**: Comprehensive test script for Llama integration validation

### 🚀 Usage Examples

#### Basic Usage
```python
from data_generation.type4_generation import Type4Generator, Type4ParaphraseMethod

# Initialize generator
generator = Type4Generator()

# Generate paraphrases using Llama
result_df = await generator.generate_for_dataset(
    df,
    method=Type4ParaphraseMethod.LLAMA,
    output_dir="output/"
)
```

#### CLI Usage
```bash
# Generate Type 4 with Llama only
python -m data_generation.orchestrator \
    --input data/input.csv \
    --type4 --no-type4-dipper --no-type4-prompt --type4-llama

# Generate Type 4 with all methods
python -m data_generation.orchestrator \
    --input data/input.csv \
    --type4 --type4-dipper --type4-prompt --type4-llama
```

#### Configuration
```python
from data_generation.config.base_model_config import create_llama_paraphrase_config

# Configure Llama model
llama_config = create_llama_paraphrase_config(
    model_id="mradermacher/Llama-3.1-8B-paraphrase-type-generation-apty-sigmoid-GGUF",
    device="auto",  # or "cuda" for GPU
    temperature=0.7
)
```

### 📊 Performance Characteristics

#### Model Sizes
- **Q4_K_S** (4.8GB): Fast, recommended for most use cases
- **Q5_K_S** (5.7GB): Better quality, good balance
- **Q6_K** (6.7GB): Very good quality
- **Q8_0** (8.6GB): Best quality, requires more memory

#### Hardware Requirements
- **CPU**: Multi-core recommended for optimal performance
- **GPU**: Optional but recommended for faster inference
- **RAM**: 6-16GB depending on quantization level
- **Storage**: 3-17GB for model files

### 🛠️ Installation Requirements

#### Core Dependencies
```bash
pip install llama-cpp-python
pip install transformers torch
pip install pandas numpy
```

#### GPU Support (Optional)
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### 🧪 Testing and Validation

#### Test Coverage
- ✅ Model loading and initialization
- ✅ Paraphrase generation with different inputs
- ✅ Error handling and retry logic
- ✅ Integration with existing pipeline
- ✅ CLI argument processing
- ✅ Output format validation

#### Test Script
Run the integration test:
```bash
python data_generation/test/test_llama_integration.py
```

### 🎉 Benefits

1. **Model Diversity**: Three different paraphrasing approaches (DIPPER, prompt-based, Llama)
2. **Quality Options**: Users can choose the best method for their specific needs
3. **Efficiency**: GGUF quantization provides good quality with lower resource usage
4. **Flexibility**: Can be used standalone or in combination with other methods
5. **Scalability**: Supports both CPU and GPU inference
6. **Integration**: Seamlessly works with existing PADBen pipeline

### 🔮 Future Enhancements

1. **Model Selection**: Automatic selection of best quantization based on available resources
2. **Batch Processing**: Optimized batch processing for Llama models
3. **Caching**: Model caching for faster subsequent runs
4. **Metrics**: Quality metrics specific to Llama paraphrases
5. **Fine-tuning**: Support for custom fine-tuned Llama models

### 📝 Notes

- The Llama model requires the `llama-cpp-python` library for GGUF support
- GPU acceleration is optional but recommended for better performance
- The model automatically handles different quantization levels
- Integration maintains backward compatibility with existing code
- All existing functionality remains unchanged

## ⚙️ Configuration

### Model Configuration

```python
from data_generation.config.generation_model_config import GenerationConfig

config = GenerationConfig(
    # LLM Provider settings
    llm_provider="gemini",
    api_key="your-api-key",
    
    # Batch processing
    batch_processing=BatchProcessingConfig(
        type2_batch_size=5,
        type4_batch_size=8,
        type5_batch_size=6
    ),
    
    # Output settings
    output_config=OutputConfig(
        output_dir="data/generated",
        format=OutputFormat.BOTH
    )
)
```

### Type-Specific Configuration

```python
# Type 2 configuration
type2_config = Type2GenerationConfig(
    sentence_completion_prompts=["Complete this sentence: {prefix}"],
    question_answer_prompts=["Answer this question: {question}"]
)

# Type 4 configuration  
type4_config = Type4GenerationConfig(
    dipper_model="kalpeshk2011/dipper-paraphraser-11B",
    dipper_settings={
        "max_length": 512,
        "num_beams": 4,
        "temperature": 0.7
    },
    llama_model="mradermacher/Llama-3.1-8B-paraphrase-type-generation-apty-sigmoid-GGUF",
    llama_settings={
        "max_length": 300,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40
    }
)

# Type 5 configuration
type5_config = Type5GenerationConfig(
    iteration_levels=[1, 3, 5],
    similarity_threshold=0.3,
    max_iteration_time=30.0
)
```

## 🔄 Data Merging

### Merge Multiple Files

```python
from data_generation.merge.merge_script import merge_padben_files

success = merge_padben_files(
    input_files=["file1.json", "file2.json", "file3.json"],
    output_path="merged_output.json",
    random_seed=42,
    merge_type="auto",
    null_fill_merge=True
)
```

### Intra-Type Merging

```python
from data_generation.merge.merge_intraType import merge_intra_type_files

result = merge_intra_type_files(
    input_files=["type2_part1.json", "type2_part2.json"],
    output_path="merged_type2.json",
    merge_type="type2"
)
```

## 🧪 Testing

### Run Individual Tests

```bash
# Test Type 2 generation
python -m data_generation.test.test_type2_generation

# Test Type 4 generation  
python -m data_generation.test.test_type4_generation

# Test Type 5 generation
python -m data_generation.test.test_type5_generation
```

### Test Configuration

```python
# Test with sample data
from data_generation.test.test_type2_generation import test_type2_generation

test_df = pd.DataFrame({
    "idx": [1, 2, 3],
    "human_original_text": ["Sample text 1", "Sample text 2", "Sample text 3"],
    "human_paraphrased_text": ["Paraphrase 1", "Paraphrase 2", "Paraphrase 3"]
})

result = await test_type2_generation(test_df)
```

## 📊 Output Structure

The pipeline generates comprehensive output with the following columns:

```python
expected_columns = [
    # Base columns
    "idx", "dataset_source", "human_original_text", "human_paraphrased_text",
    
    # Type 2 generated columns
    "llm_generated_text", "llm_generated_text_method",
    
    # Type 4 paraphrased original columns
    "llm_paraphrased_original_text_dipper", "llm_paraphrased_original_text_prompt", "llm_paraphrased_original_text_llama",
    
    # Type 5 paraphrased generated columns (DIPPER)
    "llm_paraphrased_generated_text_dipper_iter1",
    "llm_paraphrased_generated_text_dipper_iter3", 
    "llm_paraphrased_generated_text_dipper_iter5",
    
    # Type 5 paraphrased generated columns (Prompt-based)
    "llm_paraphrased_generated_text_prompt_iter1",
    "llm_paraphrased_generated_text_prompt_iter3",
    "llm_paraphrased_generated_text_prompt_iter5",
]
```

## 🎯 Command Line Interface

### Basic Usage

```bash
# Generate all types
python -m data_generation.orchestrator \
    --input data/input.csv \
    --output-dir data/generated \
    --type2 --type4 --type5

# Generate specific types
python -m data_generation.orchestrator \
    --input data/input.csv \
    --type2 --type2-method sentence_completion

# Generate Type 4 with Llama only
python -m data_generation.orchestrator \
    --input data/input.csv \
    --type4 --no-type4-dipper --no-type4-prompt --type4-llama

# Generate Type 4 with all methods (DIPPER, prompt-based, Llama)
python -m data_generation.orchestrator \
    --input data/input.csv \
    --type4 --type4-dipper --type4-prompt --type4-llama

# Generate Type 4 with only Llama method
python -m data_generation.orchestrator \
    --input data/input.csv \
    --type4 --no-type4-dipper --no-type4-prompt --type4-llama

# Type 5 with specific iterations
python -m data_generation.orchestrator \
    --input data/input.csv \
    --type5 --type5-iterations 1 3

# Generate Type 4 with all methods (DIPPER, prompt-based, Llama)
python -m data_generation.orchestrator \
    --input data/input.csv \
    --type4 --type4-dipper --type4-prompt --type4-llama
```

### Advanced Options

```bash
# Custom configuration
python -m data_generation.orchestrator \
    --input data/input.csv \
    --output-dir data/generated \
    --type2 --type4 --type5 \
    --type2-method auto \
    --type4-dipper --type4-prompt \
    --type5-dipper --type5-prompt \
    --type5-iterations 1 3 5 \
    --no-intermediate \
    --no-metadata
```

## 🦙 Llama-3.1-8B Integration

The system now includes support for the [Llama-3.1-8B paraphrase model](https://huggingface.co/mradermacher/Llama-3.1-8B-paraphrase-type-generation-apty-sigmoid-GGUF) which offers:

### Key Features

- **GGUF Format**: Optimized for efficient inference with reduced memory usage
- **Multiple Quantizations**: Choose from Q2_K to Q8_0 based on quality/size requirements
- **GPU Acceleration**: Supports CUDA for faster inference
- **Specialized Training**: Fine-tuned specifically for paraphrase generation
- **Memory Efficient**: Lower memory footprint compared to full precision models

### Model Selection

```python
# Configure Llama model with specific quantization
from data_generation.config.base_model_config import create_llama_paraphrase_config

llama_config = create_llama_paraphrase_config(
    device="cuda",  # Use GPU if available
    torch_dtype="float16",  # Use half precision
    temperature=0.7
)

# Update Type 4 config
type4_config.llama_model = llama_config
type4_config.llama_settings = {
    "max_length": 300,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "repetition_penalty": 1.1
}
```

### Performance Comparison

| Method | Speed | Quality | Memory Usage | GPU Support |
|--------|-------|---------|--------------|-------------|
| DIPPER | Medium | High | High | Yes |
| Gemini | Fast | High | Low | No |
| Llama-3.1-8B | Fast | High | Medium | Yes |

## 🔍 Error Handling

### Retry Logic

```python
# Configure error handling
config.error_handling = ErrorHandlingStrategy.RETRY
config.retry_config = RetryConfig(
    max_retries=3,
    retry_delay=1.0,
    exponential_backoff=True
)
```

### Error Analysis

```python
from data_generation.merge.error_analysis import analyze_generation_errors

error_report = analyze_generation_errors(
    input_file="data/input.csv",
    output_file="data/generated/output.csv"
)
```

## 📈 Performance Optimization

### Batch Processing

```python
# Optimize batch sizes
config.batch_processing = BatchProcessingConfig(
    type2_batch_size=10,      # Increase for faster processing
    type4_batch_size=15,      # Adjust based on GPU memory
    type5_batch_size=8,       # Balance speed vs memory
    max_concurrent_type2=5,    # Parallel processing
    max_concurrent_type4=6,
    max_concurrent_type5=4
)
```

### Memory Management

```python
# Memory-efficient configuration
from data_generation.config.type4_config import create_memory_efficient_type4_config

type4_config = create_memory_efficient_type4_config(
    max_memory_gb=8,
    device="cuda:0"
)
```

## 🛠️ Troubleshooting

### Common Issues

1. **API Key Issues**: Use `setup_api_keys_interactive()` to configure keys
2. **Memory Issues**: Reduce batch sizes or use memory-efficient configs
3. **Model Loading**: Ensure transformers and torch are properly installed
4. **Dependency Issues**: Check all required packages are installed

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
config.logging_config = LoggingConfig(
    level="DEBUG",
    log_to_file=True,
    log_file="generation.log"
)
```

## 📚 API Reference

### Main Classes

- `PADBenOrchestrator`: Main pipeline coordinator
- `EnhancedType2Generator`: Type 2 text generation
- `Type4Generator`: Type 4 paraphrasing
- `Type5Generator`: Type 5 iterative paraphrasing
- `DataStructureManager`: Output structure management
- `DependencyManager`: Generation dependencies

### Key Methods

- `run_generation_pipeline()`: Execute complete pipeline
- `generate_for_dataset()`: Generate for specific type
- `merge_padben_files()`: Merge multiple files
- `validate_input_data()`: Validate input structure

## 🤝 Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive type hints and docstrings
3. Include tests for new functionality
4. Update configuration as needed
5. Follow the error handling patterns

## 📄 License

This code is part of the PADBen benchmark project. See the main project license for details.
