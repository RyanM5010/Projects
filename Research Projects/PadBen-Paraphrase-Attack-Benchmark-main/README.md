# PadBen-Paraphrase-Attack-Benchmark

A Comprehensive Benchmark for Evaluating AI Text Detectors Against Paraphrase Attacks

## 🎯 Overview

PadBen (Paraphrase Attack Detection Benchmark) is a comprehensive benchmark designed to evaluate the robustness of AI text detectors against various types of paraphrase attacks. The benchmark includes multiple text types, evaluation methods, and quality assurance tools to provide a thorough assessment of text detection systems.

## 🏗️ Project Structure

```
PadBen-Paraphrase-Attack-Benchmark/
├── data/                           # Generated benchmark data and task-specific datasets
├── data_generation/                # LLM-based text generation pipeline (Type 2, 4, 5)
├── data_processing/                # Multi-dataset processing and standardization (MRPC, HLPC, PAWS)
├── data_quality_assurance/         # Quality metrics, similarity analysis, and RAID comparison
├── evaluation/                     # Text detection evaluation framework
│   ├── Model_Based_Detector/        # Fine-tuned detection models
│   └── Zero_Shot_Detector/         # Zero-shot detection methods (Binocular, GLTR, RADAR)
├── intrinsic_mechanism/            # Semantic space analysis experiments
│   ├── experiment1/                # Semantic vs paraphrase analysis using BGE-m3 embeddings
│   └── experiment2/                # Iterative paraphrasing semantic drift analysis
├── task_preparation/               # Task preparation and data formatting utilities
├── conda_env.yaml                  # Conda environment configuration
├── LICENSE                         # License file
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Environment Setup

#### Using Conda (Recommended)

```bash
# Create and activate the conda environment
conda env create -f conda_env.yaml
conda activate padben

# Verify installation
python -c "import pandas, numpy, torch, transformers; print('Environment ready!')"
```

### 2. API Key Setup

#### Data Generation API Keys

```bash
# Set up API keys for generation
python -c "
from data_generation.config.secrets_manager import setup_api_keys_interactive
setup_api_keys_interactive()
"
```

#### Intrinsic Mechanism API Keys

The intrinsic mechanism experiments require additional API keys for their analysis:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

**Note**: 
- Experiment 1 requires OpenAI API key for BGE-m3 embeddings
- Experiment 2 basic analysis (5 iterations) doesn't require API keys
- Experiment 2 extended analysis (10 iterations) requires Novita AI API key for original sentence comparison

### 3. Basic Usage

#### Test Data Generation

```bash
# Test Type 2 generation (LLM-generated text)
python data_generation/test/test_type2_generation.py --samples 20

# Test Type 4 generation (LLM-paraphrased original text)
python data_generation/test/test_type4_generation.py --samples 20 --methods dipper

# Test Type 5 generation (LLM-paraphrased generated text)
python data_generation/test/test_type5_generation.py --method dipper --iterations 1 --samples 20
```

#### Process Existing Data

```bash
# Process MRPC dataset
python data_processing/mrpc_analysis_process.py

# Process HLPC dataset
python data_processing/hlpc_analysis_process.py

# Process PAWS dataset
python data_processing/paws_analysis_process.py
```

#### Quality Assurance

```bash
# Run demo with small sample
python data_quality_assurance/demo.py

# Run full examination
python data_quality_assurance/run_examination.py --full

# Run with custom sample size
python data_quality_assurance/run_examination.py --sample-size 1000

# Short text analysis
python data_quality_assurance/short_length_text/run_short_text_analysis.py --threshold 10
```

## 📊 Text Types

The benchmark includes five distinct text types:

- **Type 1**: Human original text
- **Type 2**: LLM-generated text (using Gemini 2.5 Pro)
- **Type 3**: Human-paraphrased human original text
- **Type 4**: LLM-paraphrased human original text (DIPPER, prompt-based, Llama-3.1-8B)
- **Type 5**: LLM-paraphrased LLM-generated text (1/3/5 iterations)

## 🔧 Core Components

### 1. Data Generation Pipeline (`data_generation/`)

The data generation pipeline creates comprehensive benchmark data using multiple LLM models and paraphrasing techniques.

**Key Features:**
- **Type 2 Generation**: LLM-generated text using Gemini 2.5 Pro
- **Type 4 Generation**: Multiple paraphrasing methods (DIPPER, prompt-based, Llama-3.1-8B)
- **Type 5 Generation**: Iterative paraphrasing with 1, 3, and 5 iterations
- **Configurable Models**: Support for various LLM providers and models
- **Quality Control**: Built-in validation and error handling

### 2. Data Processing Module (`data_processing/`)

Unified processing pipeline for multiple benchmark datasets (MRPC, HLPC, PAWS).

**Key Features:**
- **Multi-Dataset Support**: MRPC, HLPC, PAWS datasets
- **Unified Schema**: Standardized format across all datasets
- **Quality Filtering**: Duplicate removal and data cleaning
- **Comprehensive Analysis**: Statistical analysis and quality metrics

### 3. Quality Assurance Tools (`data_quality_assurance/`)

Comprehensive quality assessment and validation tools.

**Key Features:**
- **Similarity Metrics**: Jaccard similarity, self-BLEU scores
- **Perplexity Analysis**: Language model-based quality assessment
- **RAID Comparison**: Benchmark comparison with RAID dataset
- **Visualization**: Comprehensive quality visualizations
- **Short Text Analysis**: Specialized analysis for short texts

### 4. Evaluation Framework (`evaluation/`)

Multiple evaluation approaches for text detection systems.

**Key Features:**
- **Model-Based Detectors**: Fine-tuned detection models
- **Zero-Shot Detectors**: Binocular, Fast_Detect_GPT, GLTR, RADAR
- **Comprehensive Metrics**: Accuracy, F1-score, precision, recall
- **Statistical Analysis**: Significance testing and confidence intervals

### 5. Intrinsic Mechanism Analysis (`intrinsic_mechanism/`)

Semantic space analysis experiments to understand the underlying mechanisms of paraphrase attacks.

#### Experiment 1: Semantic vs Paraphrase Analysis
**Key Features:**
- **BGE-m3 Embeddings**: High-quality semantic embeddings
- **Distance Analysis**: Cosine, Euclidean, and Manhattan distance metrics
- **Clustering Analysis**: KMeans clustering in semantic space
- **Visualization**: UMAP and t-SNE semantic space plots
- **Research Questions**: Do Type2 and Type4 form distinct clusters in semantic space?

#### Experiment 2: Iterative Paraphrasing Analysis
**Key Features:**
- **Iterative Paraphrasing**: 5-10 iterations using Qwen2.5-3B-Instruct
- **Semantic Drift Analysis**: How paraphrased text deviates from original text
- **PCA Analysis**: Principal Component Analysis on hidden states and embeddings
- **Trajectory Analysis**: Semantic trajectory visualization across iterations
- **Distance Analysis**: Inter-iteration distance measurements

### 6. Task Preparation (`task_preparation/`)

Utilities for preparing and formatting data for different task types.

**Key Features:**
- **Sentence Pair Tasks**: Paraphrase detection and similarity tasks
- **Single Sentence Tasks**: Text classification and generation tasks
- **Data Formatting**: Standardized input/output formats
- **Task-Specific Processing**: Customized processing for different task types


## 📚 Documentation

### Module Documentation

- **[Data Generation Pipeline](data_generation/README.md)**: Complete guide to LLM-based text generation (Type 2, 4, 5)
- **[Data Processing Module](data_processing/README.md)**: Multi-dataset processing and standardization
- **[Quality Assurance Tools](data_quality_assurance/README.md)**: Quality metrics, similarity analysis, and RAID comparison
- **[Task Preparation](task_preparation/README.md)**: Task preparation and data formatting utilities

### Intrinsic Mechanism Experiments

- **[Experiment 1](intrinsic_mechanism/experiment1/README.md)**: Semantic vs Paraphrase Analysis using BGE-m3 embeddings
- **[Experiment 2](intrinsic_mechanism/experiment2/README.md)**: Iterative Paraphrasing Semantic Drift Analysis

### Data Documentation

- **[Data Documentation](data/README.md)**: Generated benchmark data and task-specific datasets

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: This benchmark is designed for research purposes. Ensure compliance with model usage policies and data licensing requirements.