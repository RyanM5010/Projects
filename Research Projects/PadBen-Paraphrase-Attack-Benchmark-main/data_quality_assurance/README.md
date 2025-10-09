# Data Quality Assurance Module

This module provides comprehensive data quality assessment tools for the PADBen (Paraphrase Attack Detection Benchmark) dataset. It includes various metrics, visualizations, and comparison utilities to evaluate the quality and characteristics of different text types in the dataset.

## Overview

The data quality assurance module is designed to:
- Calculate similarity metrics between different text types
- Compute self-BLEU scores for diversity assessment
- Measure perplexity scores using language models
- Compare results with RAID benchmark
- Visualize similarity matrices and quality metrics
- Identify and analyze abnormally short texts

## Module Structure

```
data_quality_assurance/
├── __init__.py                 # Module initialization and exports
├── main.py                     # Main orchestrator class
├── config.py                   # Configuration settings
├── data_loader.py              # Data loading utilities
├── metrics.py                  # Core metrics implementation
├── comparison.py               # RAID benchmark comparison
├── visualization.py            # Visualization utilities
├── run_examination.py          # Command-line examination script
├── demo.py                    # Demo script for quick testing
└── short_length_text/          # Short text analysis submodule
    ├── short_text_analyzer.py  # Short text analysis implementation
    ├── run_short_text_analysis.py  # Command-line short text analysis
    └── demo_short_text_analysis.py  # Demo for short text analysis
```

## Core Components

### 1. DataQualityExaminer (`main.py`)

The main orchestrator class that coordinates all quality assessment tasks.

**Key Features:**
- Loads and preprocesses PADBen dataset
- Calculates Jaccard similarity, self-BLEU, and perplexity metrics
- Compares results with RAID benchmark
- Generates comprehensive visualizations
- Saves results in structured format

**Usage:**
```python
from data_quality_assurance import DataQualityExaminer

# Initialize examiner
examiner = DataQualityExaminer()

# Run complete examination
results = examiner.run_complete_examination(
    sample_size=1000,  # Optional: limit sample size
    generate_visualizations=True
)
```

### 2. Metrics Module (`metrics.py`)

Implements core quality metrics for text analysis.

**Available Metrics:**

#### JaccardSimilarityCalculator
- Calculates Jaccard similarity between text pairs
- Configurable n-gram size and case sensitivity
- Supports batch processing for efficiency

#### SelfBLEUCalculator
- Computes self-BLEU scores to measure text diversity
- Uses NLTK's BLEU implementation with smoothing
- Configurable n-gram weights and smoothing methods

#### PerplexityCalculator
- Measures perplexity using pre-trained language models
- Supports multiple models (GPT-2, etc.)
- Handles long texts with sliding window approach

#### MetricsAggregator
- Combines all metrics into comprehensive analysis
- Provides statistical summaries and distributions
- Handles missing data gracefully

### 3. RAID Comparison (`comparison.py`)

Compares PADBen metrics with RAID benchmark dataset.

**Comparison Metrics:**
- Self-BLEU scores
- Perplexity measurements
- Generation statistics
- Statistical significance testing

### 4. Visualization (`visualization.py`)

Creates comprehensive visualizations for quality metrics.

**Available Visualizations:**
- Similarity heatmaps
- Metric distribution plots
- Comparison charts with RAID benchmark
- Quality score scatter plots

### 5. Short Text Analysis (`short_length_text/`)

Specialized analysis for identifying and examining abnormally short texts.

**Features:**
- Automatic detection of short texts
- Statistical analysis of text length distributions
- Identification of potential data quality issues
- Detailed reporting of problematic records

## Quick Start

### 1. Basic Usage

```bash
# Run demo with small sample
python demo.py

# Run full examination
python run_examination.py --full

# Run with custom sample size
python run_examination.py --sample-size 1000
```

### 2. Command-Line Options

```bash
# Full examination with all metrics
python run_examination.py --full --output-dir ./results

# Specific metric analysis
python run_examination.py --quality_type jaccard_similarity --full
python run_examination.py --quality_type self-BLEU --sample-size 500
python run_examination.py --quality_type perplexity --full --no-viz

# Short text analysis
python short_length_text/run_short_text_analysis.py --threshold 10
```

### 3. Programmatic Usage

```python
from data_quality_assurance import DataQualityExaminer, ShortTextAnalyzer

# Main quality examination
examiner = DataQualityExaminer()
results = examiner.run_complete_examination(sample_size=1000)

# Short text analysis
short_analyzer = ShortTextAnalyzer(threshold=10)
short_results = short_analyzer.analyze_short_texts()
```

## Configuration

### Text Types

The module analyzes six different text types:
- `type1`: Human original text
- `type2`: LLM generated text
- `type3`: Human paraphrased text
- `type4`: LLM paraphrased original text (prompt-based)
- `type5_1st`: LLM paraphrased generated text (1st iteration)
- `type5_3rd`: LLM paraphrased generated text (3rd iteration)

### Metric Settings

**Jaccard Similarity:**
- N-gram size: 1 (unigrams)
- Case sensitivity: False

**BLEU Scores:**
- Maximum n-grams: 4
- Smoothing: Enabled
- Weights: Equal for 1-4 grams

**Perplexity:**
- Models: GPT-2 variants
- Batch processing for efficiency
- Sliding window for long texts

## Output Files

The module generates several output files:

### Main Results
- `quality_metrics.json`: Comprehensive metrics summary
- `similarity_matrix.json`: Pairwise similarity scores
- `raid_comparison.json`: Comparison with RAID benchmark
- `visualization_summary.json`: Visualization metadata

### Visualizations
- `similarity_heatmap.png`: Similarity matrix heatmap
- `metric_distributions.png`: Distribution plots
- `raid_comparison.png`: RAID benchmark comparison
- `quality_scores.png`: Quality score visualizations

### Short Text Analysis
- `short_text_analysis.json`: Short text analysis results
- `problematic_records.json`: Identified problematic records
- `length_distribution.png`: Text length distribution plots

## Dependencies

### Core Dependencies
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `matplotlib`: Basic plotting
- `seaborn`: Statistical visualizations
- `nltk`: Natural language processing
- `transformers`: Language model access
- `torch`: PyTorch for model inference
- `tqdm`: Progress bars

### Optional Dependencies
- `plotly`: Interactive visualizations
- `jupyter`: Notebook support

## Installation

```bash
# Install core dependencies
pip install numpy pandas matplotlib seaborn nltk transformers torch tqdm

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"

# Optional: Install additional visualization tools
pip install plotly jupyter
```

## Advanced Usage

### Custom Configuration

```python
from data_quality_assurance import DataQualityExaminer
from data_quality_assurance.config import JACCARD_CONFIG, BLEU_CONFIG

# Customize metric settings
JACCARD_CONFIG['n_gram'] = 2  # Use bigrams
BLEU_CONFIG['weights'] = [0.4, 0.3, 0.2, 0.1]  # Custom weights

examiner = DataQualityExaminer()
results = examiner.run_complete_examination()
```

### Batch Processing

```python
# Process multiple datasets
datasets = ['dataset1.json', 'dataset2.json', 'dataset3.json']
results = {}

for dataset in datasets:
    examiner = DataQualityExaminer(data_path=dataset)
    results[dataset] = examiner.run_complete_examination()
```

### Custom Visualizations

```python
from data_quality_assurance import SimilarityVisualizer

visualizer = SimilarityVisualizer()
fig = visualizer.create_similarity_heatmap(
    similarity_matrix=results['similarity_matrix'],
    title="Custom Similarity Analysis"
)
```

## Troubleshooting

### Common Issues

1. **Memory Issues with Large Datasets**
   - Use `sample_size` parameter to limit data
   - Process data in batches
   - Consider using smaller language models

2. **Missing Dependencies**
   - Install all required packages
   - Download NLTK data: `nltk.download('punkt')`
   - Check PyTorch installation for transformers

3. **File Path Issues**
   - Ensure data files exist at specified paths
   - Check file permissions for output directory
   - Use absolute paths when possible

### Performance Optimization

- Use GPU acceleration for perplexity calculations
- Enable batch processing for large datasets
- Consider sampling for initial analysis
- Use parallel processing for independent calculations

## Contributing

When adding new metrics or features:

1. Follow the existing code structure
2. Add comprehensive type hints
3. Include detailed docstrings
4. Add unit tests for new functionality
5. Update configuration files as needed
6. Document new features in this README

## License

This module is part of the PADBen project. See the main project LICENSE for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the demo scripts for examples
3. Examine the configuration files
4. Check the logging output for detailed error messages
