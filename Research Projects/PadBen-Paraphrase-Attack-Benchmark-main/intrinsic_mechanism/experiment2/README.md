# Semantic Space Experiment: Iterative Paraphrasing Analysis

This experiment analyzes how paraphrased text deviates from original text in semantic space through multiple iterations of paraphrasing. The goal is to understand the semantic drift that occurs when text undergoes successive paraphrasing operations.

## Experiment Overview

The experiment conducts the following steps:

1. **Data Extraction**: Extracts 100 samples each of type1 (human original text) and type2 (LLM generated text) from the dataset
2. **Iterative Paraphrasing**: Performs 5 iterations of paraphrasing using Qwen2.5-3B-Instruct model
3. **Feature Extraction**: Captures hidden states and embeddings (BGE-M3) at each iteration
4. **PCA Analysis**: Performs Principal Component Analysis on both hidden states and embeddings
5. **Visualization**: Creates comprehensive visualizations of the semantic drift

## File Structure

```
experiment2/
├── main_experiment.py      # Main experiment class and pipeline
├── run_experiment.py       # Simple runner script with CLI interface
├── config.py              # Configuration parameters and settings (no API keys)
├── utils.py               # Utility functions for data processing
├── visualization.py       # Visualization and plotting functions
├── distance_analysis.py   # Distance analysis functions
├── trajectory_analysis.py # Trajectory analysis functions
├── requirements.txt       # Python dependencies
├── README.md             # This documentation
├── more_iteration/       # Extended experiment with 10 iterations
│   ├── extended_experiment.py    # Extended experiment script
│   ├── extended_analysis.py     # Integrated analysis script
│   ├── run_extended_experiment.py # Extended experiment runner
│   └── README.md         # Extended experiment documentation
└── output/               # Experiment results (created during run)
    ├── 1/                # Iteration 1 results
    ├── 2/                # Iteration 2 results
    ├── ...
    ├── 10/               # Iteration 10 results (extended)
    ├── analysis/         # Basic analysis results
    ├── extended_analysis/ # Extended analysis results
    └── visualizations/   # Generated plots and analysis
```

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables** (for extended analysis with original sentence comparison):
   ```bash
   export NOVITA_API_KEY="your_api_key_here"
   export NOVITA_BASE_URL="https://api.novita.ai/openai"
   ```

3. **Verify data path**: Ensure the data file exists at the configured path:
   ```
   /Users/zhayiwei/Desktop/PADBen/data/test/final_generated_data.json
   ```

## Environment Variables

The experiment now uses environment variables for API keys to maintain security:

- `NOVITA_API_KEY`: API key for Novita AI (required for original sentence analysis)
- `NOVITA_BASE_URL`: Base URL for Novita AI API (default: https://api.novita.ai/openai)

**Note**: The basic experiment (5 iterations) does not require API keys. Only the extended analysis with original sentence comparison requires the Novita AI API key.

## Usage

### Experiment Modes

#### 1. Basic Experiment (5 iterations)
- **Location**: Main experiment2 folder
- **Scripts**: `main_experiment.py`, `run_experiment.py`
- **Features**: 5 iterations of paraphrasing, basic analysis
- **Requirements**: No API keys needed
- **Output**: Basic distance analysis and visualizations

#### 2. Extended Experiment (10 iterations)
- **Location**: `more_iteration/` folder
- **Scripts**: `extended_experiment.py`, `extended_analysis.py`
- **Features**: 10 iterations, comprehensive analysis, original sentence comparison
- **Requirements**: API keys for original sentence analysis (optional)
- **Output**: Extended analysis with more detailed insights

### Quick Start

Run the experiment with default settings:
```bash
python run_experiment.py
```

### Advanced Usage

Customize the experiment parameters:
```bash
# Run with fewer samples and iterations (faster)
python run_experiment.py --samples 50 --iterations 3

# Force CPU usage
python run_experiment.py --device cpu

# Custom output directory
python run_experiment.py --output-dir ./my_results

# Dry run to validate configuration
python run_experiment.py --dry-run

# Save logs to file
python run_experiment.py --log-file experiment.log
```

### Direct Python Usage

```python
from main_experiment import SemanticSpaceExperiment

# Create experiment instance
experiment = SemanticSpaceExperiment(
    data_path="/path/to/data.json",
    output_dir="./output",
    max_iterations=5,
    num_samples=100
)

# Run the complete experiment
experiment.run_full_experiment()
```

## Configuration

The experiment can be configured by modifying `config.py`:

### Key Parameters

- **NUM_SAMPLES**: Number of samples per type (default: 100)
- **MAX_ITERATIONS**: Maximum paraphrasing iterations (default: 5)
- **PARAPHRASE_MODEL**: Model for paraphrasing (default: "Qwen/Qwen2.5-3B-Instruct")
- **EMBEDDING_MODEL**: Model for embeddings (default: "BAAI/bge-m3")
- **TEMPERATURE**: Generation temperature (default: 0.7)
- **MAX_NEW_TOKENS**: Maximum tokens to generate (default: 150)

### Hardware Requirements

- **GPU**: Recommended for faster processing (10GB+ VRAM for full models)
- **CPU**: Fallback option (significantly slower)
- **RAM**: 16GB+ recommended for large datasets
- **Storage**: ~1GB for results with default settings

## Output Structure

The experiment generates the following outputs:

### Iteration Results (`output/N/typeX/`)
- `texts.json`: Paraphrased texts for each iteration
- `hidden_states.npy`: Model hidden states (numpy array)
- `embeddings.npy`: Text embeddings (numpy array)

### Visualizations (`output/visualizations/`)
- `pca_comprehensive.png`: Main PCA scatter plots
- `trajectory_analysis.png`: Semantic trajectory analysis
- `variance_analysis.png`: PCA explained variance
- `distance_analysis.png`: Inter-iteration distance analysis
- `pca_results.npz`: Raw PCA results (numpy format)
- `summary_report.txt`: Experiment summary and statistics

## Experiment Results

The experiment produces several types of analysis:

### 1. PCA Visualization
- 2D scatter plots showing how samples cluster in semantic space
- Separate plots for hidden states and embeddings
- Different colors for each iteration (1-5)
- Separate analysis for type1 and type2 texts

### 2. Trajectory Analysis
- Shows how individual samples move through semantic space across iterations
- Identifies patterns of semantic drift
- Highlights iteration centroids

### 3. Variance Analysis
- Explains how much variance is captured by PC1 and PC2
- Helps understand the dimensionality of semantic changes

### 4. Distance Analysis
- Measures semantic distance between consecutive iterations
- Shows trends in semantic drift over time
- Identifies convergence or divergence patterns

## Technical Details

### Models Used

1. **Paraphrasing Model**: Qwen2.5-3B-Instruct
   - Generates paraphrases while preserving meaning
   - Provides hidden states for analysis
   - Compatible with 10GB VRAM constraint

2. **Embedding Model**: BGE-M3
   - Generates semantic embeddings
   - Multilingual and high-quality representations
   - Used for semantic space analysis

### Analysis Methods

1. **PCA (Principal Component Analysis)**:
   - Reduces high-dimensional features to 2D
   - Standardizes features before analysis
   - Preserves maximum variance in lower dimensions

2. **Hidden States Extraction**:
   - Uses last layer hidden states from paraphrasing model
   - Mean pooling across sequence length
   - Captures model's internal representation

3. **Embedding Extraction**:
   - Uses BGE-M3 for semantic embeddings
   - Mean pooling over token embeddings
   - Captures semantic meaning

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```bash
   python run_experiment.py --device cpu
   ```

2. **Model Download Errors**:
   - Ensure internet connection
   - Check Hugging Face access
   - Try smaller models in config.py

3. **Data File Not Found**:
   - Verify data path in config.py
   - Use `--data-path` argument to specify custom path

4. **Insufficient Disk Space**:
   - Use fewer samples: `--samples 50`
   - Use fewer iterations: `--iterations 3`

### Performance Optimization

1. **Reduce Memory Usage**:
   - Lower batch sizes in utils.py
   - Use CPU instead of GPU
   - Reduce number of samples

2. **Speed Up Processing**:
   - Use GPU if available
   - Reduce max_new_tokens in config.py
   - Use smaller models

## Extending the Experiment

### Adding New Models

1. Update `config.py` with new model names
2. Ensure compatibility with existing interfaces
3. Test with small samples first

### Custom Visualizations

1. Extend `SemanticSpaceVisualizer` class
2. Add new plotting methods
3. Call from `main_experiment.py`

### Different Analysis Methods

1. Add new analysis functions to `utils.py`
2. Integrate with main experiment pipeline
3. Update visualization accordingly

## Citation

If you use this experiment in your research, please cite:

```bibtex
@misc{semantic_space_experiment_2025,
  title={Semantic Space Analysis of Iterative Paraphrasing},
  author={PADBen Research Team},
  year={2025},
  howpublished={\\url{https://github.com/your-repo/PADBen}}
}
```

## License

This project is part of the PADBen benchmark suite. Please refer to the main project license for usage terms.
