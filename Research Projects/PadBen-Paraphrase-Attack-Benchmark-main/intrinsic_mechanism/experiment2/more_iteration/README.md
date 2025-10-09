# Extended Semantic Space Experiment - More Iterations

This folder contains scripts for running extended semantic space experiments with more iterations (up to 10) to better observe semantic drift trends over multiple paraphrasing cycles.

## 📁 File Overview

### Core Experiment Files

#### 1. `extended_experiment.py`
**Purpose**: Main experiment script that runs extended semantic space experiments with more iterations.

**What it does**:
- Extracts samples from the dataset (type1 and type2 sentences)
- Performs multiple iterations of paraphrasing using Qwen3-4B model
- Captures hidden states and embeddings at each iteration
- Saves results in organized directory structure

**Input**:
- Dataset file (JSON format with type1 and type2 sentences)
- Configuration parameters (number of iterations, samples, device)
- Model configurations (Qwen3-4B for paraphrasing, BGE-M3 for embeddings)

**Output**:
```
output/
├── 1/
│   ├── type1/
│   │   ├── texts.json
│   │   ├── hidden_states.npy
│   │   └── embeddings.npy
│   └── type2/
│       ├── texts.json
│       ├── hidden_states.npy
│       └── embeddings.npy
├── 2/
│   └── ...
└── 10/
    └── ...
```

**Key Features**:
- Supports 1-10 iterations of paraphrasing
- Uses Qwen3-4B for text generation
- Uses BGE-M3 for embedding computation
- Memory-efficient processing with progress tracking
- Automatic device detection (CUDA/CPU)

#### 2. `run_extended_experiment.py`
**Purpose**: Command-line interface for running the extended experiment.

**What it does**:
- Provides easy-to-use command-line interface
- Handles argument parsing and validation
- Sets up logging and memory monitoring
- Runs the extended experiment with specified parameters

**Input**:
- Command-line arguments:
  - `--iterations`: Number of iterations (default: 10)
  - `--samples`: Number of samples per type (default: 100)
  - `--device`: Device to use (auto/cuda/cpu)
  - `--data-path`: Path to dataset file
  - `--output-dir`: Output directory

**Output**:
- Console logs with progress information
- Experiment results in organized directory structure
- Memory usage reports

**Usage Examples**:
```bash
# Run with default settings (10 iterations, 100 samples)
python run_extended_experiment.py

# Run with custom parameters
python run_extended_experiment.py --iterations 15 --samples 50 --device cuda

# Run with specific data path
python run_extended_experiment.py --data-path ../../data/data.json
```

### Analysis Files

#### 3. `extended_analysis.py` (Integrated)
**Purpose**: Comprehensive integrated analysis of extended experiment results with up to 10 iterations.

**What it does**:
- Performs pairwise distance analysis between iterations
- Creates distance trend visualizations
- Analyzes centroid trajectories over iterations
- Generates statistical summaries
- Compares type1 vs type2 baseline trajectories
- **NEW**: Compares type1 iterations against original sentences (integrated from type1_vs_original_type2_analysis.py)
- **NEW**: Provides baseline comparisons using original sentences before paraphrasing

**Input**:
- Experiment output directory containing iteration results
- Configuration for maximum iterations to analyze
- **Optional**: Original dataset file for original sentence comparison
- **Optional**: Number of samples for original sentence analysis

**Output**:
```
extended_analysis/
├── extended_distance_trends_10iter.png
├── extended_centroid_trajectories_10iter.png
├── extended_hidden_states_distance_table_10iter.csv
├── extended_embeddings_distance_table_10iter.csv
├── extended_distance_tables_10iter.json
├── original_sentence_comparison_10iter.png (if data-path provided)
├── type1_vs_original_type1_distances_10iter.csv (if data-path provided)
├── type1_vs_original_type2_distances_10iter.csv (if data-path provided)
├── original_sentence_comparison_10iter.csv (if data-path provided)
├── original_sentence_comparison_10iter.json (if data-path provided)
├── original_type1_embeddings.npy (cached, if data-path provided)
├── original_type2_embeddings.npy (cached, if data-path provided)
└── integrated_analysis_summary_10iter.txt
```

**Key Features**:
- Pairwise distance analysis (cosine, euclidean, manhattan)
- Cross-type comparison (type1 iterations vs type2 baseline)
- Centroid trajectory visualization with PCA
- Comprehensive statistical analysis
- Trend visualization across iterations
- **NEW**: Original sentence comparison analysis
- **NEW**: Baseline comparison using original sentences
- **NEW**: Cached embeddings for efficiency
- **NEW**: Integrated reporting and visualization

**Detailed Functionality**:
- **Distance Metrics**: Computes cosine, euclidean, and manhattan distances between embeddings
- **Iteration Analysis**: Tracks semantic drift from iteration 1 to iterations 2-10
- **Cross-Type Comparison**: Compares type1 iterations against type2 baseline (iteration 1)
- **Original Sentence Baseline**: When data-path is provided, compares paraphrased iterations against original sentences
- **Embedding Caching**: Saves computed embeddings to avoid recomputation
- **Visualization**: Creates comprehensive plots showing distance trends and centroid trajectories
- **Statistical Reporting**: Generates detailed summary statistics and CSV/JSON outputs
- **Error Handling**: Robust error handling with progress tracking and logging

## 🚀 Quick Start

### 1. Run Extended Experiment
```bash
# Run with default settings (10 iterations, 100 samples)
python run_extended_experiment.py

# Run with custom parameters
python run_extended_experiment.py --iterations 10 --samples 100 --device cuda
```

### 2. Run Integrated Extended Analysis
```bash
# Analyze results with up to 10 iterations (basic analysis)
python extended_analysis.py --iterations 10 --output-dir ../output

# Analyze with original sentence comparison
python extended_analysis.py --iterations 10 --output-dir ../output --data-path ../../data/data.json --samples 100
```

#### Command Line Options for `extended_analysis.py`:
- `--iterations`: Number of iterations to analyze (default: 10)
- `--output-dir`: Output directory containing experiment data (default: ../output)
- `--data-path`: Path to original dataset for original sentence comparison (optional)
- `--samples`: Number of samples for original sentence analysis (default: 100)

#### Analysis Modes:
1. **Basic Analysis Mode** (without `--data-path`):
   - Performs pairwise distance analysis between iterations
   - Creates distance trend visualizations
   - Analyzes centroid trajectories over iterations
   - Generates statistical summaries
   - Compares type1 vs type2 baseline trajectories

2. **Full Analysis Mode** (with `--data-path`):
   - All features from Basic Analysis Mode
   - **PLUS**: Original sentence comparison analysis
   - **PLUS**: Baseline comparisons using original sentences before paraphrasing
   - **PLUS**: Cached embeddings for efficiency
   - **PLUS**: Comprehensive integrated reporting

## 📊 Output Structure

### Experiment Results
```
output/
├── 1/                    # Iteration 1 (baseline)
├── 2/                    # Iteration 2
├── ...
├── 10/                   # Iteration 10
│   ├── type1/
│   │   ├── texts.json           # Paraphrased texts
│   │   ├── hidden_states.npy    # Qwen hidden states
│   │   └── embeddings.npy       # BGE-M3 embeddings
│   └── type2/
│       ├── texts.json
│       ├── hidden_states.npy
│       └── embeddings.npy
└── extended_analysis/     # Analysis results
    ├── extended_distance_trends_10iter.png
    ├── extended_centroid_trajectories_10iter.png
    ├── extended_hidden_states_distance_table_10iter.csv
    ├── extended_embeddings_distance_table_10iter.csv
    └── extended_distance_tables_10iter.json
```

### Analysis Results

#### Basic Analysis Outputs (always generated):
- **`extended_distance_trends_{iterations}iter.png`**: Line plots showing semantic drift over iterations
- **`extended_centroid_trajectories_{iterations}iter.png`**: PCA visualization of semantic space movement
- **`extended_hidden_states_distance_table_{iterations}iter.csv`**: CSV with hidden states distance metrics
- **`extended_embeddings_distance_table_{iterations}iter.csv`**: CSV with embedding distance metrics
- **`extended_distance_tables_{iterations}iter.json`**: JSON with all distance analysis results
- **`integrated_analysis_summary_{iterations}iter.txt`**: Comprehensive statistical summary

#### Additional Outputs (when `--data-path` provided):
- **`original_sentence_comparison_{iterations}iter.png`**: Plots comparing iterations vs original sentences
- **`type1_vs_original_type1_distances_{iterations}iter.csv`**: Type1 iterations vs original type1 distances
- **`type1_vs_original_type2_distances_{iterations}iter.csv`**: Type1 iterations vs original type2 distances
- **`original_sentence_comparison_{iterations}iter.csv`**: Combined original sentence comparison data
- **`original_sentence_comparison_{iterations}iter.json`**: JSON with original sentence analysis results
- **`original_type1_embeddings.npy`**: Cached original type1 embeddings
- **`original_type2_embeddings.npy`**: Cached original type2 embeddings

#### Output Interpretation:
- **Distance Trends**: Lower values indicate more similar semantics, higher values indicate semantic drift
- **Centroid Trajectories**: Show how the semantic center of each text type moves through embedding space
- **Cross-Type Comparison**: Reveals convergence/divergence between type1 and type2 over iterations
- **Original Sentence Comparison**: Provides baseline for measuring semantic drift from original meaning

## 🔧 Configuration

### Environment Variables
- `NOVITA_API_KEY`: API key for Novita AI (BGE-M3 embeddings)
- `NOVITA_BASE_URL`: Base URL for Novita AI API
- `CUDA_VISIBLE_DEVICES`: GPU device selection

### Key Parameters
- **Iterations**: Number of paraphrasing cycles (1-10)
- **Samples**: Number of samples per text type (default: 100)
- **Device**: Computing device (auto/cuda/cpu)
- **Model**: Qwen3-4B for paraphrasing, BGE-M3 for embeddings

## 📈 Analysis Types

### 1. Pairwise Distance Analysis
- Compares each iteration against iteration 1 (baseline)
- Uses cosine, euclidean, and manhattan distances
- Tracks semantic drift over iterations

### 2. Cross-Type Comparison
- Compares type1 iterations against type2 baseline
- Analyzes semantic space convergence/divergence
- Provides insights into type-specific drift patterns

### 3. Original Sentence Comparison
- Compares paraphrased iterations against original sentences
- Uses BGE-M3 embeddings for consistent comparison
- Provides baseline for semantic drift measurement

### 4. Centroid Trajectory Analysis
- Tracks movement of semantic centroids over iterations
- Uses PCA for dimensionality reduction
- Visualizes semantic space evolution

## 🎯 Use Cases

### Primary Use Cases:
1. **Semantic Drift Analysis**: Study how paraphrasing affects semantic meaning over multiple iterations
2. **Model Comparison**: Compare different paraphrasing models and their semantic stability
3. **Iteration Optimization**: Determine optimal number of paraphrasing cycles for specific tasks
4. **Quality Assessment**: Evaluate paraphrasing quality and semantic preservation over iterations
5. **Baseline Comparison**: Compare paraphrased iterations against original sentences to measure drift

### Specific Analysis Scenarios:

#### Scenario 1: Basic Semantic Drift Study
```bash
# Analyze semantic drift without original sentence comparison
python extended_analysis.py --iterations 10 --output-dir ../output
```
**Use Case**: Understanding how semantic meaning changes through paraphrasing iterations
**Outputs**: Distance trends, centroid trajectories, statistical summaries

#### Scenario 2: Original Sentence Baseline Analysis
```bash
# Full analysis with original sentence comparison
python extended_analysis.py --iterations 10 --output-dir ../output --data-path ../../data/data.json --samples 100
```
**Use Case**: Measuring semantic drift from original meaning using original sentences as baseline
**Outputs**: All basic outputs + original sentence comparison plots and data

#### Scenario 3: Cross-Type Semantic Convergence
```bash
# Analyze how type1 and type2 converge/diverge over iterations
python extended_analysis.py --iterations 10 --output-dir ../output
```
**Use Case**: Understanding semantic space convergence between different text types
**Outputs**: Cross-type comparison plots and distance metrics

#### Scenario 4: Iteration Range Analysis
```bash
# Analyze specific iteration range
python extended_analysis.py --iterations 5 --output-dir ../output
```
**Use Case**: Focused analysis on specific iteration ranges (e.g., early vs late iterations)
**Outputs**: Tailored analysis for specified iteration range

## 📝 Notes

- All scripts support both 5 and 10 iteration analyses
- Results are cached for efficiency (embeddings, analysis results)
- Memory usage is monitored and logged
- Error handling ensures robust execution
- Progress tracking provides real-time feedback

## 🔍 Troubleshooting

### Common Issues

#### General Issues:
1. **Memory Issues**: Reduce sample size or use CPU
2. **API Rate Limits**: Add delays between API calls
3. **File Not Found**: Check data paths and output directories
4. **CUDA Issues**: Use `--device cpu` for CPU-only execution

#### Specific to `extended_analysis.py`:
1. **Missing Iteration Data**: Ensure all required iteration folders (1-10) exist in output directory
2. **API Key Issues**: Set `NOVITA_API_KEY` environment variable for original sentence analysis
3. **Data Path Issues**: Ensure data path points to valid JSON file with type1/type2 structure
4. **Embedding Cache Issues**: Delete cached embedding files if corrupted (`.npy` files in analysis directory)

### Performance Tips

#### General Performance:
- Use GPU when available for faster processing
- Cache embeddings to avoid recomputation
- Monitor memory usage for large datasets
- Use smaller sample sizes for testing

#### Analysis-Specific Tips:
- **Basic Analysis**: Fast execution, no API calls required
- **Full Analysis**: Requires API calls for original sentence embeddings; consider caching
- **Large Datasets**: Use smaller sample sizes (e.g., `--samples 50`) for initial testing
- **Memory Optimization**: Process iterations in smaller batches if memory is limited
- **API Optimization**: Original sentence embeddings are cached; subsequent runs are faster

### Expected Execution Times:
- **Basic Analysis (10 iterations)**: 2-5 minutes
- **Full Analysis (10 iterations, 100 samples)**: 10-20 minutes (first run), 2-5 minutes (cached)
- **Large Dataset (10 iterations, 500 samples)**: 30-60 minutes (first run), 5-10 minutes (cached)