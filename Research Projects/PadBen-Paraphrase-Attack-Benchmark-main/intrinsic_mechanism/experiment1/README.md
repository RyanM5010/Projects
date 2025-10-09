# Semantic vs Paraphrase Experiment

This experiment explores whether semantic equivalence equals paraphrase in LLM world by comparing Type2 (LLM generated) and Type4 (LLM paraphrased) texts using BGE-m3 embeddings.

## Experimental Design

### Research Questions
- Do Type2 and Type4 form distinct clusters in semantic space?
- Are Type4 texts closer to Type1 (semantic preservation) or Type2 (generation artifact similarity)?
- What does this tell us about paraphrase attacks and detection?

### Text Types
- **Type1**: Human original sentences
- **Type2**: LLM generated sentences (using sentence completion method)
- **Type4**: LLM paraphrased sentences (based on human original sentence)

### Experimental Validity & Significance
- If Type2 ≈ Type4 semantically, then paraphrase attacks exploit the same semantic space as original generation
- If Type2 ≠ Type4, paraphrases create a distinct "attack space" that detectors must handle separately
- Understanding semantic relationships helps explain why certain detectors fail against paraphrases

## Methodology

### 1. Distance Analysis
Calculate average distances between text types:
- Type1 vs Type2
- Type1 vs Type4  
- Type2 vs Type4 (key comparison)

Distance metrics:
- Cosine similarity: `1 - cosine(emb1, emb2)`
- Euclidean distance: `euclidean_distance(emb1, emb2)`
- Manhattan distance: `manhattan_distance(emb1, emb2)`

### 2. Semantic Space Exploration
- Project all text types into shared embedding space
- Dimensionality reduction: PCA (Principal Component Analysis)
- Clustering analysis: KMeans with 3 clusters
- Visualize semantic relationships

## Setup

### Prerequisites
```bash
pip install -r requirements.txt
```

### Environment Variables
```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.probex.top/v1"
```

## Usage

### Run Main Experiment
```bash
python main_experiment.py
```

This will:
1. Load data from `data/test/final_generated_data.json` (limited to 3000 samples per type)
2. Generate BGE-m3 embeddings for Type1, Type2, and Type4 texts
3. Calculate distance metrics between all type pairs
4. Perform semantic space exploration with UMAP and t-SNE
5. Run KMeans clustering analysis
6. Save results to `results/` directory

### Generate Visualizations
```bash
python visualization.py
```

This will create:
- Distance comparison plots
- UMAP and t-SNE semantic space visualizations
- Cluster analysis plots
- Comprehensive summary report

## Results Structure

```
results/
├── distance_results.json          # Distance metrics between type pairs
├── exploration_summary.json       # Summary of semantic space exploration
├── umap_2d.npy                    # UMAP 2D coordinates
├── tsne_2d.npy                    # t-SNE 2D coordinates
├── kmeans_labels.npy             # KMeans cluster labels
├── labels.npy                     # Text type labels
├── distance_comparison.png        # Distance comparison visualization
├── umap_semantic_space.png        # UMAP semantic space plot
├── tsne_semantic_space.png         # t-SNE semantic space plot
├── cluster_analysis.png           # Cluster analysis plots
└── summary_report.png             # Comprehensive summary
```

## Key Metrics

### Distance Analysis
- **Cosine Distance**: Measures angular similarity (0 = identical, 1 = orthogonal)
- **Euclidean Distance**: Measures straight-line distance in embedding space
- **Manhattan Distance**: Measures city-block distance in embedding space

### Clustering Analysis
- **Cluster Purity**: Ratio of dominant type within each cluster
- **Cluster Distribution**: Number of samples per cluster
- **Type Distribution**: Number of samples per text type

## Interpretation Guide

### Distance Results
- **Low distance** between Type2 and Type4 → Similar semantic space
- **High distance** between Type2 and Type4 → Distinct semantic spaces
- **Type4 closer to Type1** → Paraphrases preserve original semantics
- **Type4 closer to Type2** → Paraphrases inherit generation artifacts

### Clustering Results
- **Separate clusters** for Type2 and Type4 → Distinct semantic spaces
- **Mixed clusters** → Overlapping semantic spaces
- **Type4 clustering with Type1** → Semantic preservation
- **Type4 clustering with Type2** → Generation artifact similarity

## Expected Outcomes

### Hypothesis 1: Type2 ≈ Type4
If paraphrases and generated text occupy similar semantic space:
- Low distance between Type2 and Type4
- Mixed clustering of Type2 and Type4
- Paraphrase attacks exploit same vulnerabilities as generation

### Hypothesis 2: Type2 ≠ Type4  
If paraphrases create distinct semantic space:
- High distance between Type2 and Type4
- Separate clusters for Type2 and Type4
- Paraphrase attacks require separate detection strategies

## Troubleshooting

### Common Issues
1. **API Rate Limits**: The script processes embeddings in batches to avoid rate limits
2. **Memory Issues**: Large datasets may require chunked processing
3. **Visualization Errors**: Ensure all result files are present before running visualization

### Performance Tips
- The experiment is configured to use 3000 samples per type for manageable processing
- Use smaller sample sizes for initial testing by modifying `max_samples` parameter
- Adjust batch sizes based on API limits
- Monitor memory usage for large datasets

## Contributing

When extending this experiment:
1. Follow the existing code structure
2. Add comprehensive docstrings
3. Include type hints for all functions
4. Test with smaller datasets first
5. Update this README with new features

## License

This experiment is part of the PADBen project. Please refer to the main project license for usage terms.
