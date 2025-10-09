# Sampling Method

This method implements the **sampling** approach for task preparation, where only one instance is randomly sampled from the available text types for each original sample.

## Method Description

For each task (e.g., Task1), this method:
1. For each original sample (Type1), randomly **samples** either Type3 **or** Type4 (not both)
2. Assigns labels based on the sampled text type
3. Creates a dataset of the **same size as the original** (e.g., 16k samples)
4. Allows dynamic control of label distribution ratios through sampling probabilities

## Example for Task1
- Input: 16k original samples, each having both Type3 and Type4
- Process: For each sample, randomly sample Type3 OR Type4 based on target ratio
- Output: 16k total sentences with configurable label distribution

## Label Distribution Options

### 30-70 Distribution (30% Label 1, 70% Label 0)
- Sampling probability: 30% chance to sample Type4, 70% chance to sample Type3
- Result: Fewer LLM-paraphrased sentences, more human-paraphrased sentences

### 50-50 Distribution (Balanced)
- Sampling probability: 50% chance for each type
- Result: Equal number of Type3 and Type4 sentences

### 80-20 Distribution (80% Label 1, 20% Label 0)
- Sampling probability: 80% chance to sample Type4, 20% chance to sample Type3
- Result: More LLM-paraphrased sentences, fewer human-paraphrased sentences

## Advantages
- Eliminates semantic repetition concerns through sampling
- Same dataset size as original
- Allows testing different label distributions via sampling ratios
- More realistic evaluation scenario
- True statistical sampling approach

## Research Value
Tests whether different sampling distributions affect:
- Model performance
- Detector effectiveness
- Evaluation robustness

## Usage

```bash
# Run from project root
cd /Users/zhayiwei/Desktop/PADBen

# 30% Label 1, 70% Label 0 (sample Type4 with 30% probability)
python -m task_preparation.sampling_method.main \
    --input data/test/final_generated_data.json \
    --output data/tasks/ \
    --label-1-ratio 0.3

# Balanced sampling (50-50)
python -m task_preparation.sampling_method.main \
    --input data/test/final_generated_data.json \
    --output data/tasks/ \
    --label-1-ratio 0.5

# 80% Label 1, 20% Label 0 (sample Type4 with 80% probability)
python -m task_preparation.sampling_method.main \
    --input data/test/final_generated_data.json \
    --output data/tasks/ \
    --label-1-ratio 0.8
```

## Output Structure
```
data/tasks/single-sentence/sampling_method/
├── 30-70/
│   ├── task1/
│   ├── task2/
│   ├── task3/
│   ├── task4/
│   └── pipeline_summary.json
├── 50-50/
│   ├── task1/
│   ├── task2/
│   ├── task3/
│   ├── task4/
│   └── pipeline_summary.json
└── 80-20/
    ├── task1/
    ├── task2/
    ├── task3/
    ├── task4/
    ├── task5/
    └── pipeline_summary.json
```

## Task5 - NEW!

**Task5: Original vs Deep Paraphrase Attack Detection**
- **Input**: Type1 (human original) vs Type5-3rd (deep paraphrase attack)
- **Purpose**: Detect the most sophisticated paraphrase attack against original text
- **Research Value**: Tests detector performance against the most challenging paraphrase scenario
- **Labels**: Type1 → 0 (Human original), Type5-3rd → 1 (Deep paraphrase attack)
- **Sampling**: Randomly samples either Type1 or Type5-3rd based on label_1_ratio

## Why "Sampling"?

This method is called "sampling" because it:
- **Randomly samples** one instance from available options for each original
- Uses **statistical sampling** to control label distributions
- Implements **stratified sampling** to achieve target label ratios
- Creates a **sampled subset** rather than using all available data
- Applies **sampling probabilities** to determine text type selection