# Sentence Pair Task Preparation Module

This module implements sentence pair classification tasks where models need to determine which sentence in a pair is machine-generated vs human-written. This is particularly useful for zero-shot detectors that use metrics to compare sentences and for prompt-tuned model-based approaches.

## Overview

Unlike single sentence classification, sentence pair tasks present two sentences and ask the model to determine which one is machine-generated. This format is especially useful for:

- **Zero-shot detectors**: Compare metric values between sentences
- **Model-based approaches**: Use prompt-tuning to make comparative judgments
- **Research**: Study how models perform on comparative tasks vs absolute classification

## Task Definitions

### Task 1: Paraphrase Source Attribution without Context (Sentence Pair)
- **Input**: Type3 (human paraphrased) vs Type4 (LLM paraphrased)
- **Goal**: Evaluate detectors' ability to distinguish between human and machine paraphrasing without access to original source text.
- **Research Question**: Can detectors identify the authorship of paraphrased content when the original text is unavailable for comparison? 
- **Output**: One pair per input sample with random sentence order

### Task 2: General Text Authorship Detection (Sentence Pair)
- **Input**: Type1 (human original) vs Type2 (LLM generated)
- **Goal**: Determine which sentence is machine-generated
- **Output**: One pair per input sample with random sentence order

### Task 3: AI Text Laundering Detection (Sentence Pair)
- **Input**: Type4 (LLM paraphrased original) vs Type5-1st (LLM paraphrased generated, 1st iteration)
- **Goal**: Determine which sentence is more machine-generated
- **Output**: One pair per input sample with random sentence order

### Task 4: Iterative Paraphrase Depth Detection (Sentence Pair)
- **Input**: Type5-1st (LLM paraphrased generated, 1st iteration) vs Type5-3rd (LLM paraphrased generated, 3rd iteration)
- **Goal**: Determine which sentence is more deeply paraphrased
- **Output**: One pair per input sample with random sentence order

### Task 5: Original vs Deep Paraphrase Attack Detection (Sentence Pair)
- **Input**: Type1 (human original) vs Type5-3rd (LLM paraphrased generated, 3rd iteration)
- **Goal**: Determine which sentence is machine-generated
- **Output**: One pair per input sample with random sentence order

## Output Format

Each task generates sentence pairs in the following JSON format:

```json
{
  "idx": 0,
  "sentence_pair": ["Sentence 1 text", "Sentence 2 text"],
  "label_pair": [0, 1]
}
```

Where:
- `sentence_pair`: Tuple of two sentences to compare
- `label_pair`: Tuple of labels (0=human, 1=machine) corresponding to each sentence
- The order of sentences is randomized to prevent positional bias

## Usage

### Command Line Interface

```bash
# Process all tasks
python main.py --input data.json --output results/ --all-tasks

# Process specific tasks
python main.py --input data.json --output results/ --task1 --task2

# Process with custom parameters
python main.py --input data.json --output results/ --task1 --seed 123 --min-length 10
```

### Arguments

- `--input`: Path to input JSON file (required)
- `--output`: Output directory for processed tasks (required)
- `--task1` to `--task5`: Enable specific tasks
- `--all-tasks`: Enable all tasks (default if no specific tasks selected)
- `--seed`: Random seed for reproducibility (default: 42)
- `--min-length`: Minimum text length for filtering (default: 5)
- `--log-level`: Logging level (default: INFO)
- `--batch-size`: Batch size for processing (default: 1000)

## Output Structure

```
data/tasks/sentence-pair/
├── task1/
│   ├── task1_paraphrase_source_without_context_sentence_pair.json
│   └── task1_paraphrase_source_without_context_sentence_pair_report.json
├── task2/
│   ├── task2_general_text_authorship_detection_sentence_pair.json
│   └── task2_general_text_authorship_detection_sentence_pair_report.json
├── task3/
│   ├── task3_ai_text_laundering_detection_sentence_pair.json
│   └── task3_ai_text_laundering_detection_sentence_pair_report.json
├── task4/
│   ├── task4_iterative_paraphrase_depth_detection_sentence_pair.json
│   └── task4_iterative_paraphrase_depth_detection_sentence_pair_report.json
├── task5/
│   ├── task5_original_vs_deep_paraphrase_attack_sentence_pair.json
│   └── task5_original_vs_deep_paraphrase_attack_sentence_pair_report.json
└── pipeline_summary.json
```

## Key Features

1. **One Pair Per Sample**: Each input sample generates exactly one sentence pair
2. **Randomized Order**: Sentence order within each pair is randomized to prevent positional bias
3. **Balanced Labels**: Each pair contains both human (0) and machine (1) labels
4. **Comprehensive Reporting**: Detailed statistics and validation reports
5. **Flexible Configuration**: Enable/disable specific tasks as needed
6. **Robust Validation**: Input validation and error handling

## Research Applications

- **Zero-shot Detection**: Compare metric scores between sentence pairs
- **Prompt Engineering**: Design prompts for comparative judgments
- **Model Evaluation**: Test how well models perform on comparative vs absolute tasks
- **Bias Analysis**: Study positional bias in sentence pair tasks
- **Robustness Testing**: Evaluate model performance across different task formulations
