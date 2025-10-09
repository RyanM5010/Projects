# Exhaustive Method

This method implements the **exhaustive** approach for task preparation, where all instances from both relevant text types are used to create the dataset.

## Method Description

For each task (e.g., Task1), this method:
1. Takes **ALL** instances from both relevant text types (e.g., Type3 and Type4 for Task1)
2. Assigns labels based on the text type (e.g., Type3 → Label 0, Type4 → Label 1)
3. Concatenates both sets to create a dataset of **2x the original size**
4. Shuffles the indices to create a seemingly random distribution

## Example for Task1
- Input: 16k Type3 sentences + 16k Type4 sentences
- Output: 32k total sentences (16k with label 0, 16k with label 1)
- Result: Balanced 50-50 distribution

## Advantages
- Uses all available data exhaustively
- Provides maximum training samples
- Maintains semantic diversity
- Traditional ML approach with maximum data utilization

## Potential Concerns
- Semantic repetition (Type3 and Type4 both derive from Type1)
- Models might learn semantic patterns rather than paraphrase source patterns
- Larger dataset size may not always be beneficial

## Usage

```bash
# Run from project root
cd /Users/zhayiwei/Desktop/PADBen
python -m task_preparation.exhaustive_method.main \
    --input data/test/final_generated_data.json \
    --output data/tasks/

# The output will be saved to:
# data/tasks/single-sentence/exhaustive_method/task1/
# data/tasks/single-sentence/exhaustive_method/task2/
# data/tasks/single-sentence/exhaustive_method/task5/
# etc.
```

## Output Structure
```
data/tasks/single-sentence/exhaustive_method/
├── task1/
│   ├── task1_paraphrase_source_without_context.json
│   └── task1_paraphrase_source_without_context_report.json
├── task2/
│   ├── task2_general_text_authorship_detection.json
│   └── task2_general_text_authorship_detection_report.json
├── task3/
├── task4/
├── task5/
│   ├── task5_original_vs_deep_paraphrase_attack.json
│   └── task5_original_vs_deep_paraphrase_attack_report.json
└── pipeline_summary.json
```

## Task5 - NEW!

**Task5: Original vs Deep Paraphrase Attack Detection**
- **Input**: Type1 (human original) vs Type5-3rd (deep paraphrase attack)
- **Purpose**: Detect the most sophisticated paraphrase attack against original text
- **Research Value**: Tests detector performance against the most challenging paraphrase scenario
- **Labels**: Type1 → 0 (Human original), Type5-3rd → 1 (Deep paraphrase attack)

## Why "Exhaustive"?

This method is called "exhaustive" because it:
- **Exhaustively uses** all available instances from both text types
- **Exhaustively explores** the complete dataset without sampling
- **Exhaustively includes** every possible comparison pair
- Provides the most **comprehensive** dataset possible from the available data