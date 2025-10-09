# 🤖 Model-Based AI Text Detector

A comprehensive evaluation framework for assessing Large Language Models (LLMs) on AI-generated text detection tasks using sophisticated multi-turn conversation prompts with few-shot learning.

## 📋 Overview

This project provides a robust system for evaluating multiple state-of-the-art LLMs on various AI text detection challenges. It uses task-specific expert personas and few-shot examples to guide models in distinguishing between human-written and AI-generated content.

### Key Features

- ✨ **Multi-turn Conversation Format**: Sophisticated prompting with system messages, few-shot examples, and structured queries
- 🎯 **Task-Specific Prompts**: Customized expert personas and examples for 5 different detection tasks
- 📦 **Batch Processing**: Efficient evaluation with configurable batch sizes
- 📊 **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, AUROC, and TPR@FPR
- 💾 **Robust Saving**: Automatic result saving after each task completion
- 🔄 **Retry Logic**: Intelligent error handling and API retry mechanisms

## 🎯 Supported Models

The framework currently supports the following models:

- **Claude-3.5-Haiku** - Anthropic's efficient model
- **DeepSeek-V2.5** - Advanced reasoning model
- **GLM-4.5** - Zhipu AI's latest model
- **Kimi-K2-Instruct** - Moonshot's instruction-tuned model
- **Qwen2.5-VL-72B** - Alibaba's vision-language model
- **Qwen3-32B** - Latest Qwen series model
- **GPT-OSS-120B** - Open-source GPT variant
- **Gemma-3-27B** - Google's efficient model
- **Llama-4-Scout-17B** - Meta's scout model
- **Mistral-Nemo** - Mistral AI's compact model
- **Llama-4-Maverick-17B** - Meta's maverick variant
- **Llama-3.3-70B-Instruct** - Meta's instruction-tuned model
- **WizardLM-2-8x22B** - Microsoft's mixture-of-experts model

## 📊 Evaluation Tasks

### Task Descriptions

| Task | Name | Description |
|------|------|-------------|
| **Task 1** | Paraphrase Source Attribution | Distinguish between human and LLM paraphrases |
| **Task 2** | General Text Authorship Detection | Distinguish human original from LLM-generated text |
| **Task 3** | AI Text Laundering Detection | Detect different levels of LLM processing |
| **Task 4** | Iterative Paraphrase Depth Detection | Distinguish 1st vs 3rd iteration paraphrases |
| **Task 5** | Deep Paraphrase Attack Detection | Detect sophisticated AI paraphrase attacks |

### Task Formats

- **Single-Sentence Tasks**: Classify individual texts (0 = Human, 1 = AI)
- **Sentence-Pair Tasks**: Compare two sentences and determine which is more AI-like
- **Sampling Ratios**: Multiple data distributions (20-80, 50-50, 70-30 human-AI ratios)

### Evaluation Metrics

- **Basic Metrics**: Accuracy, Precision, Recall, F1-Score
- **Advanced Metrics**: 
  - AUROC (Area Under ROC Curve)
  - TPR@1%FPR, TPR@5%FPR, TPR@10%FPR (True Positive Rate at specific False Positive Rates)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd Model_Based_Detector

# Install dependencies
pip install openai scikit-learn pandas numpy python-dotenv tqdm
```

### 2. Configuration

Create a `.env` file in the project directory:

```bash
# .env file configuration
OPENAI_API_KEY=INSERT_YOUR_API_KEY_HERE
BASE_URL=INSERT_YOUR_BASE_URL_HERE
```

**Note**: Replace the placeholders with your actual API credentials.

### 3. Run Evaluation

#### Quick Test Mode (2 samples per model)
```bash
python prompt_instucted_models.py --test-mode
```

#### Full Evaluation
```bash
# Evaluate all models on all tasks with 100 samples each
python prompt_instucted_models.py --samples 100

# Test specific model
python prompt_instucted_models.py --model "Claude-3.5-Haiku" --samples 50

# Custom batch size and output directory
python prompt_instucted_models.py --samples 200 --batch-size 20 --output-dir "results"

# Evaluate specific task
python prompt_instucted_models.py --task "task1_sampling_50_50" --samples 100
```

## 📖 Command Line Arguments

```bash
python prompt_instucted_models.py [OPTIONS]

Options:
  --model MODEL            Model to test (default: all models)
                          Choices: Claude-3.5-Haiku, DeepSeek-V2.5, GLM-4.5,
                                  Kimi-K2-Instruct, Qwen2.5-VL-72B, Qwen3-32B,
                                  GPT-OSS-120B, Gemma-3-27B, Llama-4-Scout-17B,
                                  Mistral-Nemo, Llama-4-Maverick-17B, 
                                  Llama-3.3-70B-Instruct, WizardLM-2-8x22B, all
  
  --samples N              Number of samples to test (default: all)
  --task TASK              Specific task to evaluate (e.g., 'task1_sampling_50_50')
  --batch-size N           Batch size for processing (default: 10)
  --output-dir DIR         Output directory for results (default: output)
  --test-mode              Quick test with 2 samples per model
  --api-key KEY            API key (overrides .env file)
  --base-url URL           Base URL for API (overrides .env file)
  --verbose                Enable verbose output
  --help                   Show help message
```

## 🎨 Prompt Engineering

### Multi-Turn Conversation Structure

Each evaluation uses a sophisticated multi-turn conversation format:

```python
[
    {
        "role": "system",
        "content": "You are an expert text analyst specializing in [task]..."
    },
    {
        "role": "user",
        "content": "Please analyze this text: [example 1]"
    },
    {
        "role": "assistant",
        "content": "1"  # AI-generated
    },
    {
        "role": "user",
        "content": "Please analyze this text: [example 2]"
    },
    {
        "role": "assistant",
        "content": "0"  # Human-written
    },
    # ... more few-shot examples ...
    {
        "role": "user",
        "content": "Please analyze this text: [actual text to classify]"
    }
]
```

### Task-Specific Expert Personas

- **Task 1**: Expert text analyst specializing in paraphrase detection
- **Task 2**: Expert in AI-generated text detection
- **Task 3**: Specialist in detecting AI text laundering techniques
- **Task 4**: Expert in detecting iterative AI processing depth
- **Task 5**: Cybersecurity expert specializing in sophisticated AI attacks

### Few-Shot Learning

Each task includes 3-4 carefully crafted few-shot examples that demonstrate:
- Clear classification criteria
- Edge cases and difficult examples
- Balanced representation of both classes
- Task-specific linguistic patterns

## 📁 Output Structure

### Directory Organization

```
output/
├── Claude-3.5-Haiku/
│   ├── labels/
│   │   ├── task1_sampling_20_80_labels.csv
│   │   ├── task1_sampling_50_50_labels.csv
│   │   ├── task1_sampling_70_30_labels.csv
│   │   ├── task1_sentence_pair_labels.csv
│   │   └── ... (all tasks)
│   ├── task1_sampling_20_80_predictions.csv
│   ├── task1_sampling_20_80_summary.csv
│   ├── task1_sampling_50_50_predictions.csv
│   ├── task1_sampling_50_50_summary.csv
│   └── ... (all tasks)
├── DeepSeek-V2.5/
│   └── ... (same structure)
├── ... (other models)
├── overall_summary.csv
└── consolidated_labels.csv
```

### Output Files

#### 1. Predictions Files (`{task_name}_predictions.csv`)
Detailed predictions with original text:
```csv
model,task,idx,text,true_label,predicted_label,correct
Claude-3.5-Haiku,task1_sampling_50_50,0,"Sample text...",0,0,True
```

#### 2. Summary Files (`{task_name}_summary.csv`)
Performance metrics for each task:
```csv
model,task,accuracy,precision,recall,f1,auroc,tpr_1_fpr,tpr_5_fpr,tpr_10_fpr,total_samples,errors,payment_errors
Claude-3.5-Haiku,task1_sampling_50_50,0.85,0.84,0.86,0.85,0.87,0.42,0.68,0.75,100,0,0
```

#### 3. Labels Files (`labels/{task_name}_labels.csv`)
Clean prediction labels for analysis:
```csv
task,model,sample_idx,true_label,predicted_label,correct
task1_sampling_50_50,Claude-3.5-Haiku,0,0,0,True
```

#### 4. Overall Summary (`overall_summary.csv`)
Cross-model, cross-task performance comparison

#### 5. Consolidated Labels (`consolidated_labels.csv`)
All prediction labels from all models and tasks in one file

## 🔧 Programmatic Usage

### Python API

```python
from prompt_instucted_models import main_openai_evaluation, test_all_models_retry_logic

# Full evaluation
all_results, all_dfs = main_openai_evaluation(
    api_key="INSERT_YOUR_API_KEY_HERE",  # Optional, uses .env if not provided
    base_url="INSERT_YOUR_BASE_URL_HERE",  # Optional, uses .env if not provided
    max_samples=100,
    output_dir="my_results"
)

# Quick test with 2 samples
success = test_all_models_retry_logic(
    api_key="INSERT_YOUR_API_KEY_HERE",
    base_url="INSERT_YOUR_BASE_URL_HERE",
    max_samples=2
)
```

### Custom Analysis

```python
import pandas as pd

# Load consolidated results
labels_df = pd.read_csv('output/consolidated_labels.csv')
summary_df = pd.read_csv('output/overall_summary.csv')

# Compare models on specific task
task1_results = labels_df[labels_df['task'] == 'task1_sampling_50_50']
model_accuracy = task1_results.groupby('model')['correct'].mean()
print(model_accuracy.sort_values(ascending=False))

# Find hardest examples
predictions_df = pd.read_csv('output/Claude-3.5-Haiku/task1_sampling_50_50_predictions.csv')
errors = predictions_df[~predictions_df['correct']]
print(f"Error rate: {len(errors) / len(predictions_df):.2%}")

# Analyze confusion patterns
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_df['true_label'], labels_df['predicted_label'])
print("Confusion Matrix:\n", cm)
```

## 📊 Key Features

### Robust Evaluation Pipeline

- **✅ Intermediate Saving**: Results automatically saved after each task
- **🔄 Retry Logic**: Intelligent retry for API failures and rate limits
- **📈 Progress Tracking**: Real-time progress bars with detailed status
- **⚡ Batch Processing**: Process multiple samples simultaneously
- **🛡️ Error Handling**: Graceful handling of payment errors and API issues

### Comprehensive Metrics

- **Standard ML Metrics**: Accuracy, Precision, Recall, F1-Score
- **Advanced Metrics**: AUROC, TPR@FPR (critical for security applications)
- **Confusion Matrices**: Detailed error pattern analysis
- **Per-Sample Results**: Complete prediction history for debugging

### Flexible Configuration

- **Model Selection**: Test individual models or all models
- **Task Filtering**: Focus on specific tasks or run full suite
- **Sample Size Control**: From quick tests (2 samples) to full evaluation
- **Batch Size Tuning**: Optimize for API limits and performance
- **Custom Output**: Configurable output directories and file names

## 🎯 Best Practices

### 1. Start with Test Mode

```bash
# Validate setup with quick test
python prompt_instucted_models.py --test-mode
```

This runs 2 samples per model to verify:
- ✅ API credentials are correct
- ✅ Models are accessible
- ✅ Prompts are working
- ✅ Output structure is correct

### 2. Optimize Batch Size

```bash
# Small batch size for rate-limited APIs
python prompt_instucted_models.py --batch-size 5

# Larger batch size for better throughput
python prompt_instucted_models.py --batch-size 20
```

**Recommendations**:
- Rate-limited APIs: 5-10
- Standard APIs: 10-15
- High-throughput APIs: 15-25

### 3. Monitor API Usage

- Check rate limits before large evaluations
- Monitor API costs during evaluation
- Use `--samples` to control evaluation scope
- Review payment error counts in summary files

### 4. Analyze Results Systematically

```python
# 1. Overall performance comparison
summary = pd.read_csv('output/overall_summary.csv')
best_models = summary.groupby('model')['accuracy'].mean().sort_values(ascending=False)

# 2. Task-specific analysis
task_performance = summary.pivot_table(
    index='model', 
    columns='task', 
    values='accuracy'
)

# 3. Error pattern analysis
labels = pd.read_csv('output/consolidated_labels.csv')
error_patterns = labels[~labels['correct']].groupby(['model', 'task']).size()
```

## 🔐 Security & Privacy

### API Key Management

- **Never commit** `.env` files to version control
- Use environment variables for sensitive credentials
- Rotate API keys regularly
- Monitor API usage for unauthorized access

### Data Privacy

- Evaluation data may contain sensitive text
- Review output files before sharing
- Consider data anonymization for public releases
- Follow your organization's data handling policies

## 📚 Project Structure

```
Model_Based_Detector/
├── prompt_instucted_models.py    # Main evaluation script
├── prompt_templates.py           # Multi-turn conversation templates
├── .env                          # API configuration (create this)
├── README.md                     # This file
├── requirements.txt              # Python dependencies
└── output/                       # Evaluation results (auto-created)
    ├── {model_name}/
    │   ├── labels/
    │   │   └── {task}_labels.csv
    │   ├── {task}_predictions.csv
    │   └── {task}_summary.csv
    ├── overall_summary.csv
    └── consolidated_labels.csv
```

## 🐛 Troubleshooting

### Common Issues

#### 1. API Key Not Found

```
❌ OpenAI API key not found! Please:
1. Create a .env file in the Models directory
2. Add: OPENAI_API_KEY=INSERT_YOUR_API_KEY_HERE
3. Optionally add: BASE_URL=INSERT_YOUR_BASE_URL_HERE
```

**Solution**: Create `.env` file with valid credentials

#### 2. Rate Limit Errors

```
🔄 model_name rate limit, waiting and retrying...
```

**Solution**: Reduce `--batch-size` or add delays between requests

#### 3. Payment/Credit Errors

```
💳 model_name insufficient credits
```

**Solution**: Check API account balance and add credits

#### 4. Model Not Responding

```
❌ model_name final attempt failed: 'unclear_response', defaulting to 0
```

**Solution**: Check model availability or try different model

### Debug Mode

```bash
# Enable verbose output for debugging
python prompt_instucted_models.py --verbose --test-mode
```

## 📈 Performance Benchmarks

### Typical Evaluation Times

| Configuration | Time (approx.) |
|--------------|----------------|
| Test mode (2 samples × 13 models) | 2-5 minutes |
| Quick eval (50 samples × 13 models × 20 tasks) | 2-4 hours |
| Full eval (200 samples × 13 models × 20 tasks) | 8-16 hours |

*Times vary based on API speed, batch size, and network latency*

### Recommended Sample Sizes

- **Quick validation**: 10-50 samples
- **Development**: 100-200 samples
- **Research**: 500-1000 samples
- **Production**: 1000+ samples

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- 📝 New prompt templates and strategies
- 🎯 Additional evaluation tasks
- 🔧 New model integrations
- 📊 Advanced analysis scripts
- 🐛 Bug fixes and optimizations

## 📄 License

This project is provided as-is for research and educational purposes.

## 🙏 Acknowledgments

- Built with OpenAI-compatible API infrastructure
- Inspired by advances in AI text detection research
- Uses sophisticated prompting strategies from recent LLM research

## 📞 Support

For issues, questions, or suggestions:
1. Check the troubleshooting section
2. Review output files for error messages
3. Enable `--verbose` mode for detailed logging
4. Check API service status

---

**Happy Detecting! 🚀**
