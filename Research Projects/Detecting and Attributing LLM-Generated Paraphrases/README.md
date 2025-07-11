# PLD-4: A Multi-Task Framework for Detecting and Attributing LLM-Generated Paraphrases

## Overview

Large Language Models (LLMs) have made it increasingly difficult to distinguish between human-written and machine-generated text. This challenge becomes more severe when paraphrasing is used to evade detection—posing risks to academic integrity, intellectual property, and information trustworthiness.

**PLD-4 (Paraphrase-based LLM Detection Framework)** is a novel benchmark that defines **four complementary detection tasks** to assess models' ability to identify LLM-generated paraphrases in realistic and adversarial scenarios.

## Key Contributions

- Formalizes **4 detection tasks** across sentence-pair and single-sentence settings.
- Includes **authorship attribution**, **paraphrase source detection**, and **original vs. paraphrased LLM output classification**.
- Highlights the difficulty of identifying layered AI-generated text.
- Provides benchmarks for both **feature-based models (XGBoost)** and **transformer models (DeBERTa-v3, RoBERTa)**.

## Datasets

- **MRPC (Microsoft Research Paraphrase Corpus)**
- **HLPC (Human-LLM Paraphrase Corpus)**

## Results Summary

| Task | Model | Accuracy |
|------|-------|----------|
| Sentence Pair Paraphrase Source Detection | XGBoost | 96.0% |
| Single Sentence Authorship Attribution     | RoBERTa | 93.9% |
| Paraphrased LLM vs Original LLM Detection | RoBERTa | 83.28% |

While performance is strong in attribution and source detection, results highlight the **limitation of current detectors** when facing layered or paraphrased LLM outputs.





