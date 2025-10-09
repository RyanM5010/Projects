"""
Task processors for sentence pair classification tasks.
"""

import random
from typing import List, Tuple
from base_processor import AbstractSentencePairProcessor
from data_models import InputSample, SentencePairSample


class Task1SentencePairProcessor(AbstractSentencePairProcessor):
    """Processor for TASK1: Paraphrase Source Attribution without Context (Sentence Pair).

    This task creates sentence pairs from Type3 (human paraphrased) and Type4 (LLM paraphrased)
    to determine which sentence in the pair is machine-generated.

    Research Question: Can detectors distinguish between human and LLM paraphrases when presented as pairs?

    Processing Logic:
    - Type3 (human_paraphrased_text) → Label 0
    - Type4 (llm_paraphrased_original_text) → Label 1
    - Creates two pairs per input: (type3, type4) and (type4, type3)
    - Output: 2x input size (two pairs per input sample)
    """

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed

    def process(self, input_samples: List[InputSample]) -> List[SentencePairSample]:
        if not input_samples:
            raise ValueError("Input samples cannot be empty")

        processed_samples: List[SentencePairSample] = []
        random.seed(self.random_seed)

        for input_sample in input_samples:
            if not input_sample.type3 or not input_sample.type4:
                raise ValueError(f"Sample {input_sample.idx} missing type3 or type4 text")

            if len(input_sample.type3.strip()) < 5 or len(input_sample.type4.strip()) < 5:
                raise ValueError(f"Sample {input_sample.idx} has text shorter than minimum length")

            # Randomly decide the order of sentences in the pair
            if random.random() < 0.5:
                # Order: (type3, type4) with labels (0, 1)
                sentence_pair = (input_sample.type3.strip(), input_sample.type4.strip())
                label_pair = (0, 1)
                label = 0  # For compatibility with ProcessedSample
            else:
                # Order: (type4, type3) with labels (1, 0)
                sentence_pair = (input_sample.type4.strip(), input_sample.type3.strip())
                label_pair = (1, 0)
                label = 1  # For compatibility with ProcessedSample

            # Create one pair per input sample
            pair = SentencePairSample(
                idx=0,  # Will be reassigned later
                label=label,
                sentence_pair=sentence_pair,
                label_pair=label_pair
            )
            processed_samples.append(pair)

        # Shuffle and reassign indices
        random.shuffle(processed_samples)

        for new_idx, sample in enumerate(processed_samples):
            sample.idx = new_idx

        return processed_samples

    def get_task_name(self) -> str:
        return "Paraphrase Source Attribution without Context (Sentence Pair)"

    def get_expected_output_size(self, input_size: int) -> int:
        return input_size

    def get_output_filename(self) -> str:
        return "task1_paraphrase_source_without_context_sentence_pair.json"


class Task2SentencePairProcessor(AbstractSentencePairProcessor):
    """Processor for TASK2: General Text Authorship Detection (Sentence Pair).

    This task creates sentence pairs from Type1 (human original) and Type2 (LLM generated)
    to determine which sentence in the pair is machine-generated.

    Research Question: Can detectors distinguish between human original and LLM generated text when presented as pairs?

    Processing Logic:
    - Type1 (human_original_text) → Label 0
    - Type2 (llm_generated_text) → Label 1
    - Creates two pairs per input: (type1, type2) and (type2, type1)
    - Output: 2x input size (two pairs per input sample)
    """

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed

    def process(self, input_samples: List[InputSample]) -> List[SentencePairSample]:
        if not input_samples:
            raise ValueError("Input samples cannot be empty")

        processed_samples: List[SentencePairSample] = []
        random.seed(self.random_seed)

        for input_sample in input_samples:
            if not input_sample.type1 or not input_sample.type2:
                raise ValueError(f"Sample {input_sample.idx} missing type1 or type2 text")

            if len(input_sample.type1.strip()) < 5 or len(input_sample.type2.strip()) < 5:
                raise ValueError(f"Sample {input_sample.idx} has text shorter than minimum length")

            # Randomly decide the order of sentences in the pair
            if random.random() < 0.5:
                # Order: (type1, type2) with labels (0, 1)
                sentence_pair = (input_sample.type1.strip(), input_sample.type2.strip())
                label_pair = (0, 1)
                label = 0  # For compatibility with ProcessedSample
            else:
                # Order: (type2, type1) with labels (1, 0)
                sentence_pair = (input_sample.type2.strip(), input_sample.type1.strip())
                label_pair = (1, 0)
                label = 1  # For compatibility with ProcessedSample

            # Create one pair per input sample
            pair = SentencePairSample(
                idx=0,  # Will be reassigned later
                label=label,
                sentence_pair=sentence_pair,
                label_pair=label_pair
            )
            processed_samples.append(pair)

        # Shuffle and reassign indices
        random.shuffle(processed_samples)

        for new_idx, sample in enumerate(processed_samples):
            sample.idx = new_idx

        return processed_samples

    def get_task_name(self) -> str:
        return "General Text Authorship Detection (Sentence Pair)"

    def get_expected_output_size(self, input_size: int) -> int:
        return input_size

    def get_output_filename(self) -> str:
        return "task2_general_text_authorship_detection_sentence_pair.json"


class Task3SentencePairProcessor(AbstractSentencePairProcessor):
    """Processor for TASK3: AI Text Laundering Detection (Sentence Pair).

    This task creates sentence pairs from Type4 (LLM paraphrased original) and Type5-1st (LLM paraphrased generated, 1st iteration)
    to determine which sentence in the pair is more machine-generated.

    Research Question: Can detectors distinguish between different levels of LLM paraphrasing when presented as pairs?

    Processing Logic:
    - Type4 (llm_paraphrased_original_text) → Label 0
    - Type5-1st (llm_paraphrased_generated_text-1st) → Label 1
    - Creates two pairs per input: (type4, type5-1st) and (type5-1st, type4)
    - Output: 2x input size (two pairs per input sample)
    """

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed

    def process(self, input_samples: List[InputSample]) -> List[SentencePairSample]:
        if not input_samples:
            raise ValueError("Input samples cannot be empty")

        processed_samples: List[SentencePairSample] = []
        random.seed(self.random_seed)

        for input_sample in input_samples:
            if not input_sample.type4 or not input_sample.type5_1st:
                raise ValueError(f"Sample {input_sample.idx} missing type4 or type5-1st text")

            if len(input_sample.type4.strip()) < 5 or len(input_sample.type5_1st.strip()) < 5:
                raise ValueError(f"Sample {input_sample.idx} has text shorter than minimum length")

            # Randomly decide the order of sentences in the pair
            if random.random() < 0.5:
                # Order: (type4, type5-1st) with labels (0, 1)
                sentence_pair = (input_sample.type4.strip(), input_sample.type5_1st.strip())
                label_pair = (0, 1)
                label = 0  # For compatibility with ProcessedSample
            else:
                # Order: (type5-1st, type4) with labels (1, 0)
                sentence_pair = (input_sample.type5_1st.strip(), input_sample.type4.strip())
                label_pair = (1, 0)
                label = 1  # For compatibility with ProcessedSample

            # Create one pair per input sample
            pair = SentencePairSample(
                idx=0,  # Will be reassigned later
                label=label,
                sentence_pair=sentence_pair,
                label_pair=label_pair
            )
            processed_samples.append(pair)

        # Shuffle and reassign indices
        random.shuffle(processed_samples)

        for new_idx, sample in enumerate(processed_samples):
            sample.idx = new_idx

        return processed_samples

    def get_task_name(self) -> str:
        return "AI Text Laundering Detection (Sentence Pair)"

    def get_expected_output_size(self, input_size: int) -> int:
        return input_size

    def get_output_filename(self) -> str:
        return "task3_ai_text_laundering_detection_sentence_pair.json"


class Task4SentencePairProcessor(AbstractSentencePairProcessor):
    """Processor for TASK4: Iterative Paraphrase Depth Detection (Sentence Pair).

    This task creates sentence pairs from Type5-1st (LLM paraphrased generated, 1st iteration) and Type5-3rd (LLM paraphrased generated, 3rd iteration)
    to determine which sentence in the pair is more deeply paraphrased.

    Research Question: Can detectors distinguish between different depths of iterative LLM paraphrasing when presented as pairs?

    Processing Logic:
    - Type5-1st (llm_paraphrased_generated_text-1st) → Label 0
    - Type5-3rd (llm_paraphrased_generated_text-3rd) → Label 1
    - Creates two pairs per input: (type5-1st, type5-3rd) and (type5-3rd, type5-1st)
    - Output: 2x input size (two pairs per input sample)
    """

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed

    def process(self, input_samples: List[InputSample]) -> List[SentencePairSample]:
        if not input_samples:
            raise ValueError("Input samples cannot be empty")

        processed_samples: List[SentencePairSample] = []
        random.seed(self.random_seed)

        for input_sample in input_samples:
            if not input_sample.type5_1st or not input_sample.type5_3rd:
                raise ValueError(f"Sample {input_sample.idx} missing type5-1st or type5-3rd text")

            if len(input_sample.type5_1st.strip()) < 5 or len(input_sample.type5_3rd.strip()) < 5:
                raise ValueError(f"Sample {input_sample.idx} has text shorter than minimum length")

            # Randomly decide the order of sentences in the pair
            if random.random() < 0.5:
                # Order: (type5-1st, type5-3rd) with labels (0, 1)
                sentence_pair = (input_sample.type5_1st.strip(), input_sample.type5_3rd.strip())
                label_pair = (0, 1)
                label = 0  # For compatibility with ProcessedSample
            else:
                # Order: (type5-3rd, type5-1st) with labels (1, 0)
                sentence_pair = (input_sample.type5_3rd.strip(), input_sample.type5_1st.strip())
                label_pair = (1, 0)
                label = 1  # For compatibility with ProcessedSample

            # Create one pair per input sample
            pair = SentencePairSample(
                idx=0,  # Will be reassigned later
                label=label,
                sentence_pair=sentence_pair,
                label_pair=label_pair
            )
            processed_samples.append(pair)

        # Shuffle and reassign indices
        random.shuffle(processed_samples)

        for new_idx, sample in enumerate(processed_samples):
            sample.idx = new_idx

        return processed_samples

    def get_task_name(self) -> str:
        return "Iterative Paraphrase Depth Detection (Sentence Pair)"

    def get_expected_output_size(self, input_size: int) -> int:
        return input_size

    def get_output_filename(self) -> str:
        return "task4_iterative_paraphrase_depth_detection_sentence_pair.json"


class Task5SentencePairProcessor(AbstractSentencePairProcessor):
    """Processor for TASK5: Original vs Deep Paraphrase Attack Detection (Sentence Pair).

    This task creates sentence pairs from Type1 (human original) and Type5-3rd (LLM paraphrased LLM generated text, 3rd iteration)
    to determine which sentence in the pair is machine-generated.

    Research Question: Can detectors distinguish human original text from the most sophisticated paraphrase attack when presented as pairs?

    Processing Logic:
    - Type1 (human_original_text) → Label 0
    - Type5-3rd (llm_paraphrased_generated_text-3rd) → Label 1
    - Creates two pairs per input: (type1, type5-3rd) and (type5-3rd, type1)
    - Output: 2x input size (two pairs per input sample)
    """

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed

    def process(self, input_samples: List[InputSample]) -> List[SentencePairSample]:
        if not input_samples:
            raise ValueError("Input samples cannot be empty")

        processed_samples: List[SentencePairSample] = []
        random.seed(self.random_seed)

        for input_sample in input_samples:
            if not input_sample.type1 or not input_sample.type5_3rd:
                raise ValueError(f"Sample {input_sample.idx} missing type1 or type5-3rd text")

            if len(input_sample.type1.strip()) < 5 or len(input_sample.type5_3rd.strip()) < 5:
                raise ValueError(f"Sample {input_sample.idx} has text shorter than minimum length")

            # Randomly decide the order of sentences in the pair
            if random.random() < 0.5:
                # Order: (type1, type5-3rd) with labels (0, 1)
                sentence_pair = (input_sample.type1.strip(), input_sample.type5_3rd.strip())
                label_pair = (0, 1)
                label = 0  # For compatibility with ProcessedSample
            else:
                # Order: (type5-3rd, type1) with labels (1, 0)
                sentence_pair = (input_sample.type5_3rd.strip(), input_sample.type1.strip())
                label_pair = (1, 0)
                label = 1  # For compatibility with ProcessedSample

            # Create one pair per input sample
            pair = SentencePairSample(
                idx=0,  # Will be reassigned later
                label=label,
                sentence_pair=sentence_pair,
                label_pair=label_pair
            )
            processed_samples.append(pair)

        # Shuffle and reassign indices
        random.shuffle(processed_samples)

        for new_idx, sample in enumerate(processed_samples):
            sample.idx = new_idx

        return processed_samples

    def get_task_name(self) -> str:
        return "Original vs Deep Paraphrase Attack Detection (Sentence Pair)"

    def get_expected_output_size(self, input_size: int) -> int:
        return input_size

    def get_output_filename(self) -> str:
        return "task5_original_vs_deep_paraphrase_attack_sentence_pair.json"
