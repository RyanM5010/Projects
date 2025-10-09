"""
Task-specific processors for different paraphrase detection tasks.
"""

import random
from typing import List

from base_processor import AbstractTaskProcessor
from data_models import InputSample, ProcessedSample, SingleSample


class Task1Processor(AbstractTaskProcessor):
    """Processor for TASK1: Paraphrase Source Attribution without Context.
    
    This task processes Type3 (human paraphrased) and Type4 (LLM paraphrased) texts
    to create a dataset for detecting whether a paraphrase was created by human or LLM,
    without access to the original text.
    
    Research Question: Without original sentence, can detectors distinguish 
    whether the sentence is paraphrased by human or LLM?
    
    Processing Logic:
    - Type3 (human_paraphrased_text) → Label 0 
    - Type4 (llm_paraphrased_original_text) → Label 1
    - Output: 2x input size (one sample for each type3 and type4)
    - Shuffle samples for balanced distribution
    """
    
    def __init__(self, random_seed: int = 42) -> None:
        """Initialize processor with random seed.
        
        Args:
            random_seed: Seed for reproducible shuffling
        """
        self.random_seed = random_seed
    
    def process(self, input_samples: List[InputSample]) -> List[ProcessedSample]:
        """Process input samples for TASK1.
        
        Args:
            input_samples: List of input samples containing all text types
            
        Returns:
            List of SingleSample instances with paraphrased texts and labels
            
        Raises:
            ValueError: If input data is invalid or empty
        """
        if not input_samples:
            raise ValueError("Input samples cannot be empty")
            
        processed_samples: List[ProcessedSample] = []
        
        # Process each input sample to create two output samples
        for input_sample in input_samples:
            # Validate input sample has required fields
            if not input_sample.type3 or not input_sample.type4:
                raise ValueError(f"Sample {input_sample.idx} missing type3 or type4 text")
            
            if len(input_sample.type3.strip()) < 5 or len(input_sample.type4.strip()) < 5:
                raise ValueError(f"Sample {input_sample.idx} has text shorter than minimum length")
            
            # Create sample from Type3 (human paraphrased) with label 0
            type3_sample = SingleSample(
                idx=0,  # Will be reassigned after shuffling
                sentence=input_sample.type3.strip(),
                label=0  # Human paraphrased
            )
            processed_samples.append(type3_sample)
            
            # Create sample from Type4 (LLM paraphrased) with label 1
            type4_sample = SingleSample(
                idx=0,  # Will be reassigned after shuffling
                sentence=input_sample.type4.strip(),
                label=1  # LLM paraphrased
            )
            processed_samples.append(type4_sample)
        
        # Shuffle samples for balanced distribution
        random.seed(self.random_seed)
        random.shuffle(processed_samples)
        
        # Reassign indices from 0 to N-1
        for new_idx, sample in enumerate(processed_samples):
            sample.idx = new_idx
            
        return processed_samples
    
    def get_task_name(self) -> str:
        """Get the task name.
        
        Returns:
            Human-readable task name
        """
        return "Paraphrase Source Attribution without Context"
    
    def get_expected_output_size(self, input_size: int) -> int:
        """Get expected output size.
        
        Args:
            input_size: Number of input samples
            
        Returns:
            Expected number of output samples (2x input size)
        """
        return input_size * 2
    
    def get_output_filename(self) -> str:
        """Get output filename for TASK1.
        
        Returns:
            Output filename
        """
        return "task1_paraphrase_source_without_context.json"
    
    def validate_output(self, processed_samples: List[ProcessedSample]) -> bool:
        """Validate TASK1 specific output requirements.
        
        Args:
            processed_samples: List of processed samples to validate
            
        Returns:
            True if output meets TASK1 requirements, False otherwise
        """
        # Call parent validation first
        if not super().validate_output(processed_samples):
            return False
            
        # Check that all samples are SingleSample instances
        if not all(isinstance(sample, SingleSample) for sample in processed_samples):
            return False
            
        # Check that indices are continuous from 0 to N-1
        indices = [sample.idx for sample in processed_samples]
        expected_indices = list(range(len(processed_samples)))
        if sorted(indices) != expected_indices:
            return False
            
        # Check that all samples have non-empty sentences
        for sample in processed_samples:
            if isinstance(sample, SingleSample):
                if not sample.sentence or len(sample.sentence.strip()) < 5:
                    return False
                    
        return True


class Task2Processor(AbstractTaskProcessor):
    """Processor for TASK2: General Text Authorship Detection.
    
    This task processes Type1 (human original) and Type2 (LLM generated) texts
    to create a dataset for detecting whether a text was authored by human or LLM.
    
    Research Question: Can detectors distinguish human vs LLM authorship?
    
    Processing Logic:
    - Type1 (human_original_text) → Label 0 
    - Type2 (llm_generated_text) → Label 1
    - Output: 2x input size (one sample for each type1 and type2)
    - Shuffle samples for balanced distribution
    """
    
    def __init__(self, random_seed: int = 42) -> None:
        """Initialize processor with random seed.
        
        Args:
            random_seed: Seed for reproducible shuffling
        """
        self.random_seed = random_seed
    
    def process(self, input_samples: List[InputSample]) -> List[ProcessedSample]:
        """Process input samples for TASK2.
        
        Args:
            input_samples: List of input samples containing all text types
            
        Returns:
            List of SingleSample instances with original/generated texts and labels
            
        Raises:
            ValueError: If input data is invalid or empty
        """
        if not input_samples:
            raise ValueError("Input samples cannot be empty")
            
        processed_samples: List[ProcessedSample] = []
        
        # Process each input sample to create two output samples
        for input_sample in input_samples:
            # Validate input sample has required fields
            if not input_sample.type1 or not input_sample.type2:
                raise ValueError(f"Sample {input_sample.idx} missing type1 or type2 text")
            
            if len(input_sample.type1.strip()) < 5 or len(input_sample.type2.strip()) < 5:
                raise ValueError(f"Sample {input_sample.idx} has text shorter than minimum length")
            
            # Create sample from Type1 (human original) with label 0
            type1_sample = SingleSample(
                idx=0,  # Will be reassigned after shuffling
                sentence=input_sample.type1.strip(),
                label=0  # Human authored
            )
            processed_samples.append(type1_sample)
            
            # Create sample from Type2 (LLM generated) with label 1
            type2_sample = SingleSample(
                idx=0,  # Will be reassigned after shuffling
                sentence=input_sample.type2.strip(),
                label=1  # LLM generated
            )
            processed_samples.append(type2_sample)
        
        # Shuffle samples for balanced distribution
        random.seed(self.random_seed)
        random.shuffle(processed_samples)
        
        # Reassign indices from 0 to N-1
        for new_idx, sample in enumerate(processed_samples):
            sample.idx = new_idx
            
        return processed_samples
    
    def get_task_name(self) -> str:
        """Get the task name.
        
        Returns:
            Human-readable task name
        """
        return "General Text Authorship Detection"
    
    def get_expected_output_size(self, input_size: int) -> int:
        """Get expected output size.
        
        Args:
            input_size: Number of input samples
            
        Returns:
            Expected number of output samples (2x input size)
        """
        return input_size * 2
    
    def get_output_filename(self) -> str:
        """Get output filename for TASK2.
        
        Returns:
            Output filename
        """
        return "task2_general_text_authorship_detection.json"
    
    def validate_output(self, processed_samples: List[ProcessedSample]) -> bool:
        """Validate TASK2 specific output requirements.
        
        Args:
            processed_samples: List of processed samples to validate
            
        Returns:
            True if output meets TASK2 requirements, False otherwise
        """
        # Call parent validation first
        if not super().validate_output(processed_samples):
            return False
            
        # Check that all samples are SingleSample instances
        if not all(isinstance(sample, SingleSample) for sample in processed_samples):
            return False
            
        # Check that indices are continuous from 0 to N-1
        indices = [sample.idx for sample in processed_samples]
        expected_indices = list(range(len(processed_samples)))
        if sorted(indices) != expected_indices:
            return False
            
        # Check that all samples have non-empty sentences
        for sample in processed_samples:
            if isinstance(sample, SingleSample):
                if not sample.sentence or len(sample.sentence.strip()) < 5:
                    return False
                    
        return True


class Task3Processor(AbstractTaskProcessor):
    """Processor for TASK3: AI Text Laundering Detection.
    
    This task processes Type4 (LLM paraphrased human text) and Type5-1st (LLM paraphrased LLM text)
    to create a dataset for detecting "laundering" of AI text through AI paraphrasing.
    
    Research Question: Can detectors distinguish laundered vs direct paraphrasing?
    
    Processing Logic:
    - Type4 (llm_paraphrased_original_text) → Label 0 (LLM paraphrased human text)
    - Type5-1st (llm_paraphrased_generated_text-1st) → Label 1 (LLM paraphrased LLM text)
    - Output: 2x input size (one sample for each type4 and type5-1st)
    - Shuffle samples for balanced distribution
    """
    
    def __init__(self, random_seed: int = 42) -> None:
        """Initialize processor with random seed.
        
        Args:
            random_seed: Seed for reproducible shuffling
        """
        self.random_seed = random_seed
    
    def process(self, input_samples: List[InputSample]) -> List[ProcessedSample]:
        """Process input samples for TASK3.
        
        Args:
            input_samples: List of input samples containing all text types
            
        Returns:
            List of SingleSample instances with paraphrased texts and labels
            
        Raises:
            ValueError: If input data is invalid or empty
        """
        if not input_samples:
            raise ValueError("Input samples cannot be empty")
            
        processed_samples: List[ProcessedSample] = []
        
        # Process each input sample to create two output samples
        for input_sample in input_samples:
            # Validate input sample has required fields
            if not input_sample.type4 or not input_sample.type5_1st:
                raise ValueError(f"Sample {input_sample.idx} missing type4 or type5_1st text")
            
            if len(input_sample.type4.strip()) < 5 or len(input_sample.type5_1st.strip()) < 5:
                raise ValueError(f"Sample {input_sample.idx} has text shorter than minimum length")
            
            # Create sample from Type4 (LLM paraphrased human text) with label 0
            type4_sample = SingleSample(
                idx=0,  # Will be reassigned after shuffling
                sentence=input_sample.type4.strip(),
                label=0  # LLM paraphrased human text
            )
            processed_samples.append(type4_sample)
            
            # Create sample from Type5-1st (LLM paraphrased LLM text) with label 1
            type5_sample = SingleSample(
                idx=0,  # Will be reassigned after shuffling
                sentence=input_sample.type5_1st.strip(),
                label=1  # LLM paraphrased LLM text (laundered)
            )
            processed_samples.append(type5_sample)
        
        # Shuffle samples for balanced distribution
        random.seed(self.random_seed)
        random.shuffle(processed_samples)
        
        # Reassign indices from 0 to N-1
        for new_idx, sample in enumerate(processed_samples):
            sample.idx = new_idx
            
        return processed_samples
    
    def get_task_name(self) -> str:
        """Get the task name.
        
        Returns:
            Human-readable task name
        """
        return "AI Text Laundering Detection"
    
    def get_expected_output_size(self, input_size: int) -> int:
        """Get expected output size.
        
        Args:
            input_size: Number of input samples
            
        Returns:
            Expected number of output samples (2x input size)
        """
        return input_size * 2
    
    def get_output_filename(self) -> str:
        """Get output filename for TASK3.
        
        Returns:
            Output filename
        """
        return "task3_ai_text_laundering_detection.json"
    
    def validate_output(self, processed_samples: List[ProcessedSample]) -> bool:
        """Validate TASK3 specific output requirements.
        
        Args:
            processed_samples: List of processed samples to validate
            
        Returns:
            True if output meets TASK3 requirements, False otherwise
        """
        # Call parent validation first
        if not super().validate_output(processed_samples):
            return False
            
        # Check that all samples are SingleSample instances
        if not all(isinstance(sample, SingleSample) for sample in processed_samples):
            return False
            
        # Check that indices are continuous from 0 to N-1
        indices = [sample.idx for sample in processed_samples]
        expected_indices = list(range(len(processed_samples)))
        if sorted(indices) != expected_indices:
            return False
            
        # Check that all samples have non-empty sentences
        for sample in processed_samples:
            if isinstance(sample, SingleSample):
                if not sample.sentence or len(sample.sentence.strip()) < 5:
                    return False
                    
        return True


class Task4Processor(AbstractTaskProcessor):
    """Processor for TASK4: Iterative Paraphrase Depth Detection.
    
    This task processes Type5-1st (shallow laundering) and Type5-3rd (deep laundering)
    to create a dataset for detecting the depth of iterative paraphrasing.
    
    Research Question: Are deeper paraphrases harder to detect?
    
    Processing Logic:
    - Type5-1st (llm_paraphrased_generated_text-1st) → Label 0 (Shallow laundering)
    - Type5-3rd (llm_paraphrased_generated_text-3rd) → Label 1 (Deep laundering)
    - Output: 2x input size (one sample for each type5-1st and type5-3rd)
    - Shuffle samples for balanced distribution
    """
    
    def __init__(self, random_seed: int = 42) -> None:
        """Initialize processor with random seed.
        
        Args:
            random_seed: Seed for reproducible shuffling
        """
        self.random_seed = random_seed
    
    def process(self, input_samples: List[InputSample]) -> List[ProcessedSample]:
        """Process input samples for TASK4.
        
        Args:
            input_samples: List of input samples containing all text types
            
        Returns:
            List of SingleSample instances with iteratively paraphrased texts and labels
            
        Raises:
            ValueError: If input data is invalid or empty
        """
        if not input_samples:
            raise ValueError("Input samples cannot be empty")
            
        processed_samples: List[ProcessedSample] = []
        
        # Process each input sample to create two output samples
        for input_sample in input_samples:
            # Validate input sample has required fields
            if not input_sample.type5_1st or not input_sample.type5_3rd:
                raise ValueError(f"Sample {input_sample.idx} missing type5_1st or type5_3rd text")
            
            if len(input_sample.type5_1st.strip()) < 5 or len(input_sample.type5_3rd.strip()) < 5:
                raise ValueError(f"Sample {input_sample.idx} has text shorter than minimum length")
            
            # Create sample from Type5-1st (shallow laundering) with label 0
            type5_1st_sample = SingleSample(
                idx=0,  # Will be reassigned after shuffling
                sentence=input_sample.type5_1st.strip(),
                label=0  # Shallow laundering
            )
            processed_samples.append(type5_1st_sample)
            
            # Create sample from Type5-3rd (deep laundering) with label 1
            type5_3rd_sample = SingleSample(
                idx=0,  # Will be reassigned after shuffling
                sentence=input_sample.type5_3rd.strip(),
                label=1  # Deep laundering
            )
            processed_samples.append(type5_3rd_sample)
        
        # Shuffle samples for balanced distribution
        random.seed(self.random_seed)
        random.shuffle(processed_samples)
        
        # Reassign indices from 0 to N-1
        for new_idx, sample in enumerate(processed_samples):
            sample.idx = new_idx
            
        return processed_samples
    
    def get_task_name(self) -> str:
        """Get the task name.
        
        Returns:
            Human-readable task name
        """
        return "Iterative Paraphrase Depth Detection"
    
    def get_expected_output_size(self, input_size: int) -> int:
        """Get expected output size.
        
        Args:
            input_size: Number of input samples
            
        Returns:
            Expected number of output samples (2x input size)
        """
        return input_size * 2
    
    def get_output_filename(self) -> str:
        """Get output filename for TASK4.
        
        Returns:
            Output filename
        """
        return "task4_iterative_paraphrase_depth_detection.json"
    
    def validate_output(self, processed_samples: List[ProcessedSample]) -> bool:
        """Validate TASK4 specific output requirements.
        
        Args:
            processed_samples: List of processed samples to validate
            
        Returns:
            True if output meets TASK4 requirements, False otherwise
        """
        # Call parent validation first
        if not super().validate_output(processed_samples):
            return False
            
        # Check that all samples are SingleSample instances
        if not all(isinstance(sample, SingleSample) for sample in processed_samples):
            return False
            
        # Check that indices are continuous from 0 to N-1
        indices = [sample.idx for sample in processed_samples]
        expected_indices = list(range(len(processed_samples)))
        if sorted(indices) != expected_indices:
            return False
            
        # Check that all samples have non-empty sentences
        for sample in processed_samples:
            if isinstance(sample, SingleSample):
                if not sample.sentence or len(sample.sentence.strip()) < 5:
                    return False
                    
        return True



class Task5Processor(AbstractTaskProcessor):
    """Processor for TASK5: Original vs Deep Paraphrase Attack Detection.
    
    This task processes Type1 (human original) and Type5-3rd (deep paraphrase attack)
    to create a dataset for detecting the most representative paraphrase attack.
    
    Research Question: Can detectors distinguish human original text from 
    the most sophisticated paraphrase attack (Type5-3rd)?
    
    Processing Logic:
    - Type1 (human_original_text) → Label 0 (Human original)
    - Type5-3rd (llm_paraphrased_generated_text-3rd) → Label 1 (Deep paraphrase attack)
    - Output: 2x input size (one sample for each type1 and type5-3rd)
    - Shuffle samples for balanced distribution
    """
    
    def __init__(self, random_seed: int = 42) -> None:
        """Initialize processor with random seed.
        
        Args:
            random_seed: Seed for reproducible shuffling
        """
        self.random_seed = random_seed
    
    def process(self, input_samples: List[InputSample]) -> List[ProcessedSample]:
        """Process input samples for TASK5.
        
        Args:
            input_samples: List of input samples containing all text types
            
        Returns:
            List of SingleSample instances with original and deep paraphrase attack texts
            
        Raises:
            ValueError: If input data is invalid or empty
        """
        if not input_samples:
            raise ValueError("Input samples cannot be empty")
            
        processed_samples: List[ProcessedSample] = []
        
        # Process each input sample to create two output samples
        for input_sample in input_samples:
            # Validate input sample has required fields
            if not input_sample.type1 or not input_sample.type5_3rd:
                raise ValueError(f"Sample {input_sample.idx} missing type1 or type5_3rd text")
            
            if len(input_sample.type1.strip()) < 5 or len(input_sample.type5_3rd.strip()) < 5:
                raise ValueError(f"Sample {input_sample.idx} has text shorter than minimum length")
            
            # Create sample from Type1 (human original) with label 0
            type1_sample = SingleSample(
                idx=0,  # Will be reassigned after shuffling
                sentence=input_sample.type1.strip(),
                label=0  # Human original
            )
            processed_samples.append(type1_sample)
            
            # Create sample from Type5-3rd (deep paraphrase attack) with label 1
            type5_3rd_sample = SingleSample(
                idx=0,  # Will be reassigned after shuffling
                sentence=input_sample.type5_3rd.strip(),
                label=1  # Deep paraphrase attack
            )
            processed_samples.append(type5_3rd_sample)
        
        # Shuffle samples for balanced distribution
        random.seed(self.random_seed)
        random.shuffle(processed_samples)
        
        # Reassign indices from 0 to N-1
        for new_idx, sample in enumerate(processed_samples):
            sample.idx = new_idx
            
        return processed_samples
    
    def get_task_name(self) -> str:
        """Get the task name.
        
        Returns:
            Human-readable task name
        """
        return "Original vs Deep Paraphrase Attack Detection"
    
    def get_expected_output_size(self, input_size: int) -> int:
        """Get expected output size.
        
        Args:
            input_size: Number of input samples
            
        Returns:
            Expected number of output samples (2x input for exhaustive method)
        """
        return input_size * 2
    
    def get_output_filename(self) -> str:
        """Get output filename for TASK5.
        
        Returns:
            Output filename
        """
        return "task5_original_vs_deep_paraphrase_attack.json"
    
    def get_report_filename(self) -> str:
        """Get report filename for TASK5.
        
        Returns:
            Report filename
        """
        return "task5_original_vs_deep_paraphrase_attack_report.json"
    
    def validate_output(self, processed_samples: List[ProcessedSample]) -> bool:
        """Validate processed samples for TASK5.
        
        Args:
            processed_samples: List of processed samples to validate
            
        Returns:
            True if validation passes, False otherwise
        """
        if not processed_samples:
            return False
            
        # Check for proper label balance (should be 50-50 for exhaustive method)
        label_0_count = sum(1 for sample in processed_samples 
                           if isinstance(sample, SingleSample) and sample.label == 0)
        label_1_count = sum(1 for sample in processed_samples 
                           if isinstance(sample, SingleSample) and sample.label == 1)
        
        if label_0_count != label_1_count:
            return False
            
        # Check that all samples have non-empty sentences
        for sample in processed_samples:
            if isinstance(sample, SingleSample):
                if not sample.sentence or len(sample.sentence.strip()) < 5:
                    return False
                    
        return True
