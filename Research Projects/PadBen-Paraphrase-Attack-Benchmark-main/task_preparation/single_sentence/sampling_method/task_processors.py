"""
Task-specific processors for dynamically-adjusted paraphrase detection tasks.

Each processor selects one sample per index with configurable label balance.
"""

from typing import Tuple

from base_processor import AbstractDynamicTaskProcessor
from data_models import DynamicInputSample


class DynamicTask1Processor(AbstractDynamicTaskProcessor):
    """Dynamic processor for TASK1: Paraphrase Source Attribution without Context.
    
    This task processes Type3 (human paraphrased) and Type4 (LLM paraphrased) texts
    to create a dataset for detecting whether a paraphrase was created by human or LLM,
    without access to the original text.
    
    For each input sample, randomly selects either type3 or type4 based on label_1_ratio.
    
    Research Question: Without original sentence, can detectors distinguish 
    whether the sentence is paraphrased by human or LLM?
    
    Processing Logic:
    - Type3 (human_paraphrased_text) → Label 0 
    - Type4 (llm_paraphrased_original_text) → Label 1
    - Output: Same size as input (one sample per input index)
    - Label balance controlled by label_1_ratio parameter
    """
    
    def get_text_pairs(self, input_sample: DynamicInputSample) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        """Get the text pairs for Task 1.
        
        Args:
            input_sample: Input sample containing all text types
            
        Returns:
            ((type3_text, "type3"), (type4_text, "type4"))
        """
        return (
            (input_sample.type3, "type3"),  # Label 0: Human paraphrased
            (input_sample.type4, "type4")   # Label 1: LLM paraphrased
        )
    
    def get_task_name(self) -> str:
        """Get the task name.
        
        Returns:
            Human-readable task name
        """
        return "Dynamic Paraphrase Source Attribution without Context"
    
    def get_output_filename(self) -> str:
        """Get output filename for TASK1.
        
        Returns:
            Output filename
        """
        return "dynamic_task1_paraphrase_source_without_context.json"


class DynamicTask2Processor(AbstractDynamicTaskProcessor):
    """Dynamic processor for TASK2: General Text Authorship Detection.
    
    This task processes Type1 (human original) and Type2 (LLM generated) texts
    to create a dataset for detecting whether a text was authored by human or LLM.
    
    For each input sample, randomly selects either type1 or type2 based on label_1_ratio.
    
    Research Question: Can detectors distinguish human vs LLM authorship?
    
    Processing Logic:
    - Type1 (human_original_text) → Label 0 
    - Type2 (llm_generated_text) → Label 1
    - Output: Same size as input (one sample per input index)
    - Label balance controlled by label_1_ratio parameter
    """
    
    def get_text_pairs(self, input_sample: DynamicInputSample) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        """Get the text pairs for Task 2.
        
        Args:
            input_sample: Input sample containing all text types
            
        Returns:
            ((type1_text, "type1"), (type2_text, "type2"))
        """
        return (
            (input_sample.type1, "type1"),  # Label 0: Human original
            (input_sample.type2, "type2")   # Label 1: LLM generated
        )
    
    def get_task_name(self) -> str:
        """Get the task name.
        
        Returns:
            Human-readable task name
        """
        return "Dynamic General Text Authorship Detection"
    
    def get_output_filename(self) -> str:
        """Get output filename for TASK2.
        
        Returns:
            Output filename
        """
        return "dynamic_task2_general_text_authorship_detection.json"


class DynamicTask3Processor(AbstractDynamicTaskProcessor):
    """Dynamic processor for TASK3: AI Text Laundering Detection.
    
    This task processes Type4 (LLM paraphrased human text) and Type5-1st (LLM paraphrased LLM text)
    to create a dataset for detecting "laundering" of AI text through AI paraphrasing.
    
    For each input sample, randomly selects either type4 or type5_1st based on label_1_ratio.
    
    Research Question: Can detectors distinguish laundered vs direct paraphrasing?
    
    Processing Logic:
    - Type4 (llm_paraphrased_original_text) → Label 0 (LLM paraphrased human text)
    - Type5-1st (llm_paraphrased_generated_text-1st) → Label 1 (LLM paraphrased LLM text)
    - Output: Same size as input (one sample per input index)
    - Label balance controlled by label_1_ratio parameter
    """
    
    def get_text_pairs(self, input_sample: DynamicInputSample) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        """Get the text pairs for Task 3.
        
        Args:
            input_sample: Input sample containing all text types
            
        Returns:
            ((type4_text, "type4"), (type5_1st_text, "type5_1st"))
        """
        return (
            (input_sample.type4, "type4"),        # Label 0: LLM paraphrased human text
            (input_sample.type5_1st, "type5_1st") # Label 1: LLM paraphrased LLM text (laundered)
        )
    
    def get_task_name(self) -> str:
        """Get the task name.
        
        Returns:
            Human-readable task name
        """
        return "Dynamic AI Text Laundering Detection"
    
    def get_output_filename(self) -> str:
        """Get output filename for TASK3.
        
        Returns:
            Output filename
        """
        return "dynamic_task3_ai_text_laundering_detection.json"


class DynamicTask4Processor(AbstractDynamicTaskProcessor):
    """Dynamic processor for TASK4: Iterative Paraphrase Depth Detection.
    
    This task processes Type5-1st (shallow laundering) and Type5-3rd (deep laundering)
    to create a dataset for detecting the depth of iterative paraphrasing.
    
    For each input sample, randomly selects either type5_1st or type5_3rd based on label_1_ratio.
    
    Research Question: Are deeper paraphrases harder to detect?
    
    Processing Logic:
    - Type5-1st (llm_paraphrased_generated_text-1st) → Label 0 (Shallow laundering)
    - Type5-3rd (llm_paraphrased_generated_text-3rd) → Label 1 (Deep laundering)
    - Output: Same size as input (one sample per input index)
    - Label balance controlled by label_1_ratio parameter
    """
    
    def get_text_pairs(self, input_sample: DynamicInputSample) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        """Get the text pairs for Task 4.
        
        Args:
            input_sample: Input sample containing all text types
            
        Returns:
            ((type5_1st_text, "type5_1st"), (type5_3rd_text, "type5_3rd"))
        """
        return (
            (input_sample.type5_1st, "type5_1st"), # Label 0: Shallow laundering
            (input_sample.type5_3rd, "type5_3rd")  # Label 1: Deep laundering
        )
    
    def get_task_name(self) -> str:
        """Get the task name.
        
        Returns:
            Human-readable task name
        """
        return "Dynamic Iterative Paraphrase Depth Detection"
    
    def get_output_filename(self) -> str:
        """Get output filename for TASK4.
        
        Returns:
            Output filename
        """
        return "dynamic_task4_iterative_paraphrase_depth_detection.json"


class DynamicTask5Processor(AbstractDynamicTaskProcessor):
    """Dynamic processor for TASK5: Original vs Deep Paraphrase Attack Detection.
    
    This task processes Type1 (human original) and Type5-3rd (deep paraphrase attack)
    to create a dataset for detecting the most representative paraphrase attack.
    
    For each input sample, randomly selects either type1 or type5_3rd based on label_1_ratio.
    
    Research Question: Can detectors distinguish human original text from 
    the most sophisticated paraphrase attack (Type5-3rd)?
    
    Processing Logic:
    - Type1 (human_original_text) → Label 0 (Human original)
    - Type5-3rd (llm_paraphrased_generated_text-3rd) → Label 1 (Deep paraphrase attack)
    - Output: Same size as input (one sample per input index)
    - Label balance controlled by label_1_ratio parameter
    """
    
    def get_text_pairs(self, input_sample: DynamicInputSample) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        """Get the text pairs for Task 5.
        
        Args:
            input_sample: Input sample containing all text types
            
        Returns:
            ((type1_text, "type1"), (type5_3rd_text, "type5_3rd"))
        """
        return (
            (input_sample.type1, "type1"),           # Label 0: Human original
            (input_sample.type5_3rd, "type5_3rd")    # Label 1: Deep paraphrase attack
        )
    
    def get_task_name(self) -> str:
        """Get the task name.
        
        Returns:
            Human-readable task name
        """
        return "Dynamic Original vs Deep Paraphrase Attack Detection"
    
    def get_output_filename(self) -> str:
        """Get output filename for TASK5.
        
        Returns:
            Output filename
        """
        return "dynamic_task5_original_vs_deep_paraphrase_attack.json"
