"""
PADBen Generation Orchestrator.

This module coordinates the entire text generation pipeline:
- Type 2: LLM-generated text (sentence completion & question-answer)
- Type 4: LLM-paraphrased original text (DIPPER & prompt-based)
- Type 5: LLM-paraphrased generated text (DIPPER & prompt-based, 1/3/5 iterations)

Features:
- Sequential execution with dependency management
- Flexible generation options (user can enable/disable any type)
- Comprehensive output structure with all variants
- Configuration validation and provider setup
- Progress tracking and intermediate saves
"""

import asyncio
import logging
import time
import json
import os
from typing import Dict, List, Optional, Tuple, Any, Set
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
import numpy as np

# Local imports for all generation types
from data_generation.config.generation_model_config import (
    GenerationConfig,
    DEFAULT_CONFIG,
    validate_all_configs,
    get_config_summary
)

# Import specific configurations
from data_generation.config.type2_config import Type2GenerationMethod
from data_generation.config.type4_config import Type4ParaphraseMethod
from data_generation.config.type5_config import IterationLevel

# Import generators
from data_generation.type2_generation.type2_generation import (
    EnhancedType2Generator,
    GenerationMethod as Type2Method,
    EnvironmentMode
)
from data_generation.type4_generation import Type4Generator
from data_generation.type5_generation import Type5Generator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GenerationType(Enum):
    """Available generation types."""
    TYPE2 = "type2"
    TYPE4 = "type4" 
    TYPE5 = "type5"

@dataclass
class GenerationOptions:
    """Options for controlling which generation types to run."""
    
    # Generation type flags
    type2: bool = False
    type4: bool = False
    type5: bool = False
    
    # Type 2 specific options
    type2_method: Optional[str] = "auto"  # "sentence_completion", "question_answer", "auto"
    
    # Type 4 specific options
    type4_method_dipper: bool = True
    type4_method_prompt: bool = True
    type4_method_llama: bool = True
    
    # Type 5 specific options
    type5_method_dipper: bool = True
    type5_method_prompt: bool = True
    type5_iterations: List[int] = None  # [1, 3, 5] or subset
    
    # Output options
    save_intermediate: bool = True
    include_metadata: bool = True

@dataclass
class GenerationStatus:
    """Track status of generation pipeline."""
    
    # Overall status
    total_samples: int = 0
    completed_samples: int = 0
    failed_samples: int = 0
    
    # Type-specific status
    type2_completed: int = 0
    type2_failed: int = 0
    
    type4_dipper_completed: int = 0
    type4_dipper_failed: int = 0
    type4_prompt_completed: int = 0
    type4_prompt_failed: int = 0
    type4_llama_completed: int = 0
    type4_llama_failed: int = 0
    
    type5_dipper_completed: Dict[int, int] = None
    type5_dipper_failed: Dict[int, int] = None
    type5_prompt_completed: Dict[int, int] = None
    type5_prompt_failed: Dict[int, int] = None
    
    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def __post_init__(self):
        if self.type5_dipper_completed is None:
            self.type5_dipper_completed = {1: 0, 3: 0, 5: 0}
        if self.type5_dipper_failed is None:
            self.type5_dipper_failed = {1: 0, 3: 0, 5: 0}
        if self.type5_prompt_completed is None:
            self.type5_prompt_completed = {1: 0, 3: 0, 5: 0}
        if self.type5_prompt_failed is None:
            self.type5_prompt_failed = {1: 0, 3: 0, 5: 0}

class DataStructureManager:
    """Manages the comprehensive output data structure."""
    
    @staticmethod
    def get_expected_columns() -> List[str]:
        """Get the complete list of expected output columns."""
        return [
            # Base columns (should already exist)
            "idx",
            "dataset_source", 
            "human_original_text",
            "human_paraphrased_text",
            
            # Type 2 generated columns
            "llm_generated_text",
            "llm_generated_text_method",  # sentence_completion, question_answer
            
            # Type 4 paraphrased original columns
            "llm_paraphrased_original_text_dipper",
            "llm_paraphrased_original_text_prompt",
            "llm_paraphrased_original_text_llama",
            
            # Type 5 paraphrased generated columns (DIPPER)
            "llm_paraphrased_generated_text_dipper_iter1",
            "llm_paraphrased_generated_text_dipper_iter3", 
            "llm_paraphrased_generated_text_dipper_iter5",
            
            # Type 5 paraphrased generated columns (Prompt-based)
            "llm_paraphrased_generated_text_prompt_iter1",
            "llm_paraphrased_generated_text_prompt_iter3",
            "llm_paraphrased_generated_text_prompt_iter5",
        ]
    
    @staticmethod
    def initialize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Initialize DataFrame with all required columns."""
        expected_cols = DataStructureManager.get_expected_columns()
        
        # Add missing columns
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None
        
        # Ensure correct column order
        df = df[expected_cols + [col for col in df.columns if col not in expected_cols]]
        
        logger.info(f"Initialized DataFrame with {len(expected_cols)} expected columns")
        return df
    
    @staticmethod
    def validate_input_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate that input data has required base columns."""
        required_base_cols = [
            "idx", "dataset_source", "human_original_text", "human_paraphrased_text"
        ]
        
        missing_cols = [col for col in required_base_cols if col not in df.columns]
        
        if missing_cols:
            return False, missing_cols
        
        # Check for empty required columns
        empty_cols = []
        for col in required_base_cols:
            if df[col].isna().all() or (df[col] == '').all():
                empty_cols.append(f"{col} (all empty)")
        
        if empty_cols:
            return False, empty_cols
        
        return True, []

class DependencyManager:
    """Manages dependencies between generation types."""
    
    @staticmethod
    def check_type2_dependency(df: pd.DataFrame) -> Tuple[bool, str]:
        """Check if Type 2 data is available."""
        if 'llm_generated_text' not in df.columns:
            return False, "Missing 'llm_generated_text' column"
        
        available_count = df['llm_generated_text'].notna().sum()
        total_count = len(df)
        
        if available_count == 0:
            return False, "No Type 2 data available"
        
        return True, f"Type 2 data available for {available_count}/{total_count} samples"
    
    @staticmethod
    def get_execution_plan(options: GenerationOptions) -> Tuple[List[GenerationType], Dict[GenerationType, List[str]]]:
        """
        Create execution plan based on options and dependencies.
        
        Returns:
            Tuple of (execution_order, dependency_warnings)
        """
        execution_order = []
        warnings = {}
        
        # Type 2 has no dependencies
        if options.type2:
            execution_order.append(GenerationType.TYPE2)
        
        # Type 4 depends on Type 1 (always available)
        if options.type4:
            execution_order.append(GenerationType.TYPE4)
        
        # Type 5 depends on Type 2
        if options.type5:
            if not options.type2:
                warnings[GenerationType.TYPE5] = [
                    "Type 5 generation requires Type 2 data. "
                    "Type 2 will be automatically generated if not present in input data."
                ]
            execution_order.append(GenerationType.TYPE5)
        
        return execution_order, warnings

class PADBenOrchestrator:
    """Main orchestrator for PADBen text generation pipeline."""
    
    def __init__(self, config: Optional[GenerationConfig] = None):
        """Initialize the orchestrator."""
        self.config = config or DEFAULT_CONFIG
        
        # Validate configuration
        config_results = validate_all_configs(self.config)
        if not all(config_results.values()):
            raise ValueError(f"Configuration validation failed: {config_results}")
        
        # Initialize generators (lazy loading)
        self.type2_generator = None
        self.type4_generator = None
        self.type5_generator = None
        
        # Status tracking
        self.status = GenerationStatus()
        
        # Data structure manager
        self.data_manager = DataStructureManager()
        
        logger.info("PADBen Orchestrator initialized successfully")
        logger.info(f"Configuration summary: {get_config_summary(self.config)}")
    
    def _initialize_type2_generator(self):
        """Initialize Type 2 generator (lazy loading)."""
        if self.type2_generator is None:
            try:
                # Use production environment mode for orchestrator
                self.type2_generator = EnhancedType2Generator(
                    self.config.type2_config, 
                    environment_mode=EnvironmentMode.PRODUCTION
                )
                logger.info("Type 2 generator initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Type 2 generator: {str(e)}")
                raise
    
    def _initialize_type4_generator(self):
        """Initialize Type 4 generator (lazy loading)."""
        if self.type4_generator is None:
            try:
                self.type4_generator = Type4Generator(self.config.type4_config)
                logger.info("Type 4 generator initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Type 4 generator: {str(e)}")
                raise
    
    def _initialize_type5_generator(self):
        """Initialize Type 5 generator (lazy loading)."""
        if self.type5_generator is None:
            try:
                self.type5_generator = Type5Generator(self.config.type5_config)
                logger.info("Type 5 generator initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Type 5 generator: {str(e)}")
                raise
    
    async def generate_type2(self, df: pd.DataFrame, options: GenerationOptions) -> pd.DataFrame:
        """Generate Type 2 (LLM-generated) text."""
        logger.info("=" * 50)
        logger.info("STARTING TYPE 2 GENERATION")
        logger.info("=" * 50)
        
        # Initialize generator
        self._initialize_type2_generator()
        
        # Convert method option
        method_map = {
            "sentence_completion": Type2Method.SENTENCE_COMPLETION,
            "question_answer": Type2Method.QUESTION_ANSWER,
            "auto": Type2Method.AUTO
        }
        method = method_map.get(options.type2_method, Type2Method.AUTO)
        
        # Filter samples that need Type 2 generation
        needs_type2 = df['llm_generated_text'].isna() | (df['llm_generated_text'] == '')
        target_df = df[needs_type2].copy()
        
        if target_df.empty:
            logger.info("No samples need Type 2 generation")
            return df
        
        logger.info(f"Generating Type 2 text for {len(target_df)} samples using method: {method.value}")
        
        # Get output directory with timestamp structure for production
        base_output_dir = self.config.output_config.output_dir
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        timestamped_output_dir = f"{base_output_dir}/{timestamp}"
        
        # Generate Type 2 text
        result_df = await self.type2_generator.generate_for_dataset(
            df, method, output_dir=base_output_dir
        )
        
        # Update method information
        generated_mask = result_df['llm_generated_text'].notna() & (result_df['llm_generated_text'] != '')
        result_df.loc[generated_mask, 'llm_generated_text_method'] = method.value
        
        # Update status
        completed = result_df['llm_generated_text'].notna().sum() - df['llm_generated_text'].notna().sum()
        self.status.type2_completed += completed
        self.status.type2_failed += len(target_df) - completed
        
        logger.info(f"Type 2 generation completed: {completed} successful, {len(target_df) - completed} failed")
        return result_df
    
    async def generate_type4(self, df: pd.DataFrame, options: GenerationOptions) -> pd.DataFrame:
        """Generate Type 4 (LLM-paraphrased original) text."""
        logger.info("=" * 50)
        logger.info("STARTING TYPE 4 GENERATION") 
        logger.info("=" * 50)
        
        # Initialize generator
        self._initialize_type4_generator()
        
        result_df = df.copy()
        
        # Generate DIPPER-based paraphrases
        if options.type4_method_dipper:
            logger.info("Generating Type 4 with DIPPER method...")
            
            # Filter samples that need DIPPER paraphrasing
            needs_dipper = (df['llm_paraphrased_original_text_dipper'].isna() | 
                           (df['llm_paraphrased_original_text_dipper'] == ''))
            
            if needs_dipper.sum() > 0:
                dipper_result = await self.type4_generator.generate_for_dataset(
                    result_df, 
                    method=Type4ParaphraseMethod.DIPPER,
                    output_dir=self.config.output_config.output_dir
                )
                
                # Map results to specific DIPPER column
                new_dipper = dipper_result['llm_paraphrased_original_text'].notna() & (dipper_result['llm_paraphrased_original_text'] != '')
                result_df.loc[new_dipper, 'llm_paraphrased_original_text_dipper'] = dipper_result.loc[new_dipper, 'llm_paraphrased_original_text']
                
                # Update status
                completed = new_dipper.sum()
                self.status.type4_dipper_completed += completed
                self.status.type4_dipper_failed += needs_dipper.sum() - completed
                
                logger.info(f"DIPPER Type 4 completed: {completed} successful")
        
        # Generate prompt-based paraphrases
        if options.type4_method_prompt:
            logger.info("Generating Type 4 with prompt-based method...")
            
            # Filter samples that need prompt-based paraphrasing
            needs_prompt = (df['llm_paraphrased_original_text_prompt'].isna() | 
                           (df['llm_paraphrased_original_text_prompt'] == ''))
            
            if needs_prompt.sum() > 0:
                prompt_result = await self.type4_generator.generate_for_dataset(
                    result_df,
                    method=Type4ParaphraseMethod.PROMPT_BASED,
                    output_dir=self.config.output_config.output_dir
                )
                
                # Map results to specific prompt column
                new_prompt = prompt_result['llm_paraphrased_original_text'].notna() & (prompt_result['llm_paraphrased_original_text'] != '')
                result_df.loc[new_prompt, 'llm_paraphrased_original_text_prompt'] = prompt_result.loc[new_prompt, 'llm_paraphrased_original_text']
                
                # Update status
                completed = new_prompt.sum()
                self.status.type4_prompt_completed += completed
                self.status.type4_prompt_failed += needs_prompt.sum() - completed
                
                logger.info(f"Prompt-based Type 4 completed: {completed} successful")
        
        # Generate Llama-based paraphrases
        if options.type4_method_llama:
            logger.info("Generating Type 4 with Llama method...")
            
            # Filter samples that need Llama paraphrasing
            needs_llama = (df['llm_paraphrased_original_text_llama'].isna() | 
                        (df['llm_paraphrased_original_text_llama'] == ''))
            
            if needs_llama.sum() > 0:
                llama_result = await self.type4_generator.generate_for_dataset(
                    result_df,
                    method=Type4ParaphraseMethod.LLAMA,
                    output_dir=self.config.output_config.output_dir
                )
                
                # Map results to specific Llama column
                new_llama = llama_result['llm_paraphrased_original_text'].notna() & (llama_result['llm_paraphrased_original_text'] != '')
                result_df.loc[new_llama, 'llm_paraphrased_original_text_llama'] = llama_result.loc[new_llama, 'llm_paraphrased_original_text']
                
                # Update status
                completed = new_llama.sum()
                self.status.type4_llama_completed += completed
                self.status.type4_llama_failed += needs_llama.sum() - completed
                
                logger.info(f"Llama Type 4 completed: {completed} successful")
        
        return result_df
    
    async def generate_type5(self, df: pd.DataFrame, options: GenerationOptions) -> pd.DataFrame:
        """Generate Type 5 (LLM-paraphrased generated) text."""
        logger.info("=" * 50)
        logger.info("STARTING TYPE 5 GENERATION")
        logger.info("=" * 50)
        
        # Check Type 2 dependency
        has_type2, type2_msg = DependencyManager.check_type2_dependency(df)
        if not has_type2:
            logger.warning(f"Type 2 dependency check failed: {type2_msg}")
            if not options.type2:
                logger.info("Auto-generating Type 2 data for Type 5 dependency...")
                df = await self.generate_type2(df, GenerationOptions(type2=True, type2_method="auto"))
        
        # Initialize generator
        self._initialize_type5_generator()
        
        result_df = df.copy()
        iterations_to_process = options.type5_iterations or [1, 3, 5]
        
        # Generate DIPPER-based paraphrases
        if options.type5_method_dipper:
            for iteration_count in iterations_to_process:
                logger.info(f"Generating Type 5 DIPPER with {iteration_count} iterations...")
                
                column_name = f"llm_paraphrased_generated_text_dipper_iter{iteration_count}"
                needs_generation = (df[column_name].isna() | (df[column_name] == ''))
                
                if needs_generation.sum() > 0:
                    iteration_level = IterationLevel(iteration_count)
                    dipper_result = await self.type5_generator.generate_for_dataset(
                        result_df,
                        method=Type4ParaphraseMethod.DIPPER,
                        iteration=iteration_level,
                        output_dir=self.config.output_config.output_dir
                    )
                    
                    # Map results to specific column
                    new_results = dipper_result['llm_paraphrased_generated_text'].notna() & (dipper_result['llm_paraphrased_generated_text'] != '')
                    result_df.loc[new_results, column_name] = dipper_result.loc[new_results, 'llm_paraphrased_generated_text']
                    
                    # Update status
                    completed = new_results.sum()
                    self.status.type5_dipper_completed[iteration_count] += completed
                    self.status.type5_dipper_failed[iteration_count] += needs_generation.sum() - completed
                    
                    logger.info(f"DIPPER Type 5 ({iteration_count} iter) completed: {completed} successful")
        
        # Generate prompt-based paraphrases
        if options.type5_method_prompt:
            for iteration_count in iterations_to_process:
                logger.info(f"Generating Type 5 prompt-based with {iteration_count} iterations...")
                
                column_name = f"llm_paraphrased_generated_text_prompt_iter{iteration_count}"
                needs_generation = (df[column_name].isna() | (df[column_name] == ''))
                
                if needs_generation.sum() > 0:
                    iteration_level = IterationLevel(iteration_count)
                    prompt_result = await self.type5_generator.generate_for_dataset(
                        result_df,
                        method=Type4ParaphraseMethod.PROMPT_BASED,
                        iteration=iteration_level,
                        output_dir=self.config.output_config.output_dir
                    )
                    
                    # Map results to specific column
                    new_results = prompt_result['llm_paraphrased_generated_text'].notna() & (prompt_result['llm_paraphrased_generated_text'] != '')
                    result_df.loc[new_results, column_name] = prompt_result.loc[new_results, 'llm_paraphrased_generated_text']
                    
                    # Update status
                    completed = new_results.sum()
                    self.status.type5_prompt_completed[iteration_count] += completed
                    self.status.type5_prompt_failed[iteration_count] += needs_generation.sum() - completed
                    
                    logger.info(f"Prompt-based Type 5 ({iteration_count} iter) completed: {completed} successful")
        
        return result_df
    
    async def run_generation_pipeline(self, 
                                     input_file: str, 
                                     options: GenerationOptions,
                                     output_dir: Optional[str] = None) -> Tuple[pd.DataFrame, GenerationStatus]:
        """
        Run the complete generation pipeline.
        
        Args:
            input_file: Path to input data file (CSV or JSON)
            options: Generation options specifying what to generate
            output_dir: Output directory (optional, uses config default)
            
        Returns:
            Tuple of (final_dataframe, generation_status)
        """
        self.status.start_time = datetime.now()
        output_dir = output_dir or self.config.output_config.output_dir
        
        # Set up timestamped output directory for production
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        timestamped_output_dir = f"{output_dir}/{timestamp}"
        
        logger.info("=" * 60)
        logger.info("STARTING PADBEN GENERATION PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Production output directory: {timestamped_output_dir}")
        
        # Load and validate input data
        logger.info(f"Loading input data from: {input_file}")
        input_path = Path(input_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        if input_path.suffix == '.csv':
            df = pd.read_csv(input_file)
        elif input_path.suffix == '.json':
            df = pd.read_json(input_file)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
        
        logger.info(f"Loaded {len(df)} samples")
        self.status.total_samples = len(df)
        
        # Validate input data structure
        is_valid, issues = self.data_manager.validate_input_data(df)
        if not is_valid:
            raise ValueError(f"Input data validation failed: {issues}")
        
        # Initialize comprehensive data structure
        df = self.data_manager.initialize_dataframe(df)
        
        # Get execution plan
        execution_order, warnings = DependencyManager.get_execution_plan(options)
        
        if warnings:
            for gen_type, warning_list in warnings.items():
                for warning in warning_list:
                    logger.warning(f"{gen_type.value}: {warning}")
        
        logger.info(f"Execution plan: {[gt.value for gt in execution_order]}")
        
        # Execute generation pipeline
        result_df = df.copy()
        
        for generation_type in execution_order:
            try:
                if generation_type == GenerationType.TYPE2:
                    result_df = await self.generate_type2(result_df, options)
                elif generation_type == GenerationType.TYPE4:
                    result_df = await self.generate_type4(result_df, options)
                elif generation_type == GenerationType.TYPE5:
                    result_df = await self.generate_type5(result_df, options)
                
                # Save intermediate results if configured
                if options.save_intermediate:
                    await self._save_intermediate_results(result_df, generation_type.value, timestamped_output_dir)
                    
            except Exception as e:
                logger.error(f"Failed during {generation_type.value} generation: {str(e)}")
                raise
        
        self.status.end_time = datetime.now()
        
        # Save final results
        await self._save_final_results(result_df, options, timestamped_output_dir)
        
        # Log final summary
        self._log_pipeline_summary()
        
        return result_df, self.status
    
    async def _save_intermediate_results(self, df: pd.DataFrame, stage: str, output_dir: str):
        """Save intermediate results after each generation stage."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save CSV
            csv_file = output_path / f"padben_after_{stage}_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            
            logger.info(f"Saved intermediate results after {stage}: {csv_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save intermediate results: {str(e)}")
    
    async def _save_final_results(self, df: pd.DataFrame, options: GenerationOptions, output_dir: str):
        """Save final pipeline results."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save CSV
            csv_file = output_path / f"padben_complete_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            
            # Save JSON
            json_file = output_path / f"padben_complete_{timestamp}.json"
            df.to_json(json_file, orient='records', indent=2)
            
            # Save metadata if requested
            if options.include_metadata:
                metadata_file = output_path / f"padben_pipeline_metadata_{timestamp}.json"
                metadata = {
                    "generation_options": asdict(options),
                    "generation_status": asdict(self.status),
                    "configuration_summary": get_config_summary(self.config),
                    "output_structure": {
                        "total_columns": len(df.columns),
                        "expected_columns": self.data_manager.get_expected_columns(),
                        "data_summary": {
                            "total_samples": len(df),
                            "type2_coverage": df['llm_generated_text'].notna().sum(),
                            "type4_dipper_coverage": df['llm_paraphrased_original_text_dipper'].notna().sum(),
                            "type4_prompt_coverage": df['llm_paraphrased_original_text_prompt'].notna().sum(),
                            "type4_llama_coverage": df['llm_paraphrased_original_text_llama'].notna().sum(),
                            "type5_coverage": {
                                "dipper_iter1": df['llm_paraphrased_generated_text_dipper_iter1'].notna().sum(),
                                "dipper_iter3": df['llm_paraphrased_generated_text_dipper_iter3'].notna().sum(),
                                "dipper_iter5": df['llm_paraphrased_generated_text_dipper_iter5'].notna().sum(),
                                "prompt_iter1": df['llm_paraphrased_generated_text_prompt_iter1'].notna().sum(),
                                "prompt_iter3": df['llm_paraphrased_generated_text_prompt_iter3'].notna().sum(),
                                "prompt_iter5": df['llm_paraphrased_generated_text_prompt_iter5'].notna().sum(),
                            }
                        }
                    }
                }
                
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info("=" * 60)
            logger.info("FINAL RESULTS SAVED")
            logger.info("=" * 60)
            logger.info(f"CSV: {csv_file}")
            logger.info(f"JSON: {json_file}")
            if options.include_metadata:
                logger.info(f"Metadata: {metadata_file}")
            
        except Exception as e:
            logger.error(f"Failed to save final results: {str(e)}")
    
    def _log_pipeline_summary(self):
        """Log comprehensive pipeline summary."""
        status = self.status
        total_time = (status.end_time - status.start_time).total_seconds() if status.end_time and status.start_time else 0
        
        logger.info("=" * 60)
        logger.info("PADBEN GENERATION PIPELINE COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total samples processed: {status.total_samples}")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        logger.info("")
        
        # Type 2 summary
        if status.type2_completed + status.type2_failed > 0:
            logger.info("Type 2 (LLM-generated text):")
            logger.info(f"  Completed: {status.type2_completed}")
            logger.info(f"  Failed: {status.type2_failed}")
            logger.info(f"  Success rate: {status.type2_completed / (status.type2_completed + status.type2_failed) * 100:.1f}%")
        
        # Type 4 summary
        type4_total = (status.type4_dipper_completed + status.type4_dipper_failed + 
                      status.type4_prompt_completed + status.type4_prompt_failed +
                      status.type4_llama_completed + status.type4_llama_failed)
        if type4_total > 0:
            logger.info("Type 4 (LLM-paraphrased original text):")
            logger.info(f"  DIPPER - Completed: {status.type4_dipper_completed}, Failed: {status.type4_dipper_failed}")
            logger.info(f"  Prompt - Completed: {status.type4_prompt_completed}, Failed: {status.type4_prompt_failed}")
            logger.info(f"  Llama - Completed: {status.type4_llama_completed}, Failed: {status.type4_llama_failed}")
        
        # Type 5 summary
        type5_total = sum(status.type5_dipper_completed.values()) + sum(status.type5_dipper_failed.values()) + \
                     sum(status.type5_prompt_completed.values()) + sum(status.type5_prompt_failed.values())
        if type5_total > 0:
            logger.info("Type 5 (LLM-paraphrased generated text):")
            for iterations in [1, 3, 5]:
                dipper_c = status.type5_dipper_completed[iterations]
                dipper_f = status.type5_dipper_failed[iterations]
                prompt_c = status.type5_prompt_completed[iterations]
                prompt_f = status.type5_prompt_failed[iterations]
                
                if dipper_c + dipper_f + prompt_c + prompt_f > 0:
                    logger.info(f"  {iterations} iterations - DIPPER: {dipper_c}/{dipper_c+dipper_f}, Prompt: {prompt_c}/{prompt_c+prompt_f}")
        
        logger.info("=" * 60)

def main():
    """Main function for orchestrator CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="PADBen Generation Pipeline Orchestrator")
    
    # Input/Output
    parser.add_argument("--input", required=True, help="Input data file (CSV or JSON)")
    parser.add_argument("--output-dir", default="data/generated", help="Output directory")
    
    # Generation type flags
    parser.add_argument("--type2", action="store_true", help="Generate Type 2 (LLM-generated text)")
    parser.add_argument("--type4", action="store_true", help="Generate Type 4 (LLM-paraphrased original)")
    parser.add_argument("--type5", action="store_true", help="Generate Type 5 (LLM-paraphrased generated)")
    
    # Type 2 options
    parser.add_argument("--type2-method", choices=["sentence_completion", "question_answer", "auto"], 
                       default="auto", help="Type 2 generation method")
    
    # Type 4 options
    parser.add_argument("--type4-dipper", action="store_true", default=True, help="Generate Type 4 with DIPPER")
    parser.add_argument("--type4-prompt", action="store_true", default=True, help="Generate Type 4 with prompt-based")
    parser.add_argument("--type4-llama", action="store_true", default=True, help="Generate Type 4 with Llama")
    parser.add_argument("--no-type4-dipper", action="store_false", dest="type4_dipper", help="Skip DIPPER for Type 4")
    parser.add_argument("--no-type4-prompt", action="store_false", dest="type4_prompt", help="Skip prompt-based for Type 4")
    parser.add_argument("--no-type4-llama", action="store_false", dest="type4_llama", help="Skip Llama for Type 4")
    
    # Type 5 options
    parser.add_argument("--type5-dipper", action="store_true", default=True, help="Generate Type 5 with DIPPER")
    parser.add_argument("--type5-prompt", action="store_true", default=True, help="Generate Type 5 with prompt-based")
    parser.add_argument("--no-type5-dipper", action="store_false", dest="type5_dipper", help="Skip DIPPER for Type 5")
    parser.add_argument("--no-type5-prompt", action="store_false", dest="type5_prompt", help="Skip prompt-based for Type 5")
    parser.add_argument("--type5-iterations", nargs="+", type=int, choices=[1, 3, 5], 
                       default=[1, 3, 5], help="Type 5 iteration levels")
    
    # Output options
    parser.add_argument("--no-intermediate", action="store_false", dest="save_intermediate", 
                       help="Don't save intermediate results")
    parser.add_argument("--no-metadata", action="store_false", dest="include_metadata",
                       help="Don't include metadata in output")
    
    args = parser.parse_args()
    
    async def run_pipeline():
        """Run the generation pipeline."""
        # Create generation options
        options = GenerationOptions(
            type2=args.type2,
            type4=args.type4,
            type5=args.type5,
            type2_method=args.type2_method,
            type4_method_dipper=args.type4_dipper,
            type4_method_prompt=args.type4_prompt,
            type4_method_llama=args.type4_llama,
            type5_method_dipper=args.type5_dipper,
            type5_method_prompt=args.type5_prompt,
            type5_iterations=args.type5_iterations,
            save_intermediate=args.save_intermediate,
            include_metadata=args.include_metadata
        )
        
        # Validate that at least one generation type is selected
        if not any([options.type2, options.type4, options.type5]):
            logger.error("At least one generation type must be selected (--type2, --type4, or --type5)")
            return
        
        logger.info(f"Generation options: {asdict(options)}")
        
        # Initialize orchestrator
        try:
            orchestrator = PADBenOrchestrator()
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {str(e)}")
            return
        
        # Run pipeline
        try:
            final_df, final_status = await orchestrator.run_generation_pipeline(
                args.input, options, args.output_dir
            )
            
            logger.info("Pipeline completed successfully!")
            logger.info(f"Final dataset shape: {final_df.shape}")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    # Run the pipeline
    asyncio.run(run_pipeline())

if __name__ == "__main__":
    main() 