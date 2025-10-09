"""
Centralized Logging Configuration for PADBen Data Generation Pipeline.

Provides comprehensive logging with file/line tracking and processing step visibility.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

class PADBenFormatter(logging.Formatter):
    """Custom formatter that shows file, line, and processing context."""
    
    def __init__(self):
        super().__init__()
        
    def format(self, record):
        # Get file and line information
        filename = Path(record.pathname).name
        
        # Color coding for different log levels
        colors = {
            'DEBUG': '\033[36m',    # Cyan
            'INFO': '\033[32m',     # Green
            'WARNING': '\033[33m',  # Yellow
            'ERROR': '\033[31m',    # Red
            'CRITICAL': '\033[35m'  # Magenta
        }
        reset = '\033[0m'
        
        color = colors.get(record.levelname, '')
        
        # Enhanced format with file:line tracking
        formatted = (
            f"{color}[{record.levelname:8}]{reset} "
            f"{record.asctime} | "
            f"{filename}:{record.lineno:4} | "
            f"{record.funcName:20} | "
            f"{record.getMessage()}"
        )
        
        return formatted

def setup_padben_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    include_file_handler: bool = True
) -> logging.Logger:
    """
    Set up comprehensive logging for PADBen pipeline.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        include_file_handler: Whether to log to file
        
    Returns:
        Configured logger
    """
    # Create main logger
    logger = logging.getLogger('padben')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with custom formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(PADBenFormatter())
    logger.addHandler(console_handler)
    
    # File handler if requested
    if include_file_handler:
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"logs/padben_generation_{timestamp}.log"
        
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always debug level for file
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(pathname)s:%(lineno)d | %(funcName)s | %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

class ProcessingTracker:
    """Track processing steps with detailed context."""
    
    def __init__(self, logger: logging.Logger, process_name: str):
        self.logger = logger
        self.process_name = process_name
        self.current_step = 0
        self.total_steps = 0
        self.current_sample = 0
        self.total_samples = 0
        
    def start_process(self, total_samples: int, total_steps: int = 1):
        """Start a new processing session."""
        self.total_samples = total_samples
        self.total_steps = total_steps
        self.current_sample = 0
        self.current_step = 0
        
        self.logger.info("🚀 " + "=" * 60)
        self.logger.info(f"🚀 STARTING: {self.process_name}")
        self.logger.info(f"🚀 Total samples: {total_samples}")
        self.logger.info(f"🚀 Total steps: {total_steps}")
        self.logger.info("🚀 " + "=" * 60)
    
    def start_step(self, step_name: str, step_number: int = None):
        """Start a processing step."""
        if step_number:
            self.current_step = step_number
        else:
            self.current_step += 1
            
        self.logger.info(f"📋 Step {self.current_step}/{self.total_steps}: {step_name}")
        self.logger.info("-" * 50)
    
    def start_sample(self, sample_idx: int, sample_data: dict = None):
        """Start processing a sample."""
        self.current_sample = sample_idx + 1
        
        progress = (self.current_sample / self.total_samples) * 100
        self.logger.info(f"📄 Sample {self.current_sample}/{self.total_samples} ({progress:.1f}%)")
        
        if sample_data:
            self.logger.debug(f"   Sample data: {sample_data}")
    
    def log_progress(self, message: str, level: str = "info"):
        """Log a progress message with context."""
        prefix = f"   [{self.current_sample}/{self.total_samples}]"
        full_message = f"{prefix} {message}"
        
        getattr(self.logger, level.lower())(full_message)
    
    def complete_sample(self, success: bool, result_summary: str = ""):
        """Complete processing a sample."""
        status = "✅ SUCCESS" if success else "❌ FAILED"
        self.logger.info(f"   Sample {self.current_sample}: {status} {result_summary}")
    
    def complete_step(self, step_name: str, success_count: int, failure_count: int):
        """Complete a processing step."""
        total = success_count + failure_count
        success_rate = (success_count / total * 100) if total > 0 else 0
        
        self.logger.info("-" * 50)
        self.logger.info(f"✅ Step {self.current_step} Complete: {step_name}")
        self.logger.info(f"   Success: {success_count}/{total} ({success_rate:.1f}%)")
        self.logger.info(f"   Failed: {failure_count}/{total}")
        self.logger.info("")
    
    def complete_process(self, final_stats: dict):
        """Complete the entire process."""
        self.logger.info("🏁 " + "=" * 60)
        self.logger.info(f"🏁 COMPLETED: {self.process_name}")
        self.logger.info("🏁 " + "=" * 60)
        
        for key, value in final_stats.items():
            self.logger.info(f"🏁 {key}: {value}")
        
        self.logger.info("🏁 " + "=" * 60)

# Global logger instance
padben_logger = setup_padben_logging() 