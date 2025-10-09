# data_generation/config/secrets_manager.py (New file)
"""
Secure API Key Management for PADBen Pipeline.

This module provides secure handling of API keys with multiple fallback options
and validation.
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

class SecureAPIKeyManager:
    """Secure API key manager with multiple sources and validation."""
    
    def __init__(self, project_root: Optional[str] = None):
        """Initialize the API key manager."""
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.secrets_file = self.project_root / "secrets.json"
        self.env_file = self.project_root / ".env"
        
        # Load from .env file if available
        self._load_env_file()
    
    def _load_env_file(self) -> None:
        """Load environment variables from .env file."""
        if self.env_file.exists():
            try:
                with open(self.env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            # Only set if not already in environment
                            if key not in os.environ:
                                os.environ[key] = value.strip('"\'')
                logger.info("Loaded environment variables from .env file")
            except Exception as e:
                logger.warning(f"Failed to load .env file: {e}")
    
    def get_api_key(self, key_name: str, required: bool = True) -> Optional[str]:
        """
        Get API key from multiple sources with priority order:
        1. Environment variables
        2. secrets.json file
        3. Interactive prompt (if required)
        """
        # Try environment variable first
        key = os.getenv(key_name)
        if key:
            logger.debug(f"Found {key_name} in environment variables")
            return key
        
        # Try secrets.json file
        key = self._get_from_secrets_file(key_name)
        if key:
            logger.debug(f"Found {key_name} in secrets.json")
            return key
        
        # If required and not found, prompt user
        if required:
            logger.warning(f"API key '{key_name}' not found in environment or secrets file")
            return self._prompt_for_key(key_name)
        
        return None
    
    def _get_from_secrets_file(self, key_name: str) -> Optional[str]:
        """Get API key from secrets.json file."""
        if not self.secrets_file.exists():
            return None
        
        try:
            with open(self.secrets_file, 'r') as f:
                secrets = json.load(f)
                return secrets.get(key_name)
        except Exception as e:
            logger.warning(f"Failed to read secrets.json: {e}")
            return None
    
    def _prompt_for_key(self, key_name: str) -> Optional[str]:
        """Interactively prompt for API key."""
        try:
            import getpass
            key = getpass.getpass(f"Please enter your {key_name}: ")
            
            # Optionally save to environment for this session
            if key:
                os.environ[key_name] = key
                logger.info(f"Set {key_name} for current session")
            
            return key
        except Exception as e:
            logger.error(f"Failed to prompt for {key_name}: {e}")
            return None
    
    def validate_all_keys(self) -> Dict[str, bool]:
        """Validate all required API keys are available."""
        required_keys = {
            "GEMINI_API_KEY": "Gemini 2.5 Pro for generation"
        }
        
        results = {}
        for key_name, description in required_keys.items():
            key = self.get_api_key(key_name, required=False)
            results[key_name] = bool(key)
            
            if key:
                logger.info(f"✅ {description}: Available")
            else:
                logger.error(f"❌ {description}: Missing ({key_name})")
        
        return results
    
    def create_secrets_template(self) -> None:
        """Create a template secrets.json file."""
        template = {
            "GEMINI_API_KEY": "your_gemini_api_key_here",
            "_instructions": {
                "description": "PADBen API Keys Configuration",
                "gemini_setup": "Get your API key from https://aistudio.google.com/",
                "security_note": "Keep this file secure and add to .gitignore"
            }
        }
        
        if not self.secrets_file.exists():
            with open(self.secrets_file, 'w') as f:
                json.dump(template, f, indent=2)
            logger.info(f"Created secrets template: {self.secrets_file}")
        else:
            logger.info(f"Secrets file already exists: {self.secrets_file}")

# Global instance
api_key_manager = SecureAPIKeyManager()

def get_api_key(env_var: str, required: bool = True) -> Optional[str]:
    """Enhanced API key getter with multiple fallback options."""
    return api_key_manager.get_api_key(env_var, required)

def validate_all_api_keys() -> bool:
    """Validate all required API keys are available."""
    results = api_key_manager.validate_all_keys()
    return all(results.values())

def setup_api_keys_interactive() -> None:
    """Interactive setup for API keys."""
    print("🔐 PADBen API Key Setup")
    print("=" * 40)
    
    # Create template if needed
    api_key_manager.create_secrets_template()
    
    # Validate current keys
    results = api_key_manager.validate_all_keys()
    
    if all(results.values()):
        print("✅ All API keys are properly configured!")
    else:
        print("\n❌ Missing API keys. Please set up:")
        for key, available in results.items():
            if not available:
                print(f"   - {key}")
        
        print(f"\nOptions to set up API keys:")
        print(f"1. Add to environment variables")
        print(f"2. Create .env file in project root")
        print(f"3. Edit {api_key_manager.secrets_file}")