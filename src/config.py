import os
import json
from typing import Dict, Any, Optional

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file or use defaults.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    # Default configuration
    default_config = {
        "retriever_config": {
            "use_privacy_protection": True,
            "max_results": 5,
            "embedding_model": "pritamdeka/S-BioBERT-CORD19-STS"
        },
        "generator_config": {
            "model_name": "microsoft/BioGPT-Large",
            "temperature": 0.7,
            "apply_privacy_filtering": True
        },
        "privacy_config": {
            "enabled": True,
            "use_synthetic_data": True,
            "pii_filtering_level": "standard"
        },
        "api_config": {
            "host": "0.0.0.0",
            "port": 8000,
            "enable_cors": True
        }
    }
    
    # If config path is provided and file exists, load it
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                
            # Merge with defaults (loaded config takes precedence)
            for section in default_config:
                if section in loaded_config:
                    default_config[section].update(loaded_config[section])
                
        except Exception as e:
            print(f"Error loading config from {config_path}: {e}")
            print("Using default configuration")
    
    return default_config