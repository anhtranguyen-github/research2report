"""Agent configuration module."""

import os
import yaml
from typing import Optional, Dict, Any

def get_prompt_template(agent_type: str) -> Optional[Dict[str, Any]]:
    """
    Load a prompt template from a configuration file.
    
    Args:
        agent_type: Type of agent (e.g., "generator", "query_analyzer")
        
    Returns:
        Prompt template as a dictionary or None if not found
    """
    config_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(config_dir, f"{agent_type}.yaml")
    
    try:
        if os.path.exists(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                if config and "prompt" in config:
                    return config["prompt"]
                return None
        else:
            print(f"Configuration file not found: {config_file}")
            return None
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None

__all__ = ["get_prompt_template"]
