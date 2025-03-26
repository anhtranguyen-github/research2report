"""Configuration for LLM models and their parameters."""

from typing import Dict, Any, List, Optional

# Model configuration
MODEL_CONFIGS = {
    # Qwen models
    "qwen2.5": {
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "description": "Qwen 2.5 7B LLM model - good balance of quality and performance",
    },
    "qwen2.5-14b": {
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "description": "Qwen 2.5 14B LLM model - higher quality than 7B but slower",
    },
    
    # Llama models
    "llama3": {
        "temperature": 0.6,
        "max_tokens": 1024,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "description": "Llama 3 8B LLM model - fast with good balance of capabilities",
    },
    "llama3-70b": {
        "temperature": 0.6,
        "max_tokens": 2048,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "description": "Llama 3 70B LLM model - highest quality but most resource intensive",
    },
    
    # Mistral models
    "mistral": {
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "description": "Mistral 7B LLM model - good balance for reasoning tasks",
    },
    "mistral-7b-instruct": {
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "description": "Mistral 7B Instruct LLM model - optimized for following instructions",
    },
    
    # Mixtral models
    "mixtral": {
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "description": "Mixtral 8x7B Mixture of Experts LLM - high quality general purpose model",
    },
    
    # CodeLlama models
    "codellama": {
        "temperature": 0.4,  # Lower temperature for code generation
        "max_tokens": 2048,
        "top_p": 0.95,
        "repetition_penalty": 1.1,
        "description": "CodeLlama 7B LLM model - specialized for code generation",
    },
}

# Default model to use
DEFAULT_MODEL = "qwen2.5"

# LLM parameters configuration per task type
TASK_MODEL_CONFIGS = {
    "query_analysis": {
        "temperature": 0.3,  # Lower temperature for more deterministic outputs
        "max_tokens": 512,   # Shorter responses needed
    },
    "generation": {
        "temperature": 0.7,  # Higher temperature for more creative responses
        "max_tokens": 1536,  # Longer responses possible
    },
    "retrieval_evaluation": {
        "temperature": 0.2,  # Very low temperature for consistent evaluation
        "max_tokens": 768,
    }
}

def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get the configuration for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model configuration dictionary
    """
    # Return default model config if the requested model is not found
    if model_name not in MODEL_CONFIGS:
        return MODEL_CONFIGS[DEFAULT_MODEL]
    
    return MODEL_CONFIGS[model_name]

def get_task_specific_config(model_name: str, task_type: str) -> Dict[str, Any]:
    """
    Get task-specific configuration for a model.
    
    Args:
        model_name: Name of the model
        task_type: Type of task (query_analysis, generation, etc.)
        
    Returns:
        Combined model and task configuration
    """
    # Get the base model config
    model_config = get_model_config(model_name)
    
    # If task type exists, override relevant parameters
    if task_type in TASK_MODEL_CONFIGS:
        task_config = TASK_MODEL_CONFIGS[task_type]
        # Create a new dict with model config values and override with task config
        return {**model_config, **task_config}
    
    # Otherwise return just the model config
    return model_config

def list_available_models() -> List[Dict[str, Any]]:
    """
    List all available models with their descriptions.
    
    Returns:
        List of model information dictionaries
    """
    return [
        {
            "name": name,
            "description": config.get("description", "No description available")
        }
        for name, config in MODEL_CONFIGS.items()
    ] 