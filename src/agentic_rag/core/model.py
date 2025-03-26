"""Model configuration for the Agentic RAG system."""

import os
from typing import Dict, Any, List, Optional

from langchain_ollama import OllamaLLM
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from src.agentic_rag.config.models import get_model_config, get_task_specific_config

def get_ollama_llm(
    model_name: str = "qwen2.5",
    task_type: Optional[str] = None,
    callback_manager: Optional[CallbackManager] = None,
    **kwargs: Any
) -> OllamaLLM:
    """
    Initialize and return an OllamaLLM instance.
    
    Args:
        model_name: The name of the model to use
        task_type: Optional task type for specific configurations
        callback_manager: Optional callback manager
        **kwargs: Additional keyword arguments to pass to OllamaLLM
        
    Returns:
        An initialized OllamaLLM instance
    """
    if callback_manager is None:
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    # Get Ollama host from environment
    ollama_host = os.getenv("OLLAMA_HOST")
    if not ollama_host:
        raise ValueError("OLLAMA_HOST environment variable is not set")
    
    # Get model configuration - either task-specific or general
    if task_type:
        config = get_task_specific_config(model_name, task_type)
    else:
        config = get_model_config(model_name)
    
    # Extract parameters from config
    temperature = config.get("temperature", 0.7)
    
    # Merge config with any provided kwargs
    model_kwargs = {
        "temperature": temperature,
        "top_p": config.get("top_p", 0.9),
        #"max_tokens": config.get("max_tokens", 1024),
        #"repetition_penalty": config.get("repetition_penalty", 1.1),
    }
    
    # Override with any explicitly provided kwargs
    model_kwargs.update(kwargs)
    
    # Try to find the exact model or one that starts with the requested name
    try:
        import requests
        response = requests.get(f"{ollama_host}/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            available_models = [model.get("name") for model in models]
            
            # Find exact match first
            if model_name in available_models:
                print(f"Using exact model match: {model_name}")
                actual_model_name = model_name
            else:
                # Find prefix match
                matching_models = [m for m in available_models if m.startswith(model_name)]
                if matching_models:
                    actual_model_name = matching_models[0]
                    print(f"Using model: {actual_model_name} (matched from prefix: {model_name})")
                else:
                    print(f"Warning: Requested model {model_name} not found. Using as provided.")
                    actual_model_name = model_name
        else:
            actual_model_name = model_name
    except Exception as e:
        print(f"Error finding model: {e}. Using provided model name: {model_name}")
        actual_model_name = model_name
    
    return OllamaLLM(
        model=actual_model_name,
        callback_manager=callback_manager,
        base_url=ollama_host,
        **model_kwargs
    ) 