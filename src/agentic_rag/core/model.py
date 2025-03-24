"""Model configuration for the Agentic RAG system."""

import os
from typing import Dict, Any, List, Optional

from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def get_ollama_llm(
    model_name: str = "qwen2.5",
    temperature: float = 0.7,
    callback_manager: Optional[CallbackManager] = None,
    **kwargs: Any
) -> Ollama:
    """
    Initialize and return an Ollama LLM instance.
    
    Args:
        model_name: The name of the model to use
        temperature: Controls randomness of outputs (0.0 = deterministic, 1.0 = creative)
        callback_manager: Optional callback manager
        **kwargs: Additional keyword arguments to pass to Ollama
        
    Returns:
        An initialized Ollama LLM instance
    """
    if callback_manager is None:
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    # Get Ollama host from environment
    ollama_host = os.getenv("OLLAMA_HOST")
    if not ollama_host:
        raise ValueError("OLLAMA_HOST environment variable is not set")
    
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
    
    return Ollama(
        model=actual_model_name,
        temperature=temperature,
        callback_manager=callback_manager,
        base_url=ollama_host,
        **kwargs
    ) 