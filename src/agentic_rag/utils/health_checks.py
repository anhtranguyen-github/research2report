"""Health check utilities for the Agentic RAG system."""

import requests
import os
from typing import Dict, Tuple

def check_ollama_health(model_name: str = "qwen2.5") -> Tuple[bool, str]:
    """
    Check if Ollama server is running and the specified model is available.
    
    Args:
        model_name: Name of the model to check for
        
    Returns:
        Tuple of (is_healthy, message)
    """
    try:
        # Get Ollama host from environment
        ollama_host = os.getenv("OLLAMA_HOST")
        if not ollama_host:
            return False, "OLLAMA_HOST environment variable is not set"
        
        ollama_host = ollama_host.rstrip("/")
        
        # Check if Ollama server is running
        response = requests.get(f"{ollama_host}/api/tags")
        response.raise_for_status()
        
        # Check if the model exists - more flexible check that looks for model name as a prefix
        models = response.json().get("models", [])
        available_models = [model.get("name") for model in models]
        print(f"Available models: {available_models}")
        
        # Check if any available model starts with the requested model_name
        model_found = any(m.startswith(model_name) for m in available_models)
        
        if not model_found:
            return False, f"Model '{model_name}' not found in Ollama. Available models: {', '.join(available_models)}"
        
        return True, "Ollama server is healthy"
    
    except requests.exceptions.ConnectionError:
        return False, "Ollama server is not running or not accessible"
    except Exception as e:
        return False, f"Error checking Ollama health: {str(e)}"

def check_qdrant_health(host: str = None, port: int = None) -> Tuple[bool, str]:
    """
    Check if Qdrant server is running and accessible.
    
    Args:
        host: Qdrant host (optional, defaults to parsed from QDRANT_URL)
        port: Qdrant port (optional, derived from QDRANT_URL if not provided)
        
    Returns:
        Tuple of (is_healthy, message)
    """
    try:
        # Get Qdrant URL from environment
        if host is None or port is None:
            qdrant_url = os.getenv("QDRANT_URL")
            if not qdrant_url:
                return False, "QDRANT_URL environment variable is not set"
                
            if qdrant_url.startswith("http://"):
                # Extract host and port from URL
                parts = qdrant_url.replace("http://", "").split(":")
                host = host or parts[0]
                port = port or (int(parts[1]) if len(parts) > 1 else 6333)
            else:
                return False, "QDRANT_URL must start with http://"
                
        response = requests.get(f"http://{host}:{port}/healthz")
        response.raise_for_status()
        return True, "Qdrant server is healthy"
    
    except requests.exceptions.ConnectionError:
        return False, "Qdrant server is not running or not accessible"
    except Exception as e:
        return False, f"Error checking Qdrant health: {str(e)}"

def check_all_services() -> Dict[str, Dict[str, str]]:
    """
    Check health of all required services.
    
    Returns:
        Dictionary containing health status for each service
    """
    ollama_healthy, ollama_msg = check_ollama_health()
    qdrant_healthy, qdrant_msg = check_qdrant_health()
    
    return {
        "ollama": {
            "status": "healthy" if ollama_healthy else "unhealthy",
            "message": ollama_msg
        },
        "qdrant": {
            "status": "healthy" if qdrant_healthy else "unhealthy",
            "message": qdrant_msg
        }
    } 