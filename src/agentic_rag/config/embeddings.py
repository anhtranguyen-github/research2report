"""Configuration for embedding models and their parameters."""

from typing import Dict, Any, List, Optional

# Embedding model configuration
EMBEDDING_CONFIGS = {
    # Sentence Transformers models (HuggingFace)
    "all-MiniLM-L6-v2": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimensions": 384,
        "type": "sentence_transformers",
        "description": "Lightweight all-purpose embedding model, good balance of quality and performance",
    },
    "all-mpnet-base-v2": {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "dimensions": 768,
        "type": "sentence_transformers",
        "description": "High quality general purpose embedding model, better quality than MiniLM but slower",
    },
    "e5-large-v2": {
        "model_name": "intfloat/e5-large-v2",
        "dimensions": 1024,
        "type": "sentence_transformers",
        "description": "High performance embedding model for retrieval tasks",
    },
    "multilingual-e5-base": {
        "model_name": "intfloat/multilingual-e5-base",
        "dimensions": 768,
        "type": "sentence_transformers",
        "description": "Multilingual embedding model supporting 100+ languages",
    },
    "bge-base-en-v1.5": {
        "model_name": "BAAI/bge-base-en-v1.5",
        "dimensions": 768,
        "type": "sentence_transformers",
        "description": "BGE embedding model optimized for retrieval tasks",
    },
    "bge-small-en-v1.5": {
        "model_name": "BAAI/bge-small-en-v1.5",
        "dimensions": 384,
        "type": "sentence_transformers",
        "description": "Smaller BGE embedding model for faster processing",
    },
    
    # Instructor models
    "instructor-large": {
        "model_name": "hkunlp/instructor-large",
        "dimensions": 768,
        "type": "instructor",
        "description": "Instruction-tuned text embedding model",
        "instruction": "Represent the document for retrieval:",
    },
    "instructor-xl": {
        "model_name": "hkunlp/instructor-xl",
        "dimensions": 1024,
        "type": "instructor",
        "description": "Larger instruction-tuned text embedding model, higher quality",
        "instruction": "Represent the document for retrieval:",
    },
    
    # OpenAI models (if API key available)
    "text-embedding-3-small": {
        "model_name": "text-embedding-3-small",
        "dimensions": 1536,
        "type": "openai",
        "description": "OpenAI's smaller text embedding model (requires API key)",
    },
    "text-embedding-3-large": {
        "model_name": "text-embedding-3-large",
        "dimensions": 3072,
        "type": "openai",
        "description": "OpenAI's larger text embedding model, highest quality (requires API key)",
    },
    
    # Ollama-based embedding models
    "nomic-embed-text": {
        "model_name": "nomic-embed-text",
        "dimensions": 768,
        "type": "ollama",
        "description": "Nomic AI's embedding model running through Ollama",
    },
}

# Default embedding model to use
DEFAULT_EMBEDDING = "all-MiniLM-L6-v2"

# Collection configuration for different embedding models
COLLECTION_CONFIGS = {
    # When creating a new collection, configure based on embedding model
    "all-MiniLM-L6-v2": {
        "vector_size": 384,
        "distance": "Cosine",
    },
    "all-mpnet-base-v2": {
        "vector_size": 768,
        "distance": "Cosine",
    },
    "e5-large-v2": {
        "vector_size": 1024,
        "distance": "Cosine",
    },
    "bge-base-en-v1.5": {
        "vector_size": 768,
        "distance": "Cosine",
    },
    "text-embedding-3-small": {
        "vector_size": 1536,
        "distance": "Cosine",
    },
    "text-embedding-3-large": {
        "vector_size": 3072,
        "distance": "Cosine",
    },
}

def get_embedding_config(embedding_name: str) -> Dict[str, Any]:
    """
    Get the configuration for a specific embedding model.
    
    Args:
        embedding_name: Name of the embedding model
        
    Returns:
        Embedding model configuration dictionary
    """
    # Return default embedding config if the requested model is not found
    if embedding_name not in EMBEDDING_CONFIGS:
        return EMBEDDING_CONFIGS[DEFAULT_EMBEDDING]
    
    return EMBEDDING_CONFIGS[embedding_name]

def get_collection_config(embedding_name: str) -> Dict[str, Any]:
    """
    Get collection configuration for a specific embedding model.
    
    Args:
        embedding_name: Name of the embedding model
        
    Returns:
        Collection configuration dictionary for the embedding model
    """
    # Use the embedding dimensions to determine vector size if not in collection configs
    if embedding_name not in COLLECTION_CONFIGS:
        embedding_config = get_embedding_config(embedding_name)
        dimensions = embedding_config.get("dimensions", 384)
        return {
            "vector_size": dimensions,
            "distance": "Cosine",
        }
    
    return COLLECTION_CONFIGS[embedding_name]

def list_available_embeddings() -> List[Dict[str, Any]]:
    """
    List all available embedding models with their descriptions.
    
    Returns:
        List of embedding model information dictionaries
    """
    return [
        {
            "name": name,
            "dimensions": config.get("dimensions", 0),
            "type": config.get("type", "unknown"),
            "description": config.get("description", "No description available")
        }
        for name, config in EMBEDDING_CONFIGS.items()
    ]

def create_embedding_model(embedding_name: str) -> Any:
    """
    Create an embedding model instance based on the configuration.
    
    Args:
        embedding_name: Name of the embedding model
        
    Returns:
        An embedding model instance
    """
    config = get_embedding_config(embedding_name)
    model_type = config.get("type", "sentence_transformers")
    model_name = config.get("model_name")
    
    if model_type == "sentence_transformers":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=model_name)
    
    elif model_type == "instructor":
        from langchain_huggingface import HuggingFaceInstructEmbeddings
        instruction = config.get("instruction", "Represent the document for retrieval:")
        return HuggingFaceInstructEmbeddings(
            model_name=model_name,
            embed_instruction=instruction
        )
    
    elif model_type == "openai":
        import os
        from langchain_openai import OpenAIEmbeddings
        
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        return OpenAIEmbeddings(model=model_name)
    
    elif model_type == "ollama":
        import os
        from langchain_community.embeddings.ollama import OllamaEmbeddings
        
        # Get Ollama host from environment
        ollama_host = os.getenv("OLLAMA_HOST")
        if not ollama_host:
            raise ValueError("OLLAMA_HOST environment variable is not set")
        
        return OllamaEmbeddings(
            model=model_name,
            base_url=ollama_host
        )
    
    else:
        # Default to sentence transformers
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=model_name) 