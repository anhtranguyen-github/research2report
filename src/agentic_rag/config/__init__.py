"""Configuration module for the Agentic RAG system."""

from src.agentic_rag.config.models import (
    get_model_config,
    get_task_specific_config,
    list_available_models,
    MODEL_CONFIGS,
    DEFAULT_MODEL
)

from src.agentic_rag.config.embeddings import (
    get_embedding_config,
    get_collection_config,
    list_available_embeddings,
    create_embedding_model,
    EMBEDDING_CONFIGS,
    DEFAULT_EMBEDDING
)

__all__ = [
    # LLM models
    "get_model_config",
    "get_task_specific_config", 
    "list_available_models",
    "MODEL_CONFIGS",
    "DEFAULT_MODEL",
    
    # Embedding models
    "get_embedding_config",
    "get_collection_config",
    "list_available_embeddings",
    "create_embedding_model",
    "EMBEDDING_CONFIGS",
    "DEFAULT_EMBEDDING"
] 