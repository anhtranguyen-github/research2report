# Embedding Model Configuration System

This component provides a flexible configuration system for embedding models in the Agentic RAG system.

## Overview

The embedding model configuration system allows for:

1. Easy selection between different embedding models
2. Support for multiple embedding model types (Sentence Transformers, Instructor, OpenAI, Ollama)
3. Automatic collection configuration based on embedding dimensions
4. Runtime switching between embedding models

## Key Features

- **Multiple Model Types**: Support for various embedding model libraries and APIs
- **Dimension-Aware Configuration**: Automatic vector size configuration for Qdrant collections
- **Runtime Model Switching**: Change embedding models without restarting the server
- **Collection Management**: Create new collections optimized for specific embedding models

## Available Embedding Models

The system includes configurations for:

### Sentence Transformers Models
- **all-MiniLM-L6-v2** (384 dimensions): Lightweight general-purpose model
- **all-mpnet-base-v2** (768 dimensions): Higher quality general-purpose model
- **e5-large-v2** (1024 dimensions): High performance model for retrieval
- **multilingual-e5-base** (768 dimensions): Multilingual support for 100+ languages
- **bge-base-en-v1.5** (768 dimensions): BGE model optimized for retrieval
- **bge-small-en-v1.5** (384 dimensions): Smaller BGE model for faster processing

### Instructor Models (Instruction-tuned)
- **instructor-large** (768 dimensions): Instruction-tuned embedding model
- **instructor-xl** (1024 dimensions): Larger instruction-tuned model

### OpenAI Models (requires API key)
- **text-embedding-3-small** (1536 dimensions): OpenAI's smaller embedding model
- **text-embedding-3-large** (3072 dimensions): OpenAI's larger, highest quality model

### Ollama-based Models
- **nomic-embed-text** (768 dimensions): Nomic AI's embedding model via Ollama

## Usage

### In Code

```python
from src.agentic_rag.config import (
    get_embedding_config, 
    create_embedding_model,
    get_collection_config
)

# Get configuration for a specific embedding model
config = get_embedding_config("all-MiniLM-L6-v2")

# Create an embedding model instance
embeddings = create_embedding_model("all-MiniLM-L6-v2")

# Get collection configuration for a specific embedding model
coll_config = get_collection_config("all-MiniLM-L6-v2")
```

### Via API

- List available embedding models: `GET /embeddings`
- Get embedding model configuration: `GET /embeddings/{model_name}/config`
- Change active embedding model: `POST /embeddings/change`
- Upload with specific embedding model: `POST /upload` with `embedding_model` form field

## Adding New Embedding Models

To add a new embedding model, update the `EMBEDDING_CONFIGS` dictionary in `embeddings.py`:

```python
EMBEDDING_CONFIGS = {
    # Existing models...
    
    "new-embedding-model": {
        "model_name": "path/to/model-on-huggingface",
        "dimensions": 768,  # Vector dimensions
        "type": "sentence_transformers",  # Model type
        "description": "Description of the new embedding model",
    }
}
```

## Collection Configuration

The system automatically configures Qdrant collections based on the embedding model's dimensions:

```python
COLLECTION_CONFIGS = {
    "all-MiniLM-L6-v2": {
        "vector_size": 384,
        "distance": "Cosine",
    },
    # Other models...
}
```

For models without explicit collection configurations, the system will use the embedding dimensions to determine the vector size.

## Environment Variables

- `EMBEDDING_MODEL`: Set the default embedding model to use
- `QDRANT_COLLECTION`: Collection name for vector storage
- `OPENAI_API_KEY`: Required for OpenAI embedding models
- `OLLAMA_HOST`: Required for Ollama-based embedding models 