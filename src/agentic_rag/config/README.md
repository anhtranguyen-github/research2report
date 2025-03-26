# Model Configuration System

This directory contains configuration files for the Agentic RAG system's model settings.

## Overview

The model configuration system provides a centralized way to manage settings for different LLM models and different tasks. Instead of hardcoding model parameters throughout the codebase, we use a configuration-driven approach that allows for:

1. Easy addition of new models
2. Task-specific parameter adjustment
3. Default fallbacks when requested models aren't available

## Key Features

- **Multiple Model Support**: Configure and use multiple models with different parameter sets
- **Task-Specific Parameters**: Optimize model parameters based on the task (e.g., lower temperature for query analysis, higher for response generation)
- **Centralized Configuration**: Manage all model settings in one place
- **API Exposure**: Access model configurations through the API for transparency and client-side awareness

## Configuration Files

- `models.py`: Contains the model configurations and utility functions
- `__init__.py`: Exports the configuration components

## Available Models

The system includes configurations for:

- **Qwen Models**: qwen2.5, qwen2.5-14b
- **Llama Models**: llama3, llama3-70b
- **Mistral Models**: mistral, mistral-7b-instruct
- **Mixtral Models**: mixtral
- **CodeLlama Models**: codellama

## Task-Specific Configurations

Task-specific configurations optimize parameters for different components of the RAG system:

- **query_analysis**: Lower temperature for more deterministic query understanding
- **generation**: Higher temperature for more creative and varied responses
- **retrieval_evaluation**: Very low temperature for consistent document relevance assessment

## Usage

### In Code

```python
from src.agentic_rag.config import get_model_config, get_task_specific_config

# Get general config for a model
model_config = get_model_config("qwen2.5")

# Get task-specific config
query_config = get_task_specific_config("qwen2.5", "query_analysis")
generation_config = get_task_specific_config("qwen2.5", "generation")
```

### Via API

- List available models: `GET /models`
- Get model configuration: `GET /models/{model_name}/config`

## Adding New Models

To add a new model, update the `MODEL_CONFIGS` dictionary in `models.py`:

```python
MODEL_CONFIGS = {
    # Existing models...
    
    "new-model": {
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "description": "Description of the new model",
    }
}
```

## Adding New Task Types

To add a new task type, update the `TASK_MODEL_CONFIGS` dictionary in `models.py`:

```python
TASK_MODEL_CONFIGS = {
    # Existing task types...
    
    "new_task": {
        "temperature": 0.5,
        "max_tokens": 1024,
    }
}
``` 