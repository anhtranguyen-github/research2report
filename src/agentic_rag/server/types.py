"""Types for the Agentic RAG server."""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

from src.agentic_rag.config.models import DEFAULT_MODEL

# Input and output models
class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    
    query: str = Field(..., description="The user's query string")
    model_name: str = Field(DEFAULT_MODEL, description="The LLM model to use")
    use_web_search: bool = Field(True, description="Whether to use web search")

class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    
    response: str = Field(..., description="The generated response")
    query_analysis: Optional[Dict[str, Any]] = Field(None, description="Analysis of the user's query")
    relevance_assessment: Optional[str] = Field(None, description="Assessment of the retrieved documents' relevance")
    error: Optional[str] = Field(None, description="Error message, if any")

class IngestRequest(BaseModel):
    """Request model for ingestion endpoint."""
    
    chunk_size: int = Field(1000, description="Size of document chunks")
    chunk_overlap: int = Field(200, description="Overlap between chunks")
    embedding_model: Optional[str] = Field(None, description="Embedding model to use")

class IngestResponse(BaseModel):
    """Response model for ingestion endpoint."""
    
    message: str = Field(..., description="Status message")
    num_documents: int = Field(..., description="Number of document chunks ingested")
    error: Optional[str] = Field(None, description="Error message, if any")

class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    
    status: str = Field(..., description="Overall system status")
    services: Dict[str, Dict[str, str]] = Field(..., description="Status of individual services")
    error: Optional[str] = Field(None, description="Error message, if any")

class ModelInfo(BaseModel):
    """Model for LLM information."""
    
    name: str = Field(..., description="Name of the model")
    description: str = Field(..., description="Description of the model")

class ModelsResponse(BaseModel):
    """Response model for available models endpoint."""
    
    models: List[ModelInfo] = Field(..., description="List of available models")
    default_model: str = Field(..., description="Default model name")

class ModelConfigResponse(BaseModel):
    """Response model for model configuration endpoint."""
    
    model_name: str = Field(..., description="Name of the model")
    config: Dict[str, Any] = Field(..., description="Model configuration")

class EmbeddingInfo(BaseModel):
    """Model for embedding model information."""
    
    name: str = Field(..., description="Name of the embedding model")
    dimensions: int = Field(..., description="Dimensions of the embedding vectors")
    type: str = Field(..., description="Type of embedding model")
    description: str = Field(..., description="Description of the embedding model")

class EmbeddingsResponse(BaseModel):
    """Response model for available embedding models endpoint."""
    
    models: List[EmbeddingInfo] = Field(..., description="List of available embedding models")
    default_model: str = Field(..., description="Default embedding model name")
    current_model: str = Field(..., description="Currently active embedding model")

class EmbeddingConfigResponse(BaseModel):
    """Response model for embedding model configuration endpoint."""
    
    model_name: str = Field(..., description="Name of the embedding model")
    config: Dict[str, Any] = Field(..., description="Embedding model configuration")

class ChangeEmbeddingRequest(BaseModel):
    """Request model for changing the active embedding model."""
    
    model_name: str = Field(..., description="Name of the embedding model to use")
    create_new_collection: bool = Field(False, description="Whether to create a new collection for this embedding model")
    collection_name: Optional[str] = Field(None, description="Name of the collection to use (if creating new)")

class ChangeEmbeddingResponse(BaseModel):
    """Response model for changing the active embedding model."""
    
    previous_model: str = Field(..., description="Previously active model")
    current_model: str = Field(..., description="New active model")
    collection_name: str = Field(..., description="Collection being used")
    dimensions: int = Field(..., description="Dimensions of the embedding vectors")
    message: str = Field(..., description="Status message")
