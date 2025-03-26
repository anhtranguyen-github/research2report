"""Type definitions for the FastAPI server."""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field

class HealthStatus(BaseModel):
    """Health status of a service."""
    status: str
    message: str

class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str
    services: Dict[str, HealthStatus] = {}
    error: Optional[str] = None

class ModelInfo(BaseModel):
    """Information about an LLM model."""
    name: str
    description: str
    parameters: Optional[Dict[str, Any]] = None

class ModelsResponse(BaseModel):
    """Response model for models list endpoint."""
    models: List[ModelInfo]
    default_model: str

class ModelConfigResponse(BaseModel):
    """Response model for model configuration endpoint."""
    model_name: str
    config: Dict[str, Any]

class EmbeddingInfo(BaseModel):
    """Information about an embedding model."""
    name: str
    description: str
    dimensions: int
    provider: str

class EmbeddingsResponse(BaseModel):
    """Response model for embeddings list endpoint."""
    models: List[EmbeddingInfo]
    default_model: str
    current_model: str

class EmbeddingConfigResponse(BaseModel):
    """Response model for embedding configuration endpoint."""
    model_name: str
    config: Dict[str, Any]

class ChangeEmbeddingRequest(BaseModel):
    """Request model for changing the embedding model."""
    model_name: str
    create_new_collection: bool = False
    collection_name: Optional[str] = None

class ChangeEmbeddingResponse(BaseModel):
    """Response model for changing the embedding model."""
    previous_model: str
    current_model: str
    collection_name: str
    dimensions: int
    message: str

class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str
    model_name: str = "qwen2.5"
    use_web_search: bool = True

class WebSearchResult(BaseModel):
    """Web search result model."""
    title: str
    url: str
    snippet: str = Field(alias="content", default="")
    relevance_score: float = Field(alias="score", default=0.0)
    source: str = "web"
    
    class Config:
        populate_by_name = True

class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    response: str
    query_analysis: Optional[Dict[str, Any]] = None
    retrieval_result: Optional[List[Dict[str, Any]]] = None
    web_search_results: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None

class IngestRequest(BaseModel):
    """Request model for ingestion endpoint."""
    file_path: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: Optional[str] = None

class IngestResponse(BaseModel):
    """Response model for ingestion endpoint."""
    message: str
    num_documents: int
    error: Optional[str] = None
