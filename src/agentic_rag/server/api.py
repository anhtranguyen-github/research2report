"""FastAPI server for the Agentic RAG system."""

import os
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from src.agentic_rag.utils.document_processing import load_and_split_documents
from src.agentic_rag.core.workflow import AgenticRAGWorkflow
from src.agentic_rag.utils.health_checks import check_all_services, check_ollama_health, check_qdrant_health
from src.agentic_rag.config.models import list_available_models, get_model_config, DEFAULT_MODEL

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Replaces the deprecated on_event handlers.
    """
    # Startup logic
    print("Starting API server...")
    print(f"Ollama URL: {os.getenv('OLLAMA_HOST')}")
    print(f"Qdrant URL: {os.getenv('QDRANT_URL')}")
    
    health_status = check_all_services()
    
    # Check if any service is unhealthy
    unhealthy_services = [
        service for service, status in health_status.items()
        if status["status"] == "unhealthy"
    ]
    
    if unhealthy_services:
        error_msgs = [
            f"{service}: {health_status[service]['message']}" 
            for service in unhealthy_services
        ]
        print(f"Service health check failed: {', '.join(error_msgs)}")
        print("Make sure both Ollama and Qdrant are running and accessible at the configured URLs.")
        
        raise RuntimeError(
            f"Required services are not healthy: {', '.join(unhealthy_services)}. "
            "Please ensure Ollama and Qdrant are running."
        )
    
    print("All services healthy. API server started successfully.")
    
    yield  # This is where the app runs
    
    # Shutdown logic (if any)
    print("Shutting down API server...")

# Initialize the FastAPI app
app = FastAPI(
    title="Agentic RAG API",
    description="API for the Agentic RAG system using LangGraph, Ollama (Qwen2.5), and Qdrant",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Qdrant client with environment variable for URL
qdrant_url = os.getenv("QDRANT_URL")
if not qdrant_url:
    raise ValueError("QDRANT_URL environment variable is not set")

qdrant_host = "localhost"
qdrant_port = 6333

if qdrant_url.startswith("http://"):
    # Extract host and port from URL
    parts = qdrant_url.replace("http://", "").split(":")
    qdrant_host = parts[0]
    qdrant_port = int(parts[1]) if len(parts) > 1 else 6333

# Initialize Qdrant client
qdrant_client = QdrantClient(
    host=qdrant_host,
    port=qdrant_port
)

# Global vector store instance
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=os.getenv("QDRANT_COLLECTION", "agentic_rag"),
    embedding=embeddings
)

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

# API endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Agentic RAG API is running"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check the health of all required services.
    
    Returns:
        Health status of all services
    """
    try:
        health_status = check_all_services()
        
        # Determine overall status
        all_healthy = all(
            status["status"] == "healthy"
            for status in health_status.values()
        )
        
        return HealthResponse(
            status="healthy" if all_healthy else "unhealthy",
            services=health_status
        )
    
    except Exception as e:
        return HealthResponse(
            status="error",
            services={},
            error=str(e)
        )

@app.get("/health/ollama")
async def ollama_health():
    """Check Ollama server health."""
    is_healthy, message = check_ollama_health()
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "message": message
    }

@app.get("/health/qdrant")
async def qdrant_health():
    """Check Qdrant server health."""
    # Use the already-parsed qdrant_host and qdrant_port variables
    is_healthy, message = check_qdrant_health(host=qdrant_host, port=qdrant_port)
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "message": message
    }

@app.get("/models", response_model=ModelsResponse)
async def get_available_models():
    """
    List all available LLM models.
    
    Returns:
        List of available models with descriptions
    """
    models = list_available_models()
    return ModelsResponse(
        models=models,
        default_model=DEFAULT_MODEL
    )

@app.get("/models/{model_name}/config", response_model=ModelConfigResponse)
async def get_model_configuration(model_name: str):
    """
    Get configuration for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model configuration
    """
    config = get_model_config(model_name)
    return ModelConfigResponse(
        model_name=model_name,
        config=config
    )

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process a query through the Agentic RAG system.
    
    Args:
        request: Query request containing the query, model name, and web search flag
        
    Returns:
        Generated response and analysis details
    """
    try:
        # Check service health before processing
        ollama_healthy, ollama_msg = check_ollama_health(request.model_name)
        if not ollama_healthy:
            return QueryResponse(
                response="",
                error=f"Ollama service is not healthy: {ollama_msg}"
            )
        
        # Initialize the workflow
        workflow = AgenticRAGWorkflow(
            vector_store=vector_store,
            model_name=request.model_name,
            use_web_search=request.use_web_search
        )
        
        # Run the workflow
        result = workflow.run(request.query)
        
        if result.get("error"):
            return QueryResponse(
                response="",
                error=result["error"]
            )
        
        # Extract query analysis and relevance assessment for the response
        query_analysis = result.get("query_analysis")
        relevance_assessment = None
        if result.get("retrieval_result"):
            relevance_assessment = result["retrieval_result"].relevance_assessment
        
        return QueryResponse(
            response=result.get("response", "No response generated."),
            query_analysis=query_analysis,
            relevance_assessment=relevance_assessment
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload", response_model=IngestResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200)
):
    """
    Upload and ingest a document into the vector store.
    
    Args:
        background_tasks: FastAPI background tasks
        file: Uploaded file
        chunk_size: Size of document chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        Status message and number of documents ingested
    """
    try:
        # Check Qdrant health before processing
        qdrant_healthy, qdrant_msg = check_qdrant_health(
            host=qdrant_host,
            port=qdrant_port
        )
        if not qdrant_healthy:
            return IngestResponse(
                message="Failed to ingest document",
                num_documents=0,
                error=f"Qdrant service is not healthy: {qdrant_msg}"
            )
        
        # Save the uploaded file temporarily
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Load and split the document
        documents = load_and_split_documents(
            file_paths=[temp_file_path],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Ingest the document chunks into the vector store
        vector_store.add_documents(documents)
        
        # Schedule cleanup of the temporary file
        background_tasks.add_task(os.remove, temp_file_path)
        
        return IngestResponse(
            message=f"Successfully ingested {file.filename}",
            num_documents=len(documents)
        )
    
    except Exception as e:
        # Ensure temp file cleanup even in case of error
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            background_tasks.add_task(os.remove, temp_file_path)
        
        return IngestResponse(
            message="Failed to ingest document",
            num_documents=0,
            error=str(e)
        )

@app.delete("/collection", response_model=Dict[str, str])
async def delete_collection():
    """
    Delete the entire vector store collection.
    
    Returns:
        Status message
    """
    try:
        # Check Qdrant health before processing
        qdrant_healthy, qdrant_msg = check_qdrant_health(
            host=qdrant_host,
            port=qdrant_port
        )
        if not qdrant_healthy:
            raise HTTPException(
                status_code=503,
                detail=f"Qdrant service is not healthy: {qdrant_msg}"
            )
        
        collection_name = os.getenv("QDRANT_COLLECTION", "agentic_rag")
        qdrant_client.delete_collection(collection_name=collection_name)
        return {"message": f"Collection '{collection_name}' deleted successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start():
    """Start the FastAPI server."""
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True
    )

if __name__ == "__main__":
    start() 