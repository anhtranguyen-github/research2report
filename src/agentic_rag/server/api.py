"""FastAPI server for the Agentic RAG system."""

import os
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import json
import asyncio

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from src.agentic_rag.utils.document_processing import load_and_split_documents
from src.agentic_rag.core.workflow import AgenticRAGWorkflow
from src.agentic_rag.utils.health_checks import check_all_services, check_ollama_health, check_qdrant_health
from src.agentic_rag.config.models import list_available_models, get_model_config, DEFAULT_MODEL
from src.agentic_rag.config.embeddings import (
    list_available_embeddings, 
    get_embedding_config, 
    create_embedding_model,
    DEFAULT_EMBEDDING
)
from .types import (
    QueryRequest,
    QueryResponse,
    IngestRequest,
    IngestResponse,
    HealthResponse,
    ModelsResponse,
    ModelConfigResponse,
    EmbeddingsResponse,
    EmbeddingConfigResponse,
    ChangeEmbeddingRequest,
    ChangeEmbeddingResponse,
    WebSearchResult
)

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
    description="API for the Agentic RAG system using LangGraph, Ollama, and Qdrant",
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

# Global embeddings instance (initialized with default) 
# Will be initialized based on environment or API calls
embedding_model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING)
embeddings = create_embedding_model(embedding_model_name)

# Global vector store instance
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=os.getenv("QDRANT_COLLECTION", "agentic_rag"),
    embedding=embeddings
)

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

@app.get("/embeddings", response_model=EmbeddingsResponse)
async def get_available_embeddings():
    """
    List all available embedding models.
    
    Returns:
        List of available embedding models with descriptions
    """
    models = list_available_embeddings()
    return EmbeddingsResponse(
        models=models,
        default_model=DEFAULT_EMBEDDING,
        current_model=embedding_model_name
    )

@app.get("/embeddings/{model_name}/config", response_model=EmbeddingConfigResponse)
async def get_embedding_configuration(model_name: str):
    """
    Get configuration for a specific embedding model.
    
    Args:
        model_name: Name of the embedding model
        
    Returns:
        Embedding model configuration
    """
    config = get_embedding_config(model_name)
    return EmbeddingConfigResponse(
        model_name=model_name,
        config=config
    )

@app.post("/embeddings/change", response_model=ChangeEmbeddingResponse)
async def change_embedding_model(request: ChangeEmbeddingRequest):
    """
    Change the active embedding model.
    
    Args:
        request: Change embedding model request
        
    Returns:
        Status of the change operation
    """
    global embeddings, vector_store, embedding_model_name
    
    # Store the previous model name
    previous_model = embedding_model_name
    
    # Check if the model exists
    config = get_embedding_config(request.model_name)
    
    try:
        # Create the new embeddings model
        new_embeddings = create_embedding_model(request.model_name)
        
        # Determine the collection name
        collection_name = os.getenv("QDRANT_COLLECTION", "agentic_rag")
        
        if request.create_new_collection:
            if request.collection_name:
                collection_name = request.collection_name
            else:
                collection_name = f"agentic_rag_{request.model_name}"
            
            # Create the collection if it doesn't exist
            if not qdrant_client.collection_exists(collection_name):
                from src.agentic_rag.config.embeddings import get_collection_config
                coll_config = get_collection_config(request.model_name)
                
                qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "size": coll_config["vector_size"],
                        "distance": coll_config["distance"]
                    }
                )
        
        # Update the global variables
        embeddings = new_embeddings
        embedding_model_name = request.model_name
        
        # Create a new vector store with the new embeddings
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=collection_name,
            embedding=embeddings
        )
        
        return ChangeEmbeddingResponse(
            previous_model=previous_model,
            current_model=embedding_model_name,
            collection_name=collection_name,
            dimensions=config.get("dimensions", 0),
            message=f"Successfully changed embedding model to {embedding_model_name}"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to change embedding model: {str(e)}"
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
        
        # Initialize the workflow with the agent manager approach
        workflow = AgenticRAGWorkflow(
            vector_store=vector_store,
            model_name=request.model_name,
            use_web_search=request.use_web_search,
            config_dir="src/agentic_rag/agents/config"
        )
        
        # Run the workflow
        result = workflow.run(request.query)
        
        if result.get("error"):
            return QueryResponse(
                response="",
                error=result["error"]
            )
        
        # Extract results for the response
        return QueryResponse(
            response=result.get("response", "No response generated."),
            query_analysis=result.get("query_analysis"),
            retrieval_result=result.get("retrieval_result"),
            web_search_results=result.get("web_search_result")
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/stream")
async def stream_query(request: QueryRequest):
    """
    Stream a query response from the Agentic RAG system.
    
    Args:
        request: Query request containing the query, model name, and web search flag
        
    Returns:
        Streaming response with token-by-token generation
    """
    async def generate():
        try:
            # Check service health before processing
            ollama_healthy, ollama_msg = check_ollama_health(request.model_name)
            if not ollama_healthy:
                yield f"data: Error: Ollama service is not healthy: {ollama_msg}\n\n"
                return
            
            # Initialize the workflow
            workflow = AgenticRAGWorkflow(
                vector_store=vector_store,
                model_name=request.model_name,
                use_web_search=request.use_web_search,
                config_dir="src/agentic_rag/agents/config"
            )
            
            # Define token queue to manage async stream
            token_queue = asyncio.Queue()
            
            # Create a callback that puts tokens into the queue
            async def stream_callback(token: str):
                await token_queue.put(token)
            
            # Start streaming response in background task
            asyncio.create_task(workflow.stream(request.query, stream_callback))
            
            # Keep yielding tokens until we get a special "DONE" token
            while True:
                token = await token_queue.get()
                if token == "DONE":
                    break
                yield f"data: {token}\n\n"
            
            # End the stream
            yield "data: [DONE]\n\n"
        
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
            
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

@app.post("/upload", response_model=IngestResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    embedding_model: Optional[str] = Form(None)
):
    """
    Upload and ingest a document into the vector store.
    
    Args:
        background_tasks: FastAPI background tasks
        file: Uploaded file
        chunk_size: Size of document chunks
        chunk_overlap: Overlap between chunks
        embedding_model: Optional embedding model to use for this ingestion
        
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
        
        # Use specified embedding model if provided
        local_vector_store = vector_store
        temp_embeddings = None
        
        if embedding_model and embedding_model != embedding_model_name:
            try:
                temp_embeddings = create_embedding_model(embedding_model)
                local_vector_store = QdrantVectorStore(
                    client=qdrant_client,
                    collection_name=os.getenv("QDRANT_COLLECTION", "agentic_rag"),
                    embedding=temp_embeddings
                )
            except Exception as e:
                return IngestResponse(
                    message="Failed to create embedding model",
                    num_documents=0,
                    error=f"Error creating embedding model: {str(e)}"
                )
        
        # Ingest the document chunks into the vector store
        local_vector_store.add_documents(documents)
        
        # Schedule cleanup of the temporary file
        background_tasks.add_task(os.remove, temp_file_path)
        
        return IngestResponse(
            message=f"Successfully ingested {file.filename}" + 
                    (f" using {embedding_model} embeddings" if embedding_model else ""),
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
async def delete_collection(
    collection_name: Optional[str] = Query(None, description="Name of the collection to delete")
):
    """
    Delete a vector store collection.
    
    Args:
        collection_name: Optional name of the collection to delete (uses default if not provided)
        
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
        
        # Use provided collection name or default
        target_collection = collection_name or os.getenv("QDRANT_COLLECTION", "agentic_rag")
        
        # Check if collection exists
        if not qdrant_client.collection_exists(target_collection):
            return {"message": f"Collection '{target_collection}' does not exist"}
        
        # Delete the collection
        qdrant_client.delete_collection(collection_name=target_collection)
        return {"message": f"Collection '{target_collection}' deleted successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections", response_model=List[Dict[str, Any]])
async def list_collections():
    """
    List all available collections in Qdrant.
    
    Returns:
        List of collections with their info
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
        
        # Get collections list
        collections = qdrant_client.get_collections().collections
        
        # Get details for each collection
        result = []
        for collection in collections:
            try:
                info = qdrant_client.get_collection(collection.name)
                result.append({
                    "name": collection.name,
                    "vectors_count": info.vectors_count,
                    "vector_size": info.config.params.vectors.size,
                    "distance": str(info.config.params.vectors.distance),
                    "created_at": collection.created_at.isoformat() if collection.created_at else None
                })
            except:
                # If can't get detailed info, just add the name
                result.append({"name": collection.name})
        
        return result
    
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