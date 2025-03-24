"""Script to start the Agentic RAG FastAPI server."""

import argparse
import os
import uvicorn

def main():
    """Main entry point for the server script."""
    parser = argparse.ArgumentParser(description="Agentic RAG API Server")
    
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--collection", type=str, default="agentic_rag", help="Qdrant collection name")
    parser.add_argument("--qdrant-host", type=str, default="localhost", help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["HOST"] = args.host
    os.environ["PORT"] = str(args.port)
    os.environ["QDRANT_COLLECTION"] = args.collection
    os.environ["QDRANT_HOST"] = args.qdrant_host
    os.environ["QDRANT_PORT"] = str(args.qdrant_port)
    
    print(f"Starting Agentic RAG API server on http://{args.host}:{args.port}")
    print(f"Using Qdrant collection: {args.collection} at {args.qdrant_host}:{args.qdrant_port}")
    
    # Start the server
    uvicorn.run(
        "src.agentic_rag.server.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main() 