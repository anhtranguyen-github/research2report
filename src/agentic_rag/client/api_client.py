"""Client for the Agentic RAG API."""

import argparse
import requests
import json
import os
from typing import Dict, Any, Optional

class AgenticRAGClient:
    """Client for interacting with the Agentic RAG API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the Agentic RAG client.
        
        Args:
            base_url: Base URL for the API
        """
        self.base_url = base_url
    
    def check_health(self) -> Dict[str, Any]:
        """
        Check the health of all required services.
        
        Returns:
            Dictionary containing health status for all services
        """
        url = f"{self.base_url}/health"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def check_ollama_health(self) -> Dict[str, str]:
        """
        Check Ollama server health.
        
        Returns:
            Dictionary containing Ollama health status
        """
        url = f"{self.base_url}/health/ollama"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def check_qdrant_health(self) -> Dict[str, str]:
        """
        Check Qdrant server health.
        
        Returns:
            Dictionary containing Qdrant health status
        """
        url = f"{self.base_url}/health/qdrant"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def query(
        self,
        query: str,
        model_name: str = "qwen2.5",
        use_web_search: bool = True
    ) -> Dict[str, Any]:
        """
        Send a query to the API.
        
        Args:
            query: The user's query string
            model_name: The LLM model to use
            use_web_search: Whether to use web search
            
        Returns:
            Dictionary containing the response and additional information
        """
        url = f"{self.base_url}/query"
        payload = {
            "query": query,
            "model_name": model_name,
            "use_web_search": use_web_search
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def upload_document(
        self,
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> Dict[str, Any]:
        """
        Upload and ingest a document.
        
        Args:
            file_path: Path to the document file
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            Dictionary containing the status message and number of documents ingested
        """
        url = f"{self.base_url}/upload"
        
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            data = {
                "chunk_size": str(chunk_size),
                "chunk_overlap": str(chunk_overlap)
            }
            
            response = requests.post(url, files=files, data=data)
            response.raise_for_status()
        
        return response.json()
    
    def delete_collection(self) -> Dict[str, str]:
        """
        Delete the entire vector store collection.
        
        Returns:
            Dictionary containing the status message
        """
        url = f"{self.base_url}/collection"
        
        response = requests.delete(url)
        response.raise_for_status()
        
        return response.json()

def main():
    """Main entry point for the client script."""
    parser = argparse.ArgumentParser(description="Agentic RAG API Client")
    
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="API base URL")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Health check commands
    health_parser = subparsers.add_parser("health", help="Check system health")
    health_parser.add_argument("--service", type=str, choices=["all", "ollama", "qdrant"], default="all",
                             help="Service to check (default: all)")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Send a query to the API")
    query_parser.add_argument("text", type=str, help="Query text")
    query_parser.add_argument("--model", type=str, default="qwen2.5", help="LLM model to use")
    query_parser.add_argument("--no-web-search", action="store_true", help="Disable web search")
    
    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload and ingest a document")
    upload_parser.add_argument("file", type=str, help="Path to the document file")
    upload_parser.add_argument("--chunk-size", type=int, default=1000, help="Size of document chunks")
    upload_parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between chunks")
    
    # Delete collection command
    subparsers.add_parser("delete-collection", help="Delete the entire vector store collection")
    
    args = parser.parse_args()
    
    # Initialize the client
    client = AgenticRAGClient(base_url=args.url)
    
    if args.command == "health":
        try:
            if args.service == "all":
                result = client.check_health()
                print("\nOverall System Health:")
                print(f"Status: {result['status']}")
                print("\nServices:")
                for service, status in result["services"].items():
                    print(f"\n{service.title()}:")
                    print(f"  Status: {status['status']}")
                    print(f"  Message: {status['message']}")
            
            elif args.service == "ollama":
                result = client.check_ollama_health()
                print("\nOllama Health:")
                print(f"Status: {result['status']}")
                print(f"Message: {result['message']}")
            
            elif args.service == "qdrant":
                result = client.check_qdrant_health()
                print("\nQdrant Health:")
                print(f"Status: {result['status']}")
                print(f"Message: {result['message']}")
        
        except requests.exceptions.RequestException as e:
            print(f"\nError checking health: {str(e)}")
    
    elif args.command == "query":
        result = client.query(
            query=args.text,
            model_name=args.model,
            use_web_search=not args.no_web_search
        )
        
        print("\nResponse:")
        print(result.get("response", "No response generated."))
        
        if result.get("error"):
            print(f"\nError: {result['error']}")
    
    elif args.command == "upload":
        result = client.upload_document(
            file_path=args.file,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        print(f"\nStatus: {result.get('message')}")
        print(f"Documents ingested: {result.get('num_documents')}")
        
        if result.get("error"):
            print(f"\nError: {result['error']}")
    
    elif args.command == "delete-collection":
        result = client.delete_collection()
        print(f"\nStatus: {result.get('message')}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 