"""Main entry point for the Agentic RAG system."""

import os
import argparse
from typing import List, Dict, Any, Optional

from src.agentic_rag.vector_store.qdrant_store import QdrantVectorStore
from src.agentic_rag.utils.document_processing import load_and_split_documents
from src.agentic_rag.core.workflow import AgenticRAGWorkflow

def ingest_documents(
    file_paths: List[str],
    vector_store: QdrantVectorStore,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> None:
    """
    Ingest documents into the vector store.
    
    Args:
        file_paths: List of paths to document files
        vector_store: Vector store instance
        chunk_size: Size of document chunks
        chunk_overlap: Overlap between chunks
    """
    print(f"Loading and splitting documents from {len(file_paths)} files...")
    documents = load_and_split_documents(
        file_paths=file_paths,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    print(f"Ingesting {len(documents)} document chunks into vector store...")
    vector_store.add_documents(documents)
    print("Document ingestion complete!")

def run_query(
    query: str,
    vector_store: QdrantVectorStore,
    model_name: str = "qwen2.5",
    use_web_search: bool = True
) -> Dict[str, Any]:
    """
    Run a query through the Agentic RAG system.
    
    Args:
        query: User query string
        vector_store: Vector store instance
        model_name: Name of the LLM model to use
        use_web_search: Whether to use web search
        
    Returns:
        Dictionary containing the response and intermediate results
    """
    print(f"Processing query: {query}")
    
    # Initialize the workflow
    workflow = AgenticRAGWorkflow(
        vector_store=vector_store,
        model_name=model_name,
        use_web_search=use_web_search
    )
    
    # Run the workflow
    result = workflow.run(query)
    
    return result

def interactive_mode(
    vector_store: QdrantVectorStore,
    model_name: str = "qwen2.5",
    use_web_search: bool = True
) -> None:
    """
    Run the Agentic RAG system in interactive mode.
    
    Args:
        vector_store: Vector store instance
        model_name: Name of the LLM model to use
        use_web_search: Whether to use web search
    """
    print("\n===== Agentic RAG Interactive Mode =====")
    print("Type 'exit' or 'quit' to end the session.")
    
    while True:
        query = input("\nEnter your query: ")
        
        if query.lower() in ["exit", "quit"]:
            print("Exiting interactive mode...")
            break
        
        result = run_query(
            query=query,
            vector_store=vector_store,
            model_name=model_name,
            use_web_search=use_web_search
        )
        
        if result.get("error"):
            print(f"Error: {result['error']}")
        else:
            print("\nResponse:")
            print(result.get("response", "No response generated."))

def main():
    """Main entry point for the Agentic RAG system."""
    parser = argparse.ArgumentParser(description="Agentic RAG System")
    
    # Add command-line arguments
    parser.add_argument("--ingest", action="store_true", help="Ingest documents into the vector store")
    parser.add_argument("--files", nargs="+", help="Paths to document files for ingestion")
    parser.add_argument("--query", type=str, help="Run a single query")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--model", type=str, default="qwen2.5", help="LLM model to use")
    parser.add_argument("--no-web-search", action="store_true", help="Disable web search")
    parser.add_argument("--collection", type=str, default="agentic_rag", help="Qdrant collection name")
    
    args = parser.parse_args()
    
    # Initialize vector store
    vector_store = QdrantVectorStore(collection_name=args.collection)
    
    # Determine whether to use web search
    use_web_search = not args.no_web_search
    
    # Process commands
    if args.ingest:
        if not args.files:
            print("Error: Please specify files to ingest using --files")
            return
        
        ingest_documents(file_paths=args.files, vector_store=vector_store)
    
    if args.query:
        result = run_query(
            query=args.query,
            vector_store=vector_store,
            model_name=args.model,
            use_web_search=use_web_search
        )
        
        if result.get("error"):
            print(f"Error: {result['error']}")
        else:
            print("\nResponse:")
            print(result.get("response", "No response generated."))
    
    if args.interactive:
        interactive_mode(
            vector_store=vector_store,
            model_name=args.model,
            use_web_search=use_web_search
        )

if __name__ == "__main__":
    main() 