# Agentic RAG

An Agentic RAG (Retrieval Augmented Generation) system built with LangGraph, Ollama (Qwen2.5), and Qdrant.

## Overview

This Agentic RAG system combines the power of:
- **LangGraph**: For building agentic workflows with multiple steps and decision-making
- **Ollama**: For local LLM inference using Qwen2.5
- **Qdrant**: For vector storage and semantic search
- **FastAPI**: For serving the RAG system via RESTful API

The system follows an agentic approach with specialized components that work together:

1. **Query Analyzer**: Analyzes user queries to determine intent, extract search terms, and identify if external context is needed
2. **Retriever**: Retrieves relevant information from the vector store based on the query and search terms
3. **Web Search**: Optionally supplements vector store information with web search results
4. **Generator**: Generates comprehensive responses based on the retrieved information

## Prerequisites

- Python 3.8+
- Ollama installed with Qwen2.5 model
- Qdrant running locally or remotely
- Required Python packages (see requirements.txt)

## Installation

1. Ensure you have Ollama installed and the Qwen2.5 model pulled:
   ```
   ollama pull qwen2.5
   ```

2. Start a local Qdrant instance (or use a remote instance):
   ```
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```

3. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Command-line Interface

#### Ingesting Documents

Before querying the system, you need to ingest documents into the vector store:

```bash
python -m src.agentic_rag.main --ingest --files document1.pdf document2.txt
```

#### Running Queries

You can run a single query:

```bash
python -m src.agentic_rag.main --query "What is the capital of France?"
```

#### Interactive Mode

For multiple queries, use interactive mode:

```bash
python -m src.agentic_rag.main --interactive
```

#### Additional Options

- `--model`: Specify a different Ollama model (default: qwen2.5)
- `--no-web-search`: Disable web search capability
- `--collection`: Specify a different Qdrant collection name (default: agentic_rag)

Example:
```bash
python -m src.agentic_rag.main --interactive --model qwen2.7 --no-web-search
```

### REST API Server

The system can also be deployed as a REST API server using FastAPI.

#### Starting the Server

```bash
python -m src.agentic_rag.run_server --host 0.0.0.0 --port 8000
```

Additional server options:
- `--reload`: Enable auto-reload for development
- `--collection`: Specify a Qdrant collection name
- `--qdrant-host`: Specify Qdrant host
- `--qdrant-port`: Specify Qdrant port

#### API Endpoints

The server exposes the following endpoints:

- `GET /`: Check if the API is running
- `POST /query`: Submit a query to the RAG system
- `POST /upload`: Upload and ingest a document
- `DELETE /collection`: Delete the entire vector store collection

#### Using the API Client

A Python client is provided to interact with the API:

```bash
# Query the RAG system
python -m src.agentic_rag.client.api_client query "What is the capital of France?"

# Upload a document
python -m src.agentic_rag.client.api_client upload document.pdf

# Delete the collection
python -m src.agentic_rag.client.api_client delete-collection
```

Client options:
- `--url`: Specify the API base URL (default: http://localhost:8000)
- `--model`: Specify the LLM model (for queries)
- `--no-web-search`: Disable web search (for queries)
- `--chunk-size` and `--chunk-overlap`: Control document splitting (for uploads)

## System Architecture

The system follows this workflow:

1. **Query Analysis**: Analyzes the user's query to determine intent, extract relevant questions, and identify search terms
2. **Information Retrieval**: Retrieves relevant documents from the vector store based on the query and search terms
3. **Web Search (Optional)**: If the retrieved information is insufficient, supplements with web search results
4. **Response Generation**: Generates a comprehensive response using the retrieved information

## Customization

You can customize various aspects of the system:

- Change the embedding model in `QdrantVectorStore` initialization
- Adjust chunk size and overlap for document splitting
- Modify prompts in the agent classes
- Add additional tools to enhance the system's capabilities
- Extend the API with new endpoints

## Limitations

- The web search tool currently uses a mock implementation. Replace with a real API in production.
- The system's performance depends on the quality of the ingested documents and the Qwen2.5 model.

## Future Improvements

- Add support for more document types
- Implement real-time learning and feedback mechanisms
- Enhance the web search capabilities
- Add support for streaming responses
- Implement memory for multi-turn conversations
- Add authentication to the API
- Create a web UI for interacting with the API 