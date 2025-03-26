# Testing the Agentic RAG API Server

This README explains how to set up and test the Agentic RAG API server.

## Prerequisites

- Python 3.9+ with pip
- Docker and Docker Compose (for running Ollama and Qdrant)

## Setup

1. Clone the repository and navigate to the project directory.

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with the following content (or copy from `.env.example`):
```
# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434

# GitHub API Token (for fetching repo data)
GITHUB_TOKEN= 
```

## Starting the Services

1. Start Ollama and Qdrant using Docker Compose:
```bash
docker-compose up -d
```

2. Pull the required model in Ollama:
```bash
curl http://localhost:11434/api/pull -d '{"name": "qwen2.5"}'
```

## Starting the API Server

Start the API server with:
```bash
python -m src.agentic_rag.run_server
```

The server will be accessible at http://localhost:8000

## Running the Tests

Use the provided test script to test the API:

### Basic Usage

```bash
python test_api.py
```

This will run all tests with default settings.

### Test Options

The script has several options:

```bash
# Test with a specific query
python test_api.py --query "What are the key components of a RAG system?"

# Use a different model
python test_api.py --model "llama3"

# Test file upload
python test_api.py --file sample_rag_doc.md

# Disable web search
python test_api.py --no-web-search

# Skip specific tests
python test_api.py --skip-health --skip-query
python test_api.py --skip-upload

# Use a different API URL
python test_api.py --url "http://localhost:8080"
```

### Testing All Endpoints

To do a comprehensive test of all endpoints:

```bash
python test_api.py --file sample_rag_doc.md
```

This will:
1. Test all health check endpoints
2. Run 3 sample queries
3. Upload the sample document to the vector store

## Troubleshooting

### Health Check Failures

- **Ollama**: Make sure Ollama is running and the required model is pulled
- **Qdrant**: Make sure Qdrant is running and accessible

### Connection Errors

- Verify the URLs in the `.env` file match the actual service URLs
- Check that the services are running with `docker ps`
- Check service logs with `docker logs ollama` or `docker logs qdrant`

### API Server Errors

- Check that all environment variables are properly set
- Verify that the required dependencies are installed
- Check the API server logs for error messages 