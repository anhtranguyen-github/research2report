# AgenticRAG

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced system for retrieval-augmented generation with autonomous AI agents. Developed by [Nguyen Anh Tra (Ezooo)](https://github.com/anhtranguyen-github).

## 🚀 Overview

AgenticRAG combines the power of retrieval-augmented generation (RAG) with autonomous AI agents to create a more powerful, context-aware, and adaptive information retrieval and content generation system. The system:

1. **Analyzes** - Understands complex queries beyond simple keyword matching
2. **Retrieves** - Uses sophisticated agents to gather relevant information from various sources
3. **Augments** - Enhances retrieved information with additional context and connections
4. **Generates** - Creates comprehensive, accurate responses using the augmented context

## 📁 Project Structure

```
agentic_rag/
├── src/
│   ├── agentic_rag/          # Core RAG functionality
│   └── server/               # API server components
├── tests/                    # Test suite
├── docs/                     # Documentation
├── pyproject.toml            # Project metadata and dependencies
├── .env.example              # Example environment variables
├── docker-compose.yml        # Docker configuration
└── README.md                 # Project documentation
```

## ⚙️ How It Works

The system operates through a coordinated agent workflow:

1. **Query Analysis Phase**
   - Agents analyze the user's query to understand the intent and required information
   - The query is broken down into multiple components for targeted retrieval

2. **Retrieval Phase**
   - Specialized retrieval agents search for information across different sources
   - Sources include vector databases, web searches, and structured knowledge bases
   - Results are ranked and filtered for relevance

3. **Content Generation Phase**
   - Using the retrieved context, the system generates a comprehensive response
   - Multiple agents work together to ensure accuracy and completeness
   - The final response is formatted appropriately for the user's needs

## 🔧 Installation

### Prerequisites

- Python 3.10+
- Git

### Option 1: Manual Installation

```bash
# Clone the repository
git clone https://github.com/anhtranguyen-github/agentic-rag.git
cd agentic-rag

# Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package and dependencies
pip install -e ".[dev]"
```

### Option 2: Docker Installation

```bash
# Clone the repository
git clone https://github.com/anhtranguyen-github/agentic-rag.git
cd agentic-rag

# Start the Docker containers
docker-compose up -d
```

## 🔑 Environment Setup

Copy the example environment file and update with your credentials:

```bash
cp .env.example .env
```

Required environment variables:

```
OPENAI_API_KEY=your_openai_api_key
QDRANT_URL=http://qdrant:6333
```

## 🏃‍♂️ Usage

### Command Line Interface

```bash
# Run the RAG system
kickoff

# Generate visualizations of retrieval performance
plot
```

### API Server

```bash
# Start the API server
uvicorn src.server.main:app --reload

# The API will be available at http://localhost:8000
# Swagger documentation at http://localhost:8000/docs
```

## 🔍 Configuration

### Agent Configuration

Agent behavior and tasks can be customized by modifying the YAML files in:
- `src/agentic_rag/config/agents/`
- `src/agentic_rag/config/tasks/`

### Model Configuration

The system supports multiple LLM models for different tasks:

- **Qwen Models**: qwen2.5, qwen2.5-14b
- **Llama Models**: llama3, llama3-70b
- **Mistral Models**: mistral, mistral-7b-instruct
- **Mixtral Models**: mixtral
- **OpenAI Models**: gpt-4o, gpt-4-turbo, gpt-3.5-turbo

Each task in the RAG pipeline can use optimized parameters for that specific task.

## 📊 Embedding Models

The system supports multiple embedding models for vector storage:

- **Sentence Transformers Models**: all-MiniLM-L6-v2, all-mpnet-base-v2, e5-large-v2
- **Instructor Models**: Instruction-tuned embedding models
- **OpenAI Models**: OpenAI's embedding models (requires API key)
- **Ollama Models**: Embedding models running through Ollama

## 🧪 Testing

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=src
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 About the Author

Nguyen Anh Tra (Ezooo) is a developer from Vietnam currently focused on Large Language Models (LLMs) and their applications. For more projects, check out:

- [bkai-vnlaw-rag](https://github.com/anhtranguyen-github/bkai-vnlaw-rag) - RAG system for Vietnamese legal documents
- [Other LLM-related projects](https://github.com/anhtranguyen-github)

## 📞 Contact

For questions or feedback, please [open an issue](https://github.com/anhtranguyen-github/agentic-rag/issues) on the GitHub repository.
