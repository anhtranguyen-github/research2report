# Research2Report

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced system for turning research into comprehensive reports with autonomous AI agents. Developed by [Nguyen Anh Tra (Ezooo)](https://github.com/anhtranguyen-github).

## 🚀 Overview

Research2Report combines the power of AI agents with advanced NLP to automatically generate comprehensive reports from research materials. The system:

1. **Gathers** - Collects research from various sources including papers, news, and structured databases
2. **Analyzes** - Processes and extracts key information from research materials
3. **Synthesizes** - Combines findings into a coherent narrative with proper citations
4. **Generates** - Creates professional, publication-ready reports with visualizations

## 📁 Project Structure

```
research2report/
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

1. **Research Collection Phase**
   - Agents gather materials from academic databases, news sources, and the web
   - Materials are preprocessed and normalized for analysis

2. **Information Extraction Phase**
   - Specialized agents extract key information, findings, and statistics
   - Results are structured into a knowledge graph for relationship analysis

3. **Report Generation Phase**
   - Using the structured information, the system generates a comprehensive report
   - Multiple agents ensure proper citation, coherence, and factual accuracy
   - The final report is formatted according to specified style guidelines

## 🔧 Installation

### Prerequisites

- Python 3.10+
- Git

### Option 1: Manual Installation

```bash
# Clone the repository
git clone https://github.com/anhtranguyen-github/research2report.git
cd research2report

# Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package and dependencies
pip install -e ".[dev]"
```

### Option 2: Docker Installation

```bash
# Clone the repository
git clone https://github.com/anhtranguyen-github/research2report.git
cd research2report

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
TAVILY_API_KEY=your_tavily_api_key
QDRANT_URL=http://localhost:6333
```

## 🏃‍♂️ Usage

### Command Line Interface

```bash
# Generate a report
kickoff

# Generate visualizations of report data
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

Each task in the pipeline can use optimized parameters for that specific task.

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
- [Other projects](https://github.com/anhtranguyen-github)

## 📞 Contact

For questions or feedback, please [open an issue](https://github.com/anhtranguyen-github/research2report/issues) on the GitHub repository.
