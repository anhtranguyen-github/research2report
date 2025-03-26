# AI Trends Report Generator

An AI agent system that researches and generates a comprehensive weekly report on the latest AI trends and developments.

## Overview

This project uses the CrewAI framework to orchestrate multiple AI agents that work together to create detailed, insightful reports on current trends in artificial intelligence. The system handles:

1. Researching and identifying key AI trends from the past week
2. Extracting and analyzing data about each trend
3. Compiling the findings into a structured report

## Project Structure

```
ai_trends_report/
├── crews/
│   ├── TrendDiscovery_crew/
│   │   ├── config/
│   │   │   ├── agents.yaml
│   │   │   └── tasks.yaml
│   │   └── trend_discovery_crew.py
│   ├── DataExtraction_crew/
│   │   ├── config/
│   │   │   ├── agents.yaml
│   │   │   └── tasks.yaml
│   │   └── data_extraction_crew.py
│   └── ContentGeneration_crew/
│       ├── config/
│       │   ├── agents.yaml
│       │   └── tasks.yaml
│       └── content_generation_crew.py
├── tools/
│   ├── search_tool.py
│   ├── arxiv_tool.py
│   └── github_tool.py
└── main.py
```

## How It Works

The system operates in three sequential phases:

1. **Trend Discovery Phase**: Multiple agents monitor different sources (news, research papers, social media, developer forums) to identify emerging AI trends. These findings are then analyzed and consolidated into a prioritized list of the most significant trends.

2. **Data Extraction Phase**: For each identified trend, specialized agents gather detailed information from various sources, including research papers, news articles, social media discussions, and GitHub projects. This data is analyzed to create a comprehensive view of each trend.

3. **Content Generation Phase**: Using the extracted data, content-focused agents create detailed write-ups for each trend, an executive summary highlighting strategic implications, and compile everything into a polished, coherent report.

## Getting Started

### Prerequisites

- Python 3.10+
- Dependencies listed in `requirements.txt`

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai-trends-report

# Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```
BRIGHDATA_USERNAME=your_brighdata_username
BRIGHDATA_PASSWORD=your_brighdata_password
GITHUB_TOKEN=your_github_token
```

### Usage

```bash
# Run the report generation
python -m src.ai_trends_report.main

# Or use the CLI command if you've installed the package
kickoff
```

The report will be saved as `ai_trends_report.md` in the current directory.

## Configuration

Agent behavior and tasks can be customized by modifying the YAML files in:
- `src/ai_trends_report/crews/TrendDiscovery_crew/config/`
- `src/ai_trends_report/crews/DataExtraction_crew/config/`
- `src/ai_trends_report/crews/ContentGeneration_crew/config/`

## Model Configuration

### LLM Models

The system supports multiple LLM models for different tasks:

- **Qwen Models**: qwen2.5, qwen2.5-14b
- **Llama Models**: llama3, llama3-70b
- **Mistral Models**: mistral, mistral-7b-instruct
- **Mixtral Models**: mixtral
- **CodeLlama Models**: codellama

Each task in the RAG pipeline (query analysis, retrieval evaluation, generation) can use optimized parameters for that specific task.

See [Model Configuration Documentation](src/agentic_rag/config/README.md) for details.

### Embedding Models

The system supports multiple embedding models for vector storage:

- **Sentence Transformers Models**: all-MiniLM-L6-v2, all-mpnet-base-v2, e5-large-v2, etc.
- **Instructor Models**: Instruction-tuned embedding models
- **OpenAI Models**: OpenAI's embedding models (requires API key)
- **Ollama Models**: Embedding models running through Ollama

Embedding models can be changed at runtime, and collections can be created specific to each model's dimensions.

See [Embedding Configuration Documentation](src/agentic_rag/config/EMBEDDING_README.md) for details.

## License

MIT
