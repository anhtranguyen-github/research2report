"""Agent managers for handling agent initialization and execution."""

import os
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path

from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable

from src.agentic_rag.core.model import get_ollama_llm
from src.agentic_rag.agents.types import (
    QueryAnalysis,
    RetrievedDocument,
    WebSearchResult,
    GeneratedResponse,
    AgentState
)
from src.agentic_rag.agents.query_analyzer import QueryAnalyzerAgent
from src.agentic_rag.agents.retriever import RetrieverAgent
from src.agentic_rag.agents.web_search import WebSearchAgent
from src.agentic_rag.agents.generator import GeneratorAgent
from .config import get_prompt_template

class AgentConfig:
    """Configuration for an agent."""
    
    def __init__(self, config_path: str):
        """
        Initialize agent configuration.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config from {self.config_path}: {e}")
            return {"prompt": "", "settings": {}}
    
    @property
    def prompt_template(self) -> str:
        """Get the prompt template from config."""
        return self.config.get('prompt', "")
    
    @property
    def settings(self) -> Dict[str, Any]:
        """Get the settings from config."""
        return self.config.get('settings', {})

class AgentManager:
    """Manager for handling agent initialization and execution."""
    
    def __init__(self, config_dir: str = "src/agentic_rag/agents/config"):
        """
        Initialize the agent manager.
        
        Args:
            config_dir: Directory containing agent configuration files
        """
        self.config_dir = Path(config_dir)
        self.agents = {}
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all agents with their configurations."""
        # Initialize Query Analyzer
        query_analyzer_config = AgentConfig(self.config_dir / "query_analyzer.yaml")
        self.agents['query_analyzer'] = QueryAnalyzerAgent(
            model_name=query_analyzer_config.settings.get('model_name'),
            prompt_template=query_analyzer_config.prompt_template
        )
        
        # Initialize Retriever
        retriever_config = AgentConfig(self.config_dir / "retriever.yaml")
        self.agents['retriever'] = RetrieverAgent(
            model_name=retriever_config.settings.get('model_name'),
            prompt_template=retriever_config.prompt_template,
            max_results=retriever_config.settings.get('max_results', 5)
        )
        
        # Initialize Web Search
        web_search_config = AgentConfig(self.config_dir / "web_search.yaml")
        self.agents['web_search'] = WebSearchAgent(
            model_name=web_search_config.settings.get('model_name'),
            prompt_template=web_search_config.prompt_template,
            search_engine=web_search_config.settings.get('search_engine', 'tavily'),
            num_results=web_search_config.settings.get('num_results', 5),
            base_url=web_search_config.settings.get('base_url', 'https://api.tavily.com')
        )
        
        # Initialize Generator
        generator_config = AgentConfig(self.config_dir / "generator.yaml")
        self.agents['generator'] = GeneratorAgent(
            model_name=generator_config.settings.get('model_name'),
            prompt_template=generator_config.prompt_template
        )
    
    def process_query(self, query: str) -> AgentState:
        """
        Process a user query through all agents based on query analysis.
        
        The flow:
        1. QueryAnalyzerAgent analyzes the query
        2. Based on analysis:
           - If requires retrieval, use RetrieverAgent
           - If requires web search, use WebSearchAgent
           - Can use both if needed
        3. GeneratorAgent combines all results to generate final response
        
        Args:
            query: User's query string
            
        Returns:
            Final agent state with all results
        """
        # Initialize state
        state = AgentState(query=query)
        
        # Step 1: Run query analysis
        query_analysis = self.agents['query_analyzer'].run({"query": query})
        state.query_analysis = query_analysis
        
        # Step 2a: Run document retrieval if needed
        if query_analysis.requires_retrieval:
            retrieved_docs = self.agents['retriever'].run({
                "query": query,
                "documents": [],  # TODO: Add document retrieval logic
                "analysis": query_analysis
            })
            state.retrieved_documents = retrieved_docs
        else:
            state.retrieved_documents = []
            
        # Step 2b: Run web search if needed
        if query_analysis.requires_web_search:
            web_results = self.agents['web_search'].search(
                query=query,
                analysis=query_analysis
            )
            state.web_results = [WebSearchResult(**result) for result in web_results]
        else:
            state.web_results = []
        
        # Step 3: Generate final response using all available information
        response = self.agents['generator'].run(
            state={"query": query},
            query_analysis=query_analysis.model_dump(),
            retrieved_docs=state.retrieved_documents,
            web_results=state.web_results
        )
        state.response = response
        
        return state 