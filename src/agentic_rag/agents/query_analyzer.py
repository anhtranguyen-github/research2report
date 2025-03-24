"""Query Analyzer Agent for the Agentic RAG system."""

from typing import Dict, Any, List, Optional, TypedDict, Union

from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable

from src.agentic_rag.core.model import get_ollama_llm

class QueryAnalysis(TypedDict):
    """Type for query analysis results."""
    
    intent: str
    questions: List[str]
    search_terms: List[str]
    requires_context: bool
    reasoning: str

class QueryAnalyzerAgent:
    """Agent that analyzes user queries to determine search strategy."""
    
    def __init__(self, model_name: str = "qwen2.5"):
        """
        Initialize the Query Analyzer Agent.
        
        Args:
            model_name: Name of the LLM model to use
        """
        self.llm = get_ollama_llm(model_name=model_name)
        self.prompt = self._create_prompt()
        self.chain = self.prompt | self.llm | StrOutputParser() | self._parse_output
    
    def _create_prompt(self) -> PromptTemplate:
        """Create the prompt template for query analysis."""
        template = """You are an expert query analyzer for a RAG (Retrieval Augmented Generation) system.
Your task is to analyze the user's query to determine the best search strategy.

User Query: {query}

Analyze the query and provide the following:
1. The primary intent of the query
2. A list of specific questions that need to be answered
3. Effective search terms for retrieving relevant information
4. Whether external context is required to answer the query
5. Your reasoning process

Format your response as follows:
Intent: <primary intent>
Questions: <list of specific questions>
Search Terms: <list of effective search terms>
Requires Context: <true/false>
Reasoning: <your reasoning process>

Think carefully about the query and provide a comprehensive analysis."""
        
        return PromptTemplate.from_template(template)
    
    def _parse_output(self, output: str) -> QueryAnalysis:
        """
        Parse the LLM output into a structured format.
        
        Args:
            output: Raw LLM output string
            
        Returns:
            Structured query analysis
        """
        lines = output.strip().split('\n')
        result = {}
        
        for line in lines:
            if line.startswith('Intent:'):
                result['intent'] = line.replace('Intent:', '').strip()
            elif line.startswith('Questions:'):
                questions_str = line.replace('Questions:', '').strip()
                result['questions'] = [q.strip() for q in questions_str.split(',')]
            elif line.startswith('Search Terms:'):
                terms_str = line.replace('Search Terms:', '').strip()
                result['search_terms'] = [t.strip() for t in terms_str.split(',')]
            elif line.startswith('Requires Context:'):
                context_str = line.replace('Requires Context:', '').strip().lower()
                result['requires_context'] = context_str == 'true'
            elif line.startswith('Reasoning:'):
                result['reasoning'] = line.replace('Reasoning:', '').strip()
        
        return QueryAnalysis(
            intent=result.get('intent', ''),
            questions=result.get('questions', []),
            search_terms=result.get('search_terms', []),
            requires_context=result.get('requires_context', True),
            reasoning=result.get('reasoning', '')
        )
    
    def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyze the user query.
        
        Args:
            query: The user's query string
            
        Returns:
            Structured analysis of the query
        """
        return self.chain.invoke({"query": query}) 