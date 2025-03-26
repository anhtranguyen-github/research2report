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
        # Use task-specific model for query analysis
        self.llm = get_ollama_llm(model_name=model_name, task_type="query_analysis")
        self.prompt = self._create_prompt()
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def _create_prompt(self) -> PromptTemplate:
        """Create the prompt template for query analysis."""
        template = """You are an expert AI query analyzer tasked with understanding user queries.
Analyze the given query and extract key information to help with information retrieval.

User Query: {query}

Provide an analysis with the following information:

1. Intent: What is the primary intent of the query? Is it a factual question, opinion request, how-to, etc.?

2. Specific Questions: Break down the query into specific questions that need to be answered.

3. Search Terms: List specific terms or phrases that would be most effective for searching a knowledge base.

4. Requires Context: Does this query require specific contextual information to answer properly? Yes or No.

5. Reasoning: Briefly explain your analysis.

Format your response as JSON with the following structure:
```json
{
  "intent": "string",
  "questions": ["string", "string"],
  "search_terms": ["string", "string"],
  "requires_context": true|false,
  "reasoning": "string"
}
```"""
        
        return PromptTemplate.from_template(template)
    
    def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyze a user query.
        
        Args:
            query: User query string
            
        Returns:
            Query analysis results
        """
        # Get the analysis as a JSON string
        try:
            analysis_str = self.chain.invoke({"query": query})
            
            # Parse the JSON string to get the analysis
            import json
            
            # Extract the JSON content between triple backticks if present
            if "```json" in analysis_str:
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', analysis_str, re.DOTALL)
                if json_match:
                    analysis_str = json_match.group(1)
            
            # Parse the JSON
            analysis = json.loads(analysis_str)
            
            # Ensure the result has all required fields
            return QueryAnalysis(
                intent=analysis.get("intent", ""),
                questions=analysis.get("questions", []),
                search_terms=analysis.get("search_terms", []),
                requires_context=analysis.get("requires_context", True),
                reasoning=analysis.get("reasoning", "")
            )
        except Exception as e:
            print(f"Error analyzing query: {e}")
            # Return a default analysis in case of error
            return QueryAnalysis(
                intent="Unknown",
                questions=[query],
                search_terms=[query],
                requires_context=True,
                reasoning=f"Error during analysis: {str(e)}"
            )
        
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the query analyzer within the workflow.
        
        Args:
            state: The current workflow state
            
        Returns:
            Updated workflow state
        """
        query = state.get("query", "")
        query_analysis = self.analyze(query)
        
        return {
            "query": query,
            "query_analysis": query_analysis
        } 