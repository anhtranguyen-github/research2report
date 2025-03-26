"""Query Analyzer Agent for the Agentic RAG system."""

from typing import Dict, Any, List, Optional
import json
import re

from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable

from src.agentic_rag.core.model import get_ollama_llm
from src.agentic_rag.agents.types import QueryAnalysis

class QueryAnalyzerAgent:
    """Agent that analyzes user queries to determine search strategy."""
    
    def __init__(
        self,
        model_name: str = "qwen2.5",
        prompt_template: Optional[str] = None
    ):
        """
        Initialize the QueryAnalyzerAgent.
        
        Args:
            model_name: Name of the LLM model to use
            prompt_template: Optional prompt template to use
        """
        self.model_name = model_name
        
        # Use the provided prompt template or a default one
        if prompt_template is None:
            from .config import get_prompt_template
            try:
                prompt_template = get_prompt_template("query_analyzer")
            except Exception as e:
                print(f"Error loading prompt template: {e}")
                # Default template if config can't be loaded
                prompt_template = """
                Analyze the following query: {query}
                
                Your task is to determine the user's intent, whether the query requires retrieval from a knowledge base, whether it requires web search for up-to-date information, identify specific questions, and any context requirements.
                
                Provide your analysis as a JSON with these fields:
                - intent: the main intent behind the query
                - questions: array of specific questions identified in the query
                - search_terms: array of key terms to search for
                - requires_context: boolean indicating if specific contextual information is needed
                - reasoning: brief explanation of your analysis
                """
        
        # Ensure prompt_template is a string
        if isinstance(prompt_template, dict):
            if 'template' in prompt_template:
                prompt_template = prompt_template['template']
            elif 'prompt' in prompt_template and isinstance(prompt_template['prompt'], dict) and 'template' in prompt_template['prompt']:
                prompt_template = prompt_template['prompt']['template']
            else:
                # Convert dict to string as a fallback
                prompt_template = json.dumps(prompt_template)
        
        # Create the prompt template
        self.prompt_template = prompt_template
        self.prompt = PromptTemplate.from_template(self.prompt_template)
        
        self.task_type = "query_analysis"
        
        # Use task-specific model for query analysis
        self.llm = get_ollama_llm(model_name=self.model_name, task_type=self.task_type)
        
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Analyze a user query.
        
        Args:
            query: User query string
            
        Returns:
            Query analysis results as a dictionary
        """
        try:
            # Get the analysis as a JSON string
            analysis_str = self.chain.invoke({"query": query})
            
            # Extract the JSON content between triple backticks if present
            if "```json" in analysis_str:
                json_match = re.search(r'```json\s*(.*?)\s*```', analysis_str, re.DOTALL)
                if json_match:
                    analysis_str = json_match.group(1)
            
            # Parse the JSON
            analysis = json.loads(analysis_str)
            
            # Add the original query to the analysis
            analysis["query"] = query
            
            # Set default values for backwards compatibility
            if "requires_retrieval" not in analysis:
                analysis["requires_retrieval"] = True
            if "requires_web_search" not in analysis:
                analysis["requires_web_search"] = True
            if "specific_questions" not in analysis:
                analysis["specific_questions"] = analysis.get("questions", [])
            if "context_requirements" not in analysis:
                analysis["context_requirements"] = {}
            
            # Validate against QueryAnalysis model
            return QueryAnalysis(**analysis).model_dump()
            
        except Exception as e:
            print(f"Error analyzing query: {e}")
            # Return a default analysis in case of error
            return QueryAnalysis(
                query=query,
                intent="Unknown",
                questions=[query],
                search_terms=[query],
                requires_context=True,
                reasoning=f"Error during analysis: {str(e)}",
                requires_retrieval=True,
                requires_web_search=True,
                specific_questions=[query],
                context_requirements={}
            ).model_dump()

    def run(self, state: Dict[str, Any]) -> QueryAnalysis:
        """
        Run the query analyzer within the workflow.
        
        Args:
            state: The current workflow state
            
        Returns:
            QueryAnalysis object containing the analysis results
        """
        query = state.get("query", "")
        analysis_dict = self.analyze(query)
        return QueryAnalysis(**analysis_dict) 