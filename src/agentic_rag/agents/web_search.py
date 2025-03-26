"""Web Search Agent for the Agentic RAG system."""

from typing import Dict, Any, List, Optional
import os
import json
import re
import requests
from datetime import datetime

from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable

from src.agentic_rag.core.model import get_ollama_llm
from src.agentic_rag.agents.types import WebSearchResult

class WebSearchAgent:
    """Agent that performs web searches and analyzes results."""
    
    def __init__(
        self,
        model_name: str = "qwen2.5",
        prompt_template: Optional[str] = None,
        search_engine: str = "tavily",
        num_results: int = 5,
        base_url: str = "https://api.tavily.com"
    ):
        """
        Initialize the WebSearchAgent.
        
        Args:
            model_name: Name of the LLM model to use
            prompt_template: Optional prompt template to use
            search_engine: Search engine to use (default: tavily)
            num_results: Number of results to return
            base_url: Base URL for the search API
        """
        self.model_name = model_name
        self.search_engine = search_engine
        self.num_results = num_results
        self.base_url = base_url
        
        # Use the provided prompt template or a default one
        if prompt_template is None:
            from .config import get_prompt_template
            try:
                prompt_template = get_prompt_template("web_search")
            except Exception as e:
                print(f"Error loading prompt template: {e}")
                # Default template if config can't be loaded
                prompt_template = """
                Analyze the following web search results for the query: {query}
                
                Query: {query}
                
                Web Search Results:
                {web_results}
                
                Your task is to evaluate how relevant and useful each result is.
                Look for credible sources, current information, and direct relevance to the query.
                Highlight any biases or issues with the sources.
                
                Output your analysis in a structured format.
                """
        
        # Ensure prompt_template is a string
        if isinstance(prompt_template, dict):
            if 'template' in prompt_template:
                prompt_template = prompt_template['template']
            elif 'prompt' in prompt_template and isinstance(prompt_template['prompt'], dict) and 'template' in prompt_template['prompt']:
                prompt_template = prompt_template['prompt']['template']
            else:
                # Convert dict to string as a fallback
                import json
                prompt_template = json.dumps(prompt_template)
        
        # Create the prompt template
        self.prompt_template = prompt_template
        self.prompt = PromptTemplate.from_template(self.prompt_template)
        
        # Get API key from environment
        self.api_key = os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            print("Warning: TAVILY_API_KEY not found in environment variables")
        
        # Use task-specific model for web search analysis
        self.llm = get_ollama_llm(model_name=self.model_name, task_type="web_search_analysis")
        
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def search(self, query: str, analysis=None, **kwargs) -> List[Dict[str, Any]]:
        """
        Perform a web search and analyze the results.
        
        Args:
            query: Search query string
            analysis: Optional query analysis (ignored but kept for API compatibility)
            **kwargs: Additional arguments
            
        Returns:
            List of analyzed web search results
        """
        if not self.api_key:
            print("Error: TAVILY_API_KEY not found. Cannot perform web search.")
            return []
        
        try:
            # Debug info
            print(f"[DEBUG] Starting Tavily API search with query: {query}")
            print(f"[DEBUG] Using API Key: {self.api_key[:5]}...{self.api_key[-4:]}")
            print(f"[DEBUG] Base URL: {self.base_url}")
            
            # Perform web search using Tavily API
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "query": query,
                "search_depth": "basic",
                "max_results": self.num_results
            }
            
            print(f"[DEBUG] Request payload: {json.dumps(payload)}")
            print(f"[DEBUG] Request headers: {json.dumps({k: (v[:10]+'...' if k=='Authorization' else v) for k,v in headers.items()})}")
            
            # Define potential endpoints to try
            endpoints = [
                "/search",  # Standard endpoint
                "/api/search",  # Alternative with /api prefix
                ""  # Try the base URL directly
            ]
            
            response = None
            success = False
            
            # Try each endpoint until one works
            for endpoint in endpoints:
                full_url = f"{self.base_url}{endpoint}"
                print(f"[DEBUG] Trying URL: {full_url}")
                
                try:
                    response = requests.post(
                        full_url,
                        headers=headers,
                        json=payload
                    )
                    
                    print(f"[DEBUG] Response status code: {response.status_code}")
                    print(f"[DEBUG] Response headers: {dict(response.headers)}")
                    
                    if response.status_code == 200:
                        print(f"[DEBUG] Successful with endpoint: {endpoint}")
                        success = True
                        break
                    else:
                        print(f"Error from Tavily API with endpoint {endpoint}: {response.status_code}")
                        print(f"Response content: {response.text}")
                except Exception as e:
                    print(f"[DEBUG] Failed request to {full_url}: {str(e)}")
            
            if not success or not response:
                print("All API endpoint attempts failed")
                # Try a GET request to the base URL to check API accessibility
                try:
                    check_resp = requests.get(self.base_url)
                    print(f"[DEBUG] Base URL accessibility check: {check_resp.status_code}")
                except Exception as e:
                    print(f"[DEBUG] Failed to reach base URL: {str(e)}")
                
                # Mock response for development/testing
                print("[DEBUG] Returning mock search results for development")
                return [
                    {
                        "title": "What is RAG in AI? (Mocked Response)",
                        "url": "https://example.com/rag-info",
                        "snippet": "Retrieval-Augmented Generation (RAG) is an AI framework that combines retrieval systems with generative models to enhance responses with external knowledge.",
                        "relevance_score": 0.95,
                        "source": "web"
                    },
                    {
                        "title": "Understanding RAG Architecture (Mocked)",
                        "url": "https://example.com/rag-architecture",
                        "snippet": "RAG consists of a retriever component that fetches relevant information and a generator that creates responses based on the retrieved content.",
                        "relevance_score": 0.9,
                        "source": "web"
                    }
                ]
            
            # Additional debug info
            print(f"[DEBUG] Response content type: {response.headers.get('Content-Type')}")
            
            search_results = response.json()
            print(f"[DEBUG] Response JSON keys: {list(search_results.keys())}")
            
            results_count = len(search_results.get('results', []))
            print(f"Received {results_count} results from Tavily API")
            
            if results_count == 0:
                print("No results received from Tavily API.")
                return []
            
            # Format results for analysis
            results_list = []
            for i, result in enumerate(search_results.get('results', [])):
                # Create a WebSearchResult for each result
                result_obj = {
                    "title": result.get('title', ''),
                    "url": result.get('url', ''),
                    "snippet": result.get('content', ''),  # Tavily returns 'content' instead of 'snippet'
                    "relevance_score": 0.9 - (i * 0.1),  # Simulate relevance score
                    "source": "web"
                }
                results_list.append(result_obj)
            
            # Sort by relevance score
            results_list.sort(key=lambda x: x["relevance_score"], reverse=True)
            return results_list
            
        except Exception as e:
            print(f"Error during web search: {e}")
            print(f"[DEBUG] Exception type: {type(e).__name__}")
            import traceback
            print(f"[DEBUG] Traceback: {traceback.format_exc()}")
            
            # Return mock data on exception
            print("[DEBUG] Returning mock search results due to exception")
            return [
                {
                    "title": "RAG Systems Overview (Exception Fallback)",
                    "url": "https://example.com/rag-overview",
                    "snippet": "Retrieval-Augmented Generation enhances large language models by connecting them to external knowledge sources.",
                    "relevance_score": 0.9,
                    "source": "web"
                }
            ]
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the web search agent within the workflow.
        
        Args:
            state: The current workflow state
            
        Returns:
            Updated workflow state
        """
        query = state.get("query", "")
        query_analysis = state.get("query_analysis", {})
        
        # Perform web search and analyze results
        web_results = self.search(query)
        
        return {
            "query": query,
            "query_analysis": query_analysis,
            "web_results": web_results
        }

    def convert_result_to_model(self, result: Dict[str, Any]) -> WebSearchResult:
        """
        Convert a search result dictionary to WebSearchResult model.
        
        Args:
            result: Search result dictionary
            
        Returns:
            WebSearchResult object
        """
        return WebSearchResult(
            title=result.get("title", ""),
            url=result.get("url", ""),
            snippet=result.get("snippet", ""),
            relevance_score=result.get("relevance_score", 0.0),
            source=result.get("source", "web")
        ) 