"""Web search tool for the Agentic RAG system."""

import os
import requests
from typing import Dict, Any, List, Optional

from langchain.schema.document import Document

class WebSearchTool:
    """Tool for searching the web to supplement vector store information."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        search_engine: str = "google",
        num_results: int = 5
    ):
        """
        Initialize the Web Search Tool.
        
        Args:
            api_key: API key for the search engine (optional if using environment variable)
            search_engine: The search engine to use ("google", "bing", etc.)
            num_results: Number of results to return
        """
        self.api_key = api_key or os.getenv("SEARCH_API_KEY")
        self.search_engine = search_engine
        self.num_results = num_results
        
        if not self.api_key:
            print("Warning: No search API key provided. Web search functionality will be limited.")
    
    def search(self, query: str) -> List[Document]:
        """
        Perform a web search for the given query.
        
        In a real implementation, this would connect to a search API.
        This is a placeholder that would be replaced with actual API calls.
        
        Args:
            query: Search query string
            
        Returns:
            List of Document objects containing search results
        """
        # Mock implementation - in production, replace with actual API call
        print(f"Performing web search for: {query}")
        
        # Placeholder for demonstration purposes
        # In a real implementation, you would:
        # 1. Call the appropriate search API
        # 2. Process the results
        # 3. Convert to Document objects
        
        # Example mock results
        mock_results = [
            Document(
                page_content="This is a mock search result for the query: " + query,
                metadata={"source": "web_search", "url": "https://example.com/result1"}
            ),
            Document(
                page_content="Another mock search result with different information.",
                metadata={"source": "web_search", "url": "https://example.com/result2"}
            )
        ]
        
        return mock_results
    
    def search_and_convert_to_documents(self, query: str) -> List[Document]:
        """
        Search the web and convert results to Document objects for RAG.
        
        Args:
            query: Search query string
            
        Returns:
            List of Document objects containing search results
        """
        results = self.search(query)
        
        # In a real implementation, you would process the API response here
        # For this example, we just return the mock results directly
        
        return results 
        
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the web search tool within the workflow.
        
        Args:
            state: The current workflow state
            
        Returns:
            Updated workflow state
        """
        query = state.get("query", "")
        query_analysis = state.get("query_analysis", {})
        retrieval_result = state.get("retrieval_result", None)
        
        # Use both original query and search terms from query analysis
        search_terms = []
        if query_analysis and "search_terms" in query_analysis:
            search_terms = query_analysis["search_terms"]
        
        # Combine all search terms into one query to avoid too many API calls
        combined_query = query
        if search_terms:
            combined_query = f"{query} {' '.join(search_terms)}"
            
        # Perform web search
        web_search_results = self.search_and_convert_to_documents(combined_query)
        
        return {
            "query": query,
            "query_analysis": query_analysis,
            "retrieval_result": retrieval_result,
            "web_search_result": web_search_results
        } 