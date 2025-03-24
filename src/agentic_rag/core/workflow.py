"""Core workflow implementation for the Agentic RAG system."""

from typing import Dict, Any, Optional, List
from langgraph.graph import Graph
from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore

from src.agentic_rag.agents.query_analyzer import QueryAnalysis
from src.agentic_rag.agents.retriever import RetrieverAgent
from src.agentic_rag.tools.web_search import WebSearchTool
from src.agentic_rag.agents.generator import GeneratorAgent

class AgenticRAGWorkflow:
    """Main workflow for the Agentic RAG system."""
    
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        model_name: str = "qwen2.5",
        use_web_search: bool = True
    ):
        """
        Initialize the workflow.
        
        Args:
            vector_store: Vector store instance
            model_name: Name of the LLM model to use
            use_web_search: Whether to use web search
        """
        self.vector_store = vector_store
        self.model_name = model_name
        self.use_web_search = use_web_search
        
        # Initialize agents
        self.query_analyzer = QueryAnalysis(model_name=model_name)
        self.retriever = RetrieverAgent(vector_store=vector_store)
        self.web_search = WebSearchTool() if use_web_search else None
        self.generator = GeneratorAgent(model_name=model_name)
        
        # Create the workflow graph
        self.graph = self._create_graph()
    
    def _create_graph(self) -> Graph:
        """
        Create the workflow graph.
        
        Returns:
            Graph instance
        """
        # Define the nodes
        nodes = {
            "analyze_query": self.query_analyzer.run,
            "retrieve_documents": self.retriever.run,
            "web_search": self.web_search.run if self.web_search else None,
            "generate_response": self.generator.run
        }
        
        # Create the graph
        graph = Graph()
        
        # Add nodes
        for name, func in nodes.items():
            if func is not None:
                graph.add_node(name, func)
        
        # Add edges
        graph.add_edge("analyze_query", "retrieve_documents")
        if self.web_search:
            graph.add_edge("retrieve_documents", "web_search")
            graph.add_edge("web_search", "generate_response")
        else:
            graph.add_edge("retrieve_documents", "generate_response")
        
        # Set entry and exit points
        graph.set_entry_point("analyze_query")
        graph.set_exit_point("generate_response")
        
        return graph
    
    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the workflow on a query.
        
        Args:
            query: User's query string
            
        Returns:
            Dictionary containing the results
        """
        try:
            # Run the graph
            result = self.graph.run({"query": query})
            
            return {
                "response": result.get("response", ""),
                "query_analysis": result.get("query_analysis"),
                "retrieval_result": result.get("retrieval_result"),
                "web_search_result": result.get("web_search_result")
            }
        
        except Exception as e:
            return {"error": str(e)} 