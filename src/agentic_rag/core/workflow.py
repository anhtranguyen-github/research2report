"""Core workflow implementation for the Agentic RAG system."""

from typing import Dict, Any, Optional, List, Callable, Awaitable
from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore

from ..agents.managers import AgentManager
from ..agents.types import AgentState

class AgenticRAGWorkflow:
    """Main workflow for the Agentic RAG system."""
    
    def __init__(
        self,
        vector_store: Optional[QdrantVectorStore] = None,
        model_name: str = "qwen2.5",
        use_web_search: bool = True,
        config_dir: str = "src/agentic_rag/agents/config"
    ):
        """
        Initialize the workflow.
        
        Args:
            vector_store: Vector store instance
            model_name: Name of the LLM model to use
            use_web_search: Whether to use web search
            config_dir: Directory containing agent configuration files
        """
        print(f"Initializing AgenticRAGWorkflow with model={model_name}, web_search={use_web_search}")  # Debug log
        self.vector_store = vector_store
        self.model_name = model_name
        self.use_web_search = use_web_search
        self._last_result = None
        
        # Initialize the AgentManager
        self.agent_manager = AgentManager(config_dir=config_dir)
    
    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the workflow on a query.
        
        Args:
            query: User's query string
            
        Returns:
            Dictionary containing the results
        """
        try:
            print(f"\n=== Starting Workflow for Query: {query} ===")
            print(f"Web search enabled: {self.use_web_search}")
            
            # Process the query using the AgentManager
            state = self.agent_manager.process_query(query)
            
            # Store the result for later retrieval
            self._last_result = state
            
            # Convert the agent state to a dictionary for API response
            return {
                "response": state.response.response_text if state.response else "",
                "query_analysis": state.query_analysis.model_dump() if state.query_analysis else None,
                "retrieval_result": [doc.model_dump() for doc in state.retrieved_documents] if state.retrieved_documents else [],
                "web_search_result": [result.model_dump() for result in state.web_results] if state.web_results else []
            }
        
        except Exception as e:
            print(f"Error in workflow run: {str(e)}")  # Debug log
            return {"error": str(e)}

    def get_last_result(self) -> Optional[Dict[str, Any]]:
        """
        Get the last result from the workflow.
        
        Returns:
            The last result dictionary or None if no result exists
        """
        if not self._last_result:
            return None
            
        state = self._last_result
        return {
            "response": state.response.response_text if state.response else "",
            "query_analysis": state.query_analysis.model_dump() if state.query_analysis else None,
            "retrieval_result": [doc.model_dump() for doc in state.retrieved_documents] if state.retrieved_documents else [],
            "web_search_result": [result.model_dump() for result in state.web_results] if state.web_results else []
        }

    async def stream(self, query: str, callback: Callable[[str], Awaitable[None]]) -> None:
        """
        Stream the workflow execution with token-by-token generation.
        
        Args:
            query: User's query string
            callback: Async callback function to handle streaming tokens
        """
        try:
            print(f"\n=== Starting Streaming Workflow for Query: {query} ===")
            print(f"Web search enabled: {self.use_web_search}")
            
            # Initialize the state
            state = AgentState(query=query)
            
            # Step 1: Query Analysis
            print("\nStep 1: Query Analysis")
            query_analysis = self.agent_manager.agents['query_analyzer'].run({"query": query})
            state.query_analysis = query_analysis
            
            # Step 2a: Document Retrieval (if needed)
            if query_analysis.requires_retrieval:
                print("\nStep 2a: Document Retrieval")
                retrieved_docs = self.agent_manager.agents['retriever'].run({
                    "query": query,
                    "documents": [],  # TODO: Add document retrieval logic
                    "analysis": query_analysis
                })
                state.retrieved_documents = retrieved_docs
            else:
                state.retrieved_documents = []
            
            # Step 2b: Web Search (if needed)
            if self.use_web_search and query_analysis.requires_web_search:
                print("\nStep 2b: Web Search")
                web_search_agent = self.agent_manager.agents['web_search']
                web_results = web_search_agent.search(
                    query=query,
                    analysis=query_analysis
                )
                state.web_results = [web_search_agent.convert_result_to_model(r) for r in web_results]
            else:
                state.web_results = []
            
            # Step 3: Generate Response with Streaming
            print("\nStep 3: Generate Response (Streaming)")
            
            # Prepare the inputs for the generator
            generator = self.agent_manager.agents['generator']
            
            # Start streaming generation
            await generator.stream_response(
                query=query,
                query_analysis=query_analysis.model_dump(),
                retrieved_docs=state.retrieved_documents,
                web_results=state.web_results,
                callback=callback
            )
            
            # Store the final state (without the streaming response)
            self._last_result = state
            
            print("\n=== Streaming Workflow Complete ===\n")
            
        except Exception as e:
            print(f"Error in streaming workflow: {str(e)}")
            error_message = f"Error: {str(e)}"
            await callback(error_message)
            raise 