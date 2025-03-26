"""Retriever Agent for the Agentic RAG system."""

from typing import Dict, Any, List, Optional
import json
import re

from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable

from src.agentic_rag.core.model import get_ollama_llm
from src.agentic_rag.agents.types import RetrievedDocument

class RetrieverAgent:
    """Agent that retrieves and evaluates relevant documents."""
    
    def __init__(
        self,
        model_name: str = "qwen2.5",
        prompt_template: Optional[str] = None,
        max_results: int = 5,
        vector_store: Optional[Any] = None
    ):
        """
        Initialize the RetrieverAgent.
        
        Args:
            model_name: Name of the LLM model to use
            prompt_template: Optional prompt template to use
            max_results: Maximum number of results to return
            vector_store: Optional vector store to use
        """
        self.model_name = model_name
        self.max_results = max_results
        self.vector_store = vector_store
        
        # Use the provided prompt template or a default one
        if prompt_template is None:
            from .config import get_prompt_template
            try:
                prompt_template = get_prompt_template("retriever")
            except Exception as e:
                print(f"Error loading prompt template: {e}")
                # Default template if config can't be loaded
                prompt_template = """
                Evaluate the relevance of the following documents for the query: {query}
                
                Query: {query}
                
                Documents:
                {documents}
                
                Your task is to evaluate how relevant each document is to the query.
                Output as a JSON array of documents, each with fields:
                - content: the document content
                - metadata: document metadata
                - relevance_score: a float between 0 and 1 indicating relevance
                - source: source of the document
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
        
        # Use task-specific model for retrieval evaluation
        self.llm = get_ollama_llm(model_name=self.model_name, task_type="retrieval_evaluation")
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def evaluate_relevance(self, query: str, documents: List[str]) -> List[Dict[str, Any]]:
        """
        Evaluate the relevance of retrieved documents.
        
        Args:
            query: User query string
            documents: List of document summaries to evaluate
            
        Returns:
            List of evaluated documents with relevance scores
        """
        try:
            # Format documents for prompt
            doc_text = "\n\n".join(f"Document {i+1}:\n{doc}" for i, doc in enumerate(documents))
            
            # Get the evaluation as a JSON string
            evaluation_str = self.chain.invoke({
                "query": query,
                "documents": doc_text
            })
            
            # Extract the JSON content between triple backticks if present
            if "```json" in evaluation_str:
                json_match = re.search(r'```json\s*(.*?)\s*```', evaluation_str, re.DOTALL)
                if json_match:
                    evaluation_str = json_match.group(1)
            
            # Parse the JSON
            evaluation = json.loads(evaluation_str)
            
            # Ensure we have a list of evaluated documents
            evaluated_docs = evaluation.get("evaluated_documents", [])
            
            # Validate against RetrievedDocument model and sort by relevance score
            results = [
                RetrievedDocument(**doc).model_dump()
                for doc in evaluated_docs
            ]
            
            # Sort by relevance score and limit to max_results
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            return results[:self.max_results]
            
        except Exception as e:
            print(f"Error evaluating documents: {e}")
            # Return empty list in case of error
            return []
    
    def run(self, state: Dict[str, Any]) -> List[RetrievedDocument]:
        """
        Run the retriever within the workflow.
        
        Args:
            state: The current workflow state
            
        Returns:
            List of RetrievedDocument objects containing evaluated documents
        """
        query = state.get("query", "")
        documents = state.get("documents", [])
        
        # Evaluate document relevance
        evaluated_docs = self.evaluate_relevance(query, documents)
        
        # Convert to RetrievedDocument objects
        return [RetrievedDocument(**doc) for doc in evaluated_docs] 