"""Retriever Agent for the Agentic RAG system."""

from typing import Dict, Any, List, Optional, NamedTuple
from dataclasses import dataclass, field

from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.document import Document
from langchain_qdrant import QdrantVectorStore

from src.agentic_rag.core.model import get_ollama_llm

@dataclass
class RetrievalResult:
    """Result of the retrieval process."""
    
    documents: List[Document] = field(default_factory=list)
    search_terms: List[str] = field(default_factory=list)
    relevance_assessment: str = ""

class RetrieverAgent:
    """Agent that retrieves and evaluates relevant information from the vector store."""
    
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        model_name: str = "qwen2.5",
        max_results: int = 5
    ):
        """
        Initialize the Retriever Agent.
        
        Args:
            vector_store: Vector store instance
            model_name: Name of the LLM model to use
            max_results: Maximum number of results to return
        """
        self.vector_store = vector_store
        self.max_results = max_results
        
        # Use task-specific model for relevance evaluation
        self.llm = get_ollama_llm(model_name=model_name, task_type="retrieval_evaluation")
        
        # Create the relevance assessment chain
        self.relevance_prompt = self._create_relevance_prompt()
        self.relevance_chain = self.relevance_prompt | self.llm | StrOutputParser()
    
    def _create_relevance_prompt(self) -> PromptTemplate:
        """Create the prompt template for relevance assessment."""
        template = """As an information retrieval expert, assess the relevance of the following retrieved documents to the user's query.

User Query: {query}

Retrieved Document Summaries:
{document_summaries}

Instructions:
1. Analyze how well the retrieved documents address the user's query
2. Identify any aspects of the query that are not covered
3. Rate the overall relevance of the results (low/medium/high)
4. Suggest what additional information might be needed

Your assessment:"""
        
        return PromptTemplate.from_template(template)
    
    def generate_document_summaries(self, documents: List[Document]) -> str:
        """
        Generate summaries of retrieved documents.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            String containing document summaries
        """
        if not documents:
            return "No documents retrieved."
        
        summaries = []
        for i, doc in enumerate(documents, 1):
            # Truncate document content if too long
            content = doc.page_content
            if len(content) > 500:
                content = content[:497] + "..."
            
            summaries.append(f"Document {i}:\n{content}")
        
        return "\n\n".join(summaries)
    
    def retrieve(self, query: str, search_terms: List[str] = None) -> RetrievalResult:
        """
        Retrieve relevant documents from the vector store.
        
        Args:
            query: User's query string
            search_terms: Additional search terms from query analysis
            
        Returns:
            Retrieval result containing documents and assessment
        """
        if not search_terms:
            search_terms = []
        
        # Create a set of unique search queries
        unique_search_queries = set([query])
        for term in search_terms:
            unique_search_queries.add(term)
        
        # Retrieve documents for each unique query
        all_documents = []
        for search_query in unique_search_queries:
            documents = self.vector_store.similarity_search(search_query, k=self.max_results)
            all_documents.extend(documents)
        
        # Remove duplicate documents based on content
        unique_documents = []
        seen_contents = set()
        
        for doc in all_documents:
            if doc.page_content not in seen_contents:
                unique_documents.append(doc)
                seen_contents.add(doc.page_content)
        
        # Limit to max_results
        retrieved_documents = unique_documents[:self.max_results]
        
        # Assess relevance of retrieved documents
        document_summaries = self.generate_document_summaries(retrieved_documents)
        relevance_assessment = self.relevance_chain.invoke({
            "query": query,
            "document_summaries": document_summaries
        })
        
        return RetrievalResult(
            documents=retrieved_documents,
            search_terms=list(unique_search_queries),
            relevance_assessment=relevance_assessment
        ) 
        
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the retriever agent within the workflow.
        
        Args:
            state: The current workflow state
            
        Returns:
            Updated workflow state
        """
        query = state.get("query", "")
        query_analysis = state.get("query_analysis", {})
        search_terms = query_analysis.get("search_terms", [])
        
        retrieval_result = self.retrieve(query, search_terms)
        
        return {
            "query": query,
            "query_analysis": query_analysis,
            "retrieval_result": retrieval_result
        } 