"""Retriever Agent for the Agentic RAG system."""

from typing import Dict, Any, List, Optional, TypedDict, Union
from dataclasses import dataclass

from langchain.schema.document import Document
from langchain.schema.runnable import Runnable
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_qdrant import QdrantVectorStore

from src.agentic_rag.core.model import get_ollama_llm

@dataclass
class RetrievalResult:
    """Result of the retrieval process."""
    
    documents: List[Document]
    search_terms: List[str]
    relevance_assessment: str

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
            vector_store: Vector store instance for retrieving documents
            model_name: Name of the LLM model to use
            max_results: Maximum number of results to retrieve
        """
        self.vector_store = vector_store
        self.llm = get_ollama_llm(model_name=model_name)
        self.max_results = max_results
        self.relevance_prompt = self._create_relevance_prompt()
        self.relevance_chain = self.relevance_prompt | self.llm | StrOutputParser()
    
    def _create_relevance_prompt(self) -> PromptTemplate:
        """Create the prompt template for assessing document relevance."""
        template = """You are an expert at evaluating the relevance of retrieved documents to a user's query.
You need to determine if the retrieved documents provide useful information to answer the query.

User Query: {query}

Retrieved Documents:
{document_summaries}

Evaluate the relevance of these documents to the query and explain why they are or aren't helpful.
Focus on assessing if the documents contain information that directly answers the query or provides 
important context.

Your response should be a paragraph explaining the relevance of the documents."""
        
        return PromptTemplate.from_template(template)
    
    def generate_document_summaries(self, documents: List[Document]) -> str:
        """
        Generate summaries of retrieved documents for relevance assessment.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            String containing document summaries
        """
        summaries = []
        for i, doc in enumerate(documents):
            content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            metadata_str = ", ".join([f"{k}: {v}" for k, v in doc.metadata.items()])
            summaries.append(f"Document {i+1}:\nContent: {content}\nMetadata: {metadata_str}\n")
        
        return "\n".join(summaries)
    
    def retrieve(self, query: str, search_terms: List[str]) -> RetrievalResult:
        """
        Retrieve relevant documents based on the query and search terms.
        
        Args:
            query: The user's query string
            search_terms: List of search terms extracted from the query
            
        Returns:
            RetrievalResult containing documents and relevance assessment
        """
        all_documents = []
        
        # Use both the original query and extracted search terms
        search_queries = [query] + search_terms
        
        # Remove duplicates while preserving order
        unique_search_queries = []
        for q in search_queries:
            if q not in unique_search_queries:
                unique_search_queries.append(q)
        
        # Retrieve documents for each search query
        for search_query in unique_search_queries:
            documents = self.vector_store.similarity_search(
                query=search_query,
                k=self.max_results
            )
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
            search_terms=unique_search_queries,
            relevance_assessment=relevance_assessment
        ) 