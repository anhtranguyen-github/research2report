"""Generator Agent for the Agentic RAG system."""

from typing import Dict, Any, List, Optional, TypedDict, Union

from langchain.schema.document import Document
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

from src.agentic_rag.core.model import get_ollama_llm
from src.agentic_rag.agents.query_analyzer import QueryAnalysis
from src.agentic_rag.agents.retriever import RetrievalResult

class GeneratorAgent:
    """Agent that generates responses based on retrieved information."""
    
    def __init__(self, model_name: str = "qwen2.5"):
        """
        Initialize the Generator Agent.
        
        Args:
            model_name: Name of the LLM model to use
        """
        self.llm = get_ollama_llm(model_name=model_name)
        self.prompt = self._create_prompt()
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def _create_prompt(self) -> PromptTemplate:
        """Create the prompt template for response generation."""
        template = """You are an expert AI assistant tasked with providing accurate, helpful responses to user queries.
Use the retrieved information to craft a comprehensive answer. If the retrieved information is insufficient,
state clearly what is missing and provide the best response with the available information.

User Query: {query}

Query Analysis:
- Intent: {intent}
- Specific Questions: {questions}
- Requires Context: {requires_context}

Retrieved Information:
{context}

Relevance Assessment:
{relevance_assessment}

Instructions:
1. Answer the user's query based on the retrieved information
2. Make explicit references to the sources when appropriate
3. If the retrieved information is insufficient, acknowledge this limitation
4. Provide accurate, concise information without hallucinating facts
5. Format your response appropriately for readability

Your response:"""
        
        return PromptTemplate.from_template(template)
    
    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents as context for the prompt.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant documents were found."
        
        context_parts = []
        for i, doc in enumerate(documents):
            context_parts.append(f"Document {i+1}:\n{doc.page_content}\n")
        
        return "\n".join(context_parts)
    
    def format_questions(self, questions: List[str]) -> str:
        """
        Format questions for the prompt.
        
        Args:
            questions: List of questions
            
        Returns:
            Formatted questions string
        """
        if not questions:
            return "No specific questions identified."
        
        return ", ".join(questions)
    
    def generate(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        retrieval_result: RetrievalResult
    ) -> str:
        """
        Generate a response based on the query and retrieved information.
        
        Args:
            query: The user's query string
            query_analysis: Structured analysis of the query
            retrieval_result: Result of the retrieval process
            
        Returns:
            Generated response
        """
        context = self.format_context(retrieval_result.documents)
        questions = self.format_questions(query_analysis["questions"])
        
        return self.chain.invoke({
            "query": query,
            "intent": query_analysis["intent"],
            "questions": questions,
            "requires_context": str(query_analysis["requires_context"]),
            "context": context,
            "relevance_assessment": retrieval_result.relevance_assessment
        })