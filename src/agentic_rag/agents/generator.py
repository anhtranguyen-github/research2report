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
        # Use task-specific model for generation
        self.llm = get_ollama_llm(model_name=model_name, task_type="generation")
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
    
    def generate(self, query: str, query_analysis: Dict[str, Any], retrieval_result: Any) -> str:
        """
        Generate a response based on retrieved information.
        
        Args:
            query: User's query string
            query_analysis: Analysis of the user's query
            retrieval_result: Result of document retrieval
            
        Returns:
            Generated response
        """
        # Extract information from query analysis
        intent = query_analysis.get("intent", "") if query_analysis else ""
        questions = ", ".join(query_analysis.get("questions", [])) if query_analysis else ""
        requires_context = query_analysis.get("requires_context", True) if query_analysis else True
        
        # Extract information from retrieval result
        context = ""
        relevance_assessment = ""
        
        if retrieval_result:
            # Extract document content
            docs = retrieval_result.documents if hasattr(retrieval_result, "documents") else []
            if docs:
                doc_texts = []
                for i, doc in enumerate(docs):
                    doc_texts.append(f"Source {i+1}:\n{doc.page_content}")
                context = "\n\n".join(doc_texts)
            
            # Extract relevance assessment
            if hasattr(retrieval_result, "relevance_assessment"):
                relevance_assessment = retrieval_result.relevance_assessment
        
        # Generate the response
        response = self.chain.invoke({
            "query": query,
            "intent": intent,
            "questions": questions,
            "requires_context": str(requires_context),
            "context": context or "No relevant information retrieved.",
            "relevance_assessment": relevance_assessment or "No relevance assessment available."
        })
        
        return response
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the generator agent within the workflow.
        
        Args:
            state: The current workflow state
            
        Returns:
            Updated workflow state
        """
        query = state.get("query", "")
        query_analysis = state.get("query_analysis", {})
        retrieval_result = state.get("retrieval_result", None)
        web_search_result = state.get("web_search_result", None)
        
        # Generate response using retrieved information
        response = self.generate(query, query_analysis, retrieval_result)
        
        return {
            "query": query,
            "query_analysis": query_analysis,
            "retrieval_result": retrieval_result,
            "web_search_result": web_search_result,
            "response": response
        }