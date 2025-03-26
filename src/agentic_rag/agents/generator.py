"""Generator Agent for the Agentic RAG system."""

from typing import Dict, Any, List, Optional, Callable, Awaitable
import json
import re

from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable

from src.agentic_rag.core.model import get_ollama_llm
from src.agentic_rag.agents.types import GeneratedResponse, RetrievedDocument, WebSearchResult

class GeneratorAgent:
    """Agent that generates final responses based on retrieved information."""
    
    def __init__(
        self, 
        model_name: str = "qwen2.5",
        prompt_template: Optional[str] = None
    ):
        """
        Initialize the GeneratorAgent.
        
        Args:
            model_name: Name of the LLM model to use
            prompt_template: Optional prompt template to use
        """
        self.model_name = model_name
        self.response = None
        
        # Use the provided prompt template or a default one
        if prompt_template is None:
            from .config import get_prompt_template
            try:
                prompt_template = get_prompt_template("generator")
            except Exception as e:
                print(f"Error loading prompt template: {e}")
                # Default template if config can't be loaded
                prompt_template = """
                Generate a comprehensive answer to the following query using the provided information.
                
                Query: {query}
                
                Intent: {intent}
                
                Specific Questions:
                {questions}
                
                Context Requirements:
                {requires_context}
                
                Retrieved Information:
                {context}
                
                Relevance Assessment:
                {relevance_assessment}
                
                Provide a well-structured, accurate response that directly addresses the query.
                Cite sources where appropriate and be transparent about uncertainty.
                If you cannot fully answer the query with the given information, acknowledge the limitations.
                
                Format the response in an easy-to-read manner, using appropriate formatting.
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
        
        # Use task-specific model for response generation
        self.llm = get_ollama_llm(model_name=self.model_name, task_type="generation")
        
        # Create prompt from template
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def generate_response(
        self,
        query: str,
        query_analysis: Dict[str, Any],
        retrieved_docs: List[Dict[str, Any]],
        web_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate a response based on retrieved information.
        
        Args:
            query: User's query string
            query_analysis: Analysis of the user's query
            retrieved_docs: List of retrieved documents with relevance scores
            web_results: List of web search results with relevance scores
            
        Returns:
            Generated response with sources and confidence
        """
        try:
            # Format retrieved documents
            doc_text = "\n\n".join(
                f"Document {i+1} (Relevance: {doc['relevance_score']:.2f}):\n{doc['content']}"
                for i, doc in enumerate(retrieved_docs)
            )
            
            # Format web results
            web_text = "\n\n".join(
                f"Web Result {i+1} (Relevance: {result['relevance_score']:.2f}):\nTitle: {result['title']}\nURL: {result['url']}\nContent: {result['snippet']}"
                for i, result in enumerate(web_results)
            )
            
            # Format context from documents and web results
            context = doc_text
            if web_text:
                context += "\n\nWeb Search Results:\n" + web_text
            
            # Create relevance assessment
            relevance_assessment = "No relevant documents found."
            if retrieved_docs or web_results:
                relevance_assessment = "Retrieved documents and web search results provide relevant information for answering the query."
            
            # Generate response
            response_str = self.chain.invoke({
                "query": query,
                "intent": query_analysis.get("intent", ""),
                "questions": "\n".join(query_analysis.get("questions", [])),
                "requires_context": str(query_analysis.get("requires_context", True)),
                "context": context,
                "relevance_assessment": relevance_assessment
            })
            
            # Extract the JSON content between triple backticks if present
            if "```json" in response_str:
                json_match = re.search(r'```json\s*(.*?)\s*```', response_str, re.DOTALL)
                if json_match:
                    response_str = json_match.group(1)
            
            # Create a default GeneratedResponse
            response = {
                "response_text": response_str,
                "sources_used": [doc.get("source", "unknown") for doc in retrieved_docs] + 
                               [result.get("url", "unknown") for result in web_results],
                "confidence_score": 0.8,
                "follow_up_questions": []
            }
            
            # Validate against GeneratedResponse model
            return GeneratedResponse(**response).model_dump()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            # Return a default response in case of error
            return GeneratedResponse(
                response_text=f"I apologize, but I encountered an error while generating the response: {str(e)}. Please try again.",
                sources_used=[],
                confidence_score=0.0,
                follow_up_questions=[]
            ).model_dump()
    
    def run(
        self,
        state: Dict[str, Any],
        query_analysis: Dict[str, Any],
        retrieved_docs: List[RetrievedDocument],
        web_results: List[WebSearchResult]
    ) -> GeneratedResponse:
        """
        Run the generator agent within the workflow.
        
        Args:
            state: The current workflow state
            query_analysis: Analysis of the user's query
            retrieved_docs: List of retrieved documents
            web_results: List of web search results
            
        Returns:
            GeneratedResponse object containing the generated response
        """
        query = state.get("query", "")
        
        # Generate response
        response_dict = self.generate_response(
            query=query,
            query_analysis=query_analysis,
            retrieved_docs=[doc.model_dump() for doc in retrieved_docs],
            web_results=[result.model_dump() for result in web_results]
        )
        
        return GeneratedResponse(**response_dict)

    async def stream_response(
        self,
        query: str,
        query_analysis: Dict[str, Any],
        retrieved_docs: List[RetrievedDocument],
        web_results: List[WebSearchResult],
        callback: Callable[[str], Awaitable[None]]
    ) -> None:
        """
        Stream a response token by token for a given query.
        
        Args:
            query: User's query
            query_analysis: Analysis of the query
            retrieved_docs: Retrieved documents
            web_results: Web search results
            callback: Async callback function to handle each token
        """
        try:
            # Initialize the prompt
            prompt = self._create_prompt(
                query=query,
                query_analysis=query_analysis,
                retrieved_docs=[doc.model_dump() for doc in retrieved_docs],
                web_results=[result.model_dump() for result in web_results]
            )
            
            # Get a streaming LLM
            from src.agentic_rag.core.model import get_ollama_llm
            llm = get_ollama_llm(
                model=self.model_name,
                streaming=True
            )
            
            # Stream the response
            response_tokens = []
            async for chunk in llm.astream(prompt):
                token = chunk.content
                response_tokens.append(token)
                await callback(token)
                
            # Form the complete response
            full_response = "".join(response_tokens)
            
            # Return a validated GeneratedResponse object
            self.response = GeneratedResponse(
                response_text=full_response,
                sources_used=[doc.source for doc in retrieved_docs] + [result.url for result in web_results],
                confidence_score=0.9,  # Default confidence score
                follow_up_questions=[]  # Default empty follow-up questions
            )
            
            # Signal completion to the callback
            await callback("DONE")
        
        except Exception as e:
            error_msg = f"Error generating streaming response: {str(e)}"
            print(error_msg)
            # Send error message to callback
            await callback(f"\nError: {str(e)}")
            # Signal completion to the callback
            await callback("DONE")
            # Return an error response
            self.response = GeneratedResponse(
                response_text=f"I apologize, but I encountered an error while generating a response: {str(e)}",
                sources_used=[],
                confidence_score=0.0,
                follow_up_questions=[]
            )

    def _create_prompt(
        self,
        query: str,
        query_analysis: Dict[str, Any],
        retrieved_docs: List[Dict[str, Any]],
        web_results: List[Dict[str, Any]]
    ) -> str:
        """
        Create a prompt for the LLM based on the query and supporting information.
        
        Args:
            query: User's query string
            query_analysis: Analysis of the user's query
            retrieved_docs: List of retrieved documents
            web_results: List of web search results
            
        Returns:
            Formatted prompt string
        """
        # Format retrieved documents
        doc_text = "\n\n".join(
            f"Document {i+1} (Relevance: {doc.get('relevance_score', 0):.2f}):\n{doc.get('content', '')}"
            for i, doc in enumerate(retrieved_docs)
        ) if retrieved_docs else "No documents were retrieved."
        
        # Format web results
        web_text = "\n\n".join(
            f"Web Result {i+1} (Relevance: {result.get('relevance_score', 0):.2f}):\nTitle: {result.get('title', '')}\nURL: {result.get('url', '')}\nContent: {result.get('snippet', '')}"
            for i, result in enumerate(web_results)
        ) if web_results else "No web results were retrieved."
        
        # Format context from documents and web results
        context = doc_text
        if web_text and web_text != "No web results were retrieved.":
            context += "\n\nWeb Search Results:\n" + web_text
        
        # Create relevance assessment
        relevance_assessment = "No relevant documents found."
        if retrieved_docs or web_results:
            relevance_assessment = "Retrieved documents and web search results provide relevant information for answering the query."
        
        # Extract query analysis details
        intent = query_analysis.get("intent", "")
        questions = query_analysis.get("questions", [])
        requires_context = query_analysis.get("requires_context", True)
        
        # Format questions
        questions_text = "\n".join(f"- {q}" for q in questions) if questions else "No specific questions identified."
        
        # Create the full prompt
        return self.prompt.format(
            query=query,
            intent=intent,
            questions=questions_text,
            requires_context=str(requires_context),
            context=context,
            relevance_assessment=relevance_assessment
        )