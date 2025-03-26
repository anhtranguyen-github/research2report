"""Type definitions for agent results and responses."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class QueryAnalysis(BaseModel):
    """Analysis of a user query to determine processing strategy."""
    query: Optional[str] = ""
    intent: str
    questions: List[str]
    search_terms: List[str]
    requires_context: bool
    reasoning: str
    requires_retrieval: Optional[bool] = True
    requires_web_search: Optional[bool] = True
    specific_questions: Optional[List[str]] = Field(default_factory=list)
    context_requirements: Optional[Dict[str, Any]] = Field(default_factory=dict)

class RetrievedDocument(BaseModel):
    """A document retrieved from the knowledge base."""
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    source: str

class WebSearchResult(BaseModel):
    """A result from web search."""
    title: str
    url: str
    snippet: str
    relevance_score: float
    source: str = "web"

class GeneratedResponse(BaseModel):
    """Generated response to user query."""
    response_text: str
    sources_used: List[str]
    confidence_score: float
    follow_up_questions: Optional[List[str]] = Field(default_factory=list)

class AgentState(BaseModel):
    """Current state of the agent workflow."""
    query: str
    query_analysis: Optional[QueryAnalysis] = None
    retrieved_documents: List[RetrievedDocument] = []
    web_results: List[WebSearchResult] = []
    response: Optional[GeneratedResponse] = None
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional workflow metadata") 