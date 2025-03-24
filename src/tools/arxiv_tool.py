from typing import Type
import arxiv
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class ArXivSearchToolInput(BaseModel):
    """Input schema for ArXivSearchTool."""
    query: str = Field(..., description="The query to search for AI and Big Data research papers on ArXiv.")
    max_results: int = Field(default=10, description="Maximum number of results to return.")

class ArXivSearchTool(BaseTool):
    name: str = "ArXiv Search Tool"
    description: str = "Use this tool to search for AI and Big Data research papers on ArXiv."
    args_schema: Type[BaseModel] = ArXivSearchToolInput

    def _run(self, query: str, max_results: int = 10) -> str:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        results = []
        for paper in client.results(search):
            results.append({
                "title": paper.title,
                "authors": ", ".join(author.name for author in paper.authors),
                "summary": paper.summary,
                "published": paper.published.strftime("%Y-%m-%d"),
                "url": paper.pdf_url
            })

        return results 