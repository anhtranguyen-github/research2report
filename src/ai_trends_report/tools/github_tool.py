from typing import Type
import os
import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class GitHubSearchToolInput(BaseModel):
    """Input schema for GitHubSearchTool."""
    query: str = Field(..., description="The query to search for AI and Big Data projects on GitHub.")
    sort: str = Field(default="stars", description="Sort criteria (stars, forks, updated).")
    max_results: int = Field(default=10, description="Maximum number of results to return.")

class GitHubSearchTool(BaseTool):
    name: str = "GitHub Search Tool"
    description: str = "Use this tool to search for AI and Big Data projects on GitHub."
    args_schema: Type[BaseModel] = GitHubSearchToolInput

    def _run(self, query: str, sort: str = "stars", max_results: int = 10) -> str:
        github_token = os.getenv("GITHUB_TOKEN")
        
        headers = {
            "Accept": "application/vnd.github.v3+json"
        }
        
        if github_token:
            headers["Authorization"] = f"token {github_token}"
        
        url = f"https://api.github.com/search/repositories?q={query}&sort={sort}&order=desc&per_page={max_results}"
        
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.json().get('message', 'Unknown error')}"
        
        data = response.json()
        repos = data.get("items", [])
        
        results = []
        for repo in repos:
            results.append({
                "name": repo["name"],
                "full_name": repo["full_name"],
                "description": repo["description"],
                "url": repo["html_url"],
                "stars": repo["stargazers_count"],
                "forks": repo["forks_count"],
                "language": repo["language"],
                "created_at": repo["created_at"],
                "updated_at": repo["updated_at"]
            })
        
        return results 