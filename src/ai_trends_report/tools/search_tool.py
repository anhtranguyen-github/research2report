from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import os
import ssl
import requests
from dotenv import load_dotenv

load_dotenv()

ssl._create_default_https_context = ssl._create_unverified_context

class WebSearchToolInput(BaseModel):
    """Input schema for WebSearchTool."""
    query: str = Field(..., description="The search query to look up on the web.")

class WebSearchTool(BaseTool):
    name: str = "Web Search Tool"
    description: str = "Use this tool to search Google and retrieve the top search results about AI trends, news, and innovations."
    args_schema: Type[BaseModel] = WebSearchToolInput

    def _run(self, query: str) -> str:
        
        host = 'brd.superproxy.io'
        port = 33335

        username = os.getenv("BRIGHDATA_USERNAME")
        password = os.getenv("BRIGHDATA_PASSWORD")

        proxy_url = f'http://{username}:{password}@{host}:{port}'

        proxies = {
            'http': proxy_url,
            'https': proxy_url
        }

        search_query = "+".join(query.split(" "))

        url = f"https://www.google.com/search?q={search_query}&brd_json=1&num=50"
        response = requests.get(url, proxies=proxies, verify=False)

        return response.json()['organic'] 