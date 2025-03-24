from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel

from ai_trends_report.tools.search_tool import WebSearchTool
from ai_trends_report.tools.arxiv_tool import ArXivSearchTool
from ai_trends_report.tools.github_tool import GitHubSearchTool

llm = LLM(model="ollama/gemma3:4b")

class TrendData(BaseModel):
    """Data extracted for a specific AI trend"""
    trend_id: str
    research_papers: list[str]
    news_articles: list[str]
    social_media_posts: list[str]
    github_projects: list[str]
    analysis: str

@CrewBase
class DataExtractionCrew:
    """Data Extraction Crew that fetches research papers, news articles, tweets, and GitHub projects"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def research_paper_agent(self) -> Agent:
        return Agent(config=self.agents_config["research_paper_agent"],
                    tools=[ArXivSearchTool()],
                    llm=llm)

    @task
    def extract_research_papers_task(self) -> Task:
        return Task(config=self.tasks_config["extract_research_papers_task"])
    
    @agent
    def news_article_agent(self) -> Agent:
        return Agent(config=self.agents_config["news_article_agent"],
                    tools=[WebSearchTool()],
                    llm=llm)

    @task
    def extract_news_articles_task(self) -> Task:
        return Task(config=self.tasks_config["extract_news_articles_task"])
    
    @agent
    def social_media_agent(self) -> Agent:
        return Agent(config=self.agents_config["social_media_agent"],
                    tools=[WebSearchTool()],
                    llm=llm)

    @task
    def extract_social_media_posts_task(self) -> Task:
        return Task(config=self.tasks_config["extract_social_media_posts_task"])
    
    @agent
    def github_agent(self) -> Agent:
        return Agent(config=self.agents_config["github_agent"],
                    tools=[GitHubSearchTool()],
                    llm=llm)

    @task
    def extract_github_projects_task(self) -> Task:
        return Task(config=self.tasks_config["extract_github_projects_task"])
    
    @agent
    def data_analyst_agent(self) -> Agent:
        return Agent(config=self.agents_config["data_analyst_agent"],
                    llm=llm)

    @task
    def analyze_extracted_data_task(self) -> Task:
        return Task(config=self.tasks_config["analyze_extracted_data_task"],
                    output_pydantic=TrendData)

    @crew
    def crew(self) -> Crew:
        """Creates the Data Extraction Crew"""

        return Crew(agents=self.agents,
                    tasks=self.tasks,
                    process=Process.sequential,
                    verbose=True) 