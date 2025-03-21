from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel

from ai_trends_report.tools.search_tool import WebSearchTool

llm = LLM(model="ollama/gemma3:4b")

class Trend(BaseModel):
    """An AI or Big Data trend"""
    title: str
    description: str
    sources: list[str]

class TrendList(BaseModel):
    """List of discovered AI trends"""
    trends: list[Trend]

@CrewBase
class TrendDiscoveryCrew:
    """Trend Discovery Crew that monitors AI & Big Data news, research papers, social media, and developer forums"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def news_monitor_agent(self) -> Agent:
        return Agent(config=self.agents_config["news_monitor_agent"],
                    tools=[WebSearchTool()],
                    llm=llm)

    @task
    def monitor_news_task(self) -> Task:
        return Task(config=self.tasks_config["monitor_news_task"])
    
    @agent
    def research_monitor_agent(self) -> Agent:
        return Agent(config=self.agents_config["research_monitor_agent"],
                    tools=[WebSearchTool()],
                    llm=llm)

    @task
    def monitor_research_task(self) -> Task:
        return Task(config=self.tasks_config["monitor_research_task"])
    
    @agent
    def social_media_monitor_agent(self) -> Agent:
        return Agent(config=self.agents_config["social_media_monitor_agent"],
                    tools=[WebSearchTool()],
                    llm=llm)

    @task
    def monitor_social_media_task(self) -> Task:
        return Task(config=self.tasks_config["monitor_social_media_task"])
    
    @agent
    def developer_forum_monitor_agent(self) -> Agent:
        return Agent(config=self.agents_config["developer_forum_monitor_agent"],
                    tools=[WebSearchTool()],
                    llm=llm)

    @task
    def monitor_developer_forums_task(self) -> Task:
        return Task(config=self.tasks_config["monitor_developer_forums_task"])
    
    @agent
    def trend_analyst_agent(self) -> Agent:
        return Agent(config=self.agents_config["trend_analyst_agent"],
                    llm=llm)

    @task
    def analyze_trends_task(self) -> Task:
        return Task(config=self.tasks_config["analyze_trends_task"],
                    output_pydantic=TrendList)

    @crew
    def crew(self) -> Crew:
        """Creates the Trend Discovery Crew"""

        return Crew(agents=self.agents,
                    tasks=self.tasks,
                    process=Process.sequential,
                    verbose=True) 