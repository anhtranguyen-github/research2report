from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel

llm = LLM(model="ollama/gemma3:4b")

class ReportContent(BaseModel):
    """Content of an AI trends report"""
    content: str

@CrewBase
class ContentGenerationCrew:
    """Content Generation Crew that writes AI/Big Data reports, blog posts, and summaries"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def trend_writer_agent(self) -> Agent:
        return Agent(config=self.agents_config["trend_writer_agent"],
                    llm=llm)

    @task
    def write_trend_sections_task(self) -> Task:
        return Task(config=self.tasks_config["write_trend_sections_task"])
    
    @agent
    def summary_writer_agent(self) -> Agent:
        return Agent(config=self.agents_config["summary_writer_agent"],
                    llm=llm)

    @task
    def write_executive_summary_task(self) -> Task:
        return Task(config=self.tasks_config["write_executive_summary_task"])
    
    @agent
    def editor_agent(self) -> Agent:
        return Agent(config=self.agents_config["editor_agent"],
                    llm=llm)

    @task
    def edit_report_task(self) -> Task:
        return Task(config=self.tasks_config["edit_report_task"],
                   output_pydantic=ReportContent)

    @crew
    def crew(self) -> Crew:
        """Creates the Content Generation Crew"""

        return Crew(agents=self.agents,
                    tasks=self.tasks,
                    process=Process.sequential,
                    verbose=True) 