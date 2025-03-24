#!/usr/bin/env python
import os
import asyncio

from pydantic import BaseModel

from crewai.flow import Flow, listen, start

from crews.ContentGeneration_crew.content_generation_crew import ContentGenerationCrew
from dotenv import load_dotenv

load_dotenv()

class Trend(BaseModel):
    title: str = ""
    description: str = ""
    sources: list[str] = []

class TrendData(BaseModel):
    trend_id: str = ""
    research_papers: list[str] = []
    news_articles: list[str] = []
    social_media_posts: list[str] = []
    github_projects: list[str] = []
    analysis: str = ""

class TrendReportState(BaseModel):
    week: str = ""
    discovered_trends: list[Trend] = []
    trend_data: list[TrendData] = []
    report_content: str = ""


class TrendReportFlow(Flow[TrendReportState]):

    @start()
    def discover_trends(self):
        print("Discovering AI trends")
        trends = TrendDiscoveryCrew().crew().kickoff()
        self.state.discovered_trends = trends.pydantic.trends

    @listen(discover_trends)
    async def extract_data(self):
        print("Extracting data for each trend")
        tasks = []

        async def extract_trend_data(trend: Trend):
            result = (
                DataExtractionCrew()
                .crew()
                .kickoff(inputs={
                    "trend_title": trend.title,
                    "trend_description": trend.description
                })
            )
            return result.pydantic

        # Create tasks for each trend
        for trend in self.state.discovered_trends:
            task = asyncio.create_task(
                extract_trend_data(trend)
            )
            tasks.append(task)

        # Wait for all data extraction to complete concurrently
        trend_data = await asyncio.gather(*tasks)
        print(f"Extracted data for {len(trend_data)} trends")
        self.state.trend_data.extend(trend_data)

    @listen(extract_data)
    def generate_report(self):
        print("Generating AI trends report")
        report = ContentGenerationCrew().crew().kickoff(inputs={
            "trends": self.state.discovered_trends,
            "trend_data": self.state.trend_data,
            "week": self.state.week
        })
        self.state.report_content = report.pydantic.content
        
        # Save report to file
        with open("ai_trends_report.md", "w") as f:
            f.write(self.state.report_content)
        print("Report saved to ai_trends_report.md")


def kickoff():
    trend_report_flow = TrendReportFlow()
    asyncio.run(trend_report_flow.kickoff_async())


def plot():
    trend_report_flow = TrendReportFlow()
    trend_report_flow.plot()


if __name__ == "__main__":
    kickoff() 