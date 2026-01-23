"""CrewAI Stock Analysis Crew Definition"""
import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from tools.search import WebSearchTool, WebScrapeTool


@CrewBase
class StockAnalysisCrew:
    """Stock Analysis crew for comprehensive investment research"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self):
        # Get Ollama model from environment (set by main.py)
        model_name = os.environ.get("OLLAMA_MODEL", "ollama/qwen2.5:14b")
        self.llm = LLM(
            model=model_name,
            base_url="http://localhost:11434",
            temperature=0.1,  # Low temp for factual analysis
        )

    @agent
    def research_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["research_analyst"],
            tools=[WebSearchTool(), WebScrapeTool()],
            llm=self.llm,
            verbose=True,
        )

    @agent
    def financial_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["financial_analyst"],
            tools=[WebSearchTool()],
            llm=self.llm,
            verbose=True,
        )

    @agent
    def investment_advisor(self) -> Agent:
        return Agent(
            config=self.agents_config["investment_advisor"],
            llm=self.llm,
            verbose=True,
        )

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config["research_task"])

    @task
    def analysis_task(self) -> Task:
        return Task(config=self.tasks_config["analysis_task"])

    @task
    def recommendation_task(self) -> Task:
        return Task(config=self.tasks_config["recommendation_task"])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
