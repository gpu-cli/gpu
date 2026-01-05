#!/usr/bin/env python3
"""
Multi-Agent Example with CrewAI

Demonstrates a research team of AI agents powered by local vLLM backend.
Run serve.py first, then run this script locally or on the pod.

Usage:
    # Terminal 1: Start vLLM server on GPU pod
    gpu run --publish 8000:8000 python serve.py

    # Terminal 2: Run agents (can be local or on pod)
    python agent_example.py --topic "AI trends in 2026"
"""

import argparse
import os

# Set up OpenAI-compatible endpoint before importing CrewAI
os.environ["OPENAI_API_BASE"] = os.environ.get(
    "OPENAI_API_BASE", "http://localhost:8000/v1"
)
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "not-needed")

from crewai import Agent, Crew, Process, Task
from langchain_openai import ChatOpenAI


def create_research_crew(topic: str, model_name: str = "Qwen/Qwen2.5-72B-Instruct-AWQ"):
    """Create a research team with multiple specialized agents."""

    # Configure LLM to use local vLLM server
    llm = ChatOpenAI(
        model=model_name,
        base_url=os.environ["OPENAI_API_BASE"],
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0.7,
    )

    # Define specialized agents
    researcher = Agent(
        role="Senior Research Analyst",
        goal=f"Conduct thorough research on {topic}",
        backstory="""You are an expert research analyst with deep knowledge
        across multiple domains. You excel at gathering information, identifying
        key insights, and synthesizing complex topics into clear summaries.""",
        llm=llm,
        verbose=True,
    )

    writer = Agent(
        role="Technical Writer",
        goal="Create clear, engaging reports from research findings",
        backstory="""You are a skilled technical writer who can transform
        complex research into accessible, well-structured documents. You focus
        on clarity, accuracy, and reader engagement.""",
        llm=llm,
        verbose=True,
    )

    critic = Agent(
        role="Quality Reviewer",
        goal="Ensure accuracy and completeness of the final report",
        backstory="""You are a meticulous reviewer who checks for factual
        accuracy, logical consistency, and completeness. You provide
        constructive feedback to improve the final output.""",
        llm=llm,
        verbose=True,
    )

    # Define tasks
    research_task = Task(
        description=f"""Research the topic: {topic}

        Your task:
        1. Identify the key aspects and subtopics
        2. Gather relevant information and data points
        3. Note any recent developments or trends
        4. Identify potential implications and applications

        Provide a comprehensive research summary.""",
        agent=researcher,
        expected_output="A detailed research summary with key findings and insights",
    )

    writing_task = Task(
        description="""Based on the research findings, create a well-structured report.

        Your task:
        1. Organize the information logically
        2. Write clear, engaging prose
        3. Include an executive summary
        4. Add section headers and bullet points where appropriate
        5. Conclude with key takeaways

        The report should be informative and accessible to a general audience.""",
        agent=writer,
        expected_output="A polished, well-structured report",
    )

    review_task = Task(
        description="""Review the report for quality and accuracy.

        Your task:
        1. Check for factual accuracy
        2. Verify logical consistency
        3. Ensure completeness
        4. Suggest improvements if needed
        5. Provide a final assessment

        Return the final approved report with any necessary corrections.""",
        agent=critic,
        expected_output="The final reviewed and approved report",
    )

    # Create the crew
    crew = Crew(
        agents=[researcher, writer, critic],
        tasks=[research_task, writing_task, review_task],
        process=Process.sequential,
        verbose=True,
    )

    return crew


def main():
    parser = argparse.ArgumentParser(description="Run multi-agent research team")
    parser.add_argument(
        "--topic",
        type=str,
        default="The future of AI agents in enterprise applications",
        help="Research topic",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-72B-Instruct-AWQ",
        help="Model name for the vLLM server",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/report.md",
        help="Output file path",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Multi-Agent Research Team")
    print("=" * 60)
    print(f"Topic: {args.topic}")
    print(f"Model: {args.model}")
    print(f"API Base: {os.environ['OPENAI_API_BASE']}")
    print("=" * 60)
    print()

    # Create and run the crew
    crew = create_research_crew(args.topic, args.model)
    result = crew.kickoff()

    # Save output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(f"# Research Report: {args.topic}\n\n")
        f.write(str(result))

    print()
    print("=" * 60)
    print(f"Report saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
