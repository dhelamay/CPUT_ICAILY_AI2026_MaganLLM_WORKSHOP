# CrewAI 02 — giving your agent tools
# @tool("name") turns a python function into a CrewAI tool. the docstring tells the agent when to use it.
# run: python 02_crewai/02_agent_with_tools.py

import asyncio
import os

os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from crewai import LLM, Agent, Crew, Task
from crewai.tools import tool

import workshop_common as wc

cfg = wc.get_llm_config()
llm = LLM(model=cfg.litellm_model, base_url=cfg.base_url, api_key=cfg.api_key, temperature=0)


# tool 1 — web search (free, no key)
@tool("web_search")
def web_search(query: str) -> str:
    """Search the web for up-to-date information."""
    return wc.web_search(query)


# tool 2 — a function you wrote yourself
@tool("word_count")
def word_count(text: str) -> int:
    """Count how many words are in a piece of text."""
    return len(text.split())


assistant = Agent(
    role="Research assistant",
    goal="Answer accurately, using tools when they help",
    backstory="You always check facts with a web search before answering.",
    tools=[web_search, word_count],
    llm=llm,
    verbose=True,   # prints every Thought / Action / Observation — the agent loop, live
)

task = Task(
    description="search for the latest langchain version, then tell me how many words your answer is.",
    expected_output="The latest version, plus the word count of your answer.",
    agent=assistant,
)

async def main():
    result = await Crew(agents=[assistant], tasks=[task]).kickoff_async()
    print(result.raw)


if __name__ == "__main__":
    asyncio.run(main())
