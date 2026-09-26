# CrewAI 05 — multi-agent crew: researcher -> writer -> editor
# this is what CrewAI is famous for. each agent has a role; each task's output
# is automatically passed as CONTEXT to the next task (Process.sequential).
# run: python 02_crewai/05_multi_agent.py

import asyncio
import os

os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool

import workshop_common as wc

cfg = wc.get_llm_config()
llm = LLM(model=cfg.litellm_model, base_url=cfg.base_url, api_key=cfg.api_key, temperature=0)


@tool("web_search")
def web_search(query: str) -> str:
    """Search the web for up-to-date information."""
    return wc.web_search(query)


@tool("word_count")
def word_count(text: str) -> int:
    """Count how many words are in a piece of text."""
    return len(text.split())


researcher = Agent(role="Researcher", goal="Find accurate, current facts",
                   backstory="You search the web and report facts with sources.", tools=[web_search], llm=llm)
writer = Agent(role="Writer", goal="Turn facts into a short, friendly paragraph",
               backstory="You write for workshop attendees who are new to AI.", llm=llm)
editor = Agent(role="Editor", goal="Make sure the paragraph is under 60 words",
               backstory="You are strict about length. You use word_count to check.", tools=[word_count], llm=llm)

research = Task(description="Find the latest version of {topic} and one notable feature.",
                expected_output="3 bullet facts with sources.", agent=researcher)
write = Task(description="Write one paragraph about {topic} using the research.",
             expected_output="One paragraph.", agent=writer)
edit = Task(description="Check the paragraph with word_count and shorten it to under 60 words if needed.",
            expected_output="The final paragraph and its word count.", agent=editor)

crew = Crew(agents=[researcher, writer, editor], tasks=[research, write, edit], process=Process.sequential)
async def main():
    result = await crew.kickoff_async(inputs={"topic": "langchain"})  # fills {topic} in the tasks
    print(result.raw)


if __name__ == "__main__":
    asyncio.run(main())
