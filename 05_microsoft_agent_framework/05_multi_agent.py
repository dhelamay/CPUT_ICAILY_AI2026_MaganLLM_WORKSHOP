# Microsoft Agent Framework 05 — multi-agent with a SEQUENTIAL workflow
# researcher -> writer. SequentialBuilder chains agents: each one sees the conversation so far.
# (also available: ConcurrentBuilder, GroupChatBuilder, HandoffBuilder, MagenticBuilder)
# run: python 05_microsoft_agent_framework/05_multi_agent.py

import asyncio

from agent_framework import Agent, tool
from agent_framework.orchestrations import SequentialBuilder
from agent_framework.openai import OpenAIChatCompletionClient

import workshop_common as wc

cfg = wc.get_llm_config()
client = OpenAIChatCompletionClient(model=cfg.model, api_key=cfg.api_key, base_url=cfg.base_url)


@tool
def web_search(query: str) -> str:
    """Search the web for up-to-date information."""
    return wc.web_search(query)


@tool
def word_count(text: str) -> int:
    """Count how many words are in a piece of text."""
    return len(text.split())


researcher = Agent(client, name="researcher", tools=[web_search],
                   instructions="Use web_search to find current facts about the topic. Reply with 3 short bullet facts.")
writer = Agent(client, name="writer", tools=[word_count],
               instructions="Using the researcher's facts, write one friendly paragraph under 60 words. "
                            "Check the length with word_count.")

workflow = SequentialBuilder(participants=[researcher, writer], output_from="all").build()  # show every agent


async def main():
    result = await workflow.run("the latest langchain version")
    for response in result.get_outputs():           # one AgentResponse per agent, in order
        print(f"\n[{response.messages[0].author_name}]\n{response.text}")


if __name__ == "__main__":
    asyncio.run(main())
