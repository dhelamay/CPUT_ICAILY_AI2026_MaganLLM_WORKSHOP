# Microsoft Agent Framework 02 — giving your agent tools
# plain python functions are tools. the @tool decorator is optional — use it to rename or describe a tool.
# run: python 05_microsoft_agent_framework/02_agent_with_tools.py

import asyncio

from agent_framework import Agent, tool
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


agent = Agent(
    client,
    instructions="You are a helpful assistant. Use your tools when they help answer accurately.",
    name="assistant",
    tools=[web_search, word_count],
)


async def main():
    result = await agent.run("search for the latest langchain version, then tell me how many words your answer is.")
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
