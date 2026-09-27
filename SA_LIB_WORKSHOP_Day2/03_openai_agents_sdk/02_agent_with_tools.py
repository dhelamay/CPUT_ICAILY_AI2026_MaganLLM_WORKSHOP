# OpenAI Agents SDK 02 — giving your agent tools
# @function_tool turns any python function into a tool (name, docstring and type hints -> JSON schema).
# run: python 03_openai_agents_sdk/02_agent_with_tools.py

import asyncio
import os

from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled

import workshop_common as wc

set_tracing_disabled(not os.getenv("OPENAI_API_KEY"))
cfg = wc.get_llm_config()
model = OpenAIChatCompletionsModel(model=cfg.model, openai_client=AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key))


@function_tool
def web_search(query: str) -> str:
    """Search the web for up-to-date information."""
    return wc.web_search(query)


@function_tool
def word_count(text: str) -> int:
    """Count how many words are in a piece of text."""
    return len(text.split())


agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant. Use your tools when they help answer accurately.",
    tools=[web_search, word_count],
    model=model,
)


async def main():
    result = await Runner.run(agent, "search for the latest langchain version, then tell me how many words your answer is.")
    for item in result.new_items:          # every step of the loop: tool calls, tool outputs, messages
        print(f"[{item.type}]")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
