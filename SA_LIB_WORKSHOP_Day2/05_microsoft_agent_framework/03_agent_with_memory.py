# Microsoft Agent Framework 03 — giving your agent memory
# an AgentSession holds the conversation. same session = same memory.
# run: python 05_microsoft_agent_framework/03_agent_with_memory.py

import asyncio

from agent_framework import Agent
from agent_framework.openai import OpenAIChatCompletionClient

from workshop_common import get_llm_config

cfg = get_llm_config()
client = OpenAIChatCompletionClient(model=cfg.model, api_key=cfg.api_key, base_url=cfg.base_url)
agent = Agent(client, instructions="You are a helpful assistant. Be concise and accurate.", name="assistant")


async def main():
    session = agent.create_session()                    # <- the memory
    r1 = await agent.run("hi! my name is m0h.", session=session)
    print(r1.text)
    r2 = await agent.run("what's my name?", session=session)
    print(r2.text)


if __name__ == "__main__":
    asyncio.run(main())
