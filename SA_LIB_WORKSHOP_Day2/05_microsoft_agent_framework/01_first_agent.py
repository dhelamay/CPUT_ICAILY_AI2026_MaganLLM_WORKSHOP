# Microsoft Agent Framework 01 — your first agent
# Agent Framework is Microsoft's successor to AutoGen + Semantic Kernel.
# pieces: a CHAT CLIENT (which model) + an Agent (instructions + tools) + agent.run(...)
# run: python 05_microsoft_agent_framework/01_first_agent.py

import asyncio

from agent_framework import Agent
from agent_framework.openai import OpenAIChatCompletionClient

from workshop_common import get_llm_config

cfg = get_llm_config()
# OpenAIChatCompletionClient speaks the chat-completions protocol -> works with every provider in .env
client = OpenAIChatCompletionClient(model=cfg.model, api_key=cfg.api_key, base_url=cfg.base_url)

agent = Agent(client, instructions="You are a helpful assistant. Be concise and accurate.", name="assistant")


async def main():
    result = await agent.run("explain what an AI agent is in two sentences.")
    print(f"[{cfg.provider}:{cfg.model}]", result.text)


if __name__ == "__main__":
    asyncio.run(main())
