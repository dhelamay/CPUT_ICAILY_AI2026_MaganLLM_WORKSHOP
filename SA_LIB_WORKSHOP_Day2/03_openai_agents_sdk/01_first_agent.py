# OpenAI Agents SDK 01 — your first agent
# the SDK has 3 ideas: Agent (instructions + tools), Runner (runs the loop), and results.
# it is built for OpenAI, but OpenAIChatCompletionsModel lets it use ANY OpenAI-compatible provider.
# run: python 03_openai_agents_sdk/01_first_agent.py

import asyncio
import os

from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, set_tracing_disabled

from workshop_common import get_llm_config

# tracing uploads runs to platform.openai.com — only possible with an OpenAI key
set_tracing_disabled(not os.getenv("OPENAI_API_KEY"))

cfg = get_llm_config()
model = OpenAIChatCompletionsModel(model=cfg.model, openai_client=AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key))

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant. Be concise and accurate.",
    model=model,
)


async def main():
    result = await Runner.run(agent, "explain what an AI agent is in two sentences.")
    print(f"[{cfg.provider}:{cfg.model}]", result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
