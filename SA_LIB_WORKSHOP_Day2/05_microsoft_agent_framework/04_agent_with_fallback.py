# Microsoft Agent Framework 04 — the fallback setup
# run: python 05_microsoft_agent_framework/04_agent_with_fallback.py

import asyncio

from agent_framework import Agent
from agent_framework.openai import OpenAIChatCompletionClient

from workshop_common import fallback_order, get_llm_config, ping


def get_client():
    for provider in fallback_order():
        try:
            cfg = get_llm_config(provider)
            ping(cfg)
            print(f"using {provider} ({cfg.model})")
            return OpenAIChatCompletionClient(model=cfg.model, api_key=cfg.api_key, base_url=cfg.base_url)
        except Exception as e:
            print(f"{provider} failed ({str(e)[:80]}); trying the next one")
    raise RuntimeError("every provider failed — check your keys in .env")


agent = Agent(get_client(), instructions="You are a helpful assistant. Be concise.", name="assistant")


async def main():
    print((await agent.run("say hi and tell me which model you are.")).text)


if __name__ == "__main__":
    asyncio.run(main())
