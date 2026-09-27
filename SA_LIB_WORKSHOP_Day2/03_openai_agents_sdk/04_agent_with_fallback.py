# OpenAI Agents SDK 04 — the fallback setup
# run: python 03_openai_agents_sdk/04_agent_with_fallback.py

import asyncio
import os

from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, set_tracing_disabled

from workshop_common import fallback_order, get_llm_config, ping

set_tracing_disabled(not os.getenv("OPENAI_API_KEY"))


def get_model():
    for provider in fallback_order():
        try:
            cfg = get_llm_config(provider)
            ping(cfg)
            print(f"using {provider} ({cfg.model})")
            client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
            return OpenAIChatCompletionsModel(model=cfg.model, openai_client=client)
        except Exception as e:
            print(f"{provider} failed ({str(e)[:80]}); trying the next one")
    raise RuntimeError("every provider failed — check your keys in .env")


agent = Agent(name="Assistant", instructions="You are a helpful assistant. Be concise.", model=get_model())


async def main():
    result = await Runner.run(agent, "say hi and tell me which model you are.")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
