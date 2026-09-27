# OpenAI Agents SDK — the complete agent (fallback + tools + memory)
# run: python 03_openai_agents_sdk/agent.py

import asyncio
import os

from agents import (Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, SQLiteSession, function_tool,
                    set_tracing_disabled)

import workshop_common as wc

set_tracing_disabled(not os.getenv("OPENAI_API_KEY"))


def get_model():
    for provider in wc.fallback_order():
        try:
            cfg = wc.get_llm_config(provider)
            wc.ping(cfg)
            print(f"using {provider} ({cfg.model})")
            client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
            return OpenAIChatCompletionsModel(model=cfg.model, openai_client=client)
        except Exception as e:
            print(f"{provider} failed ({str(e)[:80]}); trying the next one")
    raise RuntimeError("every provider failed — check your keys in .env")


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
    instructions="You are a helpful assistant. Use your tools when they help answer accurately, and remember the conversation.",
    tools=[web_search, word_count],
    model=get_model(),
)


async def main():
    session = SQLiteSession("chat-1")
    r1 = await Runner.run(agent, "my name is m0h. search for the latest langchain version, then tell me how many "
                                 "words your answer is.", session=session)
    print(r1.final_output)
    r2 = await Runner.run(agent, "what's my name?", session=session)
    print(r2.final_output)


if __name__ == "__main__":
    asyncio.run(main())
