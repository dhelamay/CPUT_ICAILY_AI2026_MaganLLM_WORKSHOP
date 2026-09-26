# Microsoft Agent Framework — the complete agent (fallback + tools + memory)
# run: python 05_microsoft_agent_framework/agent.py

import asyncio

from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatCompletionClient

import workshop_common as wc


def get_client():
    for provider in wc.fallback_order():
        try:
            cfg = wc.get_llm_config(provider)
            wc.ping(cfg)
            print(f"using {provider} ({cfg.model})")
            return OpenAIChatCompletionClient(model=cfg.model, api_key=cfg.api_key, base_url=cfg.base_url)
        except Exception as e:
            print(f"{provider} failed ({str(e)[:80]}); trying the next one")
    raise RuntimeError("every provider failed — check your keys in .env")


@tool
def web_search(query: str) -> str:
    """Search the web for up-to-date information."""
    return wc.web_search(query)


@tool
def word_count(text: str) -> int:
    """Count how many words are in a piece of text."""
    return len(text.split())


agent = Agent(
    get_client(),
    name="assistant",
    instructions="You are a helpful assistant. Use your tools when they help answer accurately, and remember the conversation.",
    tools=[web_search, word_count],
)


async def main():
    session = agent.create_session()
    r1 = await agent.run("my name is m0h. search for the latest langchain version, then tell me how many words your "
                         "answer is.", session=session)
    print(r1.text)
    r2 = await agent.run("what's my name?", session=session)
    print(r2.text)


if __name__ == "__main__":
    asyncio.run(main())
