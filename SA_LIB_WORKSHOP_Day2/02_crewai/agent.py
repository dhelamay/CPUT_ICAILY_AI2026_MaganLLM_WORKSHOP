# CrewAI — the complete agent (fallback + tools + memory)
# run: python 02_crewai/agent.py

import asyncio
import os

os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from crewai import LLM, Agent
from crewai.tools import tool

import workshop_common as wc


def get_llm() -> LLM:
    for provider in wc.fallback_order():
        try:
            cfg = wc.get_llm_config(provider)
            wc.ping(cfg)
            print(f"using {provider} ({cfg.model})")
            return LLM(model=cfg.litellm_model, base_url=cfg.base_url, api_key=cfg.api_key, temperature=0)
        except Exception as e:
            print(f"{provider} failed ({str(e)[:80]}); trying the next one")
    raise RuntimeError("every provider failed — check your keys in .env")


@tool("web_search")
def web_search(query: str) -> str:
    """Search the web for up-to-date information."""
    return wc.web_search(query)


@tool("word_count")
def word_count(text: str) -> int:
    """Count how many words are in a piece of text."""
    return len(text.split())


assistant = Agent(
    role="Helpful assistant",
    goal="Use your tools when they help answer accurately, and remember the conversation",
    backstory="You are friendly, precise and always check facts.",
    tools=[web_search, word_count],
    llm=get_llm(),
)
history = []


async def chat(text: str) -> str:
    history.append({"role": "user", "content": text})
    reply = (await assistant.kickoff_async(history)).raw
    history.append({"role": "assistant", "content": reply})
    return reply


async def main():
    print("\n>>> assistant:", await chat("my name is m0h. search for the latest langchain version, then tell me how many words your answer is."))
    print("\n>>> assistant:", await chat("what's my name?"))


if __name__ == "__main__":
    asyncio.run(main())
