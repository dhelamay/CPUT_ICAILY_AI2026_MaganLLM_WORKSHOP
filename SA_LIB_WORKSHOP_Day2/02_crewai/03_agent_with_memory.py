# CrewAI 03 — giving your agent memory
# the simplest memory: keep the conversation in a list and send it every time.
# (CrewAI also has Crew(memory=True) — long-term memory with a vector database — but it needs an
#  embeddings provider, so for a free workshop we show the idea with a plain list.)
# run: python 02_crewai/03_agent_with_memory.py

import asyncio
import os

os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from crewai import LLM, Agent

from workshop_common import get_llm_config

cfg = get_llm_config()
llm = LLM(model=cfg.litellm_model, base_url=cfg.base_url, api_key=cfg.api_key, temperature=0)

assistant = Agent(
    role="Helpful assistant",
    goal="Answer concisely and remember the conversation",
    backstory="You are friendly and precise.",
    llm=llm,
)

history = []  # <- the memory: same list = same conversation


async def chat(text: str) -> str:
    history.append({"role": "user", "content": text})
    reply = (await assistant.kickoff_async(history)).raw   # runs ONE agent without a crew
    history.append({"role": "assistant", "content": reply})
    return reply


async def main():
    print("\n>>> assistant:", await chat("hi! my name is m0h."))
    print("\n>>> assistant:", await chat("what's my name?"))


if __name__ == "__main__":
    asyncio.run(main())
