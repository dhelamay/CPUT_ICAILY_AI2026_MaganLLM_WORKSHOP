# CrewAI 04 — the fallback setup
# try each provider (LLM_PROVIDER, then LLM_FALLBACKS in .env) until one answers.
# run: python 02_crewai/04_agent_with_fallback.py

import asyncio
import os

os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from crewai import LLM, Agent

from workshop_common import fallback_order, get_llm_config, ping


def get_llm() -> LLM:
    for provider in fallback_order():
        try:
            cfg = get_llm_config(provider)
            ping(cfg)
            print(f"using {provider} ({cfg.model})")
            return LLM(model=cfg.litellm_model, base_url=cfg.base_url, api_key=cfg.api_key, temperature=0)
        except Exception as e:
            print(f"{provider} failed ({str(e)[:80]}); trying the next one")
    raise RuntimeError("every provider failed — check your keys in .env")


assistant = Agent(role="Helpful assistant", goal="Answer concisely", backstory="You are precise.", llm=get_llm())
async def main():
    result = await assistant.kickoff_async("say hi and tell me which model you are.")
    print("\n>>> assistant:", result.raw)


if __name__ == "__main__":
    asyncio.run(main())
