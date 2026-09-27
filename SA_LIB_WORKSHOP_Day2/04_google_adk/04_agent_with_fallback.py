# Google ADK 04 — the fallback setup
# run: python 04_google_adk/04_agent_with_fallback.py

import asyncio

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai import types

from workshop_common import fallback_order, get_llm_config, ping


def get_model():
    for provider in fallback_order():
        try:
            cfg = get_llm_config(provider)
            ping(cfg)
            print(f"using {provider} ({cfg.model})")
            if cfg.provider == "gemini":
                return cfg.model
            return LiteLlm(model=cfg.litellm_model, api_base=cfg.base_url, api_key=cfg.api_key)
        except Exception as e:
            print(f"{provider} failed ({str(e)[:80]}); trying the next one")
    raise RuntimeError("every provider failed — check your keys in .env")


agent = Agent(name="assistant", model=get_model(), instruction="You are a helpful assistant. Be concise.")


async def main():
    runner = InMemoryRunner(agent=agent, app_name="workshop")
    session = await runner.session_service.create_session(app_name="workshop", user_id="m0h")
    message = types.Content(role="user", parts=[types.Part(text="say hi and tell me which model you are.")])
    async for event in runner.run_async(user_id="m0h", session_id=session.id, new_message=message):
        if event.is_final_response() and event.content and event.content.parts:
            print(event.content.parts[0].text)


if __name__ == "__main__":
    asyncio.run(main())
