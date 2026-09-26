# Google ADK — the complete agent (fallback + tools + memory)
# run: python 04_google_adk/agent.py

import asyncio

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai import types

import workshop_common as wc


def get_model():
    for provider in wc.fallback_order():
        try:
            cfg = wc.get_llm_config(provider)
            wc.ping(cfg)
            print(f"using {provider} ({cfg.model})")
            if cfg.provider == "gemini":
                return cfg.model
            return LiteLlm(model=cfg.litellm_model, api_base=cfg.base_url, api_key=cfg.api_key)
        except Exception as e:
            print(f"{provider} failed ({str(e)[:80]}); trying the next one")
    raise RuntimeError("every provider failed — check your keys in .env")


agent = Agent(
    name="assistant",
    model=get_model(),
    instruction="You are a helpful assistant. Use your tools when they help answer accurately, and remember the conversation.",
    tools=[wc.web_search, wc.word_count],
)
runner = InMemoryRunner(agent=agent, app_name="workshop")


async def ask(session_id: str, text: str) -> str:
    message = types.Content(role="user", parts=[types.Part(text=text)])
    answer = ""
    async for event in runner.run_async(user_id="m0h", session_id=session_id, new_message=message):
        if event.is_final_response() and event.content and event.content.parts:
            answer = event.content.parts[0].text   # keep looping: let the runner finish cleanly
    return answer


async def main():
    session = await runner.session_service.create_session(app_name="workshop", user_id="m0h")
    print(await ask(session.id, "my name is m0h. search for the latest langchain version, then tell me how many "
                                "words your answer is."))
    print(await ask(session.id, "what's my name?"))


if __name__ == "__main__":
    asyncio.run(main())
