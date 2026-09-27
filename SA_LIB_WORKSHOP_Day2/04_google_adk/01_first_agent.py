# Google ADK 01 — your first agent
# ADK pieces: Agent (model + instruction + tools), Runner (runs the loop), Session (the conversation).
# Gemini is used natively; every other provider goes through LiteLLM.
# run: python 04_google_adk/01_first_agent.py

import asyncio

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai import types

from workshop_common import get_llm_config

cfg = get_llm_config()
# gemini -> native ADK model name; anything else -> LiteLLM pointing at an OpenAI-compatible endpoint
model = cfg.model if cfg.provider == "gemini" else LiteLlm(model=cfg.litellm_model, api_base=cfg.base_url, api_key=cfg.api_key)

agent = Agent(
    name="assistant",
    model=model,
    instruction="You are a helpful assistant. Be concise and accurate.",
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="workshop")
    session = await runner.session_service.create_session(app_name="workshop", user_id="m0h")
    message = types.Content(role="user", parts=[types.Part(text="explain what an AI agent is in two sentences.")])

    # the runner streams EVENTS (model calls, tool calls, ...). we print the final answer.
    async for event in runner.run_async(user_id="m0h", session_id=session.id, new_message=message):
        if event.is_final_response() and event.content and event.content.parts:
            print(f"[{cfg.provider}:{cfg.model}]", event.content.parts[0].text)


if __name__ == "__main__":
    asyncio.run(main())
