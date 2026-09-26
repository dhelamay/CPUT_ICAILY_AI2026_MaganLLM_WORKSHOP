# Google ADK 03 — giving your agent memory
# the SESSION is the memory. same session id = the agent sees the whole conversation.
# (InMemorySessionService forgets on exit; DatabaseSessionService("sqlite:///memory.db") persists.)
# run: python 04_google_adk/03_agent_with_memory.py

import asyncio

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai import types

from workshop_common import get_llm_config

cfg = get_llm_config()
model = cfg.model if cfg.provider == "gemini" else LiteLlm(model=cfg.litellm_model, api_base=cfg.base_url, api_key=cfg.api_key)
agent = Agent(name="assistant", model=model, instruction="You are a helpful assistant. Be concise and accurate.")
runner = InMemoryRunner(agent=agent, app_name="workshop")


async def ask(session_id: str, text: str) -> str:
    message = types.Content(role="user", parts=[types.Part(text=text)])
    answer = ""
    async for event in runner.run_async(user_id="m0h", session_id=session_id, new_message=message):
        if event.is_final_response() and event.content and event.content.parts:
            answer = event.content.parts[0].text   # keep looping: let the runner finish cleanly
    return answer


async def main():
    session = await runner.session_service.create_session(app_name="workshop", user_id="m0h", session_id="chat-1")
    print(await ask(session.id, "hi! my name is m0h."))
    print(await ask(session.id, "what's my name?"))


if __name__ == "__main__":
    asyncio.run(main())
