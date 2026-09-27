# Google ADK 02 — giving your agent tools
# in ADK a tool is just a python function with type hints and a docstring. no decorator needed!
# run: python 04_google_adk/02_agent_with_tools.py

import asyncio

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai import types

from workshop_common import get_llm_config, web_search

cfg = get_llm_config()
model = cfg.model if cfg.provider == "gemini" else LiteLlm(model=cfg.litellm_model, api_base=cfg.base_url, api_key=cfg.api_key)


# tool 2 — a function you wrote yourself (tool 1, web_search, comes from workshop_common.py)
def word_count(text: str) -> int:
    """Count how many words are in a piece of text."""
    return len(text.split())


agent = Agent(
    name="assistant",
    model=model,
    instruction="You are a helpful assistant. Use your tools when they help answer accurately.",
    tools=[web_search, word_count],
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="workshop")
    session = await runner.session_service.create_session(app_name="workshop", user_id="m0h")
    text = "search for the latest langchain version, then tell me how many words your answer is."
    message = types.Content(role="user", parts=[types.Part(text=text)])

    async for event in runner.run_async(user_id="m0h", session_id=session.id, new_message=message):
        for call in event.get_function_calls():            # watch the agent loop
            print(f"  tool call -> {call.name}({call.args})")
        if event.is_final_response() and event.content and event.content.parts:
            print(event.content.parts[0].text)


if __name__ == "__main__":
    asyncio.run(main())
