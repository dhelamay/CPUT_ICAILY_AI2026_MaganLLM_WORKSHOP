# Google ADK 05 — multi-agent: a SequentialAgent pipeline
# researcher -> writer. each agent saves its answer into the shared session STATE with output_key,
# and the next agent reads it with {placeholder} in its instruction.
# (ADK also has ParallelAgent, LoopAgent, and LLM-driven delegation via sub_agents=[...])
# run: python 04_google_adk/05_multi_agent.py

import asyncio

from google.adk.agents import Agent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai import types

from workshop_common import get_llm_config, web_search, word_count

cfg = get_llm_config()
model = cfg.model if cfg.provider == "gemini" else LiteLlm(model=cfg.litellm_model, api_base=cfg.base_url, api_key=cfg.api_key)

researcher = Agent(
    name="researcher",
    model=model,
    instruction="Use web_search to find current facts about the user's topic. Reply with 3 short bullet facts.",
    tools=[web_search],
    output_key="research",            # -> saved to session.state["research"]
)
writer = Agent(
    name="writer",
    model=model,
    instruction="Write one friendly paragraph under 60 words for workshop attendees, using ONLY these facts:\n"
                "{research}\nCheck the length with word_count before answering.",   # <- reads the state
    tools=[word_count],
    output_key="paragraph",
)
pipeline = SequentialAgent(name="team", sub_agents=[researcher, writer])


async def main():
    runner = InMemoryRunner(agent=pipeline, app_name="workshop")
    session = await runner.session_service.create_session(app_name="workshop", user_id="m0h")
    message = types.Content(role="user", parts=[types.Part(text="the latest langchain version")])
    async for event in runner.run_async(user_id="m0h", session_id=session.id, new_message=message):
        if event.is_final_response() and event.content and event.content.parts and event.content.parts[0].text:
            print(f"\n[{event.author}]\n{event.content.parts[0].text}")


if __name__ == "__main__":
    asyncio.run(main())
