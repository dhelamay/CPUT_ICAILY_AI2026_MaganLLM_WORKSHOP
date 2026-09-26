# OpenAI Agents SDK 03 — giving your agent memory
# a Session stores the conversation. SQLiteSession("chat-1") = the thread_id idea from LangGraph.
# pass a file path as 2nd argument (e.g. "memory.db") to keep memory after the script stops.
# run: python 03_openai_agents_sdk/03_agent_with_memory.py

import asyncio
import os

from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, SQLiteSession, set_tracing_disabled

from workshop_common import get_llm_config

set_tracing_disabled(not os.getenv("OPENAI_API_KEY"))
cfg = get_llm_config()
model = OpenAIChatCompletionsModel(model=cfg.model, openai_client=AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key))

agent = Agent(name="Assistant", instructions="You are a helpful assistant. Be concise and accurate.", model=model)


async def main():
    session = SQLiteSession("chat-1")      # in-memory SQLite; same session = same memory
    r1 = await Runner.run(agent, "hi! my name is m0h.", session=session)
    print(r1.final_output)
    r2 = await Runner.run(agent, "what's my name?", session=session)
    print(r2.final_output)


if __name__ == "__main__":
    asyncio.run(main())
