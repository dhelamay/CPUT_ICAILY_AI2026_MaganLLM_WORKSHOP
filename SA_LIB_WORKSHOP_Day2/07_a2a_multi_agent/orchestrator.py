# the ORCHESTRATOR — an ADK agent whose team members live in OTHER processes.
# RemoteA2aAgent reads each agent card and makes the remote agent look like a local sub-agent.
# SequentialAgent runs them in order: researcher -> writer.
#
# run (terminal 3, servers must be running):  python 07_a2a_multi_agent/orchestrator.py

import asyncio

from google.adk.a2a.agent import RemoteA2aAgent
from google.adk.agents import SequentialAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

CARD = "/.well-known/agent-card.json"

researcher = RemoteA2aAgent(name="researcher", agent_card=f"http://localhost:8001{CARD}",
                            description="Finds current facts on the web.")
writer = RemoteA2aAgent(name="writer", agent_card=f"http://localhost:8002{CARD}",
                        description="Writes a short paragraph from facts.")

team = SequentialAgent(name="team", sub_agents=[researcher, writer])


async def main():
    runner = InMemoryRunner(agent=team, app_name="a2a_demo")
    session = await runner.session_service.create_session(app_name="a2a_demo", user_id="m0h")
    message = types.Content(role="user", parts=[types.Part(text="the latest langchain version")])
    printed = set()  # a remote agent can report the same answer twice (status + artifact): print it once
    async for event in runner.run_async(user_id="m0h", session_id=session.id, new_message=message):
        if event.is_final_response() and event.content and event.content.parts and event.content.parts[0].text:
            key = (event.author, event.content.parts[0].text)
            if key not in printed:
                printed.add(key)
                print(f"\n[{event.author}]\n{event.content.parts[0].text}")


if __name__ == "__main__":
    asyncio.run(main())
