# A2A agent #1 — the RESEARCHER, running as its own web service on port 8001.
# to_a2a() wraps any ADK agent in an A2A server and publishes its "business card" at
#     http://localhost:8001/.well-known/agent-card.json
# (name, description, skills, how to talk to it). any A2A client, in any framework
# or language, can now discover and call this agent.
#
# run (terminal 1):  python 07_a2a_multi_agent/researcher_server.py

import uvicorn
from google.adk.agents import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a

from common_model import adk_model
from workshop_common import web_search

PORT = 8001

researcher = Agent(
    name="researcher",
    description="Finds current facts on the web and returns 3 short bullet points with sources.",
    model=adk_model(),
    instruction="Use web_search to find current facts about the user's topic. Reply with 3 short bullet facts.",
    tools=[web_search],
)

app = to_a2a(researcher, host="localhost", port=PORT)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=PORT)
