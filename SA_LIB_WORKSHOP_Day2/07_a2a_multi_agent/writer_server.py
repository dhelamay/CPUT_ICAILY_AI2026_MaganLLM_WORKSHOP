# A2A agent #2 — the WRITER, running as its own web service on port 8002.
# it knows nothing about the researcher. the orchestrator connects them over A2A.
#
# run (terminal 2):  python 07_a2a_multi_agent/writer_server.py

import uvicorn
from google.adk.agents import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a

from common_model import adk_model
from workshop_common import word_count

PORT = 8002

writer = Agent(
    name="writer",
    description="Turns facts into one friendly paragraph under 60 words for beginners.",
    model=adk_model(),
    instruction="Write one friendly paragraph under 60 words from the facts you are given. "
                "Check the length with word_count before answering.",
    tools=[word_count],
)

app = to_a2a(writer, host="localhost", port=PORT)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=PORT)
