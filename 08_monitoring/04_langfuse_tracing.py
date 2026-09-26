# Monitoring 04 — Langfuse: open-source LLM observability (traces, sessions, users, cost, scores)
# two ways to run it, both free:
#   a) Langfuse Cloud free tier: sign up at https://cloud.langfuse.com -> project -> API keys
#   b) self-host with Docker:    git clone https://github.com/langfuse/langfuse && cd langfuse && docker compose up
#                                -> http://localhost:3000, create a project, copy its keys
# then put the keys in .env:
#   LANGFUSE_PUBLIC_KEY=pk-lf-...   LANGFUSE_SECRET_KEY=sk-lf-...   LANGFUSE_HOST=https://cloud.langfuse.com (or http://localhost:3000)
#
# run: python 08_monitoring/04_langfuse_tracing.py

import os

from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

import workshop_common  # noqa: F401  (loads .env)
from agent_graph import QUESTIONS, build_agent

if not os.getenv("LANGFUSE_PUBLIC_KEY"):
    raise SystemExit("add LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST to .env (see the top of this file)")

langfuse = get_client()                 # reads the LANGFUSE_* variables
handler = CallbackHandler()             # the LangChain/LangGraph hook: records every step

agent = build_agent()
config = {"configurable": {"thread_id": "langfuse-demo"}, "callbacks": [handler]}

# user_id and session_id let you filter in the Langfuse UI: "all conversations of m0h", "this chat session"
with propagate_attributes(user_id="m0h", session_id="langfuse-demo", tags=["sa-lib", "day2"]):
    for question in QUESTIONS:
        result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config)
        print(result["messages"][-1].content)

langfuse.flush()                        # send everything before the script exits
print(f"\nopen {os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')} -> Tracing / Sessions / Users")
