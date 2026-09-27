# Monitoring 05 — LangSmith: LangChain's hosted tracing and monitoring platform
# free "Developer" plan (1 user, a few thousand traces per month — check https://www.langchain.com/pricing).
# it is NOT open source (use Langfuse or Phoenix if you need self-hosting), but it is the easiest
# option for LangChain / LangGraph: tracing turns on with environment variables, no code changes.
#
# setup:
#   1. sign up at https://smith.langchain.com  ->  Settings  ->  API Keys  ->  Create API key
#   2. add to .env:
#        LANGSMITH_TRACING=true
#        LANGSMITH_API_KEY=lsv2_...
#        LANGSMITH_PROJECT=sa-lib-day2
#
# run:  python 08_monitoring/05_langsmith_tracing.py
# then: https://smith.langchain.com -> Projects -> sa-lib-day2  (traces, tokens, latency, errors, monitoring charts)

import json
import os

import langsmith as ls
from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import OpenAI

import workshop_common as wc  # loads .env
from agent_graph import QUESTIONS, build_agent

if not os.getenv("LANGSMITH_API_KEY"):
    raise SystemExit("add LANGSMITH_TRACING=true and LANGSMITH_API_KEY=... to .env (see the top of this file)")
PROJECT = os.getenv("LANGSMITH_PROJECT") or "sa-lib-day2"
client_ls = ls.Client()   # reads LANGSMITH_API_KEY (and LANGSMITH_ENDPOINT if you self-host)

# ---------------------------------------------------------------------------------------------
# for code WITHOUT LangChain: wrap the openai client and decorate your own functions.
# this works for agents built with any framework, or none (used in part B below).
# ---------------------------------------------------------------------------------------------
cfg = wc.get_llm_config()
client = wrap_openai(OpenAI(base_url=cfg.base_url, api_key=cfg.api_key))   # every LLM call is traced


@traceable(run_type="tool")
def word_count(text: str) -> int:
    return wc.word_count(text)


@traceable(run_type="chain", name="plain_python_agent")
def plain_agent(question: str) -> str:
    spec = [{"type": "function", "function": {"name": "word_count", "description": "Count words in a text.",
             "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}}]
    messages = [{"role": "user", "content": question}]
    for _ in range(4):                                   # plan -> act -> observe
        msg = client.chat.completions.create(model=cfg.model, messages=messages, tools=spec).choices[0].message
        if not msg.tool_calls:
            return msg.content
        messages.append(msg)
        for call in msg.tool_calls:
            result = word_count(**json.loads(call.function.arguments or "{}"))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})
    return "stopped: too many steps"


# ---------------------------------------------------------------------------------------------
# run both. with LANGSMITH_TRACING=true in .env you don't even need this `with` block for LangGraph —
# we use it so the demo always traces (also in notebooks) and to name the project in one place.
# ---------------------------------------------------------------------------------------------
with ls.tracing_context(enabled=True, project_name=PROJECT, client=client_ls):
    # part A — LangGraph agent: NOTHING added to the agent itself. every node, LLM call and tool call is traced.
    agent = build_agent()
    config = {"configurable": {"thread_id": "langsmith-demo"},
              "metadata": {"user_id": "m0h"}, "tags": ["sa-lib", "langgraph"]}   # filterable in the UI
    for question in QUESTIONS:
        result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config)
        print("[langgraph]", result["messages"][-1].content)

    # part B — the plain-python agent
    print("[plain]", plain_agent("how many words are in 'agents plan act and observe'? use the tool."))

wait_for_all_tracers()   # send LangChain/LangGraph traces before exit
client_ls.flush()        # send @traceable / wrap_openai traces before exit
print(f"\nopen https://smith.langchain.com -> Projects -> {PROJECT}")
