# 10 — Agentic RAG  (no framework, just the tool-calling loop)
# in every example so far, WE decided the steps. in agentic RAG the LLM decides:
#   which tool to use (company docs? web? calculator?), what to search for,
#   whether the results are enough, and when to stop.
# this is the plan -> act -> observe loop from Day 2 — written by hand, ~40 lines.
# run: python 10_agentic_rag.py

import json

from rag_common import build_store, format_context, get_llm_client, show, web_search

store = build_store()
client, model = get_llm_client()


# --- tools: plain python functions ---
def search_company_docs(query: str) -> str:
    return format_context(store.search(query, k=3))


def calculator(expression: str) -> str:
    return str(eval(expression, {"__builtins__": {}}, {}))  # demo only: arithmetic like "18500*0.2"


TOOLS = {"search_company_docs": search_company_docs, "web_search": web_search, "calculator": calculator}

# --- tool descriptions the LLM reads to decide WHEN to use each tool ---
TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "search_company_docs",
        "description": "Search Sahara Sun Energy internal documents (products, prices, HR policy, projects, lab).",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the public web for current or general information.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression, e.g. '34000*0.2'.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
]

SYSTEM = ("You are a research assistant for Sahara Sun Energy. Use tools to find facts before answering. "
          "You may call tools several times. Cite sources. Be concise.")


def agent(question: str, max_steps: int = 6) -> str:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    for step in range(max_steps):
        msg = client.chat.completions.create(model=model, messages=messages, tools=TOOL_SPECS).choices[0].message
        if not msg.tool_calls:                       # no tool requested -> this is the final answer
            return msg.content
        messages.append(msg)                          # PLAN: the model asked for tools
        for call in msg.tool_calls:                   # ACT: run each tool
            args = json.loads(call.function.arguments or "{}")
            print(f"step {step + 1}: {call.function.name}({args})")
            try:
                result = TOOLS[call.function.name](**args)
            except Exception as e:
                result = f"tool error: {e}"
            messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})  # OBSERVE
    return "stopped: too many steps"


show("answer", agent(
    "A farmer wants the SunBox Farm kit with the Pay-as-you-save plan. "
    "How much is the deposit, and how much is each monthly instalment?"
))
