# Monitoring 01 — track your agent yourself (no extra packages, no account)
# the same agent as always (web_search + word_count), but every LLM call, tool call and run
# goes through monitor.py, which writes it to SQLite + a JSON log.
# we also run one question with a BROKEN tool, so you can see failure tracking.
#
# run:   python 08_monitoring/01_simple_monitor.py
# then:  python 08_monitoring/02_usage_report.py

import json

from openai import OpenAI

import monitor
import workshop_common as wc

cfg = wc.get_llm_config()
client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)


def broken_calculator(expression: str) -> str:
    """A tool that always fails — to show how failures are tracked."""
    raise ValueError("calculator is out of batteries")


TOOLS = {"web_search": wc.web_search, "word_count": wc.word_count, "calculator": broken_calculator}
SPECS = [
    {"type": "function", "function": {"name": "web_search", "description": "Search the web for up-to-date information.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "word_count", "description": "Count how many words are in a piece of text.",
     "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "calculator", "description": "Evaluate an arithmetic expression.",
     "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
]


def agent(question: str, tools: list, max_steps: int = 5) -> str:
    extra = {"tools": tools} if tools else {}          # some runs use no tools at all
    messages = [{"role": "system", "content": "Use your tools when they help answer accurately."},
                {"role": "user", "content": question}]
    for _ in range(max_steps):
        msg = monitor.chat(client, model=cfg.model, messages=messages, **extra).choices[0].message   # <- tracked
        if not msg.tool_calls:
            return msg.content
        messages.append(msg)
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            result = monitor.tool(TOOLS[call.function.name], **args)                                  # <- tracked
            messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})
    return "stopped: too many steps"


questions = [
    ("normal run", "search for the latest langchain version, then tell me how many words your answer is.", SPECS),
    ("broken tool", "use the calculator to compute 18500 * 0.2", [SPECS[2]]),
    ("no tools", "explain what an AI agent is in two sentences.", []),
]
for label, question, tools in questions:
    with monitor.run(label) as run_id:
        answer = agent(question, tools)
        print(f"\n[run {run_id} · {label}]\n{answer}")

print(f"\nsaved to {monitor.DB_PATH} and {monitor.LOG_PATH}. now run: python 08_monitoring/02_usage_report.py")
