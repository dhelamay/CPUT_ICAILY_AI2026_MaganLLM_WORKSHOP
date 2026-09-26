# MLflow 02 — tracing an agent you wrote yourself (no framework)
# @mlflow.trace turns any function into a SPAN. nested calls become a tree.
# mlflow.openai.autolog() adds a span for every LLM call (works with groq/deepseek/gemini too —
# they all use the openai client).
#
# run:   python 06_mlflow/02_trace_custom_agent.py
# view:  mlflow ui --backend-store-uri sqlite:///mlflow.db

import json

import mlflow
from mlflow.entities import SpanType
from openai import OpenAI

import workshop_common as wc

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("SA-LIB Day2 agents")
mlflow.openai.autolog()

cfg = wc.get_llm_config()
client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)


@mlflow.trace(span_type=SpanType.TOOL)
def web_search(query: str) -> str:
    return wc.web_search(query)


@mlflow.trace(span_type=SpanType.TOOL)
def word_count(text: str) -> int:
    return wc.word_count(text)


TOOLS = {"web_search": web_search, "word_count": word_count}
SPECS = [
    {"type": "function", "function": {"name": "web_search", "description": "Search the web for up-to-date information.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "word_count", "description": "Count how many words are in a piece of text.",
     "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
]


@mlflow.trace(span_type=SpanType.AGENT, name="my_agent")
def run_agent(question: str, max_steps: int = 5) -> str:
    messages = [{"role": "system", "content": "Use your tools when they help answer accurately."},
                {"role": "user", "content": question}]
    for _ in range(max_steps):                                     # plan -> act -> observe
        msg = client.chat.completions.create(model=cfg.model, messages=messages, tools=SPECS).choices[0].message
        if not msg.tool_calls:
            return msg.content
        messages.append(msg)
        for call in msg.tool_calls:
            result = TOOLS[call.function.name](**json.loads(call.function.arguments or "{}"))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})
    return "stopped: too many steps"


print(run_agent("search for the latest langchain version, then tell me how many words your answer is."))

mlflow.flush_trace_async_logging()
trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
print(f"\ntrace has {len(trace.data.spans)} spans:")
for span in trace.data.spans:
    print(f"  - {span.name} ({span.span_type})")
