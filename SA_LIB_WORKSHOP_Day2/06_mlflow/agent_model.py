# the agent that 04_package_agent.py logs as an MLflow model ("models from code").
# MLflow's ResponsesAgent interface = a standard input/output format, so the model can be
# served behind a REST API (mlflow models serve) no matter which framework is inside.

import json
import uuid

import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
from openai import OpenAI

import workshop_common as wc

SPECS = [
    {"type": "function", "function": {"name": "web_search", "description": "Search the web for up-to-date information.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "word_count", "description": "Count how many words are in a piece of text.",
     "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
]
TOOLS = {"web_search": wc.web_search, "word_count": wc.word_count}


class WorkshopAgent(ResponsesAgent):
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        cfg = wc.get_llm_config()
        client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
        messages = [{"role": "system", "content": "Use your tools when they help answer accurately."}]
        messages += [{"role": m.role, "content": m.content} for m in request.input]
        for _ in range(5):
            msg = client.chat.completions.create(model=cfg.model, messages=messages, tools=SPECS).choices[0].message
            if not msg.tool_calls:
                break
            messages.append(msg)
            for call in msg.tool_calls:
                result = TOOLS[call.function.name](**json.loads(call.function.arguments or "{}"))
                messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})
        item = self.create_text_output_item(text=msg.content or "", id=str(uuid.uuid4()))
        return ResponsesAgentResponse(output=[item])


mlflow.models.set_model(WorkshopAgent())
