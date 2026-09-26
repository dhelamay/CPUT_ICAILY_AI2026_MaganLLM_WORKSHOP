# 06 · MLflow — trace, evaluate and package agents

MLflow is **not** an agent framework. It's the lab notebook you use *with* any framework:

| file | what it shows |
|---|---|
| `01_trace_langgraph_agent.py` | `mlflow.langchain.autolog()`: every graph node, LLM call and tool call recorded |
| `02_trace_custom_agent.py` | `@mlflow.trace` on your own functions + `mlflow.openai.autolog()` → a span tree |
| `03_evaluate_agent.py` | `mlflow.genai.evaluate()` with code-based `@scorer`s (free, no judge model) |
| `04_package_agent.py` + `agent_model.py` | log the agent as a versioned model (`ResponsesAgent`), load it back, **serve it as a REST API** |

```bash
pip install -r 06_mlflow/requirements.txt
python 06_mlflow/01_trace_langgraph_agent.py
mlflow ui --backend-store-uri sqlite:///mlflow.db     # → http://localhost:5000 → Traces
```

Serve the packaged agent (the exact command is printed by `04_package_agent.py`):

```bash
mlflow models serve -m "models:/<model_id>" -p 5001 --env-manager local
curl -X POST localhost:5001/invocations -H "Content-Type: application/json" \
     -d '{"input": [{"role": "user", "content": "what is an AI agent?"}]}'
```

**One-line tracing for the other folders:** `mlflow.crewai.autolog()`, `mlflow.openai.autolog()` (also
covers the OpenAI Agents SDK), `mlflow.gemini.autolog()`, `mlflow.litellm.autolog()` (Google ADK via
LiteLLM), `mlflow.autogen.autolog()`.

**LLM-as-a-judge:** MLflow also has built-in judges (`from mlflow.genai.scorers import Correctness, Safety`).
They need a judge model (OpenAI by default).

⚠️ Don't set `OTEL_SDK_DISABLED=true`: it turns off MLflow tracing.

Notebook: [`06_mlflow.ipynb`](06_mlflow.ipynb). In Colab, read traces with `mlflow.search_traces()`.
