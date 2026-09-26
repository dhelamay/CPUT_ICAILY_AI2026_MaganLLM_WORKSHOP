# 03 · OpenAI Agents SDK

A small framework with few concepts: **Agent** (instructions + tools + model), **Runner** (runs the
loop), **Session** (memory), **handoffs** (pass the conversation to another agent), and **guardrails**.

| file | concept |
|---|---|
| `01_first_agent.py` | `Agent` + `await Runner.run(agent, "...")` → `result.final_output` |
| `02_agent_with_tools.py` | `@function_tool`; inspect `result.new_items` to see each step |
| `03_agent_with_memory.py` | `SQLiteSession("chat-1")` (use `SQLiteSession("chat-1", "memory.db")` to persist) |
| `04_agent_with_fallback.py` | try providers in order |
| `05_multi_agent.py` | **A)** manager with `agent.as_tool()` specialists; **B)** triage with `handoffs=[...]` |
| `agent.py` | complete: fallback + tools + memory |

```bash
pip install -r 03_openai_agents_sdk/requirements.txt
python 03_openai_agents_sdk/05_multi_agent.py
```

**Non-OpenAI providers** go through
`OpenAIChatCompletionsModel(model=..., openai_client=AsyncOpenAI(base_url=..., api_key=...))`.
Tracing uploads to the OpenAI dashboard, so it's switched off automatically when there's no
`OPENAI_API_KEY`. Use MLflow (folder 06) to trace instead.

Notebook: [`03_openai_agents_sdk.ipynb`](03_openai_agents_sdk.ipynb)
