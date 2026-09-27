# 04 · Google ADK (Agent Development Kit)

- **Agent** (`LlmAgent`): `name`, `model`, `instruction`, `tools`. **Tools are plain Python functions**:
  type hints and a docstring are enough.
- **Runner** + **SessionService**: run the agent and store conversations (the session is the memory).
- **Workflow agents**: `SequentialAgent`, `ParallelAgent`, `LoopAgent`, plus `sub_agents` for
  LLM-driven delegation.
- **`adk web`**: a built-in chat UI for debugging (events, tool calls, state).

| file | concept |
|---|---|
| `01_first_agent.py` | `Agent` + `InMemoryRunner` + `run_async` event stream |
| `02_agent_with_tools.py` | plain functions as tools; `event.get_function_calls()` |
| `03_agent_with_memory.py` | same `session_id` = same conversation |
| `04_agent_with_fallback.py` | try providers in order |
| `05_multi_agent.py` | `SequentialAgent`: researcher saves `output_key="research"`, writer reads `{research}` |
| `agent.py` | complete: fallback + tools + memory |
| `adk_web_agent/` | the complete agent packaged for `adk web` |

```bash
pip install -r 04_google_adk/requirements.txt
python 04_google_adk/05_multi_agent.py

cd 04_google_adk && adk web        # open http://localhost:8000 → pick "adk_web_agent"
```

**Models:** with `LLM_PROVIDER=gemini`, ADK uses Gemini natively (key: `GOOGLE_API_KEY`). Other
providers go through `LiteLlm(model="openai/<model>", api_base=..., api_key=...)`.

Notebook: [`04_google_adk.ipynb`](04_google_adk.ipynb)

---

*SA-LIB AI Workshop 2026 · Dr. Nasser Mooman, Magan AI Inc. · nmooman@gmail.com · written with help from Claude. Teaching code, not for production use.*
