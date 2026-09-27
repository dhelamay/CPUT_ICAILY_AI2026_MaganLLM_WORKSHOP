# 02 · CrewAI — agents with roles, working as a crew

CrewAI describes an AI **team**:

- **Agent**: `role`, `goal`, `backstory` (together they form the system prompt), plus `tools` and `llm`
- **Task**: `description` + `expected_output`, assigned to an agent
- **Crew**: runs the tasks (`Process.sequential` or `Process.hierarchical`); each task's output becomes
  context for the next one

| file | concept |
|---|---|
| `01_first_agent.py` | Agent + Task + Crew → `kickoff_async()` |
| `02_agent_with_tools.py` | `@tool("name")` functions; `verbose=True` shows Thought / Action / Observation |
| `03_agent_with_memory.py` | `Agent.kickoff_async(messages)` with a growing message list |
| `04_agent_with_fallback.py` | try providers in order |
| `05_multi_agent.py` | **researcher → writer → editor** crew, with `{topic}` filled from `inputs` |
| `agent.py` | complete: fallback + tools + memory |

```bash
pip install -r 02_crewai/requirements.txt
python 02_crewai/05_multi_agent.py
```

Notes:
- The provider is plugged in as `LLM(model="openai/<model>", base_url=..., api_key=...)`. The
  `openai/` prefix means *any OpenAI-compatible endpoint*.
- We use `kickoff_async()` so the same code runs in scripts **and** notebooks. (Notebooks already run
  an event loop, and CrewAI refuses a synchronous `kickoff()` inside one.)
- `Crew(memory=True)` gives long-term vector memory but needs an embeddings provider. We keep it
  simple and free with a message list.

Notebook: [`02_crewai.ipynb`](02_crewai.ipynb)
