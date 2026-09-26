# 05 · Microsoft Agent Framework

The successor to **AutoGen** and **Semantic Kernel** (Python and .NET).

- **Chat client**: which model to use (`OpenAIChatCompletionClient` works with any OpenAI-compatible provider)
- **Agent**: `Agent(client, instructions=..., tools=[...])`, then `await agent.run("...")` → `result.text`
- **Session**: `agent.create_session()`, passed to `run(..., session=session)`, gives memory
- **Workflows / orchestrations**: `SequentialBuilder`, `ConcurrentBuilder`, `GroupChatBuilder`,
  `HandoffBuilder`, `MagenticBuilder`

| file | concept |
|---|---|
| `01_first_agent.py` | client + `Agent` + `run` |
| `02_agent_with_tools.py` | `@tool` (optional: plain functions work too) |
| `03_agent_with_memory.py` | `AgentSession` |
| `04_agent_with_fallback.py` | try providers in order |
| `05_multi_agent.py` | `SequentialBuilder(participants=[researcher, writer])` workflow |
| `agent.py` | complete: fallback + tools + memory |

```bash
pip install -r 05_microsoft_agent_framework/requirements.txt
python 05_microsoft_agent_framework/05_multi_agent.py
```

Notebook: [`05_microsoft_agent_framework.ipynb`](05_microsoft_agent_framework.ipynb)
