# 01 · LangGraph — the agent loop as a graph

LangChain's `create_agent` is built **on** LangGraph. Here we build the graph ourselves, so nothing is
hidden:

```
STATE   the data that flows through the graph (e.g. the list of messages)
NODES   python functions: take the state, return an update
EDGES   who runs next: fixed (A -> B) or conditional (a function picks the next node)
```

```
START --> [agent] --tool call?--> [tools] --+
             ^                              |
             +------------------------------+
          [agent] --no tool call--> END
```

| file | concept |
|---|---|
| `01_first_agent.py` | `StateGraph(MessagesState)`, one node, `START -> chatbot -> END` |
| `02_agent_with_tools.py` | `bind_tools`, `ToolNode`, `tools_condition`: the ReAct loop |
| `03_agent_with_memory.py` | `compile(checkpointer=InMemorySaver())` + `thread_id`; `get_state()` |
| `04_agent_with_fallback.py` | try providers in order until one answers |
| `05_multi_agent.py` | custom `TypedDict` state; researcher → writer → reviewer with a **loop back** |
| `agent.py` | complete: fallback + tools + memory |
| **`06_langgraph_deep_dive.py`** | router with structured output, tool loop, **human-in-the-loop `interrupt()`**, `Command(resume=...)`, streaming |

👉 **Read [`../docs/langgraph_explained.html`](../docs/langgraph_explained.html)** (open it in a browser).
It walks through the deep-dive step by step, with diagrams.

```bash
pip install -r 01_langgraph/requirements.txt
python 01_langgraph/01_first_agent.py
python 01_langgraph/06_langgraph_deep_dive.py        # asks you to approve the draft
AUTO_APPROVE=1 python 01_langgraph/06_langgraph_deep_dive.py   # no prompt
```

Each graph prints Mermaid code. Paste it into https://mermaid.live to see the picture.
Notebook: [`01_langgraph.ipynb`](01_langgraph.ipynb)

---

*SA-LIB AI Workshop 2026 · Dr. Nasser Mooman, Magan AI Inc. · nmooman@gmail.com · written with help from Claude. Teaching code, not for production use.*
