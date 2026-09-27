# 07 · A2A — agents that talk to each other over the network

**A2A (Agent2Agent)** is an open protocol, now under the Linux Foundation. Agents built with *any*
framework, by *any* team, can **discover** each other and **send each other tasks** over HTTP.

```
             ┌──────────────── orchestrator.py (ADK) ─────────────────┐
             │   RemoteA2aAgent("researcher")   RemoteA2aAgent("writer")│
             └──────────┬──────────────────────────────┬──────────────┘
                 HTTP / JSON-RPC                 HTTP / JSON-RPC
      ┌─────────────────▼───────────┐      ┌────────────▼──────────────┐
      │ researcher_server.py  :8001 │      │ writer_server.py   :8002  │
      │ tools: web_search           │      │ tools: word_count          │
      │ /.well-known/agent-card.json│      │ /.well-known/agent-card.json│
      └─────────────────────────────┘      └────────────────────────────┘
```

1. **Agent card**: every A2A server publishes `/.well-known/agent-card.json` (name, description,
   skills, endpoint). It's the agent's business card.
2. **Discover**: a client downloads the card.
3. **Send a message / task**: the server runs its agent and streams back status updates and results.

| file | role |
|---|---|
| `researcher_server.py` | ADK agent with `web_search`, served with `to_a2a()` on **:8001** |
| `writer_server.py` | ADK agent with `word_count`, served on **:8002** |
| `a2a_client.py` | the protocol with **no framework**: `A2ACardResolver` → `create_client` → `send_message` |
| `orchestrator.py` | ADK `SequentialAgent` whose sub-agents are `RemoteA2aAgent`s (they live in other processes) |
| `run_demo.py` | starts both servers, runs the client + orchestrator, then stops everything |

```bash
pip install -r 07_a2a_multi_agent/requirements.txt

# easiest: one command
python 07_a2a_multi_agent/run_demo.py

# or by hand, in 3 terminals (see the agent card at http://localhost:8001/.well-known/agent-card.json)
python 07_a2a_multi_agent/researcher_server.py
python 07_a2a_multi_agent/writer_server.py
python 07_a2a_multi_agent/orchestrator.py
```

**Exercise:** replace the writer with an agent from another framework. Wrap it in an a2a-sdk
`AgentExecutor`, or use Microsoft Agent Framework's `agent_framework.a2a` module. The orchestrator
doesn't need to change, and that interchangeability is the point of A2A.

Notebook: [`07_a2a_multi_agent.ipynb`](07_a2a_multi_agent.ipynb). It works in Colab too, because the
servers run on localhost inside the Colab machine.
