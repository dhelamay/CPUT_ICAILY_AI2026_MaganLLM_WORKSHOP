# 08 · Monitoring and tracking agents (all free to start)

Once an agent runs for real users, you need answers to four questions: **What did it do? How much did
it cost? How slow was it? What failed?** This folder answers them four ways, from simplest to most
complete. All four have a free option.

| file | tool | needs | what you get |
|---|---|---|---|
| `monitor.py` + `01_simple_monitor.py` | **DIY tracker** (~100 lines) | nothing | every LLM call, tool call and run in SQLite + a JSON log: latency, tokens, cost, errors |
| `02_usage_report.py` | report | nothing | calls, failure %, avg/max latency, tokens and cost per model, tool errors, last errors |
| `03_phoenix_tracing.py` | **[Arize Phoenix](https://github.com/Arize-ai/phoenix)** (free, self-hosted) | `phoenix serve`, local, no account | a web UI with every trace: graph steps, prompts, answers, tokens, latency, errors |
| `04_langfuse_tracing.py` | **[Langfuse](https://github.com/langfuse/langfuse)** (open source) | free cloud account **or** self-host with Docker | traces + **sessions, users, cost dashboards**, scores, prompt management |
| `05_langsmith_tracing.py` | **[LangSmith](https://smith.langchain.com)** (by LangChain, hosted) | free Developer account + API key | traces, monitoring charts (latency, errors, tokens, cost), datasets and evaluation; **zero code changes** for LangChain/LangGraph |

`agent_graph.py` is the same complete LangGraph agent used in 03, 04 and 05. The monitoring tools only
*watch* the agent; its code doesn't change.

## 1. DIY tracker (start here)

```bash
pip install -r 08_monitoring/requirements.txt
python 08_monitoring/01_simple_monitor.py     # runs 3 questions; one uses a tool that always fails
python 08_monitoring/02_usage_report.py
```

Example output:

```
-- by kind ---------------------------------------------------
kind    calls  errors  fail %   avg ms   max ms
llm         6       0    0.0%      812     1904
run         3       0    0.0%     2410     4630
tool        3       1   33.3%      926     2778
-- last 5 errors ---------------------------------------------
2026-09-26T15:12:51 run=92b42628 tool:broken_calculator
    ValueError('calculator is out of batteries')
```

To track **your own** agent, make three changes:

```python
import monitor
with monitor.run("my task"):                                        # 1. group calls into a run
    resp = monitor.chat(client, model=..., messages=..., tools=...)  # 2. instead of client.chat.completions.create
    out = monitor.tool(my_tool, query="...")                          # 3. instead of my_tool(query="...")
```

Edit `PRICES` in `monitor.py` to match your provider's prices (the ones included are examples).
`agent_events.log` has one JSON line per event, ready for Grafana Loki, Elasticsearch or any log tool.

## 2. Arize Phoenix: a local tracing UI, no account

```bash
phoenix serve                                   # terminal 1 -> http://localhost:6006
python 08_monitoring/03_phoenix_tracing.py      # terminal 2
```

Open the **sa-lib-day2** project and click a trace to see its tree: `LangGraph → agent → ChatOpenAI →
tools → word_count …`, with the prompt, answer, token counts and time of each step.

**Other frameworks:** install the matching OpenInference instrumentor and call it the same way:

| framework | package | line |
|---|---|---|
| CrewAI | `openinference-instrumentation-crewai` | `CrewAIInstrumentor().instrument(tracer_provider=provider)` |
| OpenAI Agents SDK | `openinference-instrumentation-openai-agents` | `OpenAIAgentsInstrumentor().instrument(...)` |
| Google ADK | `openinference-instrumentation-google-adk` | `GoogleADKInstrumentor().instrument(...)` |
| plain `openai` client | `openinference-instrumentation-openai` | `OpenAIInstrumentor().instrument(...)` |

We set up OpenTelemetry by hand instead of calling `phoenix.otel.register()`, which crashes with some
OpenTelemetry versions. The hand-written setup is standard OpenTelemetry, so the same lines work for
any OTLP backend.

## 3. Langfuse: dashboards for users, sessions and cost

1. Get keys, choosing one:
   - **Cloud (free tier):** https://cloud.langfuse.com → new project → *API keys*
   - **Self-host:** `git clone https://github.com/langfuse/langfuse && cd langfuse && docker compose up`,
     then open http://localhost:3000
2. Add them to `.env`:
   ```bash
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_HOST=https://cloud.langfuse.com      # or http://localhost:3000
   ```
3. `python 08_monitoring/04_langfuse_tracing.py`, then check **Tracing**, **Sessions** and **Users** in Langfuse.

`CallbackHandler()` records every step. `propagate_attributes(user_id=..., session_id=..., tags=...)`
lets you filter by user or conversation.

## 4. LangSmith: tracing from the LangChain team

LangSmith is a hosted service (not open source) with a **free Developer plan** that's plenty for
learning. See https://www.langchain.com/pricing for current limits. For LangChain and LangGraph, tracing
needs **no code changes at all**: just set three environment variables.

1. Sign up at https://smith.langchain.com → **Settings → API Keys → Create API key**.
2. Add to `.env`:
   ```bash
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=lsv2_...
   LANGSMITH_PROJECT=sa-lib-day2
   ```
3. `python 08_monitoring/05_langsmith_tracing.py`, then open **Projects → sa-lib-day2** in LangSmith.

The script shows both ways to use it:

| your code | what to do |
|---|---|
| LangChain / LangGraph (folders 00, 01) | nothing: the env vars above switch tracing on |
| anything else (plain `openai`, other frameworks) | `client = wrap_openai(OpenAI(...))` to trace LLM calls, and `@traceable` on your own functions and tools |

> 💡 Once `LANGSMITH_TRACING=true` is in `.env`, **every** LangChain/LangGraph script in this repo is
> traced, including `00_langchain_original` and `01_langgraph`. Set it to `false` to stop.

## Which one should I use?

| situation | pick |
|---|---|
| learning, or a tiny app | DIY tracker |
| debugging an agent on your laptop | Phoenix |
| you build with LangChain / LangGraph and want the quickest start | LangSmith |
| a team, real users, cost per user, and you want to self-host | Langfuse |
| already using MLflow for models and evaluation | MLflow tracing (folder `06_mlflow`) |

## Other tools you can use

All of these trace LLM calls and agents and have a free option. Most of them accept **OpenTelemetry**,
so the setup in `03_phoenix_tracing.py` works with them once you change the endpoint.

| tool | open source? | free option | good for | how you connect it |
|---|---|---|---|---|
| **DIY tracker** (`monitor.py`) | yours | free | learning, tiny apps | wrap calls with `monitor.chat()` / `monitor.tool()` |
| **[LangSmith](https://smith.langchain.com)** | no (hosted) | free Developer plan | LangChain / LangGraph | env vars, `@traceable` |
| **[Langfuse](https://langfuse.com)** | yes (MIT) | self-host, or free cloud tier | teams, users, sessions, cost | callback / `@observe` / OpenTelemetry |
| **[Arize Phoenix](https://phoenix.arize.com)** | source-available | free to self-host | local debugging, evals | OpenTelemetry + OpenInference |
| **[MLflow](https://mlflow.org)** | yes (Apache-2.0) | free to self-host | tracing + evaluation + model registry | `mlflow.<framework>.autolog()` (folder 06) |
| **[Opik](https://github.com/comet-ml/opik)** (Comet) | yes (Apache-2.0) | self-host, or free cloud tier | tracing + LLM-as-judge evaluation | `@track`, framework integrations |
| **[Helicone](https://helicone.ai)** | yes (Apache-2.0) | free tier | request logs + cost with **no code changes** | it's a proxy: change the `base_url` |
| **[OpenLIT](https://github.com/openlit/openlit)** | yes (Apache-2.0) | free to self-host | OpenTelemetry-native dashboards | `openlit.init()` |
| **[AgentOps](https://www.agentops.ai)** | SDK is open source | free tier | agent session replays (CrewAI, Agents SDK, ADK …) | `agentops.init()` |
| **[W&B Weave](https://wandb.ai/site/weave)** | SDK is open source | free tier | teams already using Weights & Biases | `weave.init()`, `@weave.op` |

**Simple rule:** learn with the DIY tracker, debug locally with Phoenix, then pick **LangSmith** (hosted,
LangChain-first) or **Langfuse** (open source, self-hostable) for a real deployment.

Notebook: [`08_monitoring.ipynb`](08_monitoring.ipynb). In Colab it starts Phoenix inside the notebook
with `px.launch_app()`.

---

*SA-LIB AI Workshop 2026 · Dr. Nasser Mooman, Magan AI Inc. · nmooman@gmail.com · written with help from Claude. Teaching code, not for production use.*
