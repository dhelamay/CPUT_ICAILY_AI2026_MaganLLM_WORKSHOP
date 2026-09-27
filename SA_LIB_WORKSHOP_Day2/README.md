# SA-LIB AI Workshop 2026 — Day 2: one agent, seven frameworks

**Author:** Dr. Nasser Mooman · Magan AI Inc. · [nmooman@gmail.com](mailto:nmooman@gmail.com)  
Written with help from Claude (Anthropic).

We take **one simple agent** and build it again in every major Python agent framework. It comes from
the guide [*How to Build AI Agents Completely Free*](00_langchain_original/ORIGINAL_GUIDE.md) by m0h.
Because the example stays the same, you can compare the frameworks side by side instead of learning
seven different demos.

**The agent, in 6 steps (the same in every folder):**

| step | file | what it adds |
|---|---|---|
| 1 | `01_first_agent.py` | a model + instructions: "explain what an AI agent is in two sentences" |
| 2 | `02_agent_with_tools.py` | **tools**: free web search + a `word_count` function you wrote |
| 3 | `03_agent_with_memory.py` | **memory**: "my name is m0h" … "what's my name?" |
| 4 | `04_agent_with_fallback.py` | **fallback**: if one provider fails, try the next |
| 5 | `05_multi_agent.py` | **multi-agent**: researcher → writer (→ editor) |
| ★ | `agent.py` | the **complete agent**: fallback + tools + memory |

## The folders

| folder | framework | what makes it different | multi-agent style |
|---|---|---|---|
| [`00_langchain_original`](00_langchain_original/) | **LangChain** `create_agent` | the original code; one function builds the whole agent | — |
| [`01_langgraph`](01_langgraph/) | **LangGraph** | you build the loop as a **graph** (state, nodes, edges); plus a deep-dive and an [HTML explainer](docs/langgraph_explained.html) | graph with a feedback loop |
| [`02_crewai`](02_crewai/) | **CrewAI** | agents have a *role, goal, backstory*; you describe **tasks** | a crew of 3, sequential |
| [`03_openai_agents_sdk`](03_openai_agents_sdk/) | **OpenAI Agents SDK** | minimal: `Agent` + `Runner`; sessions; guardrails | agents-as-tools **and** handoffs |
| [`04_google_adk`](04_google_adk/) | **Google ADK** | plain functions are tools; built-in chat UI (`adk web`) | `SequentialAgent` with shared state |
| [`05_microsoft_agent_framework`](05_microsoft_agent_framework/) | **Microsoft Agent Framework** | successor of AutoGen and Semantic Kernel; sessions, workflows | `SequentialBuilder` workflow |
| [`06_mlflow`](06_mlflow/) | **MLflow** | *not* a framework: it **traces, evaluates and packages** agents from any framework | — |
| [`07_a2a_multi_agent`](07_a2a_multi_agent/) | **A2A protocol** | agents run as **separate web services** and talk over HTTP | remote researcher + writer + orchestrator |
| [`08_monitoring`](08_monitoring/) | **DIY tracker · Phoenix · Langfuse · LangSmith** | track **calls, tokens, cost, latency and failures**; all have a free option, plus a table of alternatives | — |
| [`09_mcp`](09_mcp/) | **MCP (Model Context Protocol)** | a weather **MCP server** wrapping a free API, a plain client, and **one agent** that discovers the tools | — |

Every folder has its own `README.md`, `requirements.txt` and a **notebook** (`<folder>.ipynb`) that
runs in Colab, VS Code or Jupyter.

> [!WARNING]
> **Teaching code — do not use in production.** These are sample programs for a workshop. They are kept
> short on purpose and leave out what a real system needs: authentication and access control, input
> validation, protection against prompt injection, secret management, error handling and retries,
> rate limiting, logging, tests and evaluation. LLM answers can be wrong, so check anything important.
> When you use a hosted API, you are responsible for your keys, for what data you send, for its costs
> and for following its terms. The code is provided "as is", without warranty (see `LICENSE`).

---

## Bring your own LLM API key

**Participants need to bring their own LLM API key.** The workshop does not provide API keys or GPUs.
Create one key **before the workshop** and check that it works. One key from any of these is enough:

| provider | cost | get a key |
|---|---|---|
| Groq *(default)* | free tier, no card | https://console.groq.com |
| Google Gemini | free tier, no card | https://aistudio.google.com |
| OpenRouter | free `:free` models | https://openrouter.ai/keys |
| DeepSeek | very cheap (small top-up) | https://platform.deepseek.com |
| OpenAI | paid | https://platform.openai.com/api-keys |

**No key needed only if you already run a local LLM server** on your laptop: **Ollama**, **SGLang** or
**vLLM**. All three speak the OpenAI protocol, so in `.env` set `LLM_PROVIDER=ollama`, point
`OLLAMA_BASE_URL` at your server and set `LLM_MODEL` to the model it serves:

| local server | `OLLAMA_BASE_URL` (default port) | `LLM_MODEL` example |
|---|---|---|
| Ollama | `http://localhost:11434/v1` | `qwen2.5:7b` |
| SGLang | `http://localhost:30000/v1` | `Qwen/Qwen2.5-7B-Instruct` |
| vLLM | `http://localhost:8000/v1` | `Qwen/Qwen2.5-7B-Instruct` |

Download the model before the workshop; conference Wi-Fi is too slow for multi-GB downloads.
Small local models (7B and under) often handle tool calling poorly, so the agent examples work best
with a hosted API.

---

## 1. Get an API key (free)

**You need exactly one LLM key.** Nothing else is required: web search is DuckDuckGo (free, no key), and
the LangSmith and Langfuse keys in folder 08 are optional. **OpenRouter** gives you many models behind one key: set
`LLM_PROVIDER=openrouter` and `LLM_MODEL=openai/gpt-4o-mini` (or `google/gemini-2.5-flash`, …).

All examples work with **any** of these providers. You set it with `LLM_PROVIDER` in `.env`:

| `LLM_PROVIDER` | cost | key variable | where |
|---|---|---|---|
| `groq` *(default)* | free, no card | `GROQ_API_KEY` | https://console.groq.com |
| `gemini` | free, no card | `GOOGLE_API_KEY` | https://aistudio.google.com |
| `openrouter` | free `:free` models | `OPENROUTER_API_KEY` | https://openrouter.ai/keys |
| `deepseek` | very cheap | `DEEPSEEK_API_KEY` | https://platform.deepseek.com |
| `openai` | paid | `OPENAI_API_KEY` | https://platform.openai.com |
| `ollama` | free, local | none | https://ollama.com |

**How can every framework use every provider?** All of these providers speak the *OpenAI
chat-completions* protocol. The shared helper `workshop_common.py` only swaps `base_url`, `api_key` and
`model`, and each framework plugs that in its own way:

| framework | how the provider is plugged in |
|---|---|
| LangGraph | `ChatOpenAI(model=..., base_url=..., api_key=...)` |
| CrewAI | `LLM(model="openai/<model>", base_url=..., api_key=...)` |
| OpenAI Agents SDK | `OpenAIChatCompletionsModel(model=..., openai_client=AsyncOpenAI(base_url=..., api_key=...))` |
| Google ADK | Gemini natively (`model="gemini-2.5-flash"`), others via `LiteLlm(model="openai/<model>", api_base=..., api_key=...)` |
| Microsoft Agent Framework | `OpenAIChatCompletionClient(model=..., base_url=..., api_key=...)` |

---

## 2. Setup

> **Tip:** these frameworks pin different versions of shared packages, so use **one virtual environment
> per folder**. The commands below do that.

### Option A: Linux / macOS / Windows terminal

```bash
git clone https://github.com/dhelamay/CPUT_ICAILY_AI2026_MaganLLM_WORKSHOP.git
cd CPUT_ICAILY_AI2026_MaganLLM_WORKSHOP/SA_LIB_WORKSHOP_Day2
cp .env.example .env            # Windows: copy .env.example .env   -> then paste your key

# pick a folder, e.g. CrewAI
python3 -m venv .venv-crewai
source .venv-crewai/bin/activate          # Windows: .venv-crewai\Scripts\activate
pip install -r 02_crewai/requirements.txt

python 02_crewai/01_first_agent.py        # run from the repo root
```

With [uv](https://docs.astral.sh/uv/) it's even faster: `uv venv .venv-crewai && uv pip install -r 02_crewai/requirements.txt`.

### Option B: VS Code

1. **File → Open Folder…** → `SA_LIB_WORKSHOP_Day2`, and create `.env` from `.env.example`.
2. `Ctrl+Shift+P` → **Python: Create Environment** → *Venv* → choose the folder's `requirements.txt`.
3. Open a script and press ▶. Or open `<folder>/<folder>.ipynb` and pick that `.venv` as the kernel.

### Option C: Google Colab

1. Open any `<folder>/<folder>.ipynb` in Colab (**File → Upload notebook**, or the *Open in Colab*
   badge once this repo is on GitHub).
2. 🔑 **Secrets** (left sidebar) → add `GROQ_API_KEY` (or another key) → enable *Notebook access*.
3. **Runtime → Run all.** Each notebook installs its own packages and writes its helper files, so no
   clone is needed.

---

## 3. Suggested workshop flow (≈ 6 hours)

| time | session |
|---|---|
| 30 min | What is an agent? Run `00_langchain_original` (plan → act → observe) |
| 60 min | **LangGraph**: the loop as a graph, steps 1–5, then `06_langgraph_deep_dive.py` with the HTML explainer |
| 45 min | **CrewAI** and **OpenAI Agents SDK**: roles and tasks vs. agents, handoffs |
| 45 min | **Google ADK** and **Microsoft Agent Framework**: sessions, workflows, `adk web` |
| 30 min | **MLflow**: trace every framework, evaluate, package |
| 45 min | **A2A**: agents as web services talking to each other |
| 30 min | **Monitoring**: DIY usage report, Phoenix traces, Langfuse and LangSmith dashboards |
| 30 min | **MCP**: one tool server, any client: weather API → MCP server → agent |
| 15 min | Wrap-up: choosing a framework (table below) |

## 4. Which framework should I use?

| if you want… | try |
|---|---|
| the fastest start, huge ecosystem | LangChain `create_agent` |
| full control over the loop, human-in-the-loop, durable state | **LangGraph** |
| role-based teams, readable config | **CrewAI** |
| minimal code, OpenAI-first, handoffs | **OpenAI Agents SDK** |
| Gemini, Google Cloud, built-in dev UI, A2A | **Google ADK** |
| .NET + Python, Azure, enterprise workflows | **Microsoft Agent Framework** |
| to see, test and ship any of the above | **MLflow** |
| to share tools between apps and agents (Claude Desktop, VS Code, any framework) | **MCP** |
| to watch cost, latency and failures in production | **Phoenix** (local), **LangSmith** (hosted, LangChain-first) or **Langfuse** (open source) |
| agents from different teams/frameworks cooperating | **A2A** |

---

## 5. Troubleshooting

| problem | fix |
|---|---|
| `... API_KEY is missing` | create `.env` in the **repo root** (copy `.env.example`) |
| `429` / rate limit | free tiers are limited: wait a minute, or change `LLM_PROVIDER` |
| `web search failed` | DuckDuckGo is free and rate-limited: rerun after a few seconds |
| the model doesn't call tools | small models are weaker at tool calling: use `llama-3.3-70b-versatile`, `gemini-2.5-flash`, `deepseek-chat` or `gpt-4o-mini` |
| CrewAI asks about tracing | keep `CREWAI_TRACING_ENABLED=false` in `.env` |
| ADK tries to use Google Cloud | keep `GOOGLE_GENAI_USE_VERTEXAI=FALSE` in `.env` |
| A2A: `connection refused` | start both servers first, or just run `python 07_a2a_multi_agent/run_demo.py` |
| dependency conflicts | use a separate virtual environment per folder |

## 6. For maintainers

- `shared/workshop_common.py` is the single source. After editing it, run `python tools/sync_common.py`
  to copy it into every folder.
- The notebooks are generated from the scripts: `python tools/build_notebooks.py`. After pushing to
  GitHub, set `GITHUB_REPO` in that file so the Colab badges point to your repo.
- Tested on Python 3.12 with langgraph 1.2, crewai 1.15, openai-agents 0.22, google-adk 2.10,
  agent-framework 1.19, mlflow 3.16, a2a-sdk 1.1, arize-phoenix 20, langfuse 4, langsmith 0.14 and mcp 2.2.

---

*Based on [build-ai-agents-free](https://github.com/Moh4696/build-ai-agents-free) by m0h (MIT). ·
Day 1 (RAG): `SA_LIB_WORKSHOP_Day1`*
