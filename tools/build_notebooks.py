"""
Build one Jupyter notebook per framework folder from the .py files in that folder.

The .py files are the single source of truth. After editing a script, run:
    python tools/build_notebooks.py

Every notebook is SELF-CONTAINED, so it runs in Google Colab without cloning the repo:
  1. installs the folder's requirements
  2. loads API keys (Colab Secrets -> .env -> ask with getpass)
  3. writes workshop_common.py (and any helper files) with %%writefile
  4. one section per script: the script's header comment as text, then its code
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GITHUB_REPO = "dhelamay/CPUT_ICAILY_AI2026_MaganLLM_WORKSHOP"
REPO_SUBDIR = "SA_LIB_WORKSHOP_Day2"  # this day's folder inside the repo

KEYS = ["GROQ_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY",
        "LANGSMITH_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]

FOLDERS = {
    "00_langchain_original": {
        "title": "LangChain — the original guide (Groq + Gemini)",
        "intro": "The original *How to Build AI Agents Completely Free* code, unchanged: `create_agent` with a free "
                 "Groq model, tools, memory and a Groq → Gemini fallback. Every other folder rebuilds **this same "
                 "agent** in a different framework.\n\nNeeds `GROQ_API_KEY` (and `GOOGLE_API_KEY` for the fallback).",
        "scripts": ["01_first_agent.py", "02_agent_with_tools.py", "03_agent_with_memory.py",
                    "04_agent_with_fallback.py", "agent.py"],
        "common": False,
    },
    "01_langgraph": {
        "title": "LangGraph — build the agent loop as a graph",
        "intro": "LangGraph is the engine under LangChain's `create_agent`. Here we build the graph **by hand** "
                 "(state + nodes + edges), then go further: a multi-agent loop and a deep-dive with a router, "
                 "human approval (`interrupt`) and streaming. See `docs/langgraph_explained.html`.",
        "scripts": ["01_first_agent.py", "02_agent_with_tools.py", "03_agent_with_memory.py",
                    "04_agent_with_fallback.py", "05_multi_agent.py", "agent.py", "06_langgraph_deep_dive.py"],
    },
    "02_crewai": {
        "title": "CrewAI — agents with roles, tasks and crews",
        "intro": "CrewAI models a **team**: each Agent has a role, goal and backstory; Tasks say what to do; a "
                 "Crew runs them. Great for multi-agent pipelines.",
        "scripts": ["01_first_agent.py", "02_agent_with_tools.py", "03_agent_with_memory.py",
                    "04_agent_with_fallback.py", "05_multi_agent.py", "agent.py"],
    },
    "03_openai_agents_sdk": {
        "title": "OpenAI Agents SDK — Agent, Runner, tools, sessions, handoffs",
        "intro": "OpenAI's lightweight agent framework. Works with **any** OpenAI-compatible provider through "
                 "`OpenAIChatCompletionsModel`, so Groq, Gemini, DeepSeek and Ollama all work.",
        "scripts": ["01_first_agent.py", "02_agent_with_tools.py", "03_agent_with_memory.py",
                    "04_agent_with_fallback.py", "05_multi_agent.py", "agent.py"],
    },
    "04_google_adk": {
        "title": "Google ADK — Agent Development Kit",
        "intro": "Google's agent framework. Gemini works natively; every other provider goes through LiteLLM. "
                 "Tools are plain Python functions. Locally you can also try the chat UI with `adk web`.",
        "scripts": ["01_first_agent.py", "02_agent_with_tools.py", "03_agent_with_memory.py",
                    "04_agent_with_fallback.py", "05_multi_agent.py", "agent.py"],
    },
    "05_microsoft_agent_framework": {
        "title": "Microsoft Agent Framework — the successor of AutoGen + Semantic Kernel",
        "intro": "Microsoft's agent framework: a chat client + `Agent` + `agent.run()`. Sessions give memory; "
                 "workflows (`SequentialBuilder`, `GroupChatBuilder`, `HandoffBuilder`, ...) give multi-agent.",
        "scripts": ["01_first_agent.py", "02_agent_with_tools.py", "03_agent_with_memory.py",
                    "04_agent_with_fallback.py", "05_multi_agent.py", "agent.py"],
    },
    "06_mlflow": {
        "title": "MLflow — trace, evaluate and package your agent",
        "intro": "MLflow is not an agent framework: it's the **lab notebook** for all of them. Tracing shows every "
                 "LLM and tool call, evaluation scores your agent on a test set, and packaging turns it into a "
                 "versioned model you can serve.\n\nIn Colab, look at traces with "
                 "`mlflow.search_traces()` (the UI needs a local machine or a tunnel).",
        "scripts": ["01_trace_langgraph_agent.py", "02_trace_custom_agent.py", "03_evaluate_agent.py",
                    "04_package_agent.py"],
        "files": ["agent_model.py"],
    },
    "07_a2a_multi_agent": {
        "title": "A2A — agents that talk to each other over the network",
        "intro": "The **Agent2Agent (A2A)** protocol lets agents built with *any* framework discover and call each "
                 "other over HTTP. Two agents run as separate web services (researcher on :8001, writer on :8002); "
                 "a plain A2A client and an ADK orchestrator use them.",
        "scripts": [],
        "files": ["common_model.py", "researcher_server.py", "writer_server.py", "a2a_client.py",
                  "orchestrator.py", "run_demo.py"],
        "extra": [("## Run the whole demo\nStarts both A2A servers in the background, runs the plain client "
                   "and the orchestrator, then stops the servers.", "!python run_demo.py")],
    },
    "08_monitoring": {
        "title": "Monitoring — track calls, tokens, cost, latency and failures",
        "intro": "Three free ways to watch an agent: a **DIY tracker** (SQLite + JSON log, no packages), "
                 "**Arize Phoenix** (tracing UI, local, no account), **Langfuse** (open source; free cloud or self-hosted) "
                 "and **LangSmith** (LangChain's hosted platform, free Developer plan). The agent code doesn't change; the tools only watch it.",
        "scripts": ["01_simple_monitor.py", "02_usage_report.py", "03_phoenix_tracing.py", "04_langfuse_tracing.py",
                    "05_langsmith_tracing.py"],
        "files": ["monitor.py", "agent_graph.py"],
        "before": {"03_phoenix_tracing.py": (
            "### Start Phoenix inside the notebook\nOn a laptop you can instead run `phoenix serve` in a terminal. "
            "Skip this cell if Phoenix is already running.",
            "import phoenix as px\nsession = px.launch_app()   # UI on port 6006 (in Colab: click the link it prints)")},
    },
    "09_mcp": {
        "title": "MCP — a weather MCP server, a client, and an agent that uses it",
        "intro": "**MCP (Model Context Protocol)** is a standard plug for AI tools. `weather_server.py` wraps a free "
                 "public API (Open-Meteo, no key) as MCP tools; `01` calls them directly (no LLM, no key); `02` is "
                 "**one agent** that discovers the tools at run time and lets the LLM decide when to use them.",
        "scripts": ["01_mcp_client.py", "02_mcp_agent.py"],
        "files": ["weather_server.py"],
    },
    "09_mcp_fastmcp": {          # a second notebook in the same folder
        "folder": "09_mcp",
        "title": "FastMCP — a multi-tool MCP server, a client, and an agent that picks between the tools",
        "intro": "**FastMCP** (`pip install fastmcp`) is the most popular high-level way to write MCP servers "
                 "*and* clients in Python. `fastmcp_server.py` offers **6 tools** (math, units, text, time, weather, "
                 "Wikipedia); `03` lists and calls them with FastMCP's `Client` (no LLM, no key); `04` is **one agent** "
                 "that lets the LLM choose between them. At the end, the same server runs over **HTTP**.",
        "scripts": ["03_fastmcp_client.py", "04_fastmcp_agent.py"],
        "files": ["fastmcp_server.py"],
        "extra": [(
            "## 3. The same server over HTTP\n"
            "`stdio` only works when the client starts the server. With `http`, the server is a web service that "
            "any client (another machine, Claude Desktop, VS Code, ...) can reach at `http://127.0.0.1:8000/mcp`. "
            "We start it in the background, connect with the **same** `Client` (only the URL changes), then stop it.",
            "import subprocess, sys, time\n"
            "from fastmcp import Client\n\n"
            "server = subprocess.Popen([sys.executable, \"fastmcp_server.py\", \"http\"],\n"
            "                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            "time.sleep(4)                                    # give the web server a moment to start\n"
            "try:\n"
            "    async with Client(\"http://127.0.0.1:8000/mcp\") as client:\n"
            "        print([t.name for t in await client.list_tools()])\n"
            "        result = await client.call_tool(\"convert_units\", {\"value\": 42, \"from_unit\": \"km\", \"to_unit\": \"mile\"})\n"
            "        print(result.data)\n"
            "finally:\n"
            "    server.terminate()                            # stop the background server")],
    },
}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def split_header(src: str):
    """Leading '#' comment block -> markdown text; the rest -> code."""
    lines = src.splitlines()
    header = []
    while lines and lines[0].startswith("#"):
        line = lines.pop(0)
        header.append(line[2:] if line.startswith("# ") else line.lstrip("#"))
    header = [h for h in header if not re.match(r"\s*(run|view|serve|call|extra install|needs)\b", h)]
    return "\n".join(header).strip(), "\n".join(lines).strip()


def notebook_code(src: str) -> str:
    """Make a script notebook-friendly (top-level await, no __file__)."""
    src = src.replace("asyncio.run(main())", "await main()")
    src = src.replace("Path(__file__).resolve().parent", "Path.cwd()")
    return src


def setup_cells(folder: str, cfg: dict, notebook: str) -> list:
    reqs = [line.split("#")[0].strip() for line in (ROOT / folder / "requirements.txt").read_text().splitlines()]
    reqs = " ".join(f'"{r}"' for r in reqs if r)
    colab = f"https://colab.research.google.com/github/{GITHUB_REPO}/blob/main/{REPO_SUBDIR}/{folder}/{notebook}.ipynb"
    cells = [
        md(f"# {cfg['title']}\n\n"
           f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab})\n\n"
           f"**SA-LIB AI Workshop 2026 — Day 2: AI Agents**\n\n{cfg['intro']}\n\n"
           "Same example in every framework: **(1) first agent → (2) tools → (3) memory → (4) fallback → "
           "(5) multi-agent → complete agent**."),
        md("## 0. Setup\nInstall the packages (takes ~1 minute in Colab)."),
        code(f"%pip install -q {reqs}"),
        md("### API keys\n"
           "* **Colab:** click the 🔑 *Secrets* icon on the left, add e.g. `GROQ_API_KEY`, and enable notebook access.\n"
           "* **VS Code / Jupyter / Linux:** put your keys in the `.env` file in the repo root (copy `.env.example`).\n"
           "* Otherwise you'll be asked to paste a key below.\n\n"
           "Choose the model provider with `LLM_PROVIDER`: `groq` · `gemini` · `openai` · `deepseek` · "
           "`openrouter` · `ollama`."),
        code(
            "import os, getpass\n"
            "from dotenv import load_dotenv, find_dotenv\n"
            "load_dotenv(find_dotenv(usecwd=True))          # VS Code / Linux: read ../.env\n"
            "try:                                           # Colab: read the Secrets panel\n"
            "    from google.colab import userdata\n"
            f"    for key in {KEYS + ['LLM_PROVIDER', 'LLM_MODEL']}:\n"
            "        try:\n"
            "            os.environ[key] = userdata.get(key)\n"
            "        except Exception:\n"
            "            pass\n"
            "except ImportError:\n"
            "    pass\n\n"
            "os.environ.setdefault(\"LLM_PROVIDER\", \"groq\")   # <- change the provider here if you like\n"
            "provider = os.environ[\"LLM_PROVIDER\"]\n"
            "key_name = {\"groq\": \"GROQ_API_KEY\", \"gemini\": \"GOOGLE_API_KEY\", \"openai\": \"OPENAI_API_KEY\",\n"
            "            \"deepseek\": \"DEEPSEEK_API_KEY\", \"openrouter\": \"OPENROUTER_API_KEY\"}.get(provider)\n"
            "if key_name and not os.getenv(key_name):\n"
            "    os.environ[key_name] = getpass.getpass(f\"{key_name}: \")\n"
            "print(\"provider:\", provider)"
        ),
    ]
    if folder == "00_langchain_original":
        cells[-1]["source"] = [line.replace('os.environ.setdefault("LLM_PROVIDER", "groq")',
                                            'os.environ["LLM_PROVIDER"] = "groq"          ')
                               for line in cells[-1]["source"]]
    if cfg.get("common", True):
        common = (ROOT / "shared" / "workshop_common.py").read_text()
        cells += [md("### Shared helper: `workshop_common.py`\nReads the provider settings, and holds the two "
                     "tools every example uses: `web_search` (free DuckDuckGo) and `word_count`."),
                  code("%%writefile workshop_common.py\n" + common)]
    return cells


def build(notebook: str, cfg: dict) -> Path:
    folder = cfg.get("folder", notebook)             # several notebooks can share one folder
    cells = setup_cells(folder, cfg, notebook)
    for name in cfg.get("files", []):
        header, _ = split_header((ROOT / folder / name).read_text())
        cells += [md(f"### helper file: `{name}`\n{header}"),
                  code(f"%%writefile {name}\n" + (ROOT / folder / name).read_text())]
    for i, name in enumerate(cfg["scripts"], 1):
        header, body = split_header((ROOT / folder / name).read_text())
        if name in cfg.get("before", {}):
            text, extra = cfg["before"][name]
            cells += [md(text), code(extra)]
        cells += [md(f"## {i}. `{name}`\n\n{header}"), code(notebook_code(body))]
    for text, extra in cfg.get("extra", []):
        cells += [md(text), code(extra)]
    cells.append(md("---\n*SA-LIB AI Workshop 2026 · based on "
                    "[build-ai-agents-free](https://github.com/Moh4696/build-ai-agents-free) by m0h (MIT)*"))
    for n, cell in enumerate(cells):
        cell["id"] = f"{folder[:2]}-{n:02d}"          # stable ids -> clean git diffs
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                        "name": "python3"},
                                       "language_info": {"name": "python"}, "colab": {"provenance": []}},
          "nbformat": 4, "nbformat_minor": 5}
    out = ROOT / folder / f"{notebook}.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return out


if __name__ == "__main__":
    for notebook, cfg in FOLDERS.items():
        print("built", build(notebook, cfg).relative_to(ROOT))
