"""
Build notebooks/SA_LIB_Day1_RAG_Workshop.ipynb from the .py files (the scripts are the source of truth).

    python tools/build_notebook.py

The notebook is self-contained for Google Colab: it installs packages, loads keys,
and writes rag_common.py + the data/ files with %%writefile. To use YOUR PDF in Colab,
upload it into the data/ folder (file browser on the left) before running the index cells.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GITHUB_REPO = "YOUR_GITHUB_USER/SA_LIB_WORKSHOP_Day1"  # <- change after you push to GitHub, then re-run
KEYS = ["GROQ_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY"]
SCRIPTS = sorted(p.name for p in ROOT.glob("[0-9][0-9]_*.py"))


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def split_header(src: str):
    lines, header = src.splitlines(), []
    while lines and lines[0].startswith("#"):
        line = lines.pop(0)
        header.append(line[2:] if line.startswith("# ") else line.lstrip("#"))
    header = [h for h in header if not h.strip().startswith("run:")]
    return "\n".join(header).strip(), "\n".join(lines).strip()


def build() -> Path:
    reqs = [line.split("#")[0].strip() for line in (ROOT / "requirements.txt").read_text().splitlines()]
    reqs = " ".join(f'"{r}"' for r in reqs if r)
    colab = f"https://colab.research.google.com/github/{GITHUB_REPO}/blob/main/notebooks/SA_LIB_Day1_RAG_Workshop.ipynb"
    cells = [
        md(f"# SA-LIB AI Workshop 2026 — Day 1: from simple RAG to agentic RAG\n\n"
           f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab})\n\n"
           "| # | technique | idea in one line |\n|---|---|---|\n"
           "| 00 | no RAG | the problem: the LLM doesn't know your documents |\n"
           "| 01 | naive RAG | index → retrieve → generate |\n"
           "| 02 | hybrid search | keywords (BM25) + meaning (vectors), fused with RRF |\n"
           "| 03 | re-ranking | a cross-encoder re-scores the top candidates |\n"
           "| 04 | query transformation | multi-query + HyDE rewrite the question before searching |\n"
           "| 05 | conversational RAG | memory + standalone-question rewriting |\n"
           "| 06 | corrective RAG (CRAG) | grade retrieved chunks, fall back to web search |\n"
           "| 07 | self-RAG | the model checks its own answer and retries |\n"
           "| 08 | graph RAG | LLM-built knowledge graph for multi-hop questions |\n"
           "| 09 | adaptive RAG | a router picks: no retrieval / docs / web |\n"
           "| 10 | agentic RAG | the LLM drives a tool loop (no framework) |\n"
           "| 11 | agentic RAG + LangGraph | the same agent as a graph, with memory |"),
        md("## 0. Setup"),
        code(f"%pip install -q {reqs}"),
        md("### API keys and provider\n"
           "* **Colab:** 🔑 *Secrets* panel → add e.g. `GROQ_API_KEY` or `DEEPSEEK_API_KEY` or `OPENAI_API_KEY`.\n"
           "* **VS Code / Linux:** put keys in `.env` (copy `.env.example`).\n\n"
           "`LLM_PROVIDER`: `openai` · `deepseek` · `groq` · `gemini` · `openrouter` · `ollama`  \n"
           "`EMBEDDING_PROVIDER`: `local` (free, on CPU) · `openai` · `gemini` · `ollama`"),
        code(
            "import os, getpass\n"
            "from dotenv import load_dotenv, find_dotenv\n"
            "load_dotenv(find_dotenv(usecwd=True))\n"
            "try:\n"
            "    from google.colab import userdata\n"
            f"    for key in {KEYS + ['LLM_PROVIDER', 'LLM_MODEL', 'EMBEDDING_PROVIDER']}:\n"
            "        try:\n"
            "            os.environ[key] = userdata.get(key)\n"
            "        except Exception:\n"
            "            pass\n"
            "except ImportError:\n"
            "    pass\n\n"
            "os.environ.setdefault(\"LLM_PROVIDER\", \"groq\")        # <- or \"deepseek\", \"openai\", \"gemini\" ...\n"
            "os.environ.setdefault(\"EMBEDDING_PROVIDER\", \"local\")  # free local embeddings\n"
            "provider = os.environ[\"LLM_PROVIDER\"]\n"
            "key_name = {\"groq\": \"GROQ_API_KEY\", \"gemini\": \"GOOGLE_API_KEY\", \"openai\": \"OPENAI_API_KEY\",\n"
            "            \"deepseek\": \"DEEPSEEK_API_KEY\", \"openrouter\": \"OPENROUTER_API_KEY\"}.get(provider)\n"
            "if key_name and not os.getenv(key_name):\n"
            "    os.environ[key_name] = getpass.getpass(f\"{key_name}: \")\n"
            "print(\"llm:\", provider, \"| embeddings:\", os.environ[\"EMBEDDING_PROVIDER\"])"
        ),
        md("### The sample documents (`data/`)\nA fictional solar company, *Sahara Sun Energy*, so the LLM **cannot** "
           "know the answers without RAG. Replace or add your own `.md`, `.txt` or `.pdf` files in `data/`."),
        code("import os\nos.makedirs(\"data\", exist_ok=True)"),
    ]
    for doc in sorted((ROOT / "data").glob("*")):
        if doc.suffix.lower() in {".md", ".txt"}:
            cells.append(code(f"%%writefile data/{doc.name}\n" + doc.read_text()))
    cells += [md("### The shared toolbox: `rag_common.py`\nLLM client for every provider, embeddings, loading "
                 "(md/txt/pdf), chunking, and a tiny vector store written from scratch."),
              code("%%writefile rag_common.py\n" + (ROOT / "rag_common.py").read_text())]
    for name in SCRIPTS:
        header, body = split_header((ROOT / name).read_text())
        cells += [md(f"## {name[:2]}. `{name}`\n\n{header}"), code(body)]
    cells.append(md("---\n*SA-LIB AI Workshop 2026 · Day 1*"))
    for n, cell in enumerate(cells):
        cell["id"] = f"d1-{n:02d}"
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}, "colab": {"provenance": []}},
          "nbformat": 4, "nbformat_minor": 5}
    out = ROOT / "notebooks" / "SA_LIB_Day1_RAG_Workshop.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return out


if __name__ == "__main__":
    print("built", build().relative_to(ROOT))
