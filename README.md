# SA-LIB AI Workshop 2026 — Day 1: from simple RAG to agentic RAG

Twelve small, runnable Python files that take you from "the LLM doesn't know my documents" to an
**agent that decides by itself** where to look. Each one adds one idea on top of the last.

- **Any LLM, one switch:** OpenAI, DeepSeek, Groq, Gemini, OpenRouter or a local Ollama model. You set
  it with `LLM_PROVIDER` in `.env`.
- **Free embeddings:** they run on your CPU by default (no key needed). OpenAI and Gemini embeddings are
  options.
- **No heavy framework:** the vector store is about 30 lines of numpy, so you can see exactly how
  retrieval works. Only the last example uses LangGraph.
- **Runs anywhere:** Linux / macOS / Windows terminal, **VS Code**, and **Google Colab**. One
  notebook covers everything.

---

## Contents

| # | file | technique | what you learn |
|---|------|-----------|----------------|
| 00 | [`00_no_rag_baseline.py`](00_no_rag_baseline.py) | no RAG | why we need RAG: the model guesses |
| 01 | [`01_naive_rag.py`](01_naive_rag.py) | **naive RAG** | index → retrieve → generate |
| 02 | [`02_hybrid_search_rag.py`](02_hybrid_search_rag.py) | **hybrid search** | BM25 keywords + vectors, merged with Reciprocal Rank Fusion |
| 03 | [`03_reranking_rag.py`](03_reranking_rag.py) | **re-ranking** | a cross-encoder re-scores the top 10 and keeps the best 3 |
| 04 | [`04_query_transformation_rag.py`](04_query_transformation_rag.py) | **query transformation** | multi-query rewriting and HyDE |
| 05 | [`05_conversational_rag.py`](05_conversational_rag.py) | **conversational RAG** | chat memory + rewriting follow-ups into standalone questions |
| 06 | [`06_corrective_rag.py`](06_corrective_rag.py) | **corrective RAG (CRAG)** | an LLM grader checks each chunk and falls back to web search |
| 07 | [`07_self_rag.py`](07_self_rag.py) | **self-RAG** | the model checks grounding and usefulness, then retries |
| 08 | [`08_graph_rag.py`](08_graph_rag.py) | **graph RAG** | LLM-extracted knowledge graph for multi-hop questions |
| 09 | [`09_adaptive_rag.py`](09_adaptive_rag.py) | **adaptive RAG** | a router picks no-retrieval / documents / web |
| 10 | [`10_agentic_rag.py`](10_agentic_rag.py) | **agentic RAG** | the LLM drives a tool loop (docs, web, calculator); no framework |
| 11 | [`11_agentic_rag_langgraph.py`](11_agentic_rag_langgraph.py) | **agentic RAG + LangGraph** | the same agent as a graph, with memory across turns |

Supporting files:

- [`rag_common.py`](rag_common.py): the shared toolbox (LLM client, embeddings, loader, chunker, vector store)
- [`standalone/`](standalone/): the same 12 examples, each in **one complete file** that doesn't use
  `rag_common.py`. Easier to read from top to bottom. Notebook:
  [`notebooks/SA_LIB_Day1_RAG_Standalone.ipynb`](notebooks/SA_LIB_Day1_RAG_Standalone.ipynb)
- [`data/`](data/): sample documents about *Sahara Sun Energy*, a **fictional** Libyan solar company, so
  the LLM can't answer from memory. Replace them with your own files (`.md`, `.txt`, `.pdf`).
- [`notebooks/SA_LIB_Day1_RAG_Workshop.ipynb`](notebooks/SA_LIB_Day1_RAG_Workshop.ipynb): all 12
  examples in one notebook (Colab-ready)

```
                   ┌──────── the RAG ladder ────────┐
 00 no RAG → 01 naive → 02 hybrid → 03 rerank → 04 query rewrite → 05 memory
          → 06 corrective → 07 self-check → 08 graph → 09 router → 10/11 AGENT
          (fixed pipeline, WE decide the steps)        (the LLM decides the steps)
```

---

## 1. Setup (5 minutes)

### Option A: Linux / macOS / Windows (terminal)

```bash
git clone https://github.com/YOUR_GITHUB_USER/SA_LIB_WORKSHOP_Day1.git
cd SA_LIB_WORKSHOP_Day1

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # Windows: copy .env.example .env
# open .env, pick LLM_PROVIDER and paste ONE key
python 01_naive_rag.py
```

### Option B: VS Code

1. **File → Open Folder…** → `SA_LIB_WORKSHOP_Day1`.
2. `Ctrl+Shift+P` → **Python: Create Environment** → *Venv* → tick `requirements.txt`.
3. Copy `.env.example` to `.env` and add your key.
4. Open any `0X_*.py` file and press ▶ (Run Python File). Or open the notebook and choose the `.venv`
   kernel.

### Option C: Google Colab (nothing to install)

1. Open [`notebooks/SA_LIB_Day1_RAG_Workshop.ipynb`](notebooks/SA_LIB_Day1_RAG_Workshop.ipynb) in Colab
   (**File → Upload notebook**, or the *Open in Colab* badge once the repo is on GitHub).
2. Click the 🔑 **Secrets** icon on the left, add e.g. `GROQ_API_KEY`, and switch on *Notebook access*.
3. **Runtime → Run all.** The notebook writes `rag_common.py` and `data/` for you.
4. To use your own PDF, upload it into the `data/` folder in Colab's file browser, then re-run from the
   "index" cells.

---

## 2. Choose your LLM (`.env`)

| `LLM_PROVIDER` | cost | key | get it at | default model |
|---|---|---|---|---|
| `groq` | **free** tier, no card | `GROQ_API_KEY` | https://console.groq.com | `llama-3.3-70b-versatile` |
| `gemini` | **free** tier, no card | `GOOGLE_API_KEY` | https://aistudio.google.com | `gemini-2.5-flash` |
| `openrouter` | **free** `:free` models | `OPENROUTER_API_KEY` | https://openrouter.ai/keys | `meta-llama/llama-3.3-70b-instruct:free` |
| `deepseek` | very cheap (small top-up) | `DEEPSEEK_API_KEY` | https://platform.deepseek.com | `deepseek-chat` |
| `openai` | paid | `OPENAI_API_KEY` | https://platform.openai.com | `gpt-4o-mini` |
| `ollama` | free, local | none | https://ollama.com | `qwen2.5:7b` |

```bash
# .env: example with DeepSeek
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-...
EMBEDDING_PROVIDER=local
```

Want a different model? Set `LLM_MODEL=...` (for example `LLM_MODEL=gpt-4.1-mini`).

**How can one piece of code talk to all of these?** They all speak the *OpenAI chat-completions*
protocol. `rag_common.py` changes only the `base_url` and the key:

```python
OpenAI(base_url="https://api.deepseek.com", api_key=DEEPSEEK_API_KEY).chat.completions.create(...)
```

### Embeddings

| `EMBEDDING_PROVIDER` | notes |
|---|---|
| `local` (default) | free, CPU, `BAAI/bge-small-en-v1.5` via fastembed (~70 MB, downloaded once) |
| `openai` | `text-embedding-3-small`, needs `OPENAI_API_KEY` |
| `gemini` | `gemini-embedding-001`, needs `GOOGLE_API_KEY` |
| `ollama` | `nomic-embed-text` (`ollama pull nomic-embed-text`) |

> DeepSeek and Groq don't offer embeddings. That's fine: keep `EMBEDDING_PROVIDER=local`.

---

## 3. The ideas, one by one

### 01: Naive RAG
**Index** (split files into ~120-word chunks with 20 words of overlap, then embed each chunk) →
**retrieve** (embed the question and take the 3 closest chunks by cosine similarity) →
**generate** (the LLM answers *only* from those chunks and cites `[1]`, `[2]`).

### 02: Hybrid search
Vectors understand *meaning* ("holiday" ≈ "annual leave") but miss exact tokens like `SB-F3`.
BM25 keyword search is the opposite. Run both and merge with **Reciprocal Rank Fusion**:
`score = Σ 1 / (60 + rank)`.

### 03: Re-ranking
Fast vector search picks 10 candidates. A **cross-encoder** reads *(question, chunk)* together, which
is slower but much more accurate, and we keep the top 3.

### 04: Query transformation
- **Multi-query:** the LLM rewrites the question 3 ways, we search with all of them and merge.
- **HyDE:** the LLM writes a *hypothetical* answer and we search with that. A fake answer often looks
  more like the real document than the question does.

### 05: Conversational RAG
"How much does **it** cost?" cannot be searched. The LLM first rewrites it into a standalone question
using the chat history, and then we do normal RAG.

### 06: Corrective RAG (CRAG)
An LLM **grader** labels each retrieved chunk relevant or not. If nothing is relevant, it falls back
to **web search** (DuckDuckGo, free).

### 07: Self-RAG
After answering, the LLM checks (a) *is every fact grounded in the context?* and (b) *does it answer
the question?* If not, it retrieves more and tries again.

### 08: Graph RAG
The LLM extracts `(subject, relation, object)` triples from every chunk into a **networkx** graph. For
a question, we find the entities it mentions and walk 2 hops. That answers multi-hop questions like
*"who leads the lab that invented DuneGuard, and what project does that person lead?"* The triples are
cached in `.cache/`.

### 09: Adaptive RAG
One cheap **router** call decides the path: `none` (just chat), `docs` (our documents) or `web`.

### 10 and 11: Agentic RAG
The LLM gets **tools** (`search_company_docs`, `web_search`, `calculator`) and runs the
**plan → act → observe** loop itself. It can search several times, combine sources and do the maths.
File 10 writes the loop by hand in about 40 lines. File 11 builds the same agent in **LangGraph**, with
memory across turns. This leads into **Day 2: AI Agents**.

---

## 4. Exercises

1. Put your own PDF in `data/` and ask it questions with `01_naive_rag.py`.
2. Change `chunk_size` in `build_store()` to 60 and to 300. How do the answers change?
3. Find a question where **hybrid** (02) beats **vector-only** search. (Hint: product codes.)
4. Ask `06_corrective_rag.py` something *not* in the documents and watch the web fallback.
5. Add a new tool to `10_agentic_rag.py`, for example `convert_currency(amount, rate)`.
6. Switch `LLM_PROVIDER` between `groq`, `deepseek` and `openai` and compare the answers.

---

## 5. Troubleshooting

| problem | fix |
|---|---|
| `GROQ_API_KEY is missing` | create `.env` from `.env.example` in the repo root and add the key |
| `429 rate limit` | free tiers are limited: wait a minute, or switch `LLM_PROVIDER` |
| `web search failed` | DuckDuckGo is free and rate-limited: wait a few seconds and rerun |
| slow first run | the local embedding model (~70 MB) downloads once, then it's cached |
| `json` errors in 06/07/09 | small models sometimes break JSON: use a bigger model (`LLM_MODEL`) |
| Graph RAG answers look stale | delete `.cache/graph_triples.json` after changing `data/` |

**Never commit `.env`.** It's already in `.gitignore`.

---

## 6. Rebuilding the notebook

The `.py` files are the source of truth. After editing them, run:

```bash
python tools/build_notebook.py
```

---

*SA-LIB AI Workshop 2026 · Day 1 of 2 · Day 2 (AI Agents): `SA_LIB_WORKSHOP_Day2`*
