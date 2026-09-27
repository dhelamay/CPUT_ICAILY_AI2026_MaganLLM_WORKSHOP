# Standalone versions: one complete file per technique, no `rag_common.py`

These are the **same 12 examples** as the main folder, but every file contains **everything it needs**.
They don't import `rag_common.py`. Open any file and you can read the whole program, from loading the
documents to the final answer, without jumping to another file.

Each file has the same three parts:

| part | what it does |
|---|---|
| `setup: the LLM` | picks the provider from `.env` (`LLM_PROVIDER`) and defines `chat()` |
| `setup: INDEX` | loads `../data/`, cuts it into chunks, embeds them (fastembed, free, on your CPU), and defines `search()` |
| `the technique` | the part that is different in every file: hybrid search, re-ranking, CRAG, graph, agent, ... |

The setup parts repeat in every file on purpose. Once you have read them in `01_naive_rag.py`, skip to
**the technique** in the other files.

```bash
pip install -r requirements.txt          # same packages as the main examples
python standalone/01_naive_rag.py        # run from the repo root (or from inside standalone/)
```

- **Notebook (Colab-ready):** [`../notebooks/SA_LIB_Day1_RAG_Standalone.ipynb`](../notebooks/SA_LIB_Day1_RAG_Standalone.ipynb).
  Each example is one complete code cell. Generated with `python tools/build_notebook.py`.
- They use the same `.env` file and the same `data/` folder as the main examples. Add your own files
  (`.md`, `.txt`, `.pdf`) to `data/` and every version picks them up.
- **Embeddings are always local** (`BAAI/bge-small-en-v1.5`) to keep the files short. If you want
  OpenAI or Gemini embeddings, use the main examples (`EMBEDDING_PROVIDER` in `.env`).
- With the same LLM and embeddings, each standalone file gives the same output as its twin in the main folder.

| # | file | technique |
|---|---|---|
| 00 | [`00_no_rag_baseline.py`](00_no_rag_baseline.py) | no RAG (why we need it) |
| 01 | [`01_naive_rag.py`](01_naive_rag.py) | naive RAG: index → retrieve → generate |
| 02 | [`02_hybrid_search_rag.py`](02_hybrid_search_rag.py) | hybrid search (BM25 + vectors, RRF) |
| 03 | [`03_reranking_rag.py`](03_reranking_rag.py) | re-ranking with a cross-encoder |
| 04 | [`04_query_transformation_rag.py`](04_query_transformation_rag.py) | multi-query and HyDE |
| 05 | [`05_conversational_rag.py`](05_conversational_rag.py) | conversational RAG (memory) |
| 06 | [`06_corrective_rag.py`](06_corrective_rag.py) | corrective RAG (grader + web fallback) |
| 07 | [`07_self_rag.py`](07_self_rag.py) | self-RAG (check the answer, retry) |
| 08 | [`08_graph_rag.py`](08_graph_rag.py) | graph RAG (knowledge graph) |
| 09 | [`09_adaptive_rag.py`](09_adaptive_rag.py) | adaptive RAG (router) |
| 10 | [`10_agentic_rag.py`](10_agentic_rag.py) | agentic RAG (tool loop, no framework) |
| 11 | [`11_agentic_rag_langgraph.py`](11_agentic_rag_langgraph.py) | agentic RAG with LangGraph |

---

*SA-LIB AI Workshop 2026 · Dr. Nasser Mooman, Magan AI Inc. · nmooman@gmail.com · written with help from Claude. Teaching code, not for production use.*
