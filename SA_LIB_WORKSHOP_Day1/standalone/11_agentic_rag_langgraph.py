# 11 — Agentic RAG with LangGraph  (standalone: no rag_common.py, everything is in this file)
# the same agent as 10_agentic_rag.py, but the loop is a LangGraph GRAPH:
#
#     START -> [agent] --(wants a tool?)--> [tools] --+
#                 ^                                    |
#                 +------------------------------------+
#              [agent] --(no tool call)--> END
#
# LangGraph gives you: a visible graph, memory (checkpointer), streaming, human-in-the-loop.
# run: python standalone/11_agentic_rag_langgraph.py

import os
from pathlib import Path

import numpy as np
from ddgs import DDGS
from dotenv import find_dotenv, load_dotenv
from fastembed import TextEmbedding
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv(find_dotenv(usecwd=True))   # read LLM_PROVIDER and your API key from the .env file

# ============ setup: the LLM (every provider speaks the OpenAI protocol, so ChatOpenAI works for all) ============
PROVIDERS = {  # name: (base_url, api-key env var, default model)
    "openai":     ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o-mini"),
    "deepseek":   ("https://api.deepseek.com", "DEEPSEEK_API_KEY", "deepseek-chat"),
    "groq":       ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "gemini":     ("https://generativelanguage.googleapis.com/v1beta/openai/", "GOOGLE_API_KEY", "gemini-2.5-flash"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "meta-llama/llama-3.3-70b-instruct:free"),
    "ollama":     (os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"), None, "qwen2.5:7b"),
}
PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
BASE_URL, KEY_ENV, DEFAULT_MODEL = PROVIDERS[PROVIDER]
if KEY_ENV and not os.getenv(KEY_ENV):
    raise SystemExit(f"{KEY_ENV} is missing. Add it to your .env file (see .env.example).")
llm = ChatOpenAI(model=os.getenv("LLM_MODEL") or DEFAULT_MODEL, base_url=BASE_URL,
                 api_key=os.getenv(KEY_ENV) if KEY_ENV else "ollama", temperature=0)


def show(title: str, text: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'-' * 70}\n{text}")


# ============ setup: INDEX = load data/ -> cut into chunks -> turn every chunk into a vector ============
DATA_DIR = Path(__file__).resolve().parent.parent / "data"   # the same data/ folder as the main examples


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 20) -> list[str]:
    """Split text into chunks of ~chunk_size words; neighbouring chunks share `overlap` words."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text.strip()]
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]


def load_chunks(chunk_size: int = 120, overlap: int = 20) -> list[dict]:
    """Read every .md / .txt / .pdf file in data/ and chunk it. Each chunk remembers its file."""
    chunks = []
    for path in sorted(DATA_DIR.iterdir()):
        if path.suffix.lower() in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8")
        elif path.suffix.lower() == ".pdf":
            from pypdf import PdfReader   # only needed when you put PDF files in data/
            text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        else:
            continue
        for i, piece in enumerate(chunk_text(text, chunk_size, overlap)):
            chunks.append({"id": f"{path.name}#{i}", "source": path.name, "text": piece})
    return chunks


embedder = TextEmbedding("BAAI/bge-small-en-v1.5")   # free, runs on your CPU (small download the first time)


def embed(texts: list[str]) -> np.ndarray:
    """Texts -> a (n, dim) array of vectors, each scaled to length 1."""
    vectors = np.array(list(embedder.embed(texts)))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


chunks = load_chunks()
vectors = embed([c["text"] for c in chunks])          # our whole "vector database": one numpy array
print(f"[index] {len(chunks)} chunks from {len({c['source'] for c in chunks})} files "
      f"| embeddings: local | llm: {PROVIDER}")


def search(query: str, k: int = 4) -> list[dict]:
    """Return the k most similar chunks, each with a 'score' (cosine similarity)."""
    scores = vectors @ embed([query])[0]              # dot product of length-1 vectors = cosine similarity
    return [{**chunks[i], "score": float(scores[i])} for i in np.argsort(-scores)[:k]]


# ============ the technique: tools + a LangGraph loop ============
@tool
def search_company_docs(query: str) -> str:
    """Search Sahara Sun Energy internal documents (products, prices, HR policy, projects, lab)."""
    hits = search(query, k=3)
    return "\n\n".join(f"[{i + 1}] (source: {h['source']})\n{h['text']}" for i, h in enumerate(hits))


@tool
def web_search(query: str) -> str:
    """Search the public web for current or general information."""
    try:  # free DuckDuckGo search (no key); it is rate-limited, so never crash the demo because of it
        results = DDGS().text(query, max_results=3)
        return "\n\n".join(f"{r['title']}: {r['body']} ({r['href']})" for r in results) or "no results"
    except Exception as e:
        return f"web search failed: {e}"


tools = [search_company_docs, web_search]
llm_with_tools = llm.bind_tools(tools)


# node 1: the agent (the LLM decides: answer, or call a tool)
def agent_node(state: MessagesState):
    system = {"role": "system", "content": "Use tools to find facts before answering. Cite sources."}
    return {"messages": [llm_with_tools.invoke([system] + state["messages"])]}


# build the graph
graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))       # node 2: runs whichever tools the agent asked for
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)  # tool call? -> "tools", else -> END
graph.add_edge("tools", "agent")               # after tools, go back to the agent (the loop)
app = graph.compile(checkpointer=InMemorySaver())  # checkpointer = memory between turns

print(app.get_graph().draw_mermaid())  # paste into https://mermaid.live to see the graph

config = {"configurable": {"thread_id": "farmer-1"}}
for question in ["What does the Green Wells Program do and who leads it?",
                 "How many farms has it connected so far, out of how many planned?"]:  # follow-up uses memory
    result = app.invoke({"messages": [{"role": "user", "content": question}]}, config)
    show(question, result["messages"][-1].content)
