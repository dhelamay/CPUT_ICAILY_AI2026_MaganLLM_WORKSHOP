# 10 — Agentic RAG  (no framework, just the tool-calling loop)  (standalone: no rag_common.py, everything is in this file)
# in every example so far, WE decided the steps. in agentic RAG the LLM decides:
#   which tool to use (company docs? web? calculator?), what to search for,
#   whether the results are enough, and when to stop.
# this is the plan -> act -> observe loop from Day 2 — written by hand, ~40 lines.
# run: python standalone/10_agentic_rag.py

import json
import os
from pathlib import Path

import numpy as np
from ddgs import DDGS
from dotenv import find_dotenv, load_dotenv
from fastembed import TextEmbedding
from openai import OpenAI

load_dotenv(find_dotenv(usecwd=True))   # read LLM_PROVIDER and your API key from the .env file

# ============ setup: the LLM (every provider speaks the OpenAI protocol, so one client works for all) ============
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
client = OpenAI(base_url=BASE_URL, api_key=os.getenv(KEY_ENV) if KEY_ENV else "ollama")
MODEL = os.getenv("LLM_MODEL") or DEFAULT_MODEL


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


# ============ the technique: tools + the plan -> act -> observe loop ============
# --- tools: plain python functions ---
def search_company_docs(query: str) -> str:
    hits = search(query, k=3)
    return "\n\n".join(f"[{i + 1}] (source: {h['source']})\n{h['text']}" for i, h in enumerate(hits))


def web_search(query: str, max_results: int = 3) -> str:
    """Free web search via DuckDuckGo (no key). Returns plain text snippets."""
    try:
        results = DDGS().text(query, max_results=max_results)
        return "\n\n".join(f"{r['title']}: {r['body']} ({r['href']})" for r in results) or "no results"
    except Exception as e:  # free search is rate-limited; never crash the demo because of it
        return f"web search failed: {e}"


def calculator(expression: str) -> str:
    return str(eval(expression, {"__builtins__": {}}, {}))  # demo only: arithmetic like "18500*0.2"


TOOLS = {"search_company_docs": search_company_docs, "web_search": web_search, "calculator": calculator}

# --- tool descriptions the LLM reads to decide WHEN to use each tool ---
TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "search_company_docs",
        "description": "Search Sahara Sun Energy internal documents (products, prices, HR policy, projects, lab).",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the public web for current or general information.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression, e.g. '34000*0.2'.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
]

SYSTEM = ("You are a research assistant for Sahara Sun Energy. Use tools to find facts before answering. "
          "You may call tools several times. Cite sources. Be concise.")


def agent(question: str, max_steps: int = 6) -> str:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    for step in range(max_steps):
        msg = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOL_SPECS).choices[0].message
        if not msg.tool_calls:                       # no tool requested -> this is the final answer
            return msg.content
        messages.append(msg)                          # PLAN: the model asked for tools
        for call in msg.tool_calls:                   # ACT: run each tool
            args = json.loads(call.function.arguments or "{}")
            print(f"step {step + 1}: {call.function.name}({args})")
            try:
                result = TOOLS[call.function.name](**args)
            except Exception as e:
                result = f"tool error: {e}"
            messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})  # OBSERVE
    return "stopped: too many steps"


show("answer", agent(
    "A farmer wants the SunBox Farm kit with the Pay-as-you-save plan. "
    "How much is the deposit, and how much is each monthly instalment?"
))
