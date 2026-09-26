"""
rag_common.py — the small toolbox every RAG example in this workshop shares.

It gives you five things, nothing more:
  1. chat()            -> talk to ANY LLM provider (openai, deepseek, groq, gemini, openrouter, ollama)
  2. embed()           -> turn text into vectors (local & free by default, or openai / gemini / ollama)
  3. load_documents()  -> read every .md / .txt / .pdf file in the data/ folder
  4. chunk_text()      -> split long text into overlapping chunks
  5. VectorStore       -> a tiny in-memory vector database (numpy + cosine similarity)

Everything is configured from the .env file — see .env.example.
We write the vector store ourselves (≈30 lines) so you can SEE how retrieval works.
In production you would swap it for Chroma, FAISS, Qdrant, pgvector, ...
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

# find_dotenv(usecwd=True) walks up from where you run python, so the .env in the repo root is found
load_dotenv(find_dotenv(usecwd=True))

DATA_DIR = Path(__file__).resolve().parent / "data"

# ---------------------------------------------------------------------------
# 1. LLM providers — all of them speak the OpenAI "chat completions" protocol,
#    so ONE client (the `openai` package) works for every provider.
# ---------------------------------------------------------------------------
PROVIDERS = {
    #  name        base_url                                                     api-key env var       default model
    "openai":     ("https://api.openai.com/v1",                                "OPENAI_API_KEY",     "gpt-4o-mini"),
    "deepseek":   ("https://api.deepseek.com",                                 "DEEPSEEK_API_KEY",   "deepseek-chat"),
    "groq":       ("https://api.groq.com/openai/v1",                           "GROQ_API_KEY",       "llama-3.3-70b-versatile"),
    "gemini":     ("https://generativelanguage.googleapis.com/v1beta/openai/", "GOOGLE_API_KEY",     "gemini-2.5-flash"),
    "openrouter": ("https://openrouter.ai/api/v1",                             "OPENROUTER_API_KEY", "meta-llama/llama-3.3-70b-instruct:free"),
    "ollama":     (os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),  None,                 "qwen2.5:7b"),
}

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()


def get_llm_client(provider: str | None = None) -> tuple[OpenAI, str]:
    """Return (client, model_name) for the chosen provider."""
    provider = (provider or LLM_PROVIDER).lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Choose one of: {', '.join(PROVIDERS)}")
    base_url, key_env, default_model = PROVIDERS[provider]
    api_key = os.getenv(key_env) if key_env else "ollama"  # ollama needs no key
    if not api_key:
        raise RuntimeError(f"{key_env} is missing. Add it to your .env file (see .env.example).")
    model = os.getenv("LLM_MODEL") or default_model
    return OpenAI(base_url=base_url, api_key=api_key), model


_client, MODEL = None, None


def chat(prompt: str, system: str = "You are a helpful assistant.", temperature: float = 0.0,
         history: list[dict] | None = None) -> str:
    """Send one prompt to the LLM and return the text answer."""
    global _client, MODEL
    if _client is None:
        _client, MODEL = get_llm_client()
    messages = [{"role": "system", "content": system}, *(history or []), {"role": "user", "content": prompt}]
    resp = _client.chat.completions.create(model=MODEL, messages=messages, temperature=temperature)
    return resp.choices[0].message.content.strip()


def chat_json(prompt: str, system: str = "Reply with valid JSON only.") -> dict:
    """Ask the LLM for JSON and parse it (tolerates ```json fences and extra words)."""
    text = chat(prompt, system=system)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        return json.loads(match.group(0) if match else text)
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# 2. Embeddings — "local" runs on your CPU for free (fastembed, ~70 MB model, no torch).
# ---------------------------------------------------------------------------
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
EMBEDDING_DEFAULTS = {
    "local": "BAAI/bge-small-en-v1.5",
    "openai": "text-embedding-3-small",
    "gemini": "gemini-embedding-001",
    "ollama": "nomic-embed-text",
}
_embedder = None


def embed(texts: list[str]) -> np.ndarray:
    """Turn a list of texts into a (n, dim) numpy array of unit-length vectors."""
    global _embedder
    model = os.getenv("EMBEDDING_MODEL") or EMBEDDING_DEFAULTS[EMBEDDING_PROVIDER]

    if EMBEDDING_PROVIDER == "local":
        if _embedder is None:
            from fastembed import TextEmbedding  # imported lazily: only needed for local embeddings
            _embedder = TextEmbedding(model_name=model)
        vectors = np.array(list(_embedder.embed(texts)))
    else:
        if _embedder is None:
            _embedder, _ = get_llm_client(EMBEDDING_PROVIDER)  # same OpenAI-compatible client
        vectors = []
        for i in range(0, len(texts), 64):  # send in batches of 64
            resp = _embedder.embeddings.create(model=model, input=texts[i:i + 64])
            vectors.extend(d.embedding for d in resp.data)
        vectors = np.array(vectors)

    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)  # normalise -> dot product = cosine


# ---------------------------------------------------------------------------
# 3. Loading documents — drop any .md, .txt or .pdf file into data/ and it is indexed.
# ---------------------------------------------------------------------------
def load_documents(data_dir: Path | str = DATA_DIR) -> list[dict]:
    """Return a list of {"source": filename, "text": full text}."""
    docs = []
    for path in sorted(Path(data_dir).iterdir()):
        if path.suffix.lower() in {".md", ".txt"}:
            docs.append({"source": path.name, "text": path.read_text(encoding="utf-8")})
        elif path.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            pages = [p.extract_text() or "" for p in PdfReader(path).pages]
            docs.append({"source": path.name, "text": "\n".join(pages)})
    if not docs:
        raise FileNotFoundError(f"No .md/.txt/.pdf files found in {data_dir}")
    return docs


# ---------------------------------------------------------------------------
# 4. Chunking — split by paragraphs, then pack paragraphs into ~chunk_size-word chunks.
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 120, overlap: int = 20) -> list[str]:
    """Split text into chunks of about `chunk_size` words, with `overlap` words shared between chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text.strip()]
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        start += chunk_size - overlap
    return chunks


def load_chunks(chunk_size: int = 120, overlap: int = 20) -> list[dict]:
    """Load every document and chunk it. Each chunk remembers which file it came from."""
    chunks = []
    for doc in load_documents():
        for i, piece in enumerate(chunk_text(doc["text"], chunk_size, overlap)):
            chunks.append({"id": f"{doc['source']}#{i}", "source": doc["source"], "text": piece})
    return chunks


# ---------------------------------------------------------------------------
# 5. A tiny vector store — this IS what Chroma / FAISS do, minus the speed tricks.
# ---------------------------------------------------------------------------
@dataclass
class VectorStore:
    chunks: list[dict] = field(default_factory=list)
    vectors: np.ndarray | None = None

    def add(self, chunks: list[dict]) -> "VectorStore":
        self.chunks += chunks
        new = embed([c["text"] for c in chunks])
        self.vectors = new if self.vectors is None else np.vstack([self.vectors, new])
        return self

    def search(self, query: str, k: int = 4) -> list[dict]:
        """Return the k chunks most similar to the query, each with a 'score'."""
        q = embed([query])[0]
        scores = self.vectors @ q  # cosine similarity, because every vector has length 1
        best = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in best]


def build_store(chunk_size: int = 120, overlap: int = 20) -> VectorStore:
    """Load data/, chunk it, embed it, return a ready-to-search VectorStore."""
    chunks = load_chunks(chunk_size, overlap)
    print(f"[index] {len(chunks)} chunks from {len({c['source'] for c in chunks})} files "
          f"| embeddings: {EMBEDDING_PROVIDER} | llm: {LLM_PROVIDER}")
    return VectorStore().add(chunks)


# ---------------------------------------------------------------------------
# Helpers used by several examples
# ---------------------------------------------------------------------------
def format_context(chunks: list[dict]) -> str:
    """Turn retrieved chunks into a numbered context block the LLM can cite."""
    return "\n\n".join(f"[{i + 1}] (source: {c['source']})\n{c['text']}" for i, c in enumerate(chunks))


ANSWER_SYSTEM = (
    "You answer questions using ONLY the provided context. "
    "Cite sources like [1], [2]. If the answer is not in the context, say \"I don't know\"."
)


def answer_from_context(question: str, chunks: list[dict], history: list[dict] | None = None) -> str:
    """The 'G' in RAG: generate an answer grounded in the retrieved chunks."""
    prompt = f"Context:\n{format_context(chunks)}\n\nQuestion: {question}"
    return chat(prompt, system=ANSWER_SYSTEM, history=history)


def web_search(query: str, max_results: int = 3) -> str:
    """Free web search via DuckDuckGo (no key). Returns plain text snippets."""
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=max_results)
        return "\n\n".join(f"{r['title']}: {r['body']} ({r['href']})" for r in results) or "no results"
    except Exception as e:  # free search is rate-limited; never crash the demo because of it
        return f"web search failed: {e}"


def show(title: str, text: str) -> None:
    """Pretty print a section header + text."""
    print(f"\n{'=' * 70}\n{title}\n{'-' * 70}\n{text}")
