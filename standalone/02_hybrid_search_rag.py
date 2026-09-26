# 02 — hybrid search RAG  (keywords + meaning)  (standalone: no rag_common.py, everything is in this file)
# vector search understands MEANING ("holiday" ~ "annual leave") but can miss exact tokens
# like product codes ("SB-F3") or numbers. BM25 keyword search is the opposite.
# hybrid = run both, then merge the two rankings with Reciprocal Rank Fusion (RRF).
# run: python standalone/02_hybrid_search_rag.py

import os
from pathlib import Path

import numpy as np
from dotenv import find_dotenv, load_dotenv
from fastembed import TextEmbedding
from openai import OpenAI
from rank_bm25 import BM25Okapi

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


def chat(prompt: str, system: str = "You are a helpful assistant.") -> str:
    """Send one prompt to the LLM and return its text answer."""
    resp = client.chat.completions.create(model=MODEL, temperature=0, messages=[
        {"role": "system", "content": system}, {"role": "user", "content": prompt}])
    return resp.choices[0].message.content.strip()


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
    """VECTOR search: the k chunks whose meaning is closest to the query."""
    scores = vectors @ embed([query])[0]              # dot product of length-1 vectors = cosine similarity
    return [{**chunks[i], "score": float(scores[i])} for i in np.argsort(-scores)[:k]]


ANSWER_SYSTEM = ("You answer questions using ONLY the provided context. "
                 "Cite sources like [1], [2]. If the answer is not in the context, say \"I don't know\".")


def answer_from_context(question: str, hits: list[dict]) -> str:
    context = "\n\n".join(f"[{i + 1}] (source: {h['source']})\n{h['text']}" for i, h in enumerate(hits))
    return chat(f"Context:\n{context}\n\nQuestion: {question}", system=ANSWER_SYSTEM)


# ============ the technique: KEYWORD search (BM25) + fusion ============
bm25 = BM25Okapi([c["text"].lower().split() for c in chunks])


def keyword_search(query: str, k: int = 4) -> list[dict]:
    scores = bm25.get_scores(query.lower().split())
    best = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [chunks[i] for i in best]


def hybrid_search(query: str, k: int = 4, rrf_k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion: score = sum over rankings of 1 / (rrf_k + rank)."""
    fused: dict[str, float] = {}
    by_id = {}
    for ranking in (search(query, k=10), keyword_search(query, k=10)):
        for rank, chunk in enumerate(ranking):
            fused[chunk["id"]] = fused.get(chunk["id"], 0) + 1 / (rrf_k + rank)
            by_id[chunk["id"]] = chunk
    best = sorted(fused, key=lambda cid: -fused[cid])[:k]
    return [by_id[cid] for cid in best]


# ============ the example ============
QUESTION = "What is the warranty on the SB-F3 pump controller and how much does SB-F3 cost?"

show("vector only", "\n".join(c["id"] for c in search(QUESTION, k=4)))
show("keyword (BM25) only", "\n".join(c["id"] for c in keyword_search(QUESTION)))
hits = hybrid_search(QUESTION)
show("hybrid (RRF fused)", "\n".join(c["id"] for c in hits))
show("answer", answer_from_context(QUESTION, hits))
