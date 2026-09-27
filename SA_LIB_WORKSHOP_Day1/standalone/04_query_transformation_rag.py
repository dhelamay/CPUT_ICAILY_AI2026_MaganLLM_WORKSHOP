# 04 — query transformation RAG  (multi-query + HyDE)  (standalone: no rag_common.py, everything is in this file)
# users ask short, vague questions. we let the LLM improve the QUERY before searching:
#   multi-query: rewrite the question 3 different ways, search with each, merge results
#   HyDE:        let the LLM write a *hypothetical* answer, and search with THAT text
#                (a fake answer often looks more like the real document than the question does)
# run: python standalone/04_query_transformation_rag.py

import os
from pathlib import Path

import numpy as np
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
    """Return the k most similar chunks, each with a 'score' (cosine similarity)."""
    scores = vectors @ embed([query])[0]              # dot product of length-1 vectors = cosine similarity
    return [{**chunks[i], "score": float(scores[i])} for i in np.argsort(-scores)[:k]]


ANSWER_SYSTEM = ("You answer questions using ONLY the provided context. "
                 "Cite sources like [1], [2]. If the answer is not in the context, say \"I don't know\".")


def answer_from_context(question: str, hits: list[dict]) -> str:
    context = "\n\n".join(f"[{i + 1}] (source: {h['source']})\n{h['text']}" for i, h in enumerate(hits))
    return chat(f"Context:\n{context}\n\nQuestion: {question}", system=ANSWER_SYSTEM)


# ============ the technique ============
QUESTION = "can I work from home?"


# --- technique A: multi-query ---
def multi_query_search(question: str, k: int = 4) -> list[dict]:
    rewrites = chat(
        f"Write 3 different search queries that would help answer: '{question}'. "
        "One per line, no numbering, no extra text."
    ).splitlines()
    queries = [question] + [q.strip("-• ").strip() for q in rewrites if q.strip()][:3]
    show("multi-query rewrites", "\n".join(queries))
    seen, merged = set(), []
    for q in queries:
        for hit in search(q, k=2):
            if hit["id"] not in seen:
                seen.add(hit["id"])
                merged.append(hit)
    return merged[:k]


# --- technique B: HyDE (Hypothetical Document Embeddings) ---
def hyde_search(question: str, k: int = 3) -> list[dict]:
    fake_answer = chat(f"Write a short paragraph from a company HR handbook that answers: {question}")
    show("HyDE hypothetical document", fake_answer)
    return search(fake_answer, k=k)


hits = multi_query_search(QUESTION)
show("answer (multi-query)", answer_from_context(QUESTION, hits))

hits = hyde_search(QUESTION)
show("answer (HyDE)", answer_from_context(QUESTION, hits))
