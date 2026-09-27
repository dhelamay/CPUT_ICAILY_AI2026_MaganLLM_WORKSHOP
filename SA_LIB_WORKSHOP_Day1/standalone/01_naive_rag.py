# 01 — naive (simple) RAG  (standalone: no rag_common.py, everything is in this file)
# the classic 3 steps:  INDEX  ->  RETRIEVE  ->  GENERATE
#   index:    load files, cut them into chunks, turn each chunk into a vector
#   retrieve: embed the question, find the chunks with the closest vectors
#   generate: give those chunks to the LLM as context and ask it to answer
# run: python standalone/01_naive_rag.py

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


# ============ step 1 — INDEX: load data/ -> cut into chunks -> turn every chunk into a vector ============
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


chunks = load_chunks(chunk_size=120, overlap=20)
vectors = embed([c["text"] for c in chunks])          # our whole "vector database": one numpy array
print(f"[index] {len(chunks)} chunks from {len({c['source'] for c in chunks})} files "
      f"| embeddings: local | llm: {PROVIDER}")


# ============ step 2 — RETRIEVE: the chunks whose vectors are closest to the question's vector ============
def search(query: str, k: int = 4) -> list[dict]:
    """Return the k most similar chunks, each with a 'score' (cosine similarity)."""
    scores = vectors @ embed([query])[0]              # dot product of length-1 vectors = cosine similarity
    return [{**chunks[i], "score": float(scores[i])} for i in np.argsort(-scores)[:k]]


# ============ step 3 — GENERATE: the LLM answers using ONLY the retrieved chunks ============
ANSWER_SYSTEM = ("You answer questions using ONLY the provided context. "
                 "Cite sources like [1], [2]. If the answer is not in the context, say \"I don't know\".")


def answer_from_context(question: str, hits: list[dict]) -> str:
    context = "\n\n".join(f"[{i + 1}] (source: {h['source']})\n{h['text']}" for i, h in enumerate(hits))
    return chat(f"Context:\n{context}\n\nQuestion: {question}", system=ANSWER_SYSTEM)


# ============ the example ============
QUESTION = "How many days of annual leave does a Sahara Sun Energy employee with 7 years of service get?"

hits = search(QUESTION, k=3)
show("retrieved chunks", "\n".join(f"{h['score']:.3f}  {h['id']}" for h in hits))
show("answer WITH retrieval", answer_from_context(QUESTION, hits))
