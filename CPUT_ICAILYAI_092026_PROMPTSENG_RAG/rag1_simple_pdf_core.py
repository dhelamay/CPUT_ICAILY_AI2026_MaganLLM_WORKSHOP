import argparse
import os

import numpy as np
import tiktoken
from dotenv import find_dotenv, load_dotenv
from fastembed import TextEmbedding
from openai import OpenAI
from pypdf import PdfReader

load_dotenv(find_dotenv(usecwd=True))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "qwen/qwen3.8-27b")
EMBED_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
DEFAULT_PDF = "sample_ocr.pdf"
DEFAULT_QUESTION = "What are the main findings in this document?"

if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY. Please check your .env file.")

client = OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)
embedder = TextEmbedding(model_name=EMBED_MODEL)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Read text from a searchable PDF."""
    reader = PdfReader(pdf_path)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text.strip())
    return "\n".join(text_parts)


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Split text into overlapping token chunks."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    step = chunk_size - chunk_overlap
    chunks = []
    for start in range(0, len(tokens), step):
        chunk_tokens = tokens[start : start + chunk_size]
        chunks.append(tokenizer.decode(chunk_tokens))
    return chunks


def get_embedding(text: str) -> list[float]:
    """Create a local embedding for retrieval."""
    return list(next(embedder.embed([text])))


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compare two vectors."""
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def retrieve_best_chunk(chunks: list[str], user_question: str) -> str:
    """Pick the chunk most related to the user's question."""
    chunk_embeddings = [get_embedding(chunk) for chunk in chunks]
    question_embedding = get_embedding(user_question)
    similarities = [cosine_similarity(question_embedding, chunk_embedding) for chunk_embedding in chunk_embeddings]
    best_chunk_index = int(np.argmax(similarities))
    return chunks[best_chunk_index]


def ask_groq(messages: list[dict[str, str]]) -> str:
    """Send prompt messages to Groq."""
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
    )
    return completion.choices[0].message.content or ""


def parse_args(description: str, default_question: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--pdf", default=DEFAULT_PDF, help="Searchable PDF path.")
    parser.add_argument("--question", default=default_question, help="Question to ask about the PDF.")
    return parser.parse_args()


def run_rag(
    description: str,
    default_question: str,
    build_messages,
) -> None:
    args = parse_args(description, default_question)

    print(f"Reading {args.pdf}...")
    raw_pdf_text = extract_text_from_pdf(args.pdf)
    if not raw_pdf_text.strip():
        raise ValueError(
            f"No extractable text found in {args.pdf}. "
            "These simplified versions expect a searchable PDF such as sample_ocr.pdf."
        )

    print("Chunking PDF text...")
    chunks = chunk_text(raw_pdf_text)
    if not chunks:
        raise ValueError(f"No chunks were created from {args.pdf}.")

    print(f"Retrieving context for: {args.question!r}")
    retrieved_context = retrieve_best_chunk(chunks, args.question)

    print(f"Sending the final prompt to Groq via {CHAT_MODEL}...")
    answer = ask_groq(build_messages(retrieved_context, args.question))

    print("\n--- Answer ---")
    print(answer)
