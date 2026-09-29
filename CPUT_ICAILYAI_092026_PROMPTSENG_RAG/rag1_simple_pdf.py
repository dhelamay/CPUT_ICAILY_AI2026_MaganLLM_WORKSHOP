from rag1_simple_pdf_core import run_rag

DEFAULT_QUESTION = "What are the main findings in this document?"


def build_messages(retrieved_context: str, user_question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Answer the user's question using only the context below.\n\n"
                f"{retrieved_context}"
            ),
        },
        {
            "role": "user",
            "content": user_question,
        },
    ]


if __name__ == "__main__":
    run_rag(
        description="Basic Groq-only PDF RAG demo.",
        default_question=DEFAULT_QUESTION,
        build_messages=build_messages,
    )
