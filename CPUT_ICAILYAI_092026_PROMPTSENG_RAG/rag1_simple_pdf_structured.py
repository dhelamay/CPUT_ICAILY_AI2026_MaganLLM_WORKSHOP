from rag1_simple_pdf_core import run_rag

DEFAULT_QUESTION = "Summarize this invoice in a structured way for a hospital procurement team."


def build_messages(retrieved_context: str, user_question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a careful procurement assistant. "
                "Use only the provided context.\n\n"
                "Return the answer with these headings exactly:\n"
                "1. Document Type\n"
                "2. Seller\n"
                "3. Buyer\n"
                "4. Items and Prices\n"
                "5. Total Amount\n"
                "6. Payment and Shipping Terms\n"
                "7. Risks or Data to Double-Check\n\n"
                f"Context:\n{retrieved_context}"
            ),
        },
        {
            "role": "user",
            "content": user_question,
        },
    ]


if __name__ == "__main__":
    run_rag(
        description="Structured-output Groq PDF RAG demo.",
        default_question=DEFAULT_QUESTION,
        build_messages=build_messages,
    )
