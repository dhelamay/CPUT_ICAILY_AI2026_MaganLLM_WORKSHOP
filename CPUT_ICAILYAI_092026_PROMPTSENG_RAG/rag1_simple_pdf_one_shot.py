from rag1_simple_pdf_core import run_rag

DEFAULT_QUESTION = "Extract the most important business details from this invoice."


def build_messages(retrieved_context: str, user_question: str) -> list[dict[str, str]]:
    example_context = (
        "Proforma Invoice. Seller: ABC Medical Supplies. Buyer: City Hospital. "
        "Items: 10 beds at $500 each, total $5,000. Payment: 100% at sight."
    )
    example_question = "Extract the most important business details from this invoice."
    example_answer = (
        "Summary:\n"
        "- Document type: Proforma Invoice\n"
        "- Seller: ABC Medical Supplies\n"
        "- Buyer: City Hospital\n"
        "- Main item: 10 beds at $500 each\n"
        "- Total amount: $5,000\n"
        "- Payment term: 100% at sight"
    )

    return [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Use only the provided context and follow the answer style shown in the example."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{example_context}\n\nQuestion:\n{example_question}",
        },
        {
            "role": "assistant",
            "content": example_answer,
        },
        {
            "role": "user",
            "content": f"Context:\n{retrieved_context}\n\nQuestion:\n{user_question}",
        },
    ]


if __name__ == "__main__":
    run_rag(
        description="One-shot Groq PDF RAG demo.",
        default_question=DEFAULT_QUESTION,
        build_messages=build_messages,
    )
