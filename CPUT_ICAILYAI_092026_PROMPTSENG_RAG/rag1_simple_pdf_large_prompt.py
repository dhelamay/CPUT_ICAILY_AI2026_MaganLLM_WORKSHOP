from rag1_simple_pdf_core import run_rag

DEFAULT_QUESTION = "Review this invoice like an operations and finance analyst."


def build_messages(retrieved_context: str, user_question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a senior operations, procurement, and finance analyst.\n"
                "Your job is to analyze invoice-style documents carefully and explain the result clearly.\n"
                "Use only the retrieved context below. Do not add outside facts.\n\n"
                "When you answer:\n"
                "- Start with a short plain-English summary.\n"
                "- Then provide a bullet list of the key commercial facts.\n"
                "- Then provide a section called 'Checks and Concerns'.\n"
                "- In 'Checks and Concerns', point out OCR issues, unusual numbers, ambiguous wording, or anything that should be manually verified.\n"
                "- If a fact is uncertain, say 'uncertain from context'.\n"
                "- Be concise but precise.\n\n"
                "Retrieved context:\n"
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
        description="Large-instruction Groq PDF RAG demo.",
        default_question=DEFAULT_QUESTION,
        build_messages=build_messages,
    )
