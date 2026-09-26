# 05 — conversational RAG  (RAG with memory)
# follow-up questions like "and how much does IT cost?" make no sense on their own.
# before searching, we ask the LLM to rewrite the follow-up into a STANDALONE question
# using the chat history. then we do normal RAG, and keep the history for the next turn.
# run: python 05_conversational_rag.py

from rag_common import answer_from_context, build_store, chat, show

store = build_store()
history: list[dict] = []  # the conversation memory: a plain list of messages


def ask(question: str) -> str:
    standalone = question
    if history:
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
        standalone = chat(
            f"Chat history:\n{transcript}\n\nFollow-up question: {question}\n\n"
            "Rewrite the follow-up as a standalone question. Reply with the question only."
        )
    hits = store.search(standalone, k=3)
    answer = answer_from_context(standalone, hits)
    history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
    show(f"user: {question}\n(standalone: {standalone})", answer)
    return answer


ask("What kit do you sell for farms?")
ask("How much does it cost?")                  # "it" = SunBox Farm
ask("And what does the maintenance plan for it cost per year?")
