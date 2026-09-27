# 07 — Self-RAG  (the model checks its own answer)
# after generating, we ask the LLM two questions about its own output:
#   a) is the answer GROUNDED in the retrieved context? (no hallucination)
#   b) does it actually ANSWER the question?
# if either check fails, we retrieve more context and try again (max 3 attempts).
# run: python 07_self_rag.py

from rag_common import answer_from_context, build_store, chat_json, show

store = build_store()


def is_grounded(answer: str, chunks: list[dict]) -> bool:
    context = "\n".join(c["text"] for c in chunks)
    v = chat_json(f"Context:\n{context}\n\nAnswer:\n{answer}\n\n"
                  'Is every fact in the answer supported by the context? JSON: {"grounded": "yes" or "no"}')
    return str(v.get("grounded", "no")).lower().startswith("y")


def is_useful(question: str, answer: str) -> bool:
    v = chat_json(f"Question: {question}\nAnswer: {answer}\n\n"
                  'Does the answer resolve the question? JSON: {"useful": "yes" or "no"}')
    return str(v.get("useful", "no")).lower().startswith("y")


def self_rag(question: str, max_attempts: int = 3) -> str:
    k = 2
    for attempt in range(1, max_attempts + 1):
        hits = store.search(question, k=k)
        answer = answer_from_context(question, hits)
        grounded, useful = is_grounded(answer, hits), is_useful(question, answer)
        print(f"attempt {attempt}: k={k} grounded={grounded} useful={useful}")
        if grounded and useful:
            return answer
        k += 2  # not good enough -> widen the search and try again
    return answer + "\n(warning: could not fully verify this answer)"


show("answer", self_rag("Which project does Khaled Senussi lead, and how many farms were connected by end of 2025?"))
