# 06 — Corrective RAG (CRAG)
# naive RAG trusts whatever it retrieves. CRAG adds a GRADER step:
#   1. retrieve chunks
#   2. the LLM grades each chunk: is it relevant to the question? (yes/no)
#   3. keep relevant chunks. if NONE are relevant -> "correct" by searching the web instead
#   4. generate the answer from the corrected context
# run: python 06_corrective_rag.py

from rag_common import answer_from_context, build_store, chat_json, show, web_search

store = build_store()


def grade(question: str, chunk: dict) -> bool:
    verdict = chat_json(
        f"Question: {question}\n\nDocument:\n{chunk['text']}\n\n"
        'Does the document contain information useful for answering the question? '
        'Reply as JSON: {"relevant": "yes"} or {"relevant": "no"}'
    )
    return str(verdict.get("relevant", "no")).lower().startswith("y")


def corrective_rag(question: str) -> str:
    hits = store.search(question, k=4)
    relevant = [h for h in hits if grade(question, h)]
    show(f"grader kept {len(relevant)}/{len(hits)} chunks", "\n".join(h["id"] for h in relevant) or "(none)")
    if not relevant:
        print("-> no relevant internal documents, falling back to WEB SEARCH")
        relevant = [{"source": "web", "text": web_search(question)}]
    return answer_from_context(question, relevant)


# answerable from our documents
show("answer", corrective_rag("What does the DuneGuard coating do?"))
# NOT in our documents -> triggers the web fallback
show("answer", corrective_rag("What is the capital city of Libya and its approximate population?"))
