# 04 — query transformation RAG  (multi-query + HyDE)
# users ask short, vague questions. we let the LLM improve the QUERY before searching:
#   multi-query: rewrite the question 3 different ways, search with each, merge results
#   HyDE:        let the LLM write a *hypothetical* answer, and search with THAT text
#                (a fake answer often looks more like the real document than the question does)
# run: python 04_query_transformation_rag.py

from rag_common import answer_from_context, build_store, chat, show

store = build_store()
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
        for hit in store.search(q, k=2):
            if hit["id"] not in seen:
                seen.add(hit["id"])
                merged.append(hit)
    return merged[:k]


# --- technique B: HyDE (Hypothetical Document Embeddings) ---
def hyde_search(question: str, k: int = 3) -> list[dict]:
    fake_answer = chat(f"Write a short paragraph from a company HR handbook that answers: {question}")
    show("HyDE hypothetical document", fake_answer)
    return store.search(fake_answer, k=k)


hits = multi_query_search(QUESTION)
show("answer (multi-query)", answer_from_context(QUESTION, hits))

hits = hyde_search(QUESTION)
show("answer (HyDE)", answer_from_context(QUESTION, hits))
