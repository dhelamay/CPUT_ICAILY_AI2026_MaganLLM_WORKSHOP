# 02 — hybrid search RAG  (keywords + meaning)
# vector search understands MEANING ("holiday" ~ "annual leave") but can miss exact tokens
# like product codes ("SB-F3") or numbers. BM25 keyword search is the opposite.
# hybrid = run both, then merge the two rankings with Reciprocal Rank Fusion (RRF).
# run: python 02_hybrid_search_rag.py

from rank_bm25 import BM25Okapi

from rag_common import answer_from_context, build_store, show

store = build_store()
bm25 = BM25Okapi([c["text"].lower().split() for c in store.chunks])


def keyword_search(query: str, k: int = 4) -> list[dict]:
    scores = bm25.get_scores(query.lower().split())
    best = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [store.chunks[i] for i in best]


def hybrid_search(query: str, k: int = 4, rrf_k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion: score = sum over rankings of 1 / (rrf_k + rank)."""
    fused: dict[str, float] = {}
    by_id = {}
    for ranking in (store.search(query, k=10), keyword_search(query, k=10)):
        for rank, chunk in enumerate(ranking):
            fused[chunk["id"]] = fused.get(chunk["id"], 0) + 1 / (rrf_k + rank)
            by_id[chunk["id"]] = chunk
    best = sorted(fused, key=lambda cid: -fused[cid])[:k]
    return [by_id[cid] for cid in best]


QUESTION = "What is the warranty on the SB-F3 pump controller and how much does SB-F3 cost?"

show("vector only", "\n".join(c["id"] for c in store.search(QUESTION, k=4)))
show("keyword (BM25) only", "\n".join(c["id"] for c in keyword_search(QUESTION)))
hits = hybrid_search(QUESTION)
show("hybrid (RRF fused)", "\n".join(c["id"] for c in hits))
show("answer", answer_from_context(QUESTION, hits))
