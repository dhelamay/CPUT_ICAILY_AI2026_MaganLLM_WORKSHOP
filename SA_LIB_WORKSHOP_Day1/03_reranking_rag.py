# 03 — RAG with re-ranking
# step 1: retrieve MANY candidates quickly (vector search, k=10)
# step 2: re-score each (question, chunk) pair with a slower but smarter CROSS-ENCODER
# step 3: keep only the top 3 for the LLM
# the cross-encoder reads the question and chunk TOGETHER, so it judges relevance much better.
# runs locally and free (fastembed, ~80 MB model).
# run: python 03_reranking_rag.py

from fastembed.rerank.cross_encoder import TextCrossEncoder

from rag_common import answer_from_context, build_store, show

store = build_store()
reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")

QUESTION = "Who built the model that predicts solar output, and how accurate is it?"

candidates = store.search(QUESTION, k=10)
scores = list(reranker.rerank(QUESTION, [c["text"] for c in candidates]))
reranked = [c for _, c in sorted(zip(scores, candidates), key=lambda p: -p[0])][:3]

show("before re-ranking (vector order, top 3)", "\n".join(c["id"] for c in candidates[:3]))
show("after re-ranking (cross-encoder, top 3)", "\n".join(c["id"] for c in reranked))
show("answer", answer_from_context(QUESTION, reranked))
