# 01 — naive (simple) RAG
# the classic 3 steps:  INDEX  ->  RETRIEVE  ->  GENERATE
#   index:    load files, cut them into chunks, turn each chunk into a vector
#   retrieve: embed the question, find the chunks with the closest vectors
#   generate: give those chunks to the LLM as context and ask it to answer
# run: python 01_naive_rag.py

from rag_common import answer_from_context, build_store, show

# step 1 — INDEX (done once; in real apps you save it to disk / a vector DB)
store = build_store(chunk_size=120, overlap=20)

QUESTION = "How many days of annual leave does a Sahara Sun Energy employee with 7 years of service get?"

# step 2 — RETRIEVE the 3 most similar chunks
hits = store.search(QUESTION, k=3)
show("retrieved chunks", "\n".join(f"{h['score']:.3f}  {h['id']}" for h in hits))

# step 3 — GENERATE an answer grounded in those chunks
show("answer WITH retrieval", answer_from_context(QUESTION, hits))
