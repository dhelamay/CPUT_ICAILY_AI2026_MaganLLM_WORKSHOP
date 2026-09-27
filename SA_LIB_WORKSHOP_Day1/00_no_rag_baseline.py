# 00 — the problem RAG solves
# ask the LLM about our (fictional) company WITHOUT giving it any documents.
# it has never seen these files, so it will guess, refuse, or hallucinate.
# run: python 00_no_rag_baseline.py

from rag_common import chat, show

QUESTION = "How many days of annual leave does a Sahara Sun Energy employee with 7 years of service get?"

show("question", QUESTION)
show("answer WITHOUT retrieval (the model is guessing)", chat(QUESTION))
