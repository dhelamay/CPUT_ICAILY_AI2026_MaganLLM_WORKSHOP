# 09 — Adaptive RAG  (a router picks the strategy)
# not every question needs the same pipeline:
#   "hi, how are you?"                        -> no retrieval, just answer
#   "what's our sick-leave policy?"           -> search our documents
#   "who won the last football world cup?"    -> search the web
# a ROUTER (one small LLM call) classifies the question first, then we run the right path.
# run: python 09_adaptive_rag.py

from rag_common import answer_from_context, build_store, chat, chat_json, show, web_search

store = build_store()

ROUTER_PROMPT = """Route the user question to one of:
- "none":  greetings, chit-chat, general knowledge the model surely knows
- "docs":  anything about Sahara Sun Energy (products, prices, HR, projects, lab) or the SA-LIB workshop
- "web":   recent events or public facts not about the company
Question: {q}
JSON: {{"route": "none" | "docs" | "web"}}"""


def adaptive_rag(question: str) -> str:
    route = chat_json(ROUTER_PROMPT.format(q=question)).get("route", "docs")
    print(f"router -> {route}")
    if route == "none":
        return chat(question)
    if route == "web":
        return answer_from_context(question, [{"source": "web", "text": web_search(question)}])
    return answer_from_context(question, store.search(question, k=3))


for q in ["Hello! What can you help me with?",
          "How many days of sick leave do employees get?",
          "What is the latest stable version of Python?"]:
    show(q, adaptive_rag(q))
