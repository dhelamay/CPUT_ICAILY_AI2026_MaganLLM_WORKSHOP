# LangGraph 04 — the fallback setup
# try each provider in order (LLM_PROVIDER, then LLM_FALLBACKS in .env). first one that answers wins.
# run: python 01_langgraph/04_agent_with_fallback.py

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

from workshop_common import fallback_order, get_llm_config, ping


def get_model():
    for provider in fallback_order():           # e.g. ["groq", "gemini"]
        try:
            cfg = get_llm_config(provider)
            ping(cfg)                           # quick test that it actually responds
            print(f"using {provider} ({cfg.model})")
            return ChatOpenAI(model=cfg.model, base_url=cfg.base_url, api_key=cfg.api_key, temperature=0)
        except Exception as e:
            print(f"{provider} failed ({str(e)[:80]}); trying the next one")
    raise RuntimeError("every provider failed — check your keys in .env")


model = get_model()


def chatbot(state: MessagesState):
    return {"messages": [model.invoke(state["messages"])]}


graph = StateGraph(MessagesState)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)
agent = graph.compile()

result = agent.invoke({"messages": [{"role": "user", "content": "say hi and tell me which model you are."}]})
print(result["messages"][-1].content)
