# LangGraph 01 — your first agent
# same task as the original guide, but we build the graph OURSELVES instead of calling create_agent.
# a LangGraph app is:  STATE (the data)  +  NODES (python functions)  +  EDGES (who runs next)
#
#     START --> [chatbot] --> END
#
# run: python 01_langgraph/01_first_agent.py

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

from workshop_common import get_llm_config

# the brain: any provider from your .env (groq by default) — they are all OpenAI-compatible
cfg = get_llm_config()
model = ChatOpenAI(model=cfg.model, base_url=cfg.base_url, api_key=cfg.api_key, temperature=0)

SYSTEM = {"role": "system", "content": "You are a helpful assistant. Be concise and accurate."}


# a NODE is a function: it receives the current state and returns an UPDATE to the state
def chatbot(state: MessagesState):
    reply = model.invoke([SYSTEM] + state["messages"])
    return {"messages": [reply]}  # MessagesState APPENDS new messages (it doesn't overwrite)


# wire the graph
graph = StateGraph(MessagesState)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)
agent = graph.compile()

result = agent.invoke({"messages": [{"role": "user", "content": "explain what an AI agent is in two sentences."}]})
print(f"[{cfg.provider}:{cfg.model}]", result["messages"][-1].content)
