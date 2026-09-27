# LangGraph 03 — giving your agent memory
# compile the graph with a CHECKPOINTER. after every step LangGraph saves the state.
# a thread_id picks which saved conversation to continue.
# run: python 01_langgraph/03_agent_with_memory.py

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from workshop_common import get_llm_config

cfg = get_llm_config()
model = ChatOpenAI(model=cfg.model, base_url=cfg.base_url, api_key=cfg.api_key, temperature=0)
SYSTEM = {"role": "system", "content": "You are a helpful assistant. Be concise and accurate."}


def chatbot(state: MessagesState):
    return {"messages": [model.invoke([SYSTEM] + state["messages"])]}


graph = StateGraph(MessagesState)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)
agent = graph.compile(checkpointer=InMemorySaver())   # <- the only change: memory

config = {"configurable": {"thread_id": "chat-1"}}      # same id = same conversation

r1 = agent.invoke({"messages": [{"role": "user", "content": "hi! my name is m0h."}]}, config)
print(r1["messages"][-1].content)

r2 = agent.invoke({"messages": [{"role": "user", "content": "what's my name?"}]}, config)
print(r2["messages"][-1].content)

# a different thread_id = a fresh conversation with no memory
r3 = agent.invoke({"messages": [{"role": "user", "content": "what's my name?"}]},
                  {"configurable": {"thread_id": "chat-2"}})
print("[chat-2]", r3["messages"][-1].content)

# bonus: you can read the saved state of any thread
print("messages saved in chat-1:", len(agent.get_state(config).values["messages"]))
