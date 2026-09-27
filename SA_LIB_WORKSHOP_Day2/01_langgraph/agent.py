# LangGraph — the complete agent (fallback + tools + memory), graph built by hand
# run: python 01_langgraph/agent.py

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

import workshop_common as wc


# --- fallback: try each provider until one answers ---
def get_model():
    for provider in wc.fallback_order():
        try:
            cfg = wc.get_llm_config(provider)
            wc.ping(cfg)
            print(f"using {provider} ({cfg.model})")
            return ChatOpenAI(model=cfg.model, base_url=cfg.base_url, api_key=cfg.api_key, temperature=0)
        except Exception as e:
            print(f"{provider} failed ({str(e)[:80]}); trying the next one")
    raise RuntimeError("every provider failed — check your keys in .env")


# --- tools ---
@tool
def web_search(query: str) -> str:
    """Search the web for up-to-date information."""
    return wc.web_search(query)


@tool
def word_count(text: str) -> int:
    """Count how many words are in a piece of text."""
    return len(text.split())


tools = [web_search, word_count]
model = get_model().bind_tools(tools)
SYSTEM = {"role": "system", "content": "You are a helpful assistant. Use your tools when they help answer "
                                       "accurately, and remember the conversation."}


def agent_node(state: MessagesState):
    return {"messages": [model.invoke([SYSTEM] + state["messages"])]}


# --- graph + memory ---
graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")
agent = graph.compile(checkpointer=InMemorySaver())


def main():
    config = {"configurable": {"thread_id": "chat-1"}}
    r1 = agent.invoke({"messages": [{"role": "user", "content": "my name is m0h. search for the latest "
                      "langchain version, then tell me how many words your answer is."}]}, config)
    print(r1["messages"][-1].content)
    r2 = agent.invoke({"messages": [{"role": "user", "content": "what's my name?"}]}, config)
    print(r2["messages"][-1].content)


if __name__ == "__main__":
    main()
