# the complete LangGraph agent (tools + memory), shared by the Phoenix and Langfuse examples.
# monitoring tools don't change your agent — they only watch it.

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

import workshop_common as wc


@tool
def web_search(query: str) -> str:
    """Search the web for up-to-date information."""
    return wc.web_search(query)


@tool
def word_count(text: str) -> int:
    """Count how many words are in a piece of text."""
    return len(text.split())


def build_agent():
    cfg = wc.get_llm_config()
    model = ChatOpenAI(model=cfg.model, base_url=cfg.base_url, api_key=cfg.api_key, temperature=0)
    tools = [web_search, word_count]
    llm = model.bind_tools(tools)

    def agent_node(state: MessagesState):
        return {"messages": [llm.invoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=InMemorySaver())


QUESTIONS = ["my name is m0h. search for the latest langchain version, then tell me how many words your answer is.",
             "what's my name?"]
