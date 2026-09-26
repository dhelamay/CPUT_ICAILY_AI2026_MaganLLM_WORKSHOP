# 11 — Agentic RAG with LangGraph
# the same agent as 10_agentic_rag.py, but the loop is a LangGraph GRAPH:
#
#     START -> [agent] --(wants a tool?)--> [tools] --+
#                 ^                                    |
#                 +------------------------------------+
#              [agent] --(no tool call)--> END
#
# LangGraph gives you: a visible graph, memory (checkpointer), streaming, human-in-the-loop.
# run: python 11_agentic_rag_langgraph.py

import os

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from rag_common import LLM_PROVIDER, PROVIDERS, build_store, format_context, show
from rag_common import web_search as _web_search

store = build_store()


@tool
def search_company_docs(query: str) -> str:
    """Search Sahara Sun Energy internal documents (products, prices, HR policy, projects, lab)."""
    return format_context(store.search(query, k=3))


@tool
def web_search(query: str) -> str:
    """Search the public web for current or general information."""
    return _web_search(query)


tools = [search_company_docs, web_search]

# any provider works through ChatOpenAI because they are all OpenAI-compatible
base_url, key_env, default_model = PROVIDERS[LLM_PROVIDER]
llm = ChatOpenAI(model=os.getenv("LLM_MODEL") or default_model, base_url=base_url,
                 api_key=os.getenv(key_env) if key_env else "ollama", temperature=0)
llm_with_tools = llm.bind_tools(tools)


# node 1: the agent (the LLM decides: answer, or call a tool)
def agent_node(state: MessagesState):
    system = {"role": "system", "content": "Use tools to find facts before answering. Cite sources."}
    return {"messages": [llm_with_tools.invoke([system] + state["messages"])]}


# build the graph
graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))       # node 2: runs whichever tools the agent asked for
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)  # tool call? -> "tools", else -> END
graph.add_edge("tools", "agent")               # after tools, go back to the agent (the loop)
app = graph.compile(checkpointer=InMemorySaver())  # checkpointer = memory between turns

print(app.get_graph().draw_mermaid())  # paste into https://mermaid.live to see the graph

config = {"configurable": {"thread_id": "farmer-1"}}
for question in ["What does the Green Wells Program do and who leads it?",
                 "How many farms has it connected so far, out of how many planned?"]:  # follow-up uses memory
    result = app.invoke({"messages": [{"role": "user", "content": question}]}, config)
    show(question, result["messages"][-1].content)
