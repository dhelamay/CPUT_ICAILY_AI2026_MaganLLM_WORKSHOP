# LangGraph 02 — giving your agent tools
# the agent LOOP as a graph. this is exactly what create_agent builds for you:
#
#     START --> [agent] --tool call?--> [tools] --+
#                  ^                               |
#                  +-------------------------------+
#               [agent] --no tool call--> END
#
# run: python 01_langgraph/02_agent_with_tools.py

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

import workshop_common as wc

cfg = wc.get_llm_config()
model = ChatOpenAI(model=cfg.model, base_url=cfg.base_url, api_key=cfg.api_key, temperature=0)


# tool 1 — web search (free, no key)
@tool
def web_search(query: str) -> str:
    """Search the web for up-to-date information."""
    return wc.web_search(query)


# tool 2 — a function you wrote yourself. the DOCSTRING is what the model reads.
@tool
def word_count(text: str) -> int:
    """Count how many words are in a piece of text."""
    return len(text.split())


tools = [web_search, word_count]
model_with_tools = model.bind_tools(tools)  # tells the model which tools exist
SYSTEM = {"role": "system", "content": "You are a helpful assistant. Use your tools when they help answer accurately."}


def agent_node(state: MessagesState):          # PLAN: the model decides — answer, or call tools?
    return {"messages": [model_with_tools.invoke([SYSTEM] + state["messages"])]}


graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))        # ACT: runs the tools the model asked for
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)  # tool calls? -> "tools"  else -> END
graph.add_edge("tools", "agent")                # OBSERVE: tool results go back to the model
agent = graph.compile()

result = agent.invoke({"messages": [{"role": "user", "content":
    "search for the latest langchain version, then tell me how many words your answer is."}]})

# print every step so you can SEE the loop
for msg in result["messages"]:
    calls = getattr(msg, "tool_calls", None)
    print(f"[{msg.type}]", calls if calls else str(msg.content)[:300])
