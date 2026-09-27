# MLflow 01 — see INSIDE your agent with tracing
# MLflow is not an agent framework — it's the lab notebook for ANY framework:
#   tracing (what happened, step by step), evaluation (is it good?), packaging/serving (ship it).
# one line — mlflow.langchain.autolog() — records every LLM call, tool call and graph node.
#
# run:   python 06_mlflow/01_trace_langgraph_agent.py
# view:  mlflow ui --backend-store-uri sqlite:///mlflow.db     ->  open http://localhost:5000  -> "Traces" tab
#
# other frameworks: mlflow.openai.autolog() (also covers the OpenAI Agents SDK), mlflow.crewai.autolog(),
#                   mlflow.gemini.autolog(), mlflow.autogen.autolog(), mlflow.litellm.autolog() ...

import mlflow
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

import workshop_common as wc

mlflow.set_tracking_uri("sqlite:///mlflow.db")      # a local file — no server needed
mlflow.set_experiment("SA-LIB Day2 agents")
mlflow.langchain.autolog()                          # <- that's it: everything below is traced

# ---- the same complete LangGraph agent as 01_langgraph/agent.py ----
cfg = wc.get_llm_config()
model = ChatOpenAI(model=cfg.model, base_url=cfg.base_url, api_key=cfg.api_key, temperature=0)


@tool
def web_search(query: str) -> str:
    """Search the web for up-to-date information."""
    return wc.web_search(query)


@tool
def word_count(text: str) -> int:
    """Count how many words are in a piece of text."""
    return len(text.split())


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
agent = graph.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "chat-1"}}
for question in ["my name is m0h. search for the latest langchain version, then tell me how many words your answer is.",
                 "what's my name?"]:
    result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config)
    print(result["messages"][-1].content)

# ---- check that the traces were recorded ----
mlflow.flush_trace_async_logging()
traces = mlflow.search_traces(max_results=2)
print(f"\nrecorded {len(traces)} trace(s). latest trace spans:")
for span in mlflow.get_trace(traces.iloc[0].trace_id).data.spans:
    print(f"  - {span.name} ({span.span_type})")
print("\nnow run:  mlflow ui --backend-store-uri sqlite:///mlflow.db   and open http://localhost:5000")
