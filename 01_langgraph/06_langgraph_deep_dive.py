# LangGraph 06 — deep dive: a research assistant with a router, a tool loop and human approval
# (explained step by step in docs/langgraph_explained.html)
#
#                       +--> [chat] -----------------------------+
#   START --> [router] -+                                        +--> [human_review] --approved--> [publish] --> END
#                       +--> [researcher] <--> [tools]  ---------+            |
#                                                                             +--rejected--> [researcher] (try again)
#
# what this shows, in one file:
#   1. a CUSTOM STATE with several fields (not just messages) and a reducer
#   2. a ROUTER node that uses structured output to pick a branch (conditional edges)
#   3. a TOOL-CALLING LOOP (researcher <-> tools)
#   4. HUMAN-IN-THE-LOOP with interrupt(): the graph pauses and waits for your approval
#   5. a CHECKPOINTER: the paused graph is saved, and resumed later with Command(resume=...)
#   6. STREAMING: we print each node's update as it happens
#
# run: python 01_langgraph/06_langgraph_deep_dive.py
#      (set AUTO_APPROVE=1 to skip the keyboard prompt, e.g. in CI)

import os
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

import workshop_common as wc

cfg = wc.get_llm_config()
llm = ChatOpenAI(model=cfg.model, base_url=cfg.base_url, api_key=cfg.api_key, temperature=0)


# ---------------------------------------------------------------- 1. STATE
class State(TypedDict, total=False):
    question: str
    route: str                                            # "chat" or "research"
    messages: Annotated[list[AnyMessage], add_messages]   # reducer: APPEND, don't overwrite
    draft: str
    feedback: str                                         # what the human said when rejecting


# ---------------------------------------------------------------- tools
@tool
def web_search(query: str) -> str:
    """Search the web for up-to-date information."""
    return wc.web_search(query)


@tool
def word_count(text: str) -> int:
    """Count how many words are in a piece of text."""
    return len(text.split())


TOOLS = [web_search, word_count]


# ---------------------------------------------------------------- 2. ROUTER (structured output)
class Route(BaseModel):
    route: Literal["chat", "research"] = Field(
        description="'research' if answering needs current / factual info from the web, otherwise 'chat'")


def router(state: State):
    decision = llm.with_structured_output(Route, method="function_calling").invoke(
        f"Classify this question: {state['question']}")
    return {"route": decision.route}


def pick_branch(state: State) -> str:          # a conditional edge is just a function returning a node name
    return state["route"]


# ---------------------------------------------------------------- branch A: plain chat
def chat(state: State):
    return {"draft": llm.invoke(state["question"]).content}


# ---------------------------------------------------------------- branch B: 3. tool-calling loop
RESEARCHER_PROMPT = SystemMessage("You are a careful researcher. Use web_search for facts. Answer in under 80 words.")


def researcher(state: State):
    new = []
    if not state.get("messages"):     # first visit: start the conversation with the question
        new.append(HumanMessage(state["question"]))
    if state.get("feedback"):         # coming back after the human rejected the draft
        new.append(HumanMessage(f"A reviewer rejected your answer: {state['feedback']}. Improve it."))
    history = (state.get("messages") or []) + new
    reply = llm.bind_tools(TOOLS).invoke([RESEARCHER_PROMPT] + history)
    update = {"messages": new + [reply], "feedback": ""}
    if not reply.tool_calls:          # no more tools needed -> this answer is the draft
        update["draft"] = reply.content
    return update


def after_researcher(state: State) -> str:
    return "tools" if tools_condition(state) == "tools" else "human_review"


# ---------------------------------------------------------------- 4. HUMAN-IN-THE-LOOP
def human_review(state: State):
    # interrupt() PAUSES the graph here and returns control to the caller.
    # whatever the caller later sends with Command(resume=...) becomes `decision`.
    decision = interrupt({"draft": state["draft"], "ask": "approve? (yes / or type feedback)"})
    if str(decision).strip().lower() in {"y", "yes", "ok", "approve"}:
        return Command(goto="publish")
    if state["route"] == "chat":
        return Command(goto="chat")
    return Command(goto="researcher", update={"feedback": str(decision)})


def publish(state: State):
    print(f"\n PUBLISHED ({wc.word_count(state['draft'])} words):\n{state['draft']}")
    return {}


# ---------------------------------------------------------------- build the graph
g = StateGraph(State)
g.add_node("router", router)
g.add_node("chat", chat)
g.add_node("researcher", researcher)
g.add_node("tools", ToolNode(TOOLS))
g.add_node("human_review", human_review, destinations=("publish", "researcher", "chat"))
g.add_node("publish", publish)

g.add_edge(START, "router")
g.add_conditional_edges("router", pick_branch, {"chat": "chat", "research": "researcher"})
g.add_edge("chat", "human_review")
g.add_conditional_edges("researcher", after_researcher, ["tools", "human_review"])
g.add_edge("tools", "researcher")
g.add_edge("publish", END)

app = g.compile(checkpointer=InMemorySaver())  # 5. checkpointer = pause / resume / memory


# ---------------------------------------------------------------- 6. run with streaming
def run(question: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    payload = {"question": question}
    while True:
        for update in app.stream(payload, config, stream_mode="updates"):
            for node, change in update.items():
                if node == "__interrupt__":
                    continue
                keys = ", ".join(change.keys()) if isinstance(change, dict) else ""
                print(f"  -> {node:<13} updated: {keys}")
        state = app.get_state(config)
        if not state.next:                     # nothing left to run -> finished
            return
        draft = state.values.get("draft", "")   # we are paused in human_review
        print(f"\n--- DRAFT for review ---\n{draft}\n")
        answer = "yes" if os.getenv("AUTO_APPROVE") else input("approve? (yes / or type feedback): ")
        payload = Command(resume=answer)       # resume the paused graph with the human's answer


if __name__ == "__main__":
    print(app.get_graph().draw_mermaid())
    print("\n=== question 1 (should route to CHAT) ===")
    run("Give me a friendly one-sentence welcome for workshop attendees in Libya.", "t1")
    print("\n=== question 2 (should route to RESEARCH) ===")
    run("What is the latest stable version of LangGraph, and when was it released?", "t2")
