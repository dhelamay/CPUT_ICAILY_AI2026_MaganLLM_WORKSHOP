# LangGraph 05 — multi-agent: researcher -> writer -> reviewer (with a feedback loop)
# each "agent" is a node. they share one STATE object (a typed dict) instead of just messages.
#
#     START -> [researcher] -> [writer] -> [reviewer] --too long?--> [writer]  (loop, max 3 drafts)
#                                                     --ok---------> END
#
# run: python 01_langgraph/05_multi_agent.py

from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from workshop_common import get_llm_config, web_search, word_count

cfg = get_llm_config()
model = ChatOpenAI(model=cfg.model, base_url=cfg.base_url, api_key=cfg.api_key, temperature=0)
MAX_WORDS = 60


class TeamState(TypedDict, total=False):   # the shared whiteboard all agents read & write
    topic: str
    notes: str
    draft: str
    words: int
    attempts: int


def researcher(state: TeamState):
    results = web_search(state["topic"])
    notes = model.invoke(f"Summarise these search results as 5 short bullet facts:\n{results}").content
    print(f"[researcher] collected notes ({word_count(notes)} words)")
    return {"notes": notes}


def writer(state: TeamState):
    feedback = ""
    if state.get("draft"):
        feedback = f"\nYour previous draft had {state['words']} words — too long. Make it shorter."
    draft = model.invoke(
        f"Write ONE paragraph of at most {MAX_WORDS} words about '{state['topic']}' "
        f"for workshop attendees, using these notes:\n{state['notes']}{feedback}"
    ).content
    attempts = state.get("attempts", 0) + 1
    print(f"[writer] draft #{attempts}")
    return {"draft": draft, "attempts": attempts}


def reviewer(state: TeamState):
    words = word_count(state["draft"])     # plain python can be an "agent" step too
    print(f"[reviewer] {words} words")
    return {"words": words}


def route_after_review(state: TeamState) -> str:
    if state["words"] > MAX_WORDS and state["attempts"] < 3:
        return "writer"                     # send it back
    return END


graph = StateGraph(TeamState)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_node("reviewer", reviewer)
graph.add_edge(START, "researcher")
graph.add_edge("researcher", "writer")
graph.add_edge("writer", "reviewer")
graph.add_conditional_edges("reviewer", route_after_review, ["writer", END])
team = graph.compile()

print(team.get_graph().draw_mermaid())      # paste into https://mermaid.live to see the picture
final = team.invoke({"topic": "the latest langchain version"})
print("\nFINAL:\n" + final["draft"])
