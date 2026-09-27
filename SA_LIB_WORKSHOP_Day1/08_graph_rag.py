# 08 — Graph RAG  (a knowledge graph built by the LLM)
# vector search finds SIMILAR text. but some questions need CONNECTIONS:
#   "which projects are led by people who work in the lab that invented DuneGuard?"
# graph RAG:
#   1. the LLM reads every chunk and extracts (subject, relation, object) triples
#   2. we store the triples in a graph (networkx)
#   3. for a question, find the entities it mentions and walk their neighbours (2 hops)
#   4. give those facts (+ normal vector hits) to the LLM
# triples are cached in .cache/ so you only pay for extraction once.
# run: python 08_graph_rag.py

import json
from pathlib import Path

import networkx as nx

from rag_common import answer_from_context, build_store, chat, chat_json, show

store = build_store()
CACHE = Path(".cache/graph_triples.json")


# --- 1. extract triples with the LLM ---
def extract_triples(text: str) -> list[list[str]]:
    data = chat_json(
        "Extract factual (subject, relation, object) triples about people, organisations, projects, "
        "places and products from the text. Use short names. "
        'JSON: {"triples": [["subject", "relation", "object"], ...]}\n\nText:\n' + text
    )
    return [t for t in data.get("triples", []) if isinstance(t, list) and len(t) == 3]


if CACHE.exists():
    triples = json.loads(CACHE.read_text())
else:
    triples = []
    for i, chunk in enumerate(store.chunks):
        print(f"extracting triples {i + 1}/{len(store.chunks)} ...")
        triples += extract_triples(chunk["text"])
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(triples, indent=1))

# --- 2. build the graph ---
graph = nx.Graph()
for s, r, o in triples:
    graph.add_edge(str(s).strip(), str(o).strip(), relation=str(r).strip())
print(f"[graph] {graph.number_of_nodes()} entities, {graph.number_of_edges()} relations")


# --- 3. graph retrieval ---
def graph_facts(question: str, hops: int = 2, limit: int = 40) -> list[str]:
    names = chat(f"List the named entities (people, projects, places, products) in: '{question}'. "
                 "Comma-separated, nothing else.")
    wanted = [n.strip().lower() for n in names.split(",") if n.strip()]
    seeds = [node for node in graph.nodes if any(w in node.lower() or node.lower() in w for w in wanted)]
    nearby = set(seeds)
    for seed in seeds:
        nearby |= set(nx.single_source_shortest_path_length(graph, seed, cutoff=hops))
    facts = [f"{u} --{d['relation']}--> {v}" for u, v, d in graph.subgraph(nearby).edges(data=True)]
    return facts[:limit]


QUESTION = "Who leads the lab where the DuneGuard coating was developed, and what project does that person lead?"

facts = graph_facts(QUESTION)
show("graph facts", "\n".join(facts) or "(no matching entities)")
context = [{"source": "knowledge-graph", "text": "\n".join(facts)}] + store.search(QUESTION, k=2)
show("answer (graph + vector)", answer_from_context(QUESTION, context))
