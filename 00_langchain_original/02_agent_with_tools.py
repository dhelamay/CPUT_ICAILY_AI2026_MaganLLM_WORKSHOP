# guide section — "giving your agent tools"
# adds two tools: a prebuilt web search + a function you wrote yourself.
# extra install: python3 -m pip install -U duckduckgo-search langchain-community
# run: python3 src/02_agent_with_tools.py

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

_ddg = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper(region="us-en", time=None))


# tool 1 — web search (prebuilt, free, no key)
# WORKSHOP FIX (the only change vs. the original repo): the free search is flaky — rate limits and
# random backend errors. we wrap it so a failed search is reported to the agent instead of crashing.
@tool
def search(query: str) -> str:
    """Search the web (DuckDuckGo) for up-to-date information."""
    try:
        return _ddg.invoke(query)
    except Exception as e:
        return f"search failed ({e}). try again, or answer from what you know."


# tool 2 — a function you wrote yourself
@tool
def word_count(text: str) -> int:
    """Count how many words are in a piece of text."""
    return len(text.split())


# hand both tools to the agent
agent = create_agent(
    model=model,
    tools=[search, word_count],
    system_prompt="You are a helpful assistant. Use your tools when they help answer accurately.",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "search for the latest langchain version, then tell me how many words your answer is."}]}
)

print(result["messages"][-1].content)
