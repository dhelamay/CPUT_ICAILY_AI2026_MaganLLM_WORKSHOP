# 00 · LangChain — the original guide

The code from [*How to Build AI Agents Completely Free*](ORIGINAL_GUIDE.md) by m0h, **unchanged**
except for one workshop fix (below). This is the reference: every other folder rebuilds this agent.

- Brain: Groq `llama-3.3-70b-versatile` (free), with fallback to Gemini `gemini-2.5-flash` (free)
- Framework: LangChain 1.x `create_agent` (LangGraph underneath)

```bash
pip install -r 00_langchain_original/requirements.txt
# .env needs GROQ_API_KEY (and GOOGLE_API_KEY for 04 and agent.py)
python 00_langchain_original/01_first_agent.py
python 00_langchain_original/02_agent_with_tools.py
python 00_langchain_original/03_agent_with_memory.py
python 00_langchain_original/04_agent_with_fallback.py
python 00_langchain_original/agent.py
```

Notebook: [`00_langchain_original.ipynb`](00_langchain_original.ipynb)

**Workshop fix:** in `02_agent_with_tools.py` and `agent.py`, the DuckDuckGo tool is wrapped so that a
failed search is *reported to the agent* instead of crashing the program. Free search is flaky: rate
limits, plus a `ddgs` backend that sometimes builds an invalid URL. The region is also set to `us-en`.
Nothing else was changed.

This folder always uses Groq/Gemini, as in the guide. The other folders read `LLM_PROVIDER` from `.env`.
