# SA-LIB AI Workshop 2026 — code

Hands-on code for the two-day workshop. Each day is a self-contained folder with its own README,
`requirements.txt` and `.env.example`.

| day | folder | topic | start here |
|---|---|---|---|
| 1 | [SA_LIB_WORKSHOP_Day1](SA_LIB_WORKSHOP_Day1/) | RAG, from naive RAG to agentic RAG (12 steps) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhelamay/CPUT_ICAILY_AI2026_MaganLLM_WORKSHOP/blob/main/SA_LIB_WORKSHOP_Day1/notebooks/SA_LIB_Day1_RAG_Workshop.ipynb) |
| 2 | [SA_LIB_WORKSHOP_Day2](SA_LIB_WORKSHOP_Day2/) | AI agents: one agent in LangChain, LangGraph, CrewAI, OpenAI Agents SDK, Google ADK, Microsoft Agent Framework, plus MLflow, A2A, monitoring and MCP | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhelamay/CPUT_ICAILY_AI2026_MaganLLM_WORKSHOP/blob/main/SA_LIB_WORKSHOP_Day2/01_langgraph/01_langgraph.ipynb) |

## Quick start

```bash
git clone https://github.com/dhelamay/CPUT_ICAILY_AI2026_MaganLLM_WORKSHOP.git
cd CPUT_ICAILY_AI2026_MaganLLM_WORKSHOP/SA_LIB_WORKSHOP_Day1   # or SA_LIB_WORKSHOP_Day2
cp .env.example .env    # pick LLM_PROVIDER and paste one API key
```

Then follow that folder's README. Every example runs on a hosted LLM API: Groq (default, free tier),
Gemini, OpenAI, DeepSeek or OpenRouter. Ollama is optional.
