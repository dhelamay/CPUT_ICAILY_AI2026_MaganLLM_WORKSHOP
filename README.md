# SA-LIB AI Workshop 2026 — code

**Author:** Dr. Nasser Mooman · Magan AI Inc. · [nmooman@gmail.com](mailto:nmooman@gmail.com)  
Written with help from Claude (Anthropic).

Hands-on code for the two-day workshop. Each day is a self-contained folder with its own README,
`requirements.txt` and `.env.example`.

| day | folder | topic | start here |
|---|---|---|---|
| 1 | [SA_LIB_WORKSHOP_Day1](SA_LIB_WORKSHOP_Day1/) | RAG, from naive RAG to agentic RAG (12 steps) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhelamay/CPUT_ICAILY_AI2026_MaganLLM_WORKSHOP/blob/main/SA_LIB_WORKSHOP_Day1/notebooks/SA_LIB_Day1_RAG_Workshop.ipynb) |
| 2 | [SA_LIB_WORKSHOP_Day2](SA_LIB_WORKSHOP_Day2/) | AI agents: one agent in LangChain, LangGraph, CrewAI, OpenAI Agents SDK, Google ADK, Microsoft Agent Framework, plus MLflow, A2A, monitoring and MCP | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhelamay/CPUT_ICAILY_AI2026_MaganLLM_WORKSHOP/blob/main/SA_LIB_WORKSHOP_Day2/01_langgraph/01_langgraph.ipynb) |
| — | [CPUT_ICAILYAI_092026_PROMPTSENG_RAG](CPUT_ICAILYAI_092026_PROMPTSENG_RAG/) | prompt engineering on a PDF: simple, structured, one-shot and large-prompt RAG with Groq | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhelamay/CPUT_ICAILY_AI2026_MaganLLM_WORKSHOP/blob/main/CPUT_ICAILYAI_092026_PROMPTSENG_RAG/rag1_simple_pdf_colab.ipynb) · [workshop page](https://dhelamay.github.io/CPUT_ICAILY_AI2026_MaganLLM_WORKSHOP/CPUT_ICAILYAI_092026_PROMPTSENG_RAG/groq_pdf_prompt_workshop/) · [prompt engineering guide](https://dhelamay.github.io/CPUT_ICAILY_AI2026_MaganLLM_WORKSHOP/CPUT_ICAILYAI_092026_PROMPTSENG_RAG/prompt_engineering_guide.html) |
| — | [RAG_Cheat_Sheet](RAG_Cheat_Sheet/) | one-page cheat sheet: 20 RAG techniques explained simply (PDF, PNG, HTML) | [open the PDF](<RAG_Cheat_Sheet/RAG Cheat Sheet_onepage.pdf>) |

> [!WARNING]
> **Teaching code — do not use in production.** These are sample programs for a workshop. They are kept
> short on purpose and leave out what a real system needs: authentication and access control, input
> validation, protection against prompt injection, secret management, error handling and retries,
> rate limiting, logging, tests and evaluation. LLM answers can be wrong, so check anything important.
> When you use a hosted API, you are responsible for your keys, for what data you send, for its costs
> and for following its terms. The code is provided "as is", without warranty (see `LICENSE`).

## Bring your own LLM API key

**Participants need to bring their own LLM API key.** The workshop does not provide API keys or GPUs.
Create one key **before the workshop** and check that it works. One key from any of these is enough:

| provider | cost | get a key |
|---|---|---|
| Groq *(default)* | free tier, no card | https://console.groq.com |
| Google Gemini | free tier, no card | https://aistudio.google.com |
| OpenRouter | free `:free` models | https://openrouter.ai/keys |
| DeepSeek | very cheap (small top-up) | https://platform.deepseek.com |
| OpenAI | paid | https://platform.openai.com/api-keys |

**No key needed only if you already run a local LLM server** on your laptop: **Ollama**, **SGLang** or
**vLLM**. All three speak the OpenAI protocol, so in `.env` set `LLM_PROVIDER=ollama`, point
`OLLAMA_BASE_URL` at your server and set `LLM_MODEL` to the model it serves:

| local server | `OLLAMA_BASE_URL` (default port) | `LLM_MODEL` example |
|---|---|---|
| Ollama | `http://localhost:11434/v1` | `qwen2.5:7b` |
| SGLang | `http://localhost:30000/v1` | `Qwen/Qwen2.5-7B-Instruct` |
| vLLM | `http://localhost:8000/v1` | `Qwen/Qwen2.5-7B-Instruct` |

Download the model before the workshop; conference Wi-Fi is too slow for multi-GB downloads.
Small local models (7B and under) often handle tool calling poorly, so the agent examples work best
with a hosted API.

## Quick start

```bash
git clone https://github.com/dhelamay/CPUT_ICAILY_AI2026_MaganLLM_WORKSHOP.git
cd CPUT_ICAILY_AI2026_MaganLLM_WORKSHOP/SA_LIB_WORKSHOP_Day1   # or SA_LIB_WORKSHOP_Day2
cp .env.example .env    # pick LLM_PROVIDER and paste one API key
```

Then follow that folder's README.

## Contact

Dr. Nasser Mooman · Magan AI Inc. · [nmooman@gmail.com](mailto:nmooman@gmail.com)
