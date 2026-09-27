"""
workshop_common.py — shared helpers for every framework folder in this workshop.

(the SAME file is copied into each folder so every folder works on its own —
 edit shared/workshop_common.py and run `python tools/sync_common.py` to update the copies)

What it gives you:
  get_llm_config()   -> which provider / model / base_url / api key to use (read from .env)
  fallback_order()   -> the list of providers to try, in order, for the fallback examples
  ping()             -> quick "are you alive?" check against a provider
  web_search()       -> free DuckDuckGo search (no key)
  word_count()       -> the tiny "tool you wrote yourself" from the original guide

Every provider below speaks the OpenAI chat-completions protocol, which is why every
framework in this workshop can use every provider: we just change base_url + api_key + model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv

# walks up from where you run python, so the .env in the repo root is always found
load_dotenv(find_dotenv(usecwd=True))

PROVIDERS = {
    #  name        base_url                                                     api-key env var       default model
    "groq":       ("https://api.groq.com/openai/v1",                           "GROQ_API_KEY",       "llama-3.3-70b-versatile"),
    "gemini":     ("https://generativelanguage.googleapis.com/v1beta/openai/", "GOOGLE_API_KEY",     "gemini-2.5-flash"),
    "openai":     ("https://api.openai.com/v1",                                "OPENAI_API_KEY",     "gpt-4o-mini"),
    "deepseek":   ("https://api.deepseek.com",                                 "DEEPSEEK_API_KEY",   "deepseek-chat"),
    "openrouter": ("https://openrouter.ai/api/v1",                             "OPENROUTER_API_KEY", "meta-llama/llama-3.3-70b-instruct:free"),
    "ollama":     (os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),  None,                 "qwen2.5:7b"),
}


@dataclass
class LLMConfig:
    provider: str
    model: str
    base_url: str
    api_key: str

    @property
    def litellm_model(self) -> str:
        """Model name in LiteLLM format (used by CrewAI and Google ADK): 'openai/<model>'
        means 'an OpenAI-compatible endpoint' — combined with base_url it works for every provider."""
        return f"openai/{self.model}"


def get_llm_config(provider: str | None = None) -> LLMConfig:
    """Read LLM_PROVIDER / LLM_MODEL / <PROVIDER>_API_KEY from the environment."""
    provider = (provider or os.getenv("LLM_PROVIDER") or "groq").strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Choose one of: {', '.join(PROVIDERS)}")
    base_url, key_env, default_model = PROVIDERS[provider]
    api_key = os.getenv(key_env) if key_env else "ollama"  # ollama runs locally, no key
    if not api_key:
        raise RuntimeError(f"{key_env} is missing — add it to your .env file (see .env.example).")
    # LLM_MODEL only overrides the model of the MAIN provider, not of the fallbacks
    main_provider = (os.getenv("LLM_PROVIDER") or "groq").strip().lower()
    model = (os.getenv("LLM_MODEL") if provider == main_provider else None) or default_model
    return LLMConfig(provider, model, base_url, api_key)


def fallback_order() -> list[str]:
    """Providers to try, in order. Default: your LLM_PROVIDER first, then LLM_FALLBACKS (groq,gemini)."""
    main = (os.getenv("LLM_PROVIDER") or "groq").strip().lower()
    extra = [p.strip().lower() for p in os.getenv("LLM_FALLBACKS", "groq,gemini").split(",") if p.strip()]
    return list(dict.fromkeys([main, *extra]))  # remove duplicates, keep order


def ping(cfg: LLMConfig) -> None:
    """Send a 1-token request. Raises an exception if the provider is down, rate-limited or the key is wrong."""
    from openai import OpenAI

    OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=20, max_retries=0).chat.completions.create(
        model=cfg.model, messages=[{"role": "user", "content": "ping"}], max_tokens=1
    )


def web_search(query: str) -> str:
    """Search the web with DuckDuckGo (free, no key) and return the top results as text."""
    try:
        from ddgs import DDGS

        results = DDGS().text(query, max_results=3)
        return "\n".join(f"- {r['title']}: {r['body']} ({r['href']})" for r in results) or "no results"
    except Exception as e:  # the free search is rate-limited: return the error instead of crashing
        return f"web search failed ({e}). try again in a few seconds."


def word_count(text: str) -> int:
    """Count how many words are in a piece of text."""
    return len(text.split())
