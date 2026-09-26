"""
monitor.py — a tiny agent monitor you can read in 5 minutes. no extra packages.

It records every LLM call, tool call and agent run into a SQLite file (agent_monitor.db):
    when · which run · what kind · name/model · ok or error · error text · latency · tokens · cost
and also writes one JSON line per event to agent_events.log (easy to ship to any log system).

Use it like this:
    with monitor.run("answer a question") as run_id:        # groups everything below into one run
        resp = monitor.chat(client, model=..., messages=...)  # instead of client.chat.completions.create
        result = monitor.tool(web_search, query="...")        # instead of web_search(query="...")

Then look at the numbers:  python 08_monitoring/02_usage_report.py
"""

from __future__ import annotations

import contextvars
import json
import logging
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = "agent_monitor.db"
LOG_PATH = "agent_events.log"

# price per 1 MILLION tokens in USD: (input, output). EXAMPLE values — check your provider's pricing page.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "deepseek-chat": (0.28, 0.42),
    "gemini-2.5-flash": (0.30, 2.50),
    "llama-3.3-70b-versatile": (0.0, 0.0),   # groq free tier
}

_current_run = contextvars.ContextVar("current_run", default=None)

logger = logging.getLogger("agent_monitor")
if not logger.handlers:
    handler = logging.FileHandler(LOG_PATH)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS events (
        ts TEXT, run_id TEXT, kind TEXT, name TEXT, status TEXT, error TEXT,
        latency_ms REAL, prompt_tokens INTEGER, completion_tokens INTEGER, cost_usd REAL)""")
    return con


def record(kind: str, name: str, status: str, latency_ms: float, error: str = "",
           prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
    """Save one event to SQLite and to the JSON log."""
    price_in, price_out = PRICES.get(name, (0.0, 0.0))
    cost = (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000
    event = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "run_id": _current_run.get(),
             "kind": kind, "name": name, "status": status, "error": error[:300], "latency_ms": round(latency_ms, 1),
             "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "cost_usd": round(cost, 6)}
    with _db() as con:
        con.execute("INSERT INTO events VALUES (:ts,:run_id,:kind,:name,:status,:error,:latency_ms,"
                    ":prompt_tokens,:completion_tokens,:cost_usd)", event)
    logger.info(json.dumps(event))


@contextmanager
def run(description: str = "agent run"):
    """Group all calls inside the `with` block into one run (and time the whole run)."""
    run_id = uuid.uuid4().hex[:8]
    token = _current_run.set(run_id)
    start = time.perf_counter()
    try:
        yield run_id
        record("run", description, "ok", (time.perf_counter() - start) * 1000)
    except Exception as e:
        record("run", description, "error", (time.perf_counter() - start) * 1000, error=repr(e))
        raise
    finally:
        _current_run.reset(token)


def chat(client, **kwargs):
    """Call client.chat.completions.create(**kwargs) and record latency, tokens, cost and errors."""
    start = time.perf_counter()
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        record("llm", kwargs.get("model", "?"), "error", (time.perf_counter() - start) * 1000, error=repr(e))
        raise
    usage = resp.usage
    record("llm", kwargs.get("model", "?"), "ok", (time.perf_counter() - start) * 1000,
           prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
           completion_tokens=getattr(usage, "completion_tokens", 0) or 0)
    return resp


def tool(fn, **kwargs):
    """Call a tool function and record latency and errors. Errors are returned as text (the agent keeps going)."""
    start = time.perf_counter()
    try:
        result = fn(**kwargs)
        failed = isinstance(result, str) and result.startswith("web search failed")
        record("tool", fn.__name__, "error" if failed else "ok", (time.perf_counter() - start) * 1000,
               error=result if failed else "")
        return result
    except Exception as e:
        record("tool", fn.__name__, "error", (time.perf_counter() - start) * 1000, error=repr(e))
        return f"tool error: {e}"
