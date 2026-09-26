# Monitoring 02 — the usage report: calls, tokens, cost, latency, failures
# reads agent_monitor.db written by 01_simple_monitor.py (or by your own agent using monitor.py).
# run: python 08_monitoring/02_usage_report.py

import sqlite3

from monitor import DB_PATH

con = sqlite3.connect(DB_PATH)
q = lambda sql: con.execute(sql).fetchall()  # noqa: E731

print("=" * 64 + "\nAGENT USAGE REPORT\n" + "=" * 64)

runs, run_fail = q("SELECT COUNT(*), SUM(status='error') FROM events WHERE kind='run'")[0]
print(f"runs: {runs}   failed runs: {run_fail or 0}")

print("\n-- by kind ---------------------------------------------------")
print(f"{'kind':<6} {'calls':>6} {'errors':>7} {'fail %':>7} {'avg ms':>8} {'max ms':>8}")
for kind, calls, errors, avg_ms, max_ms in q(
        "SELECT kind, COUNT(*), SUM(status='error'), AVG(latency_ms), MAX(latency_ms) "
        "FROM events GROUP BY kind ORDER BY kind"):
    print(f"{kind:<6} {calls:>6} {errors:>7} {100 * errors / calls:>6.1f}% {avg_ms:>8.0f} {max_ms:>8.0f}")

print("\n-- LLM usage by model -----------------------------------------")
print(f"{'model':<28} {'calls':>5} {'in tok':>8} {'out tok':>8} {'cost $':>9}")
for model, calls, tin, tout, cost in q(
        "SELECT name, COUNT(*), SUM(prompt_tokens), SUM(completion_tokens), SUM(cost_usd) "
        "FROM events WHERE kind='llm' GROUP BY name"):
    print(f"{model[:28]:<28} {calls:>5} {tin:>8} {tout:>8} {cost:>9.5f}")

print("\n-- tools -------------------------------------------------------")
for name, calls, errors in q("SELECT name, COUNT(*), SUM(status='error') FROM events WHERE kind='tool' GROUP BY name"):
    print(f"{name:<20} calls={calls:<4} errors={errors}")

print("\n-- last 5 errors ------------------------------------------------")
for ts, run_id, kind, name, error in q(
        "SELECT ts, run_id, kind, name, error FROM events WHERE status='error' ORDER BY ts DESC LIMIT 5"):
    print(f"{ts} run={run_id} {kind}:{name}\n    {error[:110]}")
