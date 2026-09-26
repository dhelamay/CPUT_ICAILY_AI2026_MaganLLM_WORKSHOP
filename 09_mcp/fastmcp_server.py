# a MULTI-TOOL MCP server built with FastMCP (the `fastmcp` package: pip install fastmcp)
#
# FastMCP is the most popular high-level way to write MCP servers in Python. one decorator,
# @mcp.tool, turns a normal function into a tool any MCP client can discover and call.
# the docstring becomes the tool description, the type hints become the input schema.
#
# this server bundles 6 tools of different kinds, so you can see an agent CHOOSE between them:
#   - calculate            pure Python (no internet)
#   - convert_units        pure Python, with a fixed list of allowed values (Literal -> enum in the schema)
#   - text_stats           pure Python, returns structured data (a dict -> JSON)
#   - current_time         standard library (zoneinfo)
#   - get_weather          public API: Open-Meteo (free, no key)
#   - wikipedia_summary    public API: Wikipedia (free, no key)
#
# this file is started BY the client (03_fastmcp_client.py / 04_fastmcp_agent.py) over stdio.
# to serve it over HTTP instead: python 09_mcp/fastmcp_server.py http   (-> http://127.0.0.1:8000/mcp)

import ast
import operator
import sys
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

import httpx
from fastmcp import FastMCP

mcp = FastMCP("workshop-toolbox", instructions="General-purpose tools: math, units, text, time, weather, Wikipedia.")

HEADERS = {"User-Agent": "SA-LIB-workshop/1.0 (MCP demo)"}   # Wikipedia asks every client to identify itself


# ---------- 1. calculator (safe: we parse the expression, we never call eval) ----------
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.operand))
    raise ValueError("only numbers and + - * / ** % ( ) are allowed")


@mcp.tool
def calculate(expression: str) -> float:
    """Evaluate a math expression exactly, e.g. '(1250 * 0.15) + 40' or '2 ** 10'. Use this instead of doing math in your head."""
    return _eval(ast.parse(expression, mode="eval").body)


# ---------- 2. unit converter ----------
TO_BASE = {"km": 1000, "m": 1, "mile": 1609.344, "ft": 0.3048,     # length -> metres
           "kg": 1, "lb": 0.45359237}                              # mass   -> kilograms


@mcp.tool
def convert_units(value: float, from_unit: Literal["km", "m", "mile", "ft", "kg", "lb", "C", "F"],
                  to_unit: Literal["km", "m", "mile", "ft", "kg", "lb", "C", "F"]) -> str:
    """Convert a value between units of length (km, m, mile, ft), mass (kg, lb) or temperature (C, F)."""
    if {from_unit, to_unit} <= {"C", "F"}:
        result = value if from_unit == to_unit else (value * 9 / 5 + 32 if from_unit == "C" else (value - 32) * 5 / 9)
    elif from_unit in TO_BASE and to_unit in TO_BASE and ({from_unit, to_unit} <= {"kg", "lb"}) == (from_unit in {"kg", "lb"}):
        result = value * TO_BASE[from_unit] / TO_BASE[to_unit]
    else:
        return f"cannot convert {from_unit} to {to_unit} (different kinds of unit)"
    return f"{value} {from_unit} = {round(result, 4)} {to_unit}"


# ---------- 3. text statistics (returns a dict: clients get it as structured JSON) ----------
@mcp.tool
def text_stats(text: str) -> dict:
    """Count the words, characters and sentences in a piece of text, and find the longest word."""
    words = text.split()
    return {"words": len(words), "characters": len(text),
            "sentences": sum(text.count(p) for p in ".!?") or 1,
            "longest_word": max(words, key=len) if words else ""}


# ---------- 4. current time in any time zone ----------
@mcp.tool
def current_time(timezone: str = "Africa/Tripoli") -> str:
    """Get the current date and time in an IANA time zone, e.g. 'Africa/Tripoli', 'Europe/London', 'Asia/Tokyo'."""
    try:
        now = datetime.now(ZoneInfo(timezone))
    except Exception:
        return f"unknown time zone '{timezone}'. use IANA names like 'Africa/Tripoli'."
    return now.strftime(f"%A %d %B %Y, %H:%M ({timezone})")


# ---------- 5. weather (public API, no key) ----------
@mcp.tool
def get_weather(city: str) -> str:
    """Get the current temperature, wind and humidity for a city, e.g. 'Tripoli' or 'Benghazi'."""
    found = httpx.get("https://geocoding-api.open-meteo.com/v1/search",
                      params={"name": city, "count": 1}, timeout=15).json().get("results")
    if not found:
        return f"I could not find a city called '{city}'."
    place = found[0]
    now = httpx.get("https://api.open-meteo.com/v1/forecast", timeout=15, params={
        "latitude": place["latitude"], "longitude": place["longitude"],
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m"}).json()["current"]
    return (f"{place['name']}, {place.get('country', '')}: {now['temperature_2m']} °C, "
            f"humidity {now['relative_humidity_2m']}%, wind {now['wind_speed_10m']} km/h")


# ---------- 6. Wikipedia summary (public API, no key) ----------
@mcp.tool
def wikipedia_summary(topic: str) -> str:
    """Get a short encyclopedia summary of a topic, place or person from Wikipedia."""
    title = topic.strip().replace(" ", "_")
    r = httpx.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
                  headers=HEADERS, timeout=15, follow_redirects=True)
    if r.status_code != 200:
        return f"no Wikipedia page found for '{topic}'."
    data = r.json()
    return f"{data.get('title', topic)}: {data.get('extract', 'no summary available')}"


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "http":
        mcp.run("http", host="127.0.0.1", port=8000)   # clients connect to http://127.0.0.1:8000/mcp
    else:
        mcp.run(show_banner=False, log_level="WARNING")   # default: stdio (the client starts us as a subprocess)
