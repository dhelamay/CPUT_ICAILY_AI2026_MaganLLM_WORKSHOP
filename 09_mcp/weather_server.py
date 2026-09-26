# a simple MCP SERVER that wraps a real public API (Open-Meteo weather — free, no key)
#
# MCP (Model Context Protocol) is a standard plug for AI tools: write your tools ONCE as an MCP
# server, and any MCP client can use them — our agent, Claude Desktop, VS Code, Cursor, the
# OpenAI Agents SDK, Google ADK, LangGraph ...
#
# this file is started BY the client (01_mcp_client.py / 02_mcp_agent.py). it talks over
# stdin/stdout ("stdio" transport), so you normally don't run it by hand.
# to test it on its own:  python 09_mcp/weather_server.py   (it waits silently for a client; Ctrl+C to stop)

import logging

import httpx
from mcp.server.mcpserver import MCPServer

logging.getLogger("httpx").setLevel(logging.WARNING)   # keep the output clean

server = MCPServer("weather")   # the name clients will see

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_CODES = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 51: "light drizzle",
                 61: "light rain", 63: "rain", 65: "heavy rain", 80: "rain showers", 95: "thunderstorm"}


def find_city(city: str) -> dict | None:
    """Step 1 of the API: city name -> latitude / longitude."""
    results = httpx.get(GEOCODE_URL, params={"name": city, "count": 1}, timeout=15).json().get("results")
    return results[0] if results else None


# every @server.tool() becomes a tool that MCP clients can discover and call.
# the docstring and type hints are what the client (and the LLM) sees.
@server.tool()
def get_current_weather(city: str) -> str:
    """Get the current weather (temperature, wind, conditions) for a city, e.g. 'Tripoli' or 'Benghazi'."""
    place = find_city(city)
    if not place:
        return f"I could not find a city called '{city}'."
    now = httpx.get(WEATHER_URL, timeout=15, params={
        "latitude": place["latitude"], "longitude": place["longitude"],
        "current": "temperature_2m,wind_speed_10m,weather_code"}).json()["current"]
    sky = WEATHER_CODES.get(now["weather_code"], f"weather code {now['weather_code']}")
    return (f"{place['name']}, {place.get('country', '')}: {now['temperature_2m']} °C, "
            f"wind {now['wind_speed_10m']} km/h, {sky} (at {now['time']} UTC)")


@server.tool()
def get_forecast(city: str, days: int = 3) -> str:
    """Get the daily min/max temperature forecast for a city for the next 1-7 days."""
    place = find_city(city)
    if not place:
        return f"I could not find a city called '{city}'."
    daily = httpx.get(WEATHER_URL, timeout=15, params={
        "latitude": place["latitude"], "longitude": place["longitude"], "timezone": "auto",
        "daily": "temperature_2m_min,temperature_2m_max", "forecast_days": max(1, min(days, 7))}).json()["daily"]
    rows = [f"{d}: {lo}–{hi} °C" for d, lo, hi in zip(daily["time"], daily["temperature_2m_min"], daily["temperature_2m_max"])]
    return f"{place['name']} forecast:\n" + "\n".join(rows)


@server.tool()
def word_count(text: str) -> int:
    """Count how many words are in a piece of text."""
    return len(text.split())


if __name__ == "__main__":
    server.run("stdio")   # other transports: "streamable-http" (a web server other machines can reach)
