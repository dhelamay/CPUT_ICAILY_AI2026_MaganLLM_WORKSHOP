# 09 · MCP — give any agent new tools through a standard plug

**MCP (Model Context Protocol)** is an open standard for connecting AI apps to tools and data. You write
your tools **once** as an *MCP server*, and any *MCP client* can use them: your own agent, Claude
Desktop, VS Code / Copilot, Cursor, the OpenAI Agents SDK, Google ADK, LangGraph and others.

```
 user ──> [ agent + LLM ] ──tool call──> [ MCP client ] ══stdio══> [ weather_server.py ] ──HTTP──> Open-Meteo API
                ^                                                         (MCP server)              (free, no key)
                └────────────────────────── result ─────────────────────────────┘
```

| file | what it shows |
|---|---|
| `weather_server.py` | a **simple MCP server** that wraps a real public **API** (Open-Meteo weather, free, no key). Tools: `get_current_weather`, `get_forecast`, `word_count` |
| `01_mcp_client.py` | **MCP + API, no LLM**: start the server, handshake, **list tools** (with their input schemas), **call tools** |
| `02_mcp_agent.py` | **MCP + API + one agent**: the LLM discovers the server's tools at run time and decides when to call them |

```bash
pip install -r 09_mcp/requirements.txt
python 09_mcp/01_mcp_client.py      # no API key needed at all
python 09_mcp/02_mcp_agent.py       # needs your LLM key (LLM_PROVIDER in .env)
```

Example output of `01_mcp_client.py`:

```
tools offered by the server:
  - get_current_weather: Get the current weather (temperature, wind, conditions) for a city ...
      input schema: {"city": {"title": "City", "type": "string"}}
  - get_forecast: Get the daily min/max temperature forecast for a city for the next 1-7 days.
...
>>> get_current_weather({'city': 'Tripoli'})
Tripoli, Libya: 25.0 °C, wind 11.6 km/h, partly cloudy (at 2026-09-26T15:30 UTC)
```

## The key ideas

1. **Server:** `@server.tool()` turns a Python function into an MCP tool. The docstring and type hints
   become the tool's description and **input schema**.
2. **Transport:** here the client starts the server as a subprocess and they talk over stdin/stdout
   (`stdio`). For a server on another machine, use `server.run("streamable-http")`.
3. **Discovery:** the client calls `list_tools()`, so the agent never hard-codes its tools.
4. **Bridge to the LLM:** `mcp_to_openai()` in `02_mcp_agent.py` converts each MCP tool into the
   standard *function-calling* format, which works with every provider in `.env`.

## Use the same server in other apps

**Claude Desktop** (`claude_desktop_config.json`) or **VS Code** (`.vscode/mcp.json`, where the key is
`"servers"` instead of `"mcpServers"`):

```json
{
  "mcpServers": {
    "weather": { "command": "python", "args": ["/full/path/to/09_mcp/weather_server.py"] }
  }
}
```

**In the agent frameworks from this workshop** (one or two lines each):

| framework | how it connects to an MCP server |
|---|---|
| OpenAI Agents SDK | `Agent(..., mcp_servers=[MCPServerStdio(params={"command": "python", "args": ["weather_server.py"]})])` (use `async with` on the server) |
| Google ADK | `tools=[McpToolset(connection_params=StdioConnectionParams(server_params=StdioServerParameters(command="python", args=["weather_server.py"])))]` (also `pip install mcp`) |
| LangGraph / LangChain | `langchain-mcp-adapters`: `MultiServerMCPClient({...}).get_tools()` |
| Microsoft Agent Framework | `MCPStdioTool(name="weather", command="python", args=[...])` |
| CrewAI | `crewai-tools` `MCPServerAdapter(...)` |

## Exercises

1. Add a tool to `weather_server.py`, for example `get_sunrise(city)` using Open-Meteo's `daily=sunrise,sunset`. Re-run `01_mcp_client.py` and it appears automatically.
2. Ask the agent: *"Should I bring a jacket to Benghazi tomorrow evening?"*
3. Wrap a different free API (currency rates, public holidays, Wikipedia) in a new MCP server.

> Note: the official `mcp` SDK is now at version **2.x**. Many tutorials still show 1.x code
> (`from mcp.server.fastmcp import FastMCP`), which was renamed to `from mcp.server.mcpserver import MCPServer`.
> Tool fields are snake_case now (`tool.input_schema`).

Notebook: [`09_mcp.ipynb`](09_mcp.ipynb)
