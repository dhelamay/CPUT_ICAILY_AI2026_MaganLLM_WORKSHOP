# MCP 03 — FastMCP CLIENT, no LLM: connect to the multi-tool server, list its tools, call each one
# same 4 steps as 01_mcp_client.py (start, handshake, list tools, call tools), but FastMCP's Client
# does steps 1 + 2 for you: give it a path to a .py server and it starts it over stdio.
#
# run: python 09_mcp/03_fastmcp_client.py
# run (HTTP): python 09_mcp/03_fastmcp_client.py http://127.0.0.1:8000/mcp   (after: python 09_mcp/fastmcp_server.py http)

import asyncio
import json
import sys
from pathlib import Path

from fastmcp import Client

# a file path -> stdio (the client starts the server); a URL -> HTTP (the server is already running)
URL = next((a for a in sys.argv[1:] if a.startswith("http")), None)
SERVER = URL or str(Path(__file__).resolve().parent / "fastmcp_server.py")

CALLS = [("calculate", {"expression": "(1250 * 0.15) + 40"}),
         ("convert_units", {"value": 100, "from_unit": "km", "to_unit": "mile"}),
         ("convert_units", {"value": 38, "from_unit": "C", "to_unit": "F"}),
         ("text_stats", {"text": "MCP is a standard plug for AI tools. Write once, use everywhere!"}),
         ("current_time", {"timezone": "Africa/Tripoli"}),
         ("get_weather", {"city": "Benghazi"}),
         ("wikipedia_summary", {"topic": "Leptis Magna"})]


async def main():
    async with Client(SERVER) as client:                      # 1 + 2: start/connect and handshake
        tools = await client.list_tools()                     # 3: discover
        print(f"{len(tools)} tools offered by the server:")
        for t in tools:
            print(f"  - {t.name}: {t.description}")
            print(f"      input schema: {json.dumps(t.input_schema.get('properties', {}))}")

        for name, args in CALLS:                              # 4: call
            result = await client.call_tool(name, args)
            print(f"\n>>> {name}({args})")
            print(result.data if result.data is not None else result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
