# MCP 01 — MCP + API, no LLM: connect to the MCP server, list its tools, call them
# this is what every MCP client does under the hood (Claude Desktop, VS Code, agent frameworks):
#   1. START / CONNECT to the server       2. INITIALIZE (handshake)
#   3. LIST TOOLS (names, descriptions, input schemas)       4. CALL TOOLS
#
# run: python 09_mcp/01_mcp_client.py

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = StdioServerParameters(command=sys.executable, args=[str(Path(__file__).resolve().parent / "weather_server.py")])


async def main():
    async with stdio_client(SERVER) as (read, write):          # 1. start the server as a subprocess
        async with ClientSession(read, write) as session:
            await session.initialize()                          # 2. handshake

            tools = (await session.list_tools()).tools         # 3. what can this server do?
            print("tools offered by the server:")
            for t in tools:
                print(f"  - {t.name}: {t.description}")
                print(f"      input schema: {json.dumps(t.input_schema.get('properties', {}))}")

            for name, args in [("get_current_weather", {"city": "Tripoli"}),     # 4. call them
                               ("get_forecast", {"city": "Benghazi", "days": 3}),
                               ("word_count", {"text": "MCP is a standard plug for AI tools"})]:
                result = await session.call_tool(name, args)
                print(f"\n>>> {name}({args})\n{result.content[0].text}")


if __name__ == "__main__":
    asyncio.run(main())
