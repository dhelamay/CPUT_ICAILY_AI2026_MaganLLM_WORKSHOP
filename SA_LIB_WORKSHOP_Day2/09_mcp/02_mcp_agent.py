# MCP 02 — MCP + API + ONE AGENT
# the agent does NOT define any tools itself. it asks the MCP server what tools exist, gives them
# to the LLM, and when the LLM wants a tool, the agent forwards the call to the MCP server.
#
#   user --> [agent + LLM] --tool call--> [MCP client] --stdio--> [weather_server.py] --HTTP--> Open-Meteo API
#
# swap weather_server.py for ANY other MCP server (files, GitHub, databases, ...) and the agent
# gets new abilities without changing a line of agent code. that is the point of MCP.
#
# run: python 09_mcp/02_mcp_agent.py

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

from workshop_common import get_llm_config

SERVER = StdioServerParameters(command=sys.executable, args=[str(Path(__file__).resolve().parent / "weather_server.py")])
cfg = get_llm_config()
llm = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)


def mcp_to_openai(tool) -> dict:
    """An MCP tool description -> the 'function calling' format every LLM provider understands."""
    return {"type": "function", "function": {
        "name": tool.name, "description": tool.description or "", "parameters": tool.input_schema}}


async def run_agent(session: ClientSession, question: str, max_steps: int = 6) -> str:
    tools = [mcp_to_openai(t) for t in (await session.list_tools()).tools]    # discover tools at run time
    messages = [{"role": "system", "content": "You are a helpful weather assistant. Use your tools for facts. Be concise."},
                {"role": "user", "content": question}]
    for _ in range(max_steps):                                                 # plan -> act -> observe
        msg = llm.chat.completions.create(model=cfg.model, messages=messages, tools=tools).choices[0].message
        if not msg.tool_calls:
            return msg.content
        messages.append(msg)
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            print(f"  [agent -> MCP] {call.function.name}({args})")
            result = await session.call_tool(call.function.name, args)          # the MCP server does the work
            text = "\n".join(c.text for c in result.content if hasattr(c, "text"))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
    return "stopped: too many steps"


async def main():
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for question in ["What's the weather like in Tripoli right now?",
                             "Compare the next 3 days in Benghazi and Sabha. Which city will be hotter?"]:
                print(f"\nuser: {question}")
                print(f"agent: {await run_agent(session, question)}")


if __name__ == "__main__":
    asyncio.run(main())
