# MCP 04 — FastMCP CLIENT + ONE AGENT that picks from MANY tools
# the agent gets all 6 tools from fastmcp_server.py and the LLM decides which ones to use —
# often several in one question (weather + unit conversion + math ...).
#
#   user --> [agent + LLM] --tool calls--> [FastMCP Client] --stdio--> [fastmcp_server.py] --> Open-Meteo / Wikipedia
#
# run: python 09_mcp/04_fastmcp_agent.py      (needs your LLM key: LLM_PROVIDER in .env)

import asyncio
import json
from pathlib import Path

from fastmcp import Client
from openai import OpenAI

from workshop_common import get_llm_config

SERVER = str(Path(__file__).resolve().parent / "fastmcp_server.py")
cfg = get_llm_config()
llm = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

SYSTEM = ("You are a helpful assistant. Use your tools for facts, time, weather and ALL arithmetic. "
          "You may call several tools. Be concise.")


def mcp_to_openai(tool) -> dict:
    """An MCP tool description -> the 'function calling' format every LLM provider understands."""
    return {"type": "function", "function": {
        "name": tool.name, "description": tool.description or "", "parameters": tool.input_schema}}


async def run_agent(client: Client, question: str, max_steps: int = 8) -> str:
    tools = [mcp_to_openai(t) for t in await client.list_tools()]              # discover tools at run time
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    for _ in range(max_steps):                                                 # plan -> act -> observe
        msg = llm.chat.completions.create(model=cfg.model, messages=messages, tools=tools).choices[0].message
        if not msg.tool_calls:
            return msg.content
        messages.append(msg)
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            print(f"  [agent -> MCP] {call.function.name}({args})")
            result = await client.call_tool(call.function.name, args, raise_on_error=False)
            text = "\n".join(c.text for c in result.content if hasattr(c, "text"))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
    return "stopped: too many steps"


async def main():
    async with Client(SERVER) as client:
        for question in ["What time is it in Tripoli, and what's the weather there in Fahrenheit?",
                         "Tell me briefly about Leptis Magna. If I visit with 4 friends and a ticket costs 12.5 dinars, what is the total?",
                         "How many words are in this sentence: 'Agents choose tools, servers provide them'?"]:
            print(f"\nuser: {question}")
            print(f"agent: {await run_agent(client, question)}")


if __name__ == "__main__":
    asyncio.run(main())
