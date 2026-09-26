# OpenAI Agents SDK 05 — multi-agent, two patterns
#
#  A) AGENTS AS TOOLS (manager pattern): a manager agent calls specialist agents like tools,
#     gets their answers back, and stays in control.
#  B) HANDOFFS: a triage agent hands the WHOLE conversation over to a specialist,
#     who then answers the user directly.
#
# run: python 03_openai_agents_sdk/05_multi_agent.py

import asyncio
import os

from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled

import workshop_common as wc

set_tracing_disabled(not os.getenv("OPENAI_API_KEY"))
cfg = wc.get_llm_config()
model = OpenAIChatCompletionsModel(model=cfg.model, openai_client=AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key))


@function_tool
def web_search(query: str) -> str:
    """Search the web for up-to-date information."""
    return wc.web_search(query)


@function_tool
def word_count(text: str) -> int:
    """Count how many words are in a piece of text."""
    return len(text.split())


researcher = Agent(name="Researcher", model=model, tools=[web_search],
                   instructions="Find current facts with web_search. Reply with 3 short bullet facts.")
writer = Agent(name="Writer", model=model, tools=[word_count],
               instructions="Write one friendly paragraph under 60 words. Check the length with word_count.")

# ---- A) agents as tools ----
manager = Agent(
    name="Manager",
    model=model,
    instructions="First ask the researcher for facts, then give those facts to the writer. Return the writer's paragraph.",
    tools=[
        researcher.as_tool(tool_name="research", tool_description="Research a topic on the web."),
        writer.as_tool(tool_name="write", tool_description="Write a short paragraph from facts."),
    ],
)

# ---- B) handoffs ----
math_tutor = Agent(name="Math tutor", model=model, handoff_description="Questions about maths",
                   instructions="You explain maths step by step, briefly.")
history_tutor = Agent(name="History tutor", model=model, handoff_description="Questions about history",
                      instructions="You explain history clearly and briefly.")
triage = Agent(name="Triage", model=model, handoffs=[math_tutor, history_tutor],
               instructions="Decide which tutor should answer and hand off to them. Do not answer yourself.")


async def main():
    print("=== A) agents as tools ===")
    result = await Runner.run(manager, "Write about the latest langchain version.")
    print(result.final_output)

    print("\n=== B) handoffs ===")
    result = await Runner.run(triage, "Who founded the ancient city of Cyrene in Libya?")
    print(f"answered by: {result.last_agent.name}\n{result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
