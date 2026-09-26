# CrewAI 01 — your first agent
# CrewAI thinks in TEAMS: an Agent (who), a Task (what), and a Crew (runs the tasks).
# run: python 02_crewai/01_first_agent.py

import asyncio
import os

os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")  # no cloud tracing prompt during the workshop

from crewai import LLM, Agent, Crew, Task

from workshop_common import get_llm_config

# the brain — "openai/<model>" + base_url means "any OpenAI-compatible provider"
cfg = get_llm_config()
llm = LLM(model=cfg.litellm_model, base_url=cfg.base_url, api_key=cfg.api_key, temperature=0)

# the agent: CrewAI describes an agent with a role, a goal and a backstory (= the system prompt)
assistant = Agent(
    role="Helpful assistant",
    goal="Answer questions concisely and accurately",
    backstory="You explain technical ideas in plain language.",
    llm=llm,
)

# the task: what to do + what a good answer looks like
task = Task(
    description="explain what an AI agent is in two sentences.",
    expected_output="Two clear sentences.",
    agent=assistant,
)

async def main():
    # kickoff_async works both in scripts and in notebooks (which already run an event loop)
    result = await Crew(agents=[assistant], tasks=[task]).kickoff_async()
    print(f"[{cfg.provider}:{cfg.model}]", result.raw)


if __name__ == "__main__":
    asyncio.run(main())
