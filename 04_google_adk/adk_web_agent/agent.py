# the same complete agent, packaged for ADK's built-in chat UI.
# run from the 04_google_adk folder:   adk web
# then open http://localhost:8000 and pick "adk_web_agent" in the top-left menu.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so we can import workshop_common

from google.adk.agents import Agent  # noqa: E402
from google.adk.models.lite_llm import LiteLlm  # noqa: E402

from workshop_common import get_llm_config, web_search, word_count  # noqa: E402

cfg = get_llm_config()
model = cfg.model if cfg.provider == "gemini" else LiteLlm(model=cfg.litellm_model, api_base=cfg.base_url, api_key=cfg.api_key)

root_agent = Agent(
    name="assistant",
    model=model,
    instruction="You are a helpful assistant. Use your tools when they help answer accurately.",
    tools=[web_search, word_count],
)
