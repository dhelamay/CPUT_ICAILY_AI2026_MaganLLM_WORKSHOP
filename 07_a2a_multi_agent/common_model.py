# picks the model for the ADK agents in this folder (gemini natively, everything else via LiteLLM)
from google.adk.models.lite_llm import LiteLlm

from workshop_common import get_llm_config


def adk_model():
    cfg = get_llm_config()
    if cfg.provider == "gemini":
        return cfg.model
    return LiteLlm(model=cfg.litellm_model, api_base=cfg.base_url, api_key=cfg.api_key)
