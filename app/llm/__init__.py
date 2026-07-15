from app.core.config import get_settings
from app.llm.base import LLMProvider
from app.llm.mock import MockLLM
from app.llm.openai_compatible import OpenAICompatibleLLM


def build_llm() -> LLMProvider:
    settings = get_settings()
    if settings.llm_provider == "openai_compatible":
        return OpenAICompatibleLLM(settings)
    return MockLLM()

