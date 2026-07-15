"""OpenAI-compatible provider placeholder with safe fallback behavior."""

import json

import httpx
from pydantic import ValidationError

from app.core.config import Settings
from app.llm.base import LLMProvider, T


class OpenAICompatibleLLM(LLMProvider):
    """Minimal OpenAI-compatible structured output client."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.openai_api_key or settings.dashscope_api_key
        self.base_url = settings.openai_base_url if settings.openai_api_key else settings.dashscope_compatible_base_url
        self.model = settings.openai_model if settings.openai_api_key else settings.dashscope_model

    def structured(self, prompt_name: str, variables: dict, schema: type[T]) -> T:
        if not self.api_key:
            raise RuntimeError("No API key configured for OpenAI-compatible LLM.")
        system = (
            "Return only valid JSON matching this schema. "
            "Do not diagnose, do not provide medication instructions, and do not invent clinical claims."
        )
        user = f"Prompt: {prompt_name}\nVariables: {json.dumps(variables, ensure_ascii=False)}"
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(timeout=30) as client:
            response = client.post(f"{self.base_url.rstrip('/')}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        try:
            return schema.model_validate_json(content)
        except ValidationError as exc:
            raise RuntimeError(f"LLM output failed schema validation: {exc}") from exc

