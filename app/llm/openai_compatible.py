"""OpenAI-compatible provider placeholder with safe fallback behavior."""

import json

import httpx
from pydantic import ValidationError

from app.core.config import Settings
from app.domain.intents import IntentLabel
from app.domain.safety import SafetyAction
from app.llm.base import LLMProvider, T
from app.llm.schemas import GeneratedAnswer, IntentClassification, SafetyClassification


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
        schema_instruction = self._schema_instruction(schema)
        system = (
            "Return only valid JSON. Do not wrap it in Markdown. "
            "Do not diagnose, do not provide medication instructions, and do not invent clinical claims. "
            f"Required JSON schema: {schema_instruction}"
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
            coerced = self._coerce_content(content, schema)
            if coerced is not None:
                return coerced
            raise RuntimeError(f"LLM output failed schema validation: {exc}") from exc

    def _schema_instruction(self, schema: type[T]) -> str:
        if schema is SafetyClassification:
            actions = ", ".join(action.value for action in SafetyAction)
            return f'{{"action": one of [{actions}], "confidence": number between 0 and 1}}'
        if schema is IntentClassification:
            labels = ", ".join(label.value for label in IntentLabel)
            return f'{{"label": one of [{labels}], "confidence": number between 0 and 1}}'
        if schema is GeneratedAnswer:
            return '{"answer": short non-diagnostic answer string}'
        return json.dumps(schema.model_json_schema(), ensure_ascii=False)

    def _coerce_content(self, content: str, schema: type[T]) -> T | None:
        try:
            raw = json.loads(content)
        except json.JSONDecodeError:
            if schema is GeneratedAnswer and content.strip():
                return schema(answer=content.strip())  # type: ignore[return-value]
            return None

        if schema is SafetyClassification:
            value = str(raw.get("action") or raw.get("safety") or raw.get("risk") or "").lower()
            if raw.get("safe") is True or value in {"safe", "continue", "normal", "low"}:
                return schema(action=SafetyAction.continue_normal, confidence=float(raw.get("confidence", 0.6)))  # type: ignore[return-value]
            if value in {"crisis", "unsafe", "self_harm", "high"}:
                return schema(action=SafetyAction.fixed_crisis_response, confidence=float(raw.get("confidence", 0.7)))  # type: ignore[return-value]
            if value in {"medical", "medication", "drug"}:
                return schema(action=SafetyAction.refuse_medical_advice, confidence=float(raw.get("confidence", 0.7)))  # type: ignore[return-value]

        if schema is IntentClassification:
            value = raw.get("label") or raw.get("intent")
            if value in {label.value for label in IntentLabel}:
                return schema(label=value, confidence=float(raw.get("confidence", 0.6)))  # type: ignore[return-value]

        if schema is GeneratedAnswer:
            answer = raw.get("answer") or raw.get("content") or raw.get("message")
            if isinstance(answer, str) and answer.strip():
                return schema(answer=answer.strip())  # type: ignore[return-value]
        return None
