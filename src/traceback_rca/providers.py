"""Minimal LLM provider boundary with Gemini and offline scripted implementations."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Iterable, Mapping, Optional, Protocol, Tuple, Union


class ProviderConfigurationError(RuntimeError):
    """Raised when a real provider is requested without valid configuration."""


class ProviderResponseError(RuntimeError):
    """Raised when a provider returns no usable response body."""


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Provider-neutral text and optional operational metadata."""

    text: str
    latency_ms: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class LLMProvider(Protocol):
    """Small interface used by baseline and investigation reasoning."""

    def generate(
        self, prompt: str, response_schema: Mapping[str, Any]
    ) -> LLMResponse:
        """Generate one structured response for a prompt."""


@dataclass(frozen=True, slots=True)
class GeminiSettings:
    """Validated environment configuration for a real Gemini request."""

    api_key: str = field(repr=False)
    model: str

    @classmethod
    def from_environment(cls, load_local_env: bool = True) -> "GeminiSettings":
        if load_local_env:
            from dotenv import load_dotenv

            load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        model = os.getenv("GEMINI_MODEL", "").strip()
        missing = [
            name
            for name, value in (
                ("GEMINI_API_KEY", api_key),
                ("GEMINI_MODEL", model),
            )
            if not value
        ]
        if missing:
            raise ProviderConfigurationError(
                "Real Gemini execution requires non-empty environment variables: "
                + ", ".join(missing)
            )
        return cls(api_key=api_key, model=model)


class GeminiProvider:
    """Official google-genai provider using JSON-schema constrained responses."""

    def __init__(self, settings: GeminiSettings, client: Any = None) -> None:
        self._settings = settings
        if client is None:
            from google import genai

            client = genai.Client(api_key=settings.api_key)
        self._client = client

    @classmethod
    def from_environment(cls) -> "GeminiProvider":
        return cls(GeminiSettings.from_environment())

    @property
    def model(self) -> str:
        return self._settings.model

    def generate(
        self, prompt: str, response_schema: Mapping[str, Any]
    ) -> LLMResponse:
        started_at = perf_counter()
        response = self._client.models.generate_content(
            model=self._settings.model,
            contents=prompt,
            config={
                "temperature": 0,
                "response_mime_type": "application/json",
                "response_json_schema": dict(response_schema),
            },
        )
        latency_ms = (perf_counter() - started_at) * 1000.0
        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise ProviderResponseError("Gemini returned no structured response text")

        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", None) if usage else None
        output_tokens = (
            getattr(usage, "candidates_token_count", None) if usage else None
        )
        return LLMResponse(
            text=text,
            latency_ms=round(latency_ms, 2),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


ScriptedValue = Union[str, Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class LLMRequest:
    prompt: str
    response_schema: Mapping[str, Any]


class ScriptedProvider:
    """Deterministic provider for unit tests and offline demonstrations."""

    def __init__(self, responses: Iterable[ScriptedValue]) -> None:
        self._responses = tuple(responses)
        self._next_response = 0
        self._requests: list[LLMRequest] = []

    @property
    def requests(self) -> Tuple[LLMRequest, ...]:
        return tuple(self._requests)

    @property
    def call_count(self) -> int:
        return len(self._requests)

    def generate(
        self, prompt: str, response_schema: Mapping[str, Any]
    ) -> LLMResponse:
        if self._next_response >= len(self._responses):
            raise ProviderResponseError("ScriptedProvider has no response remaining")
        self._requests.append(LLMRequest(prompt, response_schema))
        response = self._responses[self._next_response]
        self._next_response += 1
        text = response if isinstance(response, str) else json.dumps(response)
        return LLMResponse(text=text, latency_ms=0.0)
