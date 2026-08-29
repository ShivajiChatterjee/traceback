"""Strict helpers for validating provider-generated JSON without silent defaults."""

import json
from typing import Any, Mapping, Sequence, Tuple

from traceback_rca.providers import LLMResponse


class StructuredOutputError(ValueError):
    """Raised when provider output cannot be safely parsed into the required schema."""


def parse_json_object(response: LLMResponse) -> Mapping[str, Any]:
    try:
        parsed = json.loads(response.text)
    except json.JSONDecodeError as error:
        raise StructuredOutputError(
            f"LLM response is not valid JSON: {error.msg}"
        ) from error
    if not isinstance(parsed, dict):
        raise StructuredOutputError("LLM response must be a JSON object")
    return parsed


def require_string(data: Mapping[str, Any], field_name: str) -> str:
    value = data.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise StructuredOutputError(f"{field_name} must be a non-empty string")
    return value.strip()


def require_confidence(data: Mapping[str, Any], field_name: str = "confidence") -> float:
    value = data.get(field_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StructuredOutputError(f"{field_name} must be numeric")
    confidence = float(value)
    if not 0.0 <= confidence <= 1.0:
        raise StructuredOutputError(f"{field_name} must be between 0.0 and 1.0")
    return confidence


def require_string_tuple(
    data: Mapping[str, Any], field_name: str
) -> Tuple[str, ...]:
    value = data.get(field_name)
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise StructuredOutputError(f"{field_name} must be an array of strings")
    return tuple(item.strip() for item in value)


def require_object_sequence(
    data: Mapping[str, Any], field_name: str
) -> Sequence[Mapping[str, Any]]:
    value = data.get(field_name)
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise StructuredOutputError(f"{field_name} must be an array of objects")
    return value

