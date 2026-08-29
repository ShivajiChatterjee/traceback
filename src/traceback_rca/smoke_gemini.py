"""Opt-in minimal Gemini authentication and structured-response smoke test."""

from traceback_rca.providers import GeminiProvider
from traceback_rca.structured import StructuredOutputError, parse_json_object, require_string


_SMOKE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status"],
    "properties": {"status": {"type": "string", "enum": ["ok"]}},
}


def main() -> None:
    provider = GeminiProvider.from_environment()
    response = provider.generate(
        "Return the structured status value that confirms this request was received.",
        _SMOKE_SCHEMA,
    )
    data = parse_json_object(response)
    if require_string(data, "status") != "ok":
        raise StructuredOutputError("Gemini smoke response did not return status=ok")
    print("Gemini authentication and structured response parsing succeeded.")


if __name__ == "__main__":
    main()

