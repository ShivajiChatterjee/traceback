"""Offline provider, configuration, and single-request baseline tests."""

from types import SimpleNamespace

import pytest

from traceback_rca.baseline import Baseline
from traceback_rca.incidents import get_incident
from traceback_rca.providers import (
    GeminiProvider,
    GeminiSettings,
    ProviderConfigurationError,
    ScriptedProvider,
)
from traceback_rca.structured import StructuredOutputError
from traceback_rca.taxonomy import RootCauseCategory, root_cause_labels


def _valid_baseline_response(root_cause: str = "prompt_regression") -> dict:
    return {
        "predicted_root_cause": root_cause,
        "confidence": 0.72,
        "summary": "The prompt change is temporally associated with lower quality.",
        "supporting_evidence": ["prompt_profile changed after deployment"],
    }


def test_taxonomy_contains_current_and_future_categories() -> None:
    labels = set(root_cause_labels())

    assert {
        "retriever_top_k_regression",
        "prompt_regression",
        "no_incident",
        "embedding_regression",
        "stale_index",
        "reranker_disabled",
        "guardrail_regression",
        "tool_latency_regression",
        "context_truncation",
        "ingestion_failure",
        "model_regression",
    } <= labels


def test_baseline_makes_exactly_one_request_and_validates_response() -> None:
    provider = ScriptedProvider([_valid_baseline_response()])
    diagnosis = Baseline(provider).diagnose(get_incident("I10"))

    assert provider.call_count == 1
    assert diagnosis.predicted_root_cause is RootCauseCategory.PROMPT_REGRESSION
    assert diagnosis.confidence == 0.72
    assert "retriever_top_k_regression" not in provider.requests[0].prompt
    assert "expected_root_cause" not in provider.requests[0].prompt


@pytest.mark.parametrize(
    "response",
    [
        "not-json",
        {"predicted_root_cause": "not-in-taxonomy", "confidence": 0.5,
         "summary": "x", "supporting_evidence": []},
        {"predicted_root_cause": "prompt_regression", "confidence": 2.0,
         "summary": "x", "supporting_evidence": []},
        {"predicted_root_cause": "prompt_regression", "confidence": 0.5,
         "summary": "x"},
    ],
)
def test_baseline_surfaces_malformed_output(response) -> None:
    with pytest.raises(StructuredOutputError):
        Baseline(ScriptedProvider([response])).diagnose(get_incident("I10"))


def test_missing_real_gemini_configuration_fails_clearly(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    with pytest.raises(
        ProviderConfigurationError, match="GEMINI_API_KEY, GEMINI_MODEL"
    ):
        GeminiSettings.from_environment(load_local_env=False)


def test_gemini_settings_load_environment_without_exposing_key(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-secret")
    monkeypatch.setenv("GEMINI_MODEL", "test-model-from-environment")

    settings = GeminiSettings.from_environment()

    assert settings.model == "test-model-from-environment"
    assert "test-only-secret" not in repr(settings)


def test_gemini_provider_uses_configured_model_and_json_schema_without_network() -> None:
    response = SimpleNamespace(
        text='{"status":"ok"}',
        usage_metadata=SimpleNamespace(
            prompt_token_count=4, candidates_token_count=2
        ),
    )

    class FakeModels:
        def __init__(self) -> None:
            self.arguments = None

        def generate_content(self, **kwargs):
            self.arguments = kwargs
            return response

    fake_models = FakeModels()
    fake_client = SimpleNamespace(models=fake_models)
    provider = GeminiProvider(
        GeminiSettings("test-only-key", "configured-model"), client=fake_client
    )

    result = provider.generate("hello", {"type": "object"})

    assert fake_models.arguments["model"] == "configured-model"
    assert fake_models.arguments["config"]["response_mime_type"] == "application/json"
    assert result.text == '{"status":"ok"}'
    assert result.input_tokens == 4
    assert result.output_tokens == 2
