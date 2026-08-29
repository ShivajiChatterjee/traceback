"""Plausible offline LLM fixtures; production workflows never import this module."""

from typing import Any


CASE_CAUSES = {
    "I01": ("retriever_top_k_regression", "retriever_top_k", 8),
    "I02": ("embedding_regression", "embedding_profile", "aligned_v1"),
    "I03": ("prompt_regression", "prompt_profile", "stable_v1"),
    "I04": ("stale_index", "index_profile", "current_v2"),
    "I05": ("reranker_disabled", "reranker_enabled", True),
    "I06": ("guardrail_regression", "guardrail_profile", "balanced"),
    "I07": ("tool_latency_regression", "tool_latency_profile", "healthy"),
    "I08": ("context_truncation", "context_profile", "standard"),
    "I09": ("no_incident", None, None),
    "I10": ("retriever_top_k_regression", "retriever_top_k", 8),
}


def baseline_response(incident_id: str) -> dict[str, Any]:
    category = CASE_CAUSES[incident_id][0]
    return {
        "predicted_root_cause": category,
        "confidence": 0.70,
        "summary": "Scripted offline prediction from visible telemetry.",
        "supporting_evidence": ["The visible metrics and configuration changed together."],
    }


def investigator_response(incident_id: str) -> dict[str, Any]:
    if incident_id == "I09":
        raise ValueError("I09 must bypass hypothesis generation")
    category, field, before_value = CASE_CAUSES[incident_id]
    if incident_id == "I10":
        hypotheses = [
            hypothesis(
                "prompt_regression", "prompt_profile", "stable_v1", confidence=0.78
            ),
            hypothesis(category, field, before_value, confidence=0.62),
        ]
    else:
        hypotheses = [
            hypothesis(category, field, before_value, confidence=0.72),
            hypothesis("model_regression", None, None, confidence=0.28),
        ]
    return {"hypotheses": hypotheses}


def hypothesis(
    category: str,
    field: str | None,
    value: Any,
    confidence: float,
) -> dict[str, Any]:
    return {
        "root_cause": category,
        "prior_confidence": confidence,
        "supporting_evidence": ["Visible telemetry is consistent with this mechanism."],
        "contradicting_evidence": ["Temporal association requires controlled replay."],
        "proposed_intervention": (
            {"field": field, "value": value} if field is not None else None
        ),
    }

