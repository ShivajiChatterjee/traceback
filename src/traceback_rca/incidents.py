"""The ten deterministic investigator-visible benchmark incidents."""

from dataclasses import replace
from datetime import datetime, timezone
from typing import Dict, Tuple

from traceback_rca.evaluator import evaluate_configuration
from traceback_rca.models import (
    ConfigurationChange,
    Incident,
    SystemConfiguration,
    TelemetrySnapshot,
)


def _at(day: int, hour: int) -> datetime:
    return datetime(2026, 8, day, hour, tzinfo=timezone.utc)


def _telemetry(
    captured_at: datetime, configuration: SystemConfiguration
) -> TelemetrySnapshot:
    evaluation = evaluate_configuration(configuration)
    metrics = evaluation.metrics
    return TelemetrySnapshot(
        captured_at=captured_at,
        answer_quality=metrics.answer_quality,
        retrieval_relevance=metrics.retrieval_relevance,
        groundedness=metrics.groundedness,
        aggregate_quality=metrics.aggregate_quality,
        p95_latency_ms=metrics.latency_ms,
        tool_latency_ms=metrics.tool_latency_ms,
        guardrail_block_rate=metrics.guardrail_rejection_rate,
        usable_answer_rate=metrics.usable_answer_rate,
        context_inclusion_rate=metrics.context_inclusion_rate,
        fresh_evidence_rate=metrics.fresh_evidence_rate,
        retrieved_evidence_count=metrics.retrieved_evidence_count,
        included_context_count=metrics.included_context_count,
        truncated_context_count=metrics.truncated_context_count,
        blocked_answer_count=metrics.blocked_answer_count,
        stale_document_count=metrics.stale_document_count,
        sample_size=len(evaluation.query_results),
    )


def _change(
    field_name: str,
    before,
    after,
    day: int,
    deployment_id: str,
) -> ConfigurationChange:
    return ConfigurationChange(
        field_name=field_name,
        before=before,
        after=after,
        changed_at=_at(day, 10),
        deployment_id=deployment_id,
    )


def _incident(
    incident_id: str,
    day: int,
    title: str,
    description: str,
    before: SystemConfiguration,
    after: SystemConfiguration,
    changes: Tuple[ConfigurationChange, ...],
) -> Incident:
    return Incident(
        incident_id=incident_id,
        title=title,
        description=description,
        detected_at=_at(day, 11),
        telemetry_before=_telemetry(_at(day, 9), before),
        telemetry_after=_telemetry(_at(day, 11), after),
        configuration_before=before,
        configuration_after=after,
        changes=changes,
    )


_HEALTHY = SystemConfiguration(prompt_profile="stable_v1", retriever_top_k=8)

_I01_AFTER = replace(_HEALTHY, retriever_top_k=2)
_I01 = _incident(
    "I01",
    11,
    "Retrieval and answer quality decline after deployment",
    "The service remains available, but retrieval relevance and answer quality drop.",
    _HEALTHY,
    _I01_AFTER,
    (_change("retriever_top_k", 8, 2, 11, "dep-i01-042"),),
)

_I02_AFTER = replace(_HEALTHY, embedding_profile="mismatched_v2")
_I02 = _incident(
    "I02",
    12,
    "Retrieval ranking quality declines after embedding rollout",
    (
        "The prompt and top-k are unchanged, while several gold evidence documents "
        "move below the usable retrieval window after the embedding profile rollout."
    ),
    _HEALTHY,
    _I02_AFTER,
    (
        _change(
            "embedding_profile",
            "aligned_v1",
            "mismatched_v2",
            12,
            "dep-i02-031",
        ),
    ),
)

_I03_AFTER = replace(_HEALTHY, prompt_profile="regressed")
_I03 = _incident(
    "I03",
    13,
    "Answer quality decline after prompt deployment",
    (
        "Retrieval remains healthy, while groundedness and answer completeness decline "
        "after the prompt profile change."
    ),
    _HEALTHY,
    _I03_AFTER,
    (
        _change(
            "prompt_profile", "stable_v1", "regressed", 13, "dep-i03-017"
        ),
    ),
)

_I04_AFTER = replace(_HEALTHY, index_profile="stale_v1")
_I04 = _incident(
    "I04",
    14,
    "Current evidence missing after index deployment",
    (
        "Retrieval still returns documents, but freshness telemetry reports stale "
        "documents and missing current evidence after an older index was activated."
    ),
    _HEALTHY,
    _I04_AFTER,
    (
        _change(
            "index_profile", "current_v2", "stale_v1", 14, "dep-i04-055"
        ),
    ),
)

_I05_AFTER = replace(_HEALTHY, reranker_enabled=False)
_I05 = _incident(
    "I05",
    15,
    "Moderate retrieval degradation after reranking change",
    (
        "Initial candidates are still produced, but post-rerank evidence positions "
        "worsen for part of the workload after reranking is disabled."
    ),
    _HEALTHY,
    _I05_AFTER,
    (_change("reranker_enabled", True, False, 15, "dep-i05-064"),),
)

_I06_AFTER = replace(_HEALTHY, guardrail_profile="over_strict")
_I06 = _incident(
    "I06",
    16,
    "Valid grounded answers rejected after guardrail update",
    (
        "Retrieval and evidence generation remain healthy, while blocked-answer rate "
        "increases and usable-answer rate declines."
    ),
    _HEALTHY,
    _I06_AFTER,
    (
        _change(
            "guardrail_profile",
            "balanced",
            "over_strict",
            16,
            "dep-i06-023",
        ),
    ),
)

_I07_AFTER = replace(_HEALTHY, tool_latency_profile="slow")
_I07 = _incident(
    "I07",
    17,
    "Tool latency breaches service objective",
    (
        "Answer quality remains stable, but external-tool and end-to-end latency rise "
        "well beyond the normal service range."
    ),
    _HEALTHY,
    _I07_AFTER,
    (
        _change(
            "tool_latency_profile", "healthy", "slow", 17, "dep-i07-088"
        ),
    ),
)

_I08_AFTER = replace(_HEALTHY, context_profile="truncated")
_I08 = _incident(
    "I08",
    18,
    "Retrieved evidence omitted during context construction",
    (
        "Retrieval relevance stays high, but included-context count falls and retrieved "
        "evidence is truncated before answer generation."
    ),
    _HEALTHY,
    _I08_AFTER,
    (
        _change(
            "context_profile", "standard", "truncated", 18, "dep-i08-071"
        ),
    ),
)

_I09_AFTER = replace(_HEALTHY, prompt_profile="stable_canary")
_I09 = _incident(
    "I09",
    19,
    "Routine canary revision with metrics inside healthy tolerance",
    (
        "A prompt canary revision produces small normal answer variation while core "
        "quality, retrieval, guardrail, context, and latency remain within tolerance."
    ),
    _HEALTHY,
    _I09_AFTER,
    (
        _change(
            "prompt_profile", "stable_v1", "stable_canary", 19, "dep-i09-012"
        ),
    ),
)

_I10_AFTER = replace(
    _HEALTHY, prompt_profile="stable_v1_1", retriever_top_k=2
)
_I10 = _incident(
    "I10",
    20,
    "Quality decline following two nearby configuration changes",
    (
        "A neutral prompt revision and a retrieval-depth change ship together. "
        "Retrieval, groundedness, and answer quality then decline."
    ),
    _HEALTHY,
    _I10_AFTER,
    (
        _change(
            "prompt_profile", "stable_v1", "stable_v1_1", 20, "dep-i10-099"
        ),
        _change("retriever_top_k", 8, 2, 20, "dep-i10-099"),
    ),
)


_INCIDENTS: Dict[str, Incident] = {
    incident.incident_id: incident
    for incident in (_I01, _I02, _I03, _I04, _I05, _I06, _I07, _I08, _I09, _I10)
}


def list_incidents() -> Tuple[Incident, ...]:
    """Return the complete ten-case investigator-visible dataset."""

    return tuple(_INCIDENTS.values())


def get_incident(incident_id: str) -> Incident:
    """Return investigator-visible data for one incident."""

    return _INCIDENTS[incident_id]


def display_incident_id(incident_id: str) -> str:
    """Convert an internal I01 identifier to its human-facing INC-01 form."""

    if incident_id.startswith("I") and incident_id[1:].isdigit():
        return f"INC-{int(incident_id[1:]):02d}"
    return incident_id
