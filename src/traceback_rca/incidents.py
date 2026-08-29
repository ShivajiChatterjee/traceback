"""The three deterministic synthetic incidents in milestone one."""

from datetime import datetime, timezone
from typing import Dict, Tuple

from traceback_rca.models import (
    ConfigurationChange,
    Incident,
    SystemConfiguration,
    TelemetrySnapshot,
)
from traceback_rca.evaluator import evaluate_configuration


def _at(day: int, hour: int) -> datetime:
    return datetime(2026, 8, day, hour, tzinfo=timezone.utc)


def _telemetry(
    captured_at: datetime,
    configuration: SystemConfiguration,
) -> TelemetrySnapshot:
    evaluation = evaluate_configuration(configuration)
    metrics = evaluation.metrics
    return TelemetrySnapshot(
        captured_at=captured_at,
        answer_quality=metrics.answer_quality,
        retrieval_relevance=metrics.retrieval_relevance,
        groundedness=metrics.groundedness,
        p95_latency_ms=metrics.latency_ms,
        guardrail_block_rate=0.0,
        sample_size=len(evaluation.query_results),
    )


_I01_BEFORE = SystemConfiguration(prompt_profile="stable_v1", retriever_top_k=8)
_I01_AFTER = SystemConfiguration(prompt_profile="stable_v1", retriever_top_k=2)
_I01 = Incident(
    incident_id="I01",
    title="Retrieval and answer quality decline after deployment",
    description=(
        "The service remains available, but retrieval relevance and answer quality "
        "drop after deployment dep-i01-042."
    ),
    detected_at=_at(11, 11),
    telemetry_before=_telemetry(_at(11, 9), _I01_BEFORE),
    telemetry_after=_telemetry(_at(11, 11), _I01_AFTER),
    configuration_before=_I01_BEFORE,
    configuration_after=_I01_AFTER,
    changes=(
        ConfigurationChange(
            field_name="retriever_top_k",
            before=8,
            after=2,
            changed_at=_at(11, 10),
            deployment_id="dep-i01-042",
        ),
    ),
)

_I03_BEFORE = SystemConfiguration(prompt_profile="stable_v1", retriever_top_k=8)
_I03_AFTER = SystemConfiguration(prompt_profile="regressed", retriever_top_k=8)
_I03 = Incident(
    incident_id="I03",
    title="Answer quality decline after prompt deployment",
    description=(
        "The service remains healthy at the transport layer, while groundedness and "
        "answer quality decline after deployment dep-i03-017."
    ),
    detected_at=_at(13, 15),
    telemetry_before=_telemetry(_at(13, 13), _I03_BEFORE),
    telemetry_after=_telemetry(_at(13, 15), _I03_AFTER),
    configuration_before=_I03_BEFORE,
    configuration_after=_I03_AFTER,
    changes=(
        ConfigurationChange(
            field_name="prompt_profile",
            before="stable_v1",
            after="regressed",
            changed_at=_at(13, 14),
            deployment_id="dep-i03-017",
        ),
    ),
)

_I10_BEFORE = SystemConfiguration(prompt_profile="stable_v1", retriever_top_k=8)
_I10_AFTER = SystemConfiguration(prompt_profile="stable_v1_1", retriever_top_k=2)
_I10 = Incident(
    incident_id="I10",
    title="Quality decline following two nearby configuration changes",
    description=(
        "Two configuration values changed in deployment dep-i10-099. Retrieval, "
        "groundedness, and answer quality then declined without availability errors."
    ),
    detected_at=_at(20, 18),
    telemetry_before=_telemetry(_at(20, 16), _I10_BEFORE),
    telemetry_after=_telemetry(_at(20, 18), _I10_AFTER),
    configuration_before=_I10_BEFORE,
    configuration_after=_I10_AFTER,
    changes=(
        ConfigurationChange(
            field_name="prompt_profile",
            before="stable_v1",
            after="stable_v1_1",
            changed_at=_at(20, 17),
            deployment_id="dep-i10-099",
        ),
        ConfigurationChange(
            field_name="retriever_top_k",
            before=8,
            after=2,
            changed_at=_at(20, 17),
            deployment_id="dep-i10-099",
        ),
    ),
)


_INCIDENTS: Dict[str, Incident] = {
    incident.incident_id: incident for incident in (_I01, _I03, _I10)
}

def list_incidents() -> Tuple[Incident, ...]:
    """Return the complete milestone-one investigator dataset."""

    return tuple(_INCIDENTS.values())


def get_incident(incident_id: str) -> Incident:
    """Return investigator-visible data for one incident."""

    return _INCIDENTS[incident_id]
