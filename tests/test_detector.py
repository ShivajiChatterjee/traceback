"""Deterministic multi-metric incident detector tests."""

import pytest

from traceback_rca.detector import (
    AGGREGATE_QUALITY_DROP_THRESHOLD,
    CONTEXT_RATE_DROP_THRESHOLD,
    FRESHNESS_RATE_DROP_THRESHOLD,
    GUARDRAIL_RATE_CHANGE_THRESHOLD,
    LATENCY_ABSOLUTE_INCREASE_MS,
    IncidentDetector,
    MetricGroup,
)
from traceback_rca.incidents import get_incident


@pytest.mark.parametrize(
    "incident_id", ("I01", "I02", "I03", "I04", "I05", "I06", "I07", "I08", "I10")
)
def test_detector_finds_every_true_incident(incident_id: str) -> None:
    detection = IncidentDetector().detect(get_incident(incident_id))

    assert detection.material_incident
    assert detection.affected_metrics


@pytest.mark.parametrize(
    ("incident_id", "metric_group"),
    [
        ("I02", MetricGroup.QUALITY),
        ("I04", MetricGroup.FRESHNESS),
        ("I06", MetricGroup.GUARDRAIL),
        ("I07", MetricGroup.PERFORMANCE),
        ("I08", MetricGroup.CONTEXT),
    ],
)
def test_detector_preserves_degradation_type(
    incident_id: str, metric_group: MetricGroup
) -> None:
    assert metric_group in IncidentDetector().detect(
        get_incident(incident_id)
    ).degradation_types


def test_latency_incident_does_not_require_quality_regression() -> None:
    incident = get_incident("I07")
    detection = IncidentDetector().detect(incident)

    assert incident.telemetry_after.aggregate_quality == incident.telemetry_before.aggregate_quality
    assert (
        incident.telemetry_after.p95_latency_ms
        - incident.telemetry_before.p95_latency_ms
        >= LATENCY_ABSOLUTE_INCREASE_MS
    )
    assert detection.material_incident


def test_healthy_canary_stays_below_every_relevant_threshold() -> None:
    incident = get_incident("I09")
    detection = IncidentDetector().detect(incident)

    assert (
        incident.telemetry_before.aggregate_quality
        - incident.telemetry_after.aggregate_quality
        < AGGREGATE_QUALITY_DROP_THRESHOLD
    )
    assert incident.telemetry_after.guardrail_block_rate < GUARDRAIL_RATE_CHANGE_THRESHOLD
    assert (
        incident.telemetry_before.context_inclusion_rate
        - incident.telemetry_after.context_inclusion_rate
        < CONTEXT_RATE_DROP_THRESHOLD
    )
    assert (
        incident.telemetry_before.fresh_evidence_rate
        - incident.telemetry_after.fresh_evidence_rate
        < FRESHNESS_RATE_DROP_THRESHOLD
    )
    assert not detection.material_incident
    assert detection.affected_metrics == ()

