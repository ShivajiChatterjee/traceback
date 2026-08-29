"""Deterministic multi-metric incident detection from visible telemetry."""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Tuple

from traceback_rca.models import Incident, TelemetrySnapshot


AGGREGATE_QUALITY_DROP_THRESHOLD = 0.10
QUALITY_COMPONENT_DROP_THRESHOLD = 0.15
GUARDRAIL_RATE_CHANGE_THRESHOLD = 0.15
CONTEXT_RATE_DROP_THRESHOLD = 0.15
FRESHNESS_RATE_DROP_THRESHOLD = 0.15
LATENCY_ABSOLUTE_INCREASE_MS = 200.0
LATENCY_RELATIVE_INCREASE = 0.30


class MetricGroup(str, Enum):
    QUALITY = "quality"
    GUARDRAIL = "guardrail"
    PERFORMANCE = "performance"
    CONTEXT = "context"
    FRESHNESS = "freshness"


class IncidentSeverity(str, Enum):
    NONE = "none"
    MODERATE = "moderate"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class IncidentDetection:
    material_incident: bool
    affected_metrics: Tuple[str, ...]
    degradation_types: Tuple[MetricGroup, ...]
    severity: IncidentSeverity
    evidence: Tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["degradation_types"] = [group.value for group in self.degradation_types]
        result["severity"] = self.severity.value
        return result


class IncidentDetector:
    """Detect material regressions without LLM reasoning or hidden annotations."""

    def detect(self, incident: Incident) -> IncidentDetection:
        before = incident.telemetry_before
        after = incident.telemetry_after
        affected: list[str] = []
        groups: list[MetricGroup] = []
        evidence: list[str] = []

        aggregate_drop = before.aggregate_quality - after.aggregate_quality
        if aggregate_drop >= AGGREGATE_QUALITY_DROP_THRESHOLD:
            affected.append("aggregate_quality")
            _append_once(groups, MetricGroup.QUALITY)
            evidence.append(f"aggregate_quality dropped by {aggregate_drop:.4f}")

        for name in ("retrieval_relevance", "groundedness", "answer_quality"):
            drop = getattr(before, name) - getattr(after, name)
            if drop >= QUALITY_COMPONENT_DROP_THRESHOLD:
                affected.append(name)
                _append_once(groups, MetricGroup.QUALITY)
                evidence.append(f"{name} dropped by {drop:.4f}")

        guardrail_increase = after.guardrail_block_rate - before.guardrail_block_rate
        usable_drop = before.usable_answer_rate - after.usable_answer_rate
        if guardrail_increase >= GUARDRAIL_RATE_CHANGE_THRESHOLD:
            affected.append("guardrail_block_rate")
            _append_once(groups, MetricGroup.GUARDRAIL)
            evidence.append(
                f"guardrail_block_rate increased by {guardrail_increase:.4f}"
            )
        if usable_drop >= GUARDRAIL_RATE_CHANGE_THRESHOLD:
            affected.append("usable_answer_rate")
            _append_once(groups, MetricGroup.GUARDRAIL)
            evidence.append(f"usable_answer_rate dropped by {usable_drop:.4f}")

        context_drop = before.context_inclusion_rate - after.context_inclusion_rate
        if context_drop >= CONTEXT_RATE_DROP_THRESHOLD:
            affected.append("context_inclusion_rate")
            _append_once(groups, MetricGroup.CONTEXT)
            evidence.append(f"context_inclusion_rate dropped by {context_drop:.4f}")

        freshness_drop = before.fresh_evidence_rate - after.fresh_evidence_rate
        if freshness_drop >= FRESHNESS_RATE_DROP_THRESHOLD:
            affected.append("fresh_evidence_rate")
            _append_once(groups, MetricGroup.FRESHNESS)
            evidence.append(f"fresh_evidence_rate dropped by {freshness_drop:.4f}")

        latency_increase = after.p95_latency_ms - before.p95_latency_ms
        relative_latency_increase = (
            latency_increase / before.p95_latency_ms
            if before.p95_latency_ms > 0
            else float("inf")
        )
        if (
            latency_increase >= LATENCY_ABSOLUTE_INCREASE_MS
            and relative_latency_increase >= LATENCY_RELATIVE_INCREASE
        ):
            affected.extend(("p95_latency_ms", "tool_latency_ms"))
            _append_once(groups, MetricGroup.PERFORMANCE)
            evidence.append(
                f"p95_latency_ms increased by {latency_increase:.2f} ms "
                f"({relative_latency_increase:.1%})"
            )

        material_incident = bool(groups)
        high_severity = aggregate_drop >= 0.40 or relative_latency_increase >= 1.0
        severity = (
            IncidentSeverity.HIGH
            if material_incident and high_severity
            else IncidentSeverity.MODERATE
            if material_incident
            else IncidentSeverity.NONE
        )
        if not material_incident:
            evidence.append("all metric changes remain inside deterministic tolerances")
        return IncidentDetection(
            material_incident=material_incident,
            affected_metrics=tuple(dict.fromkeys(affected)),
            degradation_types=tuple(groups),
            severity=severity,
            evidence=tuple(evidence),
        )


def _append_once(values: list[MetricGroup], value: MetricGroup) -> None:
    if value not in values:
        values.append(value)

