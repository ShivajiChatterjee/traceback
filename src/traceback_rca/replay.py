"""Evidence-only controlled counterfactual replay infrastructure."""

from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Mapping

from traceback_rca.evaluator import Evaluator, QualityMetrics
from traceback_rca.models import (
    ConfigurationValue,
    ExperimentStatus,
    Incident,
    SystemConfiguration,
)


_INTERVENTION_ALIASES = {"top_k": "retriever_top_k"}


@dataclass(frozen=True, slots=True)
class MetricDelta:
    """Signed replay-minus-reference deltas for every preserved aggregate metric."""

    retrieval_relevance: float
    groundedness: float
    answer_quality: float
    aggregate_quality: float
    latency_ms: float
    tool_latency_ms: float
    guardrail_rejection_rate: float
    usable_answer_rate: float
    context_inclusion_rate: float
    fresh_evidence_rate: float
    retrieved_evidence_count: float
    included_context_count: float
    truncated_context_count: float
    blocked_answer_count: float
    allowed_answer_count: float
    stale_document_count: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def between(cls, result: QualityMetrics, reference: QualityMetrics) -> "MetricDelta":
        values = {}
        for metric_field in fields(cls):
            name = metric_field.name
            precision = 2 if name.endswith("latency_ms") else 4
            values[name] = round(getattr(result, name) - getattr(reference, name), precision)
        return cls(**values)

    @classmethod
    def progress_toward(
        cls,
        healthy: QualityMetrics,
        degraded: QualityMetrics,
        replayed: QualityMetrics,
    ) -> "MetricDelta":
        degraded_gap = cls.between(degraded, healthy)
        replay_gap = cls.between(replayed, healthy)
        values = {}
        for metric_field in fields(cls):
            name = metric_field.name
            precision = 2 if name.endswith("latency_ms") else 4
            values[name] = round(
                abs(getattr(degraded_gap, name)) - abs(getattr(replay_gap, name)),
                precision,
            )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """Counterfactual measurements and configurations, with no causal verdict."""

    incident_id: str
    intervention: Mapping[str, ConfigurationValue]
    before_configuration: SystemConfiguration
    degraded_configuration: SystemConfiguration
    replay_configuration: SystemConfiguration
    before_values: Mapping[str, ConfigurationValue]
    degraded_values: Mapping[str, ConfigurationValue]
    replay_values: Mapping[str, ConfigurationValue]
    healthy_metrics: QualityMetrics
    degraded_metrics: QualityMetrics
    replay_metrics: QualityMetrics
    delta_from_degraded: MetricDelta
    delta_toward_healthy: MetricDelta
    experiment_status: ExperimentStatus = ExperimentStatus.COMPLETED

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "intervention": dict(self.intervention),
            "before_configuration": self.before_configuration.to_dict(),
            "degraded_configuration": self.degraded_configuration.to_dict(),
            "replay_configuration": self.replay_configuration.to_dict(),
            "before_values": dict(self.before_values),
            "degraded_values": dict(self.degraded_values),
            "replay_values": dict(self.replay_values),
            "healthy_metrics": self.healthy_metrics.to_dict(),
            "degraded_metrics": self.degraded_metrics.to_dict(),
            "replay_metrics": self.replay_metrics.to_dict(),
            "delta_from_degraded": self.delta_from_degraded.to_dict(),
            "delta_toward_healthy": self.delta_toward_healthy.to_dict(),
            "experiment_status": self.experiment_status.value,
        }


class ControlledReplayEngine:
    """Apply explicit interventions to an incident's after-configuration and evaluate."""

    def __init__(self, evaluator: Evaluator | None = None) -> None:
        self._evaluator = evaluator or Evaluator()

    def replay(
        self,
        incident: Incident,
        intervention: Mapping[str, ConfigurationValue],
    ) -> ReplayResult:
        if not intervention:
            raise ValueError("intervention cannot be empty")

        valid_fields = {
            configuration_field.name for configuration_field in fields(SystemConfiguration)
        }
        normalized: dict[str, ConfigurationValue] = {}
        for requested_field, value in intervention.items():
            field_name = _INTERVENTION_ALIASES.get(requested_field, requested_field)
            if field_name not in valid_fields:
                raise ValueError(f"unknown configuration field: {requested_field}")
            if field_name in normalized:
                raise ValueError(f"duplicate intervention for field: {field_name}")
            normalized[field_name] = value

        replay_configuration = replace(incident.configuration_after, **normalized)
        healthy = self._evaluator.evaluate(incident.configuration_before).metrics
        degraded = self._evaluator.evaluate(incident.configuration_after).metrics
        replayed = self._evaluator.evaluate(replay_configuration).metrics
        before_values = {
            name: getattr(incident.configuration_before, name) for name in normalized
        }
        degraded_values = {
            name: getattr(incident.configuration_after, name) for name in normalized
        }
        replay_values = {name: getattr(replay_configuration, name) for name in normalized}

        return ReplayResult(
            incident_id=incident.incident_id,
            intervention=normalized,
            before_configuration=incident.configuration_before,
            degraded_configuration=incident.configuration_after,
            replay_configuration=replay_configuration,
            before_values=before_values,
            degraded_values=degraded_values,
            replay_values=replay_values,
            healthy_metrics=healthy,
            degraded_metrics=degraded,
            replay_metrics=replayed,
            delta_from_degraded=MetricDelta.between(replayed, degraded),
            delta_toward_healthy=MetricDelta.progress_toward(
                healthy, degraded, replayed
            ),
        )


def replay(
    incident: Incident, intervention: Mapping[str, ConfigurationValue]
) -> ReplayResult:
    """Replay one intervention using the default deterministic evaluator."""

    return ControlledReplayEngine().replay(incident, intervention)

