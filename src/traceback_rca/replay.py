"""Evidence-only controlled counterfactual replay infrastructure."""

from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Mapping

from traceback_rca.evaluator import Evaluator, QualityMetrics
from traceback_rca.models import ConfigurationValue, Incident, SystemConfiguration


_INTERVENTION_ALIASES = {"top_k": "retriever_top_k"}


@dataclass(frozen=True, slots=True)
class MetricDelta:
    """Signed changes for quality metrics and latency."""

    retrieval_relevance: float
    groundedness: float
    answer_quality: float
    aggregate_quality: float
    latency_ms: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def between(cls, result: QualityMetrics, reference: QualityMetrics) -> "MetricDelta":
        return cls(
            retrieval_relevance=round(
                result.retrieval_relevance - reference.retrieval_relevance, 4
            ),
            groundedness=round(result.groundedness - reference.groundedness, 4),
            answer_quality=round(result.answer_quality - reference.answer_quality, 4),
            aggregate_quality=round(
                result.aggregate_quality - reference.aggregate_quality, 4
            ),
            latency_ms=round(result.latency_ms - reference.latency_ms, 2),
        )


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """Counterfactual measurements, with no causal verdict attached."""

    incident_id: str
    intervention: Mapping[str, ConfigurationValue]
    replay_configuration: SystemConfiguration
    healthy_metrics: QualityMetrics
    degraded_metrics: QualityMetrics
    replay_metrics: QualityMetrics
    delta_from_degraded: MetricDelta
    delta_toward_healthy: MetricDelta

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "intervention": dict(self.intervention),
            "replay_configuration": self.replay_configuration.to_dict(),
            "healthy_metrics": self.healthy_metrics.to_dict(),
            "degraded_metrics": self.degraded_metrics.to_dict(),
            "replay_metrics": self.replay_metrics.to_dict(),
            "delta_from_degraded": self.delta_from_degraded.to_dict(),
            "delta_toward_healthy": self.delta_toward_healthy.to_dict(),
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

        valid_fields = {configuration_field.name for configuration_field in fields(SystemConfiguration)}
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

        degraded_gap = MetricDelta.between(degraded, healthy)
        replay_gap = MetricDelta.between(replayed, healthy)
        delta_toward_healthy = MetricDelta(
            retrieval_relevance=round(
                abs(degraded_gap.retrieval_relevance)
                - abs(replay_gap.retrieval_relevance),
                4,
            ),
            groundedness=round(
                abs(degraded_gap.groundedness) - abs(replay_gap.groundedness), 4
            ),
            answer_quality=round(
                abs(degraded_gap.answer_quality) - abs(replay_gap.answer_quality), 4
            ),
            aggregate_quality=round(
                abs(degraded_gap.aggregate_quality)
                - abs(replay_gap.aggregate_quality),
                4,
            ),
            latency_ms=round(
                abs(degraded_gap.latency_ms) - abs(replay_gap.latency_ms), 2
            ),
        )

        return ReplayResult(
            incident_id=incident.incident_id,
            intervention=normalized,
            replay_configuration=replay_configuration,
            healthy_metrics=healthy,
            degraded_metrics=degraded,
            replay_metrics=replayed,
            delta_from_degraded=MetricDelta.between(replayed, degraded),
            delta_toward_healthy=delta_toward_healthy,
        )


def replay(
    incident: Incident, intervention: Mapping[str, ConfigurationValue]
) -> ReplayResult:
    """Replay one intervention using the default deterministic evaluator."""

    return ControlledReplayEngine().replay(incident, intervention)

