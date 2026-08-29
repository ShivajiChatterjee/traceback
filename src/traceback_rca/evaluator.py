"""Aggregation of deterministic RAG testbed runs."""

from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Any, Tuple

from traceback_rca.models import SystemConfiguration
from traceback_rca.testbed import QueryEvaluation, SyntheticRAGTestbed


MATERIAL_QUALITY_IMPROVEMENT = 0.15


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    """Aggregate deterministic quality and latency metrics."""

    retrieval_relevance: float
    groundedness: float
    answer_quality: float
    aggregate_quality: float
    latency_ms: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Aggregate metrics plus per-query evidence for one configuration."""

    configuration: SystemConfiguration
    metrics: QualityMetrics
    query_results: Tuple[QueryEvaluation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration": self.configuration.to_dict(),
            "metrics": self.metrics.to_dict(),
            "query_results": [result.to_dict() for result in self.query_results],
        }


class Evaluator:
    """Run and aggregate a SyntheticRAGTestbed without probabilistic components."""

    def __init__(self, testbed: SyntheticRAGTestbed | None = None) -> None:
        self._testbed = testbed or SyntheticRAGTestbed()

    def evaluate(self, configuration: SystemConfiguration) -> EvaluationResult:
        query_results = self._testbed.run(configuration)
        retrieval_relevance = fmean(
            result.retrieval_relevance for result in query_results
        )
        groundedness = fmean(result.groundedness for result in query_results)
        answer_quality = fmean(result.answer_quality for result in query_results)
        latency_ms = fmean(result.latency_ms for result in query_results)

        # Retrieval and groundedness each contribute 30%; answer completeness 40%.
        aggregate_quality = (
            0.30 * retrieval_relevance
            + 0.30 * groundedness
            + 0.40 * answer_quality
        )
        metrics = QualityMetrics(
            retrieval_relevance=round(retrieval_relevance, 4),
            groundedness=round(groundedness, 4),
            answer_quality=round(answer_quality, 4),
            aggregate_quality=round(aggregate_quality, 4),
            latency_ms=round(latency_ms, 2),
        )
        return EvaluationResult(configuration, metrics, query_results)


def evaluate_configuration(configuration: SystemConfiguration) -> EvaluationResult:
    """Convenience entry point for the default fixed workload."""

    return Evaluator().evaluate(configuration)

