"""Aggregation of deterministic RAG testbed evidence."""

from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Any, Tuple

from traceback_rca.models import SystemConfiguration
from traceback_rca.testbed import QueryEvaluation, SyntheticRAGTestbed


MATERIAL_QUALITY_IMPROVEMENT = 0.15


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    """Quality, freshness, context, guardrail, and performance measurements."""

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
    retrieved_evidence_count: int
    included_context_count: int
    truncated_context_count: int
    blocked_answer_count: int
    allowed_answer_count: int
    stale_document_count: int

    def to_dict(self) -> dict[str, float | int]:
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
        sample_size = len(query_results)
        retrieval_relevance = fmean(
            result.retrieval_relevance for result in query_results
        )
        groundedness = fmean(result.groundedness for result in query_results)
        answer_quality = fmean(result.answer_quality for result in query_results)
        latency_ms = fmean(result.latency_ms for result in query_results)
        tool_latency_ms = fmean(result.tool_latency_ms for result in query_results)
        blocked_answer_count = sum(result.guardrail_blocked for result in query_results)
        retrieved_evidence_count = sum(
            result.gold_document_retrieved for result in query_results
        )
        included_context_count = sum(
            result.context_evidence_included for result in query_results
        )
        truncated_context_count = sum(
            result.context_truncated for result in query_results
        )
        stale_document_count = sum(
            not result.fresh_evidence_available for result in query_results
        )
        usable_answer_rate = fmean(
            float(result.usable_answer) for result in query_results
        )
        context_inclusion_rate = fmean(
            result.context_inclusion for result in query_results
        )
        fresh_evidence_rate = fmean(
            float(result.fresh_evidence_available) for result in query_results
        )

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
            tool_latency_ms=round(tool_latency_ms, 2),
            guardrail_rejection_rate=round(blocked_answer_count / sample_size, 4),
            usable_answer_rate=round(usable_answer_rate, 4),
            context_inclusion_rate=round(context_inclusion_rate, 4),
            fresh_evidence_rate=round(fresh_evidence_rate, 4),
            retrieved_evidence_count=retrieved_evidence_count,
            included_context_count=included_context_count,
            truncated_context_count=truncated_context_count,
            blocked_answer_count=blocked_answer_count,
            allowed_answer_count=sample_size - blocked_answer_count,
            stale_document_count=stale_document_count,
        )
        return EvaluationResult(configuration, metrics, query_results)


def evaluate_configuration(configuration: SystemConfiguration) -> EvaluationResult:
    """Convenience entry point for the default fixed workload."""

    return Evaluator().evaluate(configuration)

