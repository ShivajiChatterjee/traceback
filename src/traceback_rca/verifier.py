"""Deterministic metric-aware replay support and falsification classification."""

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Tuple

from traceback_rca.detector import MetricGroup
from traceback_rca.models import Incident
from traceback_rca.planner import MAX_EXPERIMENTS, PlannedExperiment
from traceback_rca.replay import ControlledReplayEngine, ReplayResult
from traceback_rca.taxonomy import RootCauseCategory


class VerificationOutcome(str, Enum):
    SUPPORTED = "supported_by_replay"
    FAILS_TO_SUPPORT = "fails_to_support"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class MetricCriterion:
    metric: str
    higher_is_better: bool
    material_recovery: float
    non_material_max: float


@dataclass(frozen=True, slots=True)
class VerificationRule:
    metric_group: MetricGroup
    criteria: Tuple[MetricCriterion, ...]


_RULES: Mapping[RootCauseCategory, VerificationRule] = {
    RootCauseCategory.RETRIEVER_TOP_K_REGRESSION: VerificationRule(
        MetricGroup.QUALITY,
        (
            MetricCriterion("retrieval_relevance", True, 0.15, 0.05),
            MetricCriterion("aggregate_quality", True, 0.15, 0.05),
        ),
    ),
    RootCauseCategory.EMBEDDING_REGRESSION: VerificationRule(
        MetricGroup.QUALITY,
        (
            MetricCriterion("retrieval_relevance", True, 0.15, 0.05),
            MetricCriterion("answer_quality", True, 0.10, 0.05),
        ),
    ),
    RootCauseCategory.PROMPT_REGRESSION: VerificationRule(
        MetricGroup.QUALITY,
        (
            MetricCriterion("answer_quality", True, 0.15, 0.05),
            MetricCriterion("groundedness", True, 0.15, 0.05),
        ),
    ),
    RootCauseCategory.STALE_INDEX: VerificationRule(
        MetricGroup.FRESHNESS,
        (
            MetricCriterion("fresh_evidence_rate", True, 0.15, 0.05),
            MetricCriterion("answer_quality", True, 0.10, 0.05),
        ),
    ),
    RootCauseCategory.RERANKER_DISABLED: VerificationRule(
        MetricGroup.QUALITY,
        (
            MetricCriterion("retrieval_relevance", True, 0.15, 0.05),
            MetricCriterion("answer_quality", True, 0.10, 0.05),
        ),
    ),
    RootCauseCategory.GUARDRAIL_REGRESSION: VerificationRule(
        MetricGroup.GUARDRAIL,
        (
            MetricCriterion("guardrail_rejection_rate", False, 0.15, 0.05),
            MetricCriterion("usable_answer_rate", True, 0.15, 0.05),
        ),
    ),
    RootCauseCategory.TOOL_LATENCY_REGRESSION: VerificationRule(
        MetricGroup.PERFORMANCE,
        (
            MetricCriterion("latency_ms", False, 200.0, 50.0),
            MetricCriterion("tool_latency_ms", False, 200.0, 50.0),
        ),
    ),
    RootCauseCategory.CONTEXT_TRUNCATION: VerificationRule(
        MetricGroup.CONTEXT,
        (
            MetricCriterion("context_inclusion_rate", True, 0.15, 0.05),
            MetricCriterion("answer_quality", True, 0.10, 0.05),
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class VerificationResult:
    experiment: PlannedExperiment
    replay: ReplayResult
    quality_delta: float
    relevant_metric_deltas: Mapping[str, float]
    decision_thresholds: Mapping[str, float]
    metric_group: MetricGroup
    support_score: float
    outcome: VerificationOutcome
    rationale: str


class ReplayVerifier:
    """Execute validated local replays and classify category-relevant recovery."""

    def __init__(self, replay_engine: ControlledReplayEngine | None = None) -> None:
        self._replay_engine = replay_engine or ControlledReplayEngine()

    def verify(
        self, incident: Incident, experiments: Tuple[PlannedExperiment, ...]
    ) -> Tuple[VerificationResult, ...]:
        if len(experiments) > MAX_EXPERIMENTS:
            raise ValueError(f"cannot run more than {MAX_EXPERIMENTS} experiments")
        return tuple(self._run_one(incident, experiment) for experiment in experiments)

    def _run_one(
        self, incident: Incident, experiment: PlannedExperiment
    ) -> VerificationResult:
        category = experiment.hypothesis.root_cause
        try:
            rule = _RULES[category]
        except KeyError as error:
            raise ValueError(
                f"no deterministic verification rule for {category.value}"
            ) from error

        replay_result = self._replay_engine.replay(
            incident, experiment.intervention.to_dict()
        )
        recoveries = {
            criterion.metric: _recovery_for(criterion, replay_result)
            for criterion in rule.criteria
        }
        thresholds = {
            criterion.metric: criterion.material_recovery
            for criterion in rule.criteria
        }
        if all(
            recoveries[criterion.metric] >= criterion.material_recovery
            for criterion in rule.criteria
        ):
            outcome = VerificationOutcome.SUPPORTED
        elif all(
            recoveries[criterion.metric] <= criterion.non_material_max
            for criterion in rule.criteria
        ):
            outcome = VerificationOutcome.FAILS_TO_SUPPORT
        else:
            outcome = VerificationOutcome.INCONCLUSIVE

        support_score = round(
            sum(
                min(
                    max(recoveries[criterion.metric], 0.0)
                    / criterion.material_recovery,
                    1.0,
                )
                for criterion in rule.criteria
            )
            / len(rule.criteria),
            4,
        )
        recovery_text = ", ".join(
            f"{criterion.metric}={recoveries[criterion.metric]:+.4f} "
            f"(threshold {criterion.material_recovery:.4f})"
            for criterion in rule.criteria
        )
        rationale = f"{rule.metric_group.value} replay recovery: {recovery_text}"
        return VerificationResult(
            experiment=experiment,
            replay=replay_result,
            quality_delta=replay_result.delta_from_degraded.aggregate_quality,
            relevant_metric_deltas=recoveries,
            decision_thresholds=thresholds,
            metric_group=rule.metric_group,
            support_score=support_score,
            outcome=outcome,
            rationale=rationale,
        )


def verification_rule(category: RootCauseCategory) -> VerificationRule:
    """Expose documented deterministic criteria for tests and reporting."""

    return _RULES[category]


def _recovery_for(
    criterion: MetricCriterion, replay_result: ReplayResult
) -> float:
    degraded = getattr(replay_result.degraded_metrics, criterion.metric)
    replayed = getattr(replay_result.replay_metrics, criterion.metric)
    recovery = replayed - degraded if criterion.higher_is_better else degraded - replayed
    precision = 2 if criterion.metric.endswith("latency_ms") else 4
    return round(recovery, precision)

