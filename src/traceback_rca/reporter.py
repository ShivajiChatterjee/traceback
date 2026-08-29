"""Deterministic evidence-backed RCA reporting with mandatory human approval."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Tuple

from traceback_rca.evaluator import QualityMetrics, evaluate_configuration
from traceback_rca.investigator import InvestigationHypothesis
from traceback_rca.models import Incident
from traceback_rca.planner import PlanningResult
from traceback_rca.taxonomy import RootCauseCategory
from traceback_rca.verifier import VerificationOutcome, VerificationResult


class ReportStatus(str, Enum):
    SUPPORTED_BY_REPLAY = "supported_by_replay"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class RCAReport:
    incident_id: str
    status: ReportStatus
    affected_metrics: Tuple[str, ...]
    predicted_root_cause: RootCauseCategory
    confidence: float
    supporting_evidence: Tuple[str, ...]
    contradicting_evidence: Tuple[str, ...]
    experiments_run: int
    before_metrics: QualityMetrics
    degraded_metrics: QualityMetrics
    replay_results: Tuple[VerificationResult, ...]
    observed: Tuple[str, ...]
    inferred: Tuple[str, ...]
    replay_evidence: Tuple[str, ...]
    final_assessment: str
    recommended_action: str
    human_approval_required: bool = field(default=True, init=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "status": self.status.value,
            "affected_metrics": list(self.affected_metrics),
            "predicted_root_cause": self.predicted_root_cause.value,
            "confidence": self.confidence,
            "supporting_evidence": list(self.supporting_evidence),
            "contradicting_evidence": list(self.contradicting_evidence),
            "experiments_run": self.experiments_run,
            "before_metrics": self.before_metrics.to_dict(),
            "degraded_metrics": self.degraded_metrics.to_dict(),
            "replay_results": [
                {
                    "experiment_id": result.experiment.experiment_id,
                    "hypothesis": result.experiment.hypothesis.root_cause.value,
                    "intervention": result.experiment.intervention.to_dict(),
                    "outcome": result.outcome.value,
                    "quality_delta": result.quality_delta,
                    "metrics": result.replay.replay_metrics.to_dict(),
                    "rationale": result.rationale,
                }
                for result in self.replay_results
            ],
            "observed": list(self.observed),
            "inferred": list(self.inferred),
            "replay_evidence": list(self.replay_evidence),
            "final_assessment": self.final_assessment,
            "recommended_action": self.recommended_action,
            "human_approval_required": self.human_approval_required,
        }


class RCAReporter:
    """Select the strongest replay-supported hypothesis and produce a safe report."""

    def report(
        self,
        incident: Incident,
        hypotheses: Tuple[InvestigationHypothesis, ...],
        planning: PlanningResult,
        verifications: Tuple[VerificationResult, ...],
    ) -> RCAReport:
        before = evaluate_configuration(incident.configuration_before).metrics
        degraded = evaluate_configuration(incident.configuration_after).metrics
        selected, supporting_verification = _select_hypothesis(
            hypotheses, verifications
        )
        if supporting_verification is not None:
            status = ReportStatus.SUPPORTED_BY_REPLAY
            delta = supporting_verification.quality_delta
            confidence = round(
                min(1.0, (0.5 * selected.prior_confidence) + (0.5 * min(delta / 0.5, 1.0))),
                4,
            )
            final_assessment = (
                f"The {selected.root_cause.value} hypothesis has strong causal support "
                f"from a controlled local replay (aggregate quality delta {delta:+.4f})."
            )
            recommended_action = _recommendation(selected.root_cause)
        else:
            status = ReportStatus.INCONCLUSIVE
            confidence = round(selected.prior_confidence * 0.5, 4)
            final_assessment = (
                f"The leading inference is {selected.root_cause.value}, but no replay "
                "met the material-recovery threshold; the RCA remains inconclusive."
            )
            recommended_action = (
                "Do not change production configuration; gather additional evidence "
                "and obtain human approval before any remediation."
            )

        observed = _observed_evidence(incident, before, degraded)
        inferred = tuple(
            f"{hypothesis.root_cause.value}: prior={hypothesis.prior_confidence:.4f}"
            for hypothesis in hypotheses
        )
        replay_evidence = tuple(
            f"{result.experiment.hypothesis.root_cause.value}: "
            f"{result.outcome.value}, aggregate_delta={result.quality_delta:+.4f}"
            for result in verifications
        )
        selected_failed_evidence = tuple(
            result.rationale
            for result in verifications
            if result.experiment.hypothesis.root_cause is selected.root_cause
            and result.outcome is VerificationOutcome.FAILS_TO_SUPPORT
        )
        selected_rejected_evidence = tuple(
            rejection.reason
            for rejection in planning.rejected
            if rejection.hypothesis.root_cause is selected.root_cause
        )
        support_from_replay = (
            (supporting_verification.rationale,)
            if supporting_verification is not None
            else ()
        )

        return RCAReport(
            incident_id=incident.incident_id,
            status=status,
            affected_metrics=_affected_metrics(before, degraded),
            predicted_root_cause=selected.root_cause,
            confidence=confidence,
            supporting_evidence=selected.supporting_evidence + support_from_replay,
            contradicting_evidence=(
                selected.contradicting_evidence
                + selected_failed_evidence
                + selected_rejected_evidence
            ),
            experiments_run=len(verifications),
            before_metrics=before,
            degraded_metrics=degraded,
            replay_results=verifications,
            observed=observed,
            inferred=inferred,
            replay_evidence=replay_evidence,
            final_assessment=final_assessment,
            recommended_action=recommended_action,
        )


def _select_hypothesis(
    hypotheses: Tuple[InvestigationHypothesis, ...],
    verifications: Tuple[VerificationResult, ...],
) -> tuple[InvestigationHypothesis, Optional[VerificationResult]]:
    if not hypotheses:
        raise ValueError("at least one hypothesis is required for reporting")
    supported = tuple(
        result
        for result in verifications
        if result.outcome is VerificationOutcome.SUPPORTED
    )
    if supported:
        strongest = max(supported, key=lambda result: result.quality_delta)
        return strongest.experiment.hypothesis, strongest
    return hypotheses[0], None


def _affected_metrics(
    before: QualityMetrics, degraded: QualityMetrics
) -> Tuple[str, ...]:
    affected = []
    for name in (
        "retrieval_relevance",
        "groundedness",
        "answer_quality",
        "aggregate_quality",
    ):
        if getattr(degraded, name) < getattr(before, name) - 0.01:
            affected.append(name)
    if degraded.latency_ms > before.latency_ms + 1.0:
        affected.append("latency_ms")
    return tuple(affected)


def _observed_evidence(
    incident: Incident, before: QualityMetrics, degraded: QualityMetrics
) -> Tuple[str, ...]:
    metrics = (
        f"aggregate_quality changed from {before.aggregate_quality:.4f} "
        f"to {degraded.aggregate_quality:.4f}",
        f"retrieval_relevance changed from {before.retrieval_relevance:.4f} "
        f"to {degraded.retrieval_relevance:.4f}",
        f"groundedness changed from {before.groundedness:.4f} "
        f"to {degraded.groundedness:.4f}",
        f"answer_quality changed from {before.answer_quality:.4f} "
        f"to {degraded.answer_quality:.4f}",
    )
    changes = tuple(
        f"{change.field_name} changed from {change.before!r} to {change.after!r}"
        for change in incident.changes
    )
    return metrics + changes


def _recommendation(root_cause: RootCauseCategory) -> str:
    if root_cause is RootCauseCategory.RETRIEVER_TOP_K_REGRESSION:
        return (
            "Restore retriever_top_k to the previous value in an approved change, "
            "then run the complete regression suite."
        )
    if root_cause is RootCauseCategory.PROMPT_REGRESSION:
        return (
            "Restore prompt_profile to the previous value in an approved change, "
            "then run the complete regression suite."
        )
    return (
        "Review the replay evidence and prepare a scoped remediation for explicit "
        "human approval; do not change production automatically."
    )
