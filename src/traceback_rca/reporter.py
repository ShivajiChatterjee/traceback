"""Deterministic evidence-backed RCA reporting with explicit approval semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple

from traceback_rca.detector import IncidentDetection
from traceback_rca.evaluator import QualityMetrics, evaluate_configuration
from traceback_rca.investigator import InvestigationHypothesis
from traceback_rca.models import Incident
from traceback_rca.planner import PlanningResult
from traceback_rca.taxonomy import RootCauseCategory
from traceback_rca.verifier import VerificationOutcome, VerificationResult


class ReportStatus(str, Enum):
    SUPPORTED_BY_REPLAY = "supported_by_replay"
    INCONCLUSIVE = "inconclusive"
    NO_MATERIAL_INCIDENT = "no_material_incident"


@dataclass(frozen=True, slots=True)
class RCAReport:
    incident_id: str
    status: ReportStatus
    detection: IncidentDetection
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
    human_approval_required: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "status": self.status.value,
            "detection": self.detection.to_dict(),
            "affected_metrics": list(self.affected_metrics),
            "predicted_root_cause": self.predicted_root_cause.value,
            "confidence": self.confidence,
            "supporting_evidence": list(self.supporting_evidence),
            "contradicting_evidence": list(self.contradicting_evidence),
            "experiments_run": self.experiments_run,
            "before_metrics": self.before_metrics.to_dict(),
            "degraded_metrics": self.degraded_metrics.to_dict(),
            "replay_results": [_verification_dict(result) for result in self.replay_results],
            "observed": list(self.observed),
            "inferred": list(self.inferred),
            "replay_evidence": list(self.replay_evidence),
            "final_assessment": self.final_assessment,
            "recommended_action": self.recommended_action,
            "human_approval_required": self.human_approval_required,
        }


class RCAReporter:
    """Select the strongest metric-aware replay-supported hypothesis."""

    def report(
        self,
        incident: Incident,
        detection: IncidentDetection,
        hypotheses: Tuple[InvestigationHypothesis, ...],
        planning: PlanningResult,
        verifications: Tuple[VerificationResult, ...],
    ) -> RCAReport:
        if not detection.material_incident:
            return self.no_incident_report(incident, detection)

        before = evaluate_configuration(incident.configuration_before).metrics
        degraded = evaluate_configuration(incident.configuration_after).metrics
        selected, supporting_verification = _select_hypothesis(
            hypotheses, verifications
        )
        if supporting_verification is not None:
            status = ReportStatus.SUPPORTED_BY_REPLAY
            confidence = round(
                min(
                    1.0,
                    (0.5 * selected.prior_confidence)
                    + (0.5 * supporting_verification.support_score),
                ),
                4,
            )
            final_assessment = (
                f"The {selected.root_cause.value} hypothesis has strong causal support "
                f"from controlled local replay: {supporting_verification.rationale}."
            )
            recommended_action = _recommendation(selected.root_cause)
        else:
            status = ReportStatus.INCONCLUSIVE
            confidence = round(selected.prior_confidence * 0.5, 4)
            final_assessment = (
                f"The leading inference is {selected.root_cause.value}, but no replay "
                "met all category-specific recovery thresholds; the RCA is inconclusive."
            )
            recommended_action = (
                "Do not change production configuration; gather additional evidence "
                "and obtain human approval before any remediation."
            )

        observed = _observed_evidence(incident, detection)
        inferred = tuple(
            f"{hypothesis.root_cause.value}: prior={hypothesis.prior_confidence:.4f}"
            for hypothesis in hypotheses
        )
        replay_evidence = tuple(
            f"{result.experiment.hypothesis.root_cause.value}: "
            f"{result.outcome.value}; {result.rationale}"
            for result in verifications
        )
        selected_failed = tuple(
            result.rationale
            for result in verifications
            if result.experiment.hypothesis.root_cause is selected.root_cause
            and result.outcome is VerificationOutcome.FAILS_TO_SUPPORT
        )
        selected_rejected = tuple(
            rejection.reason
            for rejection in planning.rejected
            if rejection.hypothesis.root_cause is selected.root_cause
        )
        replay_support = (
            (supporting_verification.rationale,)
            if supporting_verification is not None
            else ()
        )
        return RCAReport(
            incident_id=incident.incident_id,
            status=status,
            detection=detection,
            affected_metrics=detection.affected_metrics,
            predicted_root_cause=selected.root_cause,
            confidence=confidence,
            supporting_evidence=selected.supporting_evidence + replay_support,
            contradicting_evidence=(
                selected.contradicting_evidence + selected_failed + selected_rejected
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
            human_approval_required=True,
        )

    def no_incident_report(
        self, incident: Incident, detection: IncidentDetection
    ) -> RCAReport:
        before = evaluate_configuration(incident.configuration_before).metrics
        after = evaluate_configuration(incident.configuration_after).metrics
        return RCAReport(
            incident_id=incident.incident_id,
            status=ReportStatus.NO_MATERIAL_INCIDENT,
            detection=detection,
            affected_metrics=(),
            predicted_root_cause=RootCauseCategory.NO_INCIDENT,
            confidence=1.0,
            supporting_evidence=detection.evidence,
            contradicting_evidence=(),
            experiments_run=0,
            before_metrics=before,
            degraded_metrics=after,
            replay_results=(),
            observed=_observed_evidence(incident, detection),
            inferred=("No causal hypothesis was generated because detection was negative.",),
            replay_evidence=("No replay was required.",),
            final_assessment=(
                "Deterministic telemetry comparison found no material regression; "
                "the appropriate classification is no_incident."
            ),
            recommended_action="No remediation is recommended; continue normal monitoring.",
            human_approval_required=False,
        )


def _verification_dict(result: VerificationResult) -> dict[str, Any]:
    return {
        "experiment_id": result.experiment.experiment_id,
        "hypothesis": result.experiment.hypothesis.root_cause.value,
        "intervention": result.experiment.intervention.to_dict(),
        "outcome": result.outcome.value,
        "metric_group": result.metric_group.value,
        "quality_delta": result.quality_delta,
        "relevant_metric_deltas": dict(result.relevant_metric_deltas),
        "decision_thresholds": dict(result.decision_thresholds),
        "support_score": result.support_score,
        "rationale": result.rationale,
        "replay": result.replay.to_dict(),
    }


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
        strongest = max(supported, key=lambda result: result.support_score)
        return strongest.experiment.hypothesis, strongest
    return hypotheses[0], None


def _observed_evidence(
    incident: Incident, detection: IncidentDetection
) -> Tuple[str, ...]:
    changes = tuple(
        f"{change.field_name} changed from {change.before!r} to {change.after!r}"
        for change in incident.changes
    )
    return detection.evidence + changes


def _recommendation(root_cause: RootCauseCategory) -> str:
    field_by_category = {
        RootCauseCategory.RETRIEVER_TOP_K_REGRESSION: "retriever_top_k",
        RootCauseCategory.EMBEDDING_REGRESSION: "embedding_profile",
        RootCauseCategory.PROMPT_REGRESSION: "prompt_profile",
        RootCauseCategory.STALE_INDEX: "index_profile",
        RootCauseCategory.RERANKER_DISABLED: "reranker_enabled",
        RootCauseCategory.GUARDRAIL_REGRESSION: "guardrail_profile",
        RootCauseCategory.TOOL_LATENCY_REGRESSION: "tool_latency_profile",
        RootCauseCategory.CONTEXT_TRUNCATION: "context_profile",
    }
    field_name = field_by_category.get(root_cause)
    if field_name is None:
        return (
            "Prepare a scoped remediation from the evidence for explicit human "
            "approval; do not change production automatically."
        )
    return (
        f"Restore {field_name} to the previous value in an approved change, then run "
        "the complete regression suite."
    )
