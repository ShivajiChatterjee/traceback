"""Bounded end-to-end Traceback investigation orchestration."""

from dataclasses import dataclass
from typing import Optional, Tuple

from traceback_rca.detector import IncidentDetection, IncidentDetector
from traceback_rca.investigator import HypothesisInvestigator, InvestigatorResult
from traceback_rca.models import Incident
from traceback_rca.planner import MAX_EXPERIMENTS, PlanningResult, SafeExperimentPlanner
from traceback_rca.providers import LLMProvider
from traceback_rca.reporter import RCAReport, RCAReporter
from traceback_rca.verifier import ReplayVerifier, VerificationResult


@dataclass(frozen=True, slots=True)
class InvestigationRun:
    detection: IncidentDetection
    investigator: InvestigatorResult | None
    planning: PlanningResult
    verifications: Tuple[VerificationResult, ...]
    report: RCAReport


class TracebackWorkflow:
    """Detect first, then run one reasoning pass and at most three local experiments."""

    def __init__(
        self,
        provider: LLMProvider,
        max_experiments: int = MAX_EXPERIMENTS,
        detector: Optional[IncidentDetector] = None,
        verifier: Optional[ReplayVerifier] = None,
        reporter: Optional[RCAReporter] = None,
    ) -> None:
        self._detector = detector or IncidentDetector()
        self._investigator = HypothesisInvestigator(provider)
        self._planner = SafeExperimentPlanner(max_experiments)
        self._verifier = verifier or ReplayVerifier()
        self._reporter = reporter or RCAReporter()

    def investigate(self, incident: Incident) -> InvestigationRun:
        detection = self._detector.detect(incident)
        if not detection.material_incident:
            planning = PlanningResult((), ())
            report = self._reporter.no_incident_report(incident, detection)
            return InvestigationRun(
                detection=detection,
                investigator=None,
                planning=planning,
                verifications=(),
                report=report,
            )

        investigator_result = self._investigator.investigate(incident)
        planning = self._planner.plan(incident, investigator_result.hypotheses)
        verifications = self._verifier.verify(incident, planning.experiments)
        report = self._reporter.report(
            incident,
            detection,
            investigator_result.hypotheses,
            planning,
            verifications,
        )
        return InvestigationRun(
            detection=detection,
            investigator=investigator_result,
            planning=planning,
            verifications=verifications,
            report=report,
        )

