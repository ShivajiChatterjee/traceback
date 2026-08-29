"""Bounded end-to-end Traceback investigation orchestration."""

from dataclasses import dataclass
from typing import Optional, Tuple

from traceback_rca.investigator import HypothesisInvestigator, InvestigatorResult
from traceback_rca.models import Incident
from traceback_rca.planner import MAX_EXPERIMENTS, PlanningResult, SafeExperimentPlanner
from traceback_rca.providers import LLMProvider
from traceback_rca.reporter import RCAReport, RCAReporter
from traceback_rca.verifier import ReplayVerifier, VerificationResult


@dataclass(frozen=True, slots=True)
class InvestigationRun:
    investigator: InvestigatorResult
    planning: PlanningResult
    verifications: Tuple[VerificationResult, ...]
    report: RCAReport


class TracebackWorkflow:
    """Run one reasoning pass and at most three deterministic experiments."""

    def __init__(
        self,
        provider: LLMProvider,
        max_experiments: int = MAX_EXPERIMENTS,
        verifier: Optional[ReplayVerifier] = None,
        reporter: Optional[RCAReporter] = None,
    ) -> None:
        self._investigator = HypothesisInvestigator(provider)
        self._planner = SafeExperimentPlanner(max_experiments)
        self._verifier = verifier or ReplayVerifier()
        self._reporter = reporter or RCAReporter()

    def investigate(self, incident: Incident) -> InvestigationRun:
        investigator_result = self._investigator.investigate(incident)
        planning = self._planner.plan(incident, investigator_result.hypotheses)
        verifications = self._verifier.verify(incident, planning.experiments)
        report = self._reporter.report(
            incident,
            investigator_result.hypotheses,
            planning,
            verifications,
        )
        return InvestigationRun(
            investigator=investigator_result,
            planning=planning,
            verifications=verifications,
            report=report,
        )

