"""Deterministic replay-based support and falsification classification."""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

from traceback_rca.evaluator import MATERIAL_QUALITY_IMPROVEMENT
from traceback_rca.models import Incident
from traceback_rca.planner import MAX_EXPERIMENTS, PlannedExperiment
from traceback_rca.replay import ControlledReplayEngine, ReplayResult


NON_MATERIAL_MAX_DELTA = 0.05


class VerificationOutcome(str, Enum):
    SUPPORTED = "supported_by_replay"
    FAILS_TO_SUPPORT = "fails_to_support"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class VerificationResult:
    experiment: PlannedExperiment
    replay: ReplayResult
    quality_delta: float
    material_recovery_threshold: float
    outcome: VerificationOutcome
    rationale: str


class ReplayVerifier:
    """Execute validated local replays and classify only measured quality movement."""

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
        replay_result = self._replay_engine.replay(
            incident, experiment.intervention.to_dict()
        )
        delta = replay_result.delta_from_degraded.aggregate_quality
        if delta >= MATERIAL_QUALITY_IMPROVEMENT:
            outcome = VerificationOutcome.SUPPORTED
            rationale = (
                f"aggregate quality improved by {delta:+.4f}, meeting the "
                f"{MATERIAL_QUALITY_IMPROVEMENT:.4f} material-recovery threshold"
            )
        elif delta <= NON_MATERIAL_MAX_DELTA:
            outcome = VerificationOutcome.FAILS_TO_SUPPORT
            rationale = (
                f"aggregate quality changed by only {delta:+.4f}, no more than the "
                f"{NON_MATERIAL_MAX_DELTA:.4f} non-material bound"
            )
        else:
            outcome = VerificationOutcome.INCONCLUSIVE
            rationale = (
                f"aggregate quality improved by {delta:+.4f}, above the non-material "
                f"bound but below the {MATERIAL_QUALITY_IMPROVEMENT:.4f} threshold"
            )
        return VerificationResult(
            experiment=experiment,
            replay=replay_result,
            quality_delta=delta,
            material_recovery_threshold=MATERIAL_QUALITY_IMPROVEMENT,
            outcome=outcome,
            rationale=rationale,
        )

