"""Ten-case offline scorer for same-model baseline versus Traceback."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from statistics import fmean
from time import perf_counter
from typing import Any, Iterable, Mapping, Optional, Tuple

from traceback_rca.baseline import Baseline, BaselineDiagnosis
from traceback_rca.ground_truth import get_ground_truth
from traceback_rca.incidents import get_incident, list_incidents
from traceback_rca.providers import GeminiProvider, LLMProvider
from traceback_rca.taxonomy import RootCauseCategory
from traceback_rca.verifier import VerificationOutcome
from traceback_rca.workflow import InvestigationRun, TracebackWorkflow


@dataclass(frozen=True, slots=True)
class BenchmarkPrediction:
    predicted_root_cause: RootCauseCategory
    confidence: float
    correct: bool
    experiments_used: int
    latency_ms: float
    summary: str
    final_assessment: str | None = None
    supported_hypotheses: Tuple[str, ...] = ()
    fails_to_support_hypotheses: Tuple[str, ...] = ()
    inconclusive_hypotheses: Tuple[str, ...] = ()
    replay_evidence: Tuple[Mapping[str, Any], ...] = ()
    input_tokens: int | None = None
    output_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_root_cause": self.predicted_root_cause.value,
            "confidence": self.confidence,
            "correct": self.correct,
            "experiments_used": self.experiments_used,
            "latency_ms": self.latency_ms,
            "summary": self.summary,
            "final_assessment": self.final_assessment,
            "supported_hypotheses": list(self.supported_hypotheses),
            "fails_to_support_hypotheses": list(
                self.fails_to_support_hypotheses
            ),
            "inconclusive_hypotheses": list(self.inconclusive_hypotheses),
            "replay_evidence": list(self.replay_evidence),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass(frozen=True, slots=True)
class CaseBenchmarkResult:
    incident_id: str
    ground_truth: RootCauseCategory
    baseline: Optional[BenchmarkPrediction]
    traceback: Optional[BenchmarkPrediction]
    baseline_error: Optional[str] = None
    traceback_error: Optional[str] = None

    @property
    def expected_root_cause(self) -> RootCauseCategory:
        """Compatibility alias for earlier benchmark consumers."""

        return self.ground_truth

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "ground_truth": self.ground_truth.value,
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "traceback": self.traceback.to_dict() if self.traceback else None,
            "baseline_error": self.baseline_error,
            "traceback_error": self.traceback_error,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    model: str
    cases: Tuple[CaseBenchmarkResult, ...]
    baseline_accuracy: float
    traceback_accuracy: float
    baseline_correct: int
    traceback_correct: int
    healthy_false_positive_baseline: bool | None
    healthy_false_positive_traceback: bool | None
    mean_traceback_experiments_per_true_incident: float
    replay_supported_diagnosis_count: int
    replay_supported_diagnosis_rate: float
    mean_baseline_latency_ms: float | None
    mean_traceback_latency_ms: float | None
    baseline_input_tokens: int | None
    baseline_output_tokens: int | None
    traceback_input_tokens: int | None
    traceback_output_tokens: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "case_count": len(self.cases),
            "primary_metrics": {
                "baseline_root_cause_accuracy": self.baseline_accuracy,
                "traceback_root_cause_accuracy": self.traceback_accuracy,
                "baseline_correct": self.baseline_correct,
                "traceback_correct": self.traceback_correct,
            },
            "secondary_metrics": {
                "healthy_false_positive_baseline": self.healthy_false_positive_baseline,
                "healthy_false_positive_traceback": self.healthy_false_positive_traceback,
                "mean_traceback_experiments_per_true_incident": (
                    self.mean_traceback_experiments_per_true_incident
                ),
                "replay_supported_diagnosis_count": (
                    self.replay_supported_diagnosis_count
                ),
                "replay_supported_diagnosis_rate": (
                    self.replay_supported_diagnosis_rate
                ),
                "mean_baseline_latency_ms": self.mean_baseline_latency_ms,
                "mean_traceback_latency_ms": self.mean_traceback_latency_ms,
                "baseline_input_tokens": self.baseline_input_tokens,
                "baseline_output_tokens": self.baseline_output_tokens,
                "traceback_input_tokens": self.traceback_input_tokens,
                "traceback_output_tokens": self.traceback_output_tokens,
            },
            "cases": [case.to_dict() for case in self.cases],
        }


class BenchmarkRunner:
    """Complete both predictions before retrieving each hidden answer for scoring."""

    def __init__(
        self,
        baseline_provider: LLMProvider,
        traceback_provider: LLMProvider,
        model: str = "scripted-offline",
    ) -> None:
        self._baseline = Baseline(baseline_provider)
        self._traceback = TracebackWorkflow(traceback_provider)
        self._model = model

    def run(self, incident_ids: Iterable[str] | None = None) -> BenchmarkResult:
        selected_ids = (
            tuple(incident.incident_id for incident in list_incidents())
            if incident_ids is None
            else tuple(incident_ids)
        )
        if not selected_ids:
            raise ValueError("benchmark requires at least one incident")

        cases = []
        for incident_id in selected_ids:
            incident = get_incident(incident_id)
            baseline_diagnosis: BaselineDiagnosis | None = None
            traceback_run: InvestigationRun | None = None
            baseline_error = None
            traceback_error = None

            baseline_started = perf_counter()
            try:
                baseline_diagnosis = self._baseline.diagnose(incident)
            except Exception as error:  # Per-case real API/parse failure is persisted.
                baseline_error = _safe_error(error)
            baseline_latency = (perf_counter() - baseline_started) * 1000.0

            traceback_started = perf_counter()
            try:
                traceback_run = self._traceback.investigate(incident)
            except Exception as error:  # Continue the benchmark after one failed case.
                traceback_error = _safe_error(error)
            traceback_latency = (perf_counter() - traceback_started) * 1000.0

            # Evaluation-only data is accessed after both workflow attempts finish.
            expected = RootCauseCategory(get_ground_truth(incident_id).root_cause)
            cases.append(
                CaseBenchmarkResult(
                    incident_id=incident_id,
                    ground_truth=expected,
                    baseline=(
                        _baseline_prediction(
                            baseline_diagnosis, expected, baseline_latency
                        )
                        if baseline_diagnosis
                        else None
                    ),
                    traceback=(
                        _traceback_prediction(traceback_run, expected, traceback_latency)
                        if traceback_run
                        else None
                    ),
                    baseline_error=baseline_error,
                    traceback_error=traceback_error,
                )
            )
        return _aggregate(self._model, tuple(cases))


def _baseline_prediction(
    diagnosis: BaselineDiagnosis,
    expected: RootCauseCategory,
    elapsed_ms: float,
) -> BenchmarkPrediction:
    return BenchmarkPrediction(
        predicted_root_cause=diagnosis.predicted_root_cause,
        confidence=diagnosis.confidence,
        correct=diagnosis.predicted_root_cause is expected,
        experiments_used=0,
        latency_ms=round(elapsed_ms, 2),
        summary=diagnosis.summary,
        input_tokens=diagnosis.input_tokens,
        output_tokens=diagnosis.output_tokens,
    )


def _traceback_prediction(
    run: InvestigationRun,
    expected: RootCauseCategory,
    elapsed_ms: float,
) -> BenchmarkPrediction:
    report_payload = run.report.to_dict()
    return BenchmarkPrediction(
        predicted_root_cause=run.report.predicted_root_cause,
        confidence=run.report.confidence,
        correct=run.report.predicted_root_cause is expected,
        experiments_used=run.report.experiments_run,
        latency_ms=round(elapsed_ms, 2),
        summary=run.report.final_assessment,
        final_assessment=run.report.final_assessment,
        supported_hypotheses=_outcome_categories(
            run, VerificationOutcome.SUPPORTED
        ),
        fails_to_support_hypotheses=_outcome_categories(
            run, VerificationOutcome.FAILS_TO_SUPPORT
        ),
        inconclusive_hypotheses=_outcome_categories(
            run, VerificationOutcome.INCONCLUSIVE
        ),
        replay_evidence=tuple(report_payload["replay_results"]),
        input_tokens=run.investigator.input_tokens if run.investigator else None,
        output_tokens=run.investigator.output_tokens if run.investigator else None,
    )


def _outcome_categories(
    run: InvestigationRun, outcome: VerificationOutcome
) -> Tuple[str, ...]:
    return tuple(
        result.experiment.hypothesis.root_cause.value
        for result in run.verifications
        if result.outcome is outcome
    )


def _aggregate(model: str, cases: Tuple[CaseBenchmarkResult, ...]) -> BenchmarkResult:
    total = len(cases)
    baseline_correct = sum(bool(case.baseline and case.baseline.correct) for case in cases)
    traceback_correct = sum(bool(case.traceback and case.traceback.correct) for case in cases)
    true_incidents = tuple(
        case for case in cases if case.ground_truth is not RootCauseCategory.NO_INCIDENT
    )
    healthy = next(
        (case for case in cases if case.ground_truth is RootCauseCategory.NO_INCIDENT),
        None,
    )
    supported_count = sum(
        bool(
            case.traceback
            and case.traceback.predicted_root_cause.value
            in case.traceback.supported_hypotheses
        )
        for case in true_incidents
    )
    return BenchmarkResult(
        model=model,
        cases=cases,
        baseline_accuracy=round(baseline_correct / total, 4),
        traceback_accuracy=round(traceback_correct / total, 4),
        baseline_correct=baseline_correct,
        traceback_correct=traceback_correct,
        healthy_false_positive_baseline=(
            healthy.baseline.predicted_root_cause is not RootCauseCategory.NO_INCIDENT
            if healthy and healthy.baseline
            else None
        ),
        healthy_false_positive_traceback=(
            healthy.traceback.predicted_root_cause is not RootCauseCategory.NO_INCIDENT
            if healthy and healthy.traceback
            else None
        ),
        mean_traceback_experiments_per_true_incident=round(
            sum(
                case.traceback.experiments_used if case.traceback else 0
                for case in true_incidents
            )
            / len(true_incidents)
            if true_incidents
            else 0.0,
            4,
        ),
        replay_supported_diagnosis_count=supported_count,
        replay_supported_diagnosis_rate=round(
            supported_count / len(true_incidents) if true_incidents else 0.0, 4
        ),
        mean_baseline_latency_ms=_mean_optional(
            case.baseline.latency_ms for case in cases if case.baseline
        ),
        mean_traceback_latency_ms=_mean_optional(
            case.traceback.latency_ms for case in cases if case.traceback
        ),
        baseline_input_tokens=_sum_optional(
            case.baseline.input_tokens for case in cases if case.baseline
        ),
        baseline_output_tokens=_sum_optional(
            case.baseline.output_tokens for case in cases if case.baseline
        ),
        traceback_input_tokens=_sum_optional(
            case.traceback.input_tokens for case in cases if case.traceback
        ),
        traceback_output_tokens=_sum_optional(
            case.traceback.output_tokens for case in cases if case.traceback
        ),
    )


def _mean_optional(values: Iterable[float]) -> float | None:
    collected = tuple(values)
    return round(fmean(collected), 2) if collected else None


def _sum_optional(values: Iterable[int | None]) -> int | None:
    collected = tuple(value for value in values if value is not None)
    return sum(collected) if collected else None


def _safe_error(error: Exception) -> str:
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    suffix = f" (status={status})" if status is not None else ""
    return f"{type(error).__name__}{suffix}: request or workflow failed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-real-api",
        action="store_true",
        help="required acknowledgement that this command makes multiple Gemini calls",
    )
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    if not args.confirm_real_api:
        parser.error("pass --confirm-real-api to explicitly authorize real Gemini calls")

    provider = GeminiProvider.from_environment()
    result = BenchmarkRunner(provider, provider, model=provider.model).run()
    from traceback_rca.export import save_benchmark

    output_directory = save_benchmark(result, args.results_dir)
    for case in result.cases:
        baseline_label = (
            case.baseline.predicted_root_cause.value if case.baseline else "ERROR"
        )
        traceback_label = (
            case.traceback.predicted_root_cause.value if case.traceback else "ERROR"
        )
        experiments = case.traceback.experiments_used if case.traceback else 0
        print(
            f"{case.incident_id}: expected={case.ground_truth.value} "
            f"baseline={baseline_label} traceback={traceback_label} "
            f"experiments={experiments}"
        )
    print(
        f"baseline_accuracy={result.baseline_correct}/{len(result.cases)} "
        f"({result.baseline_accuracy:.1%})"
    )
    print(
        f"traceback_accuracy={result.traceback_correct}/{len(result.cases)} "
        f"({result.traceback_accuracy:.1%})"
    )
    print(f"results_saved_to={output_directory}")


if __name__ == "__main__":
    main()
