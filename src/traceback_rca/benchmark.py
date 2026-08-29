"""Ten-case offline scorer for same-model baseline versus Traceback."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any, Iterable, Mapping, Optional, Tuple

from traceback_rca.baseline import Baseline, BaselineDiagnosis
from traceback_rca.ground_truth import get_ground_truth
from traceback_rca.incidents import display_incident_id, get_incident, list_incidents
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
    baseline_accuracy: float | None
    traceback_accuracy: float | None
    baseline_correct: int
    traceback_correct: int
    baseline_completed_cases: int
    traceback_completed_cases: int
    baseline_provider_errors: int
    traceback_provider_errors: int
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
                "baseline_completed_cases": self.baseline_completed_cases,
                "traceback_completed_cases": self.traceback_completed_cases,
                "baseline_provider_errors": self.baseline_provider_errors,
                "traceback_provider_errors": self.traceback_provider_errors,
                "benchmark_complete": (
                    self.baseline_completed_cases == len(self.cases)
                    and self.traceback_completed_cases == len(self.cases)
                ),
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


@dataclass(frozen=True, slots=True)
class BenchmarkResumeResult:
    """Result and exact system/case combinations attempted during a resume."""

    benchmark: BenchmarkResult
    retried: Tuple[Tuple[str, str], ...]


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


def resume_failed_benchmark(
    benchmark_directory: str | Path,
    baseline_provider: LLMProvider,
    traceback_provider: LLMProvider,
    model: str,
) -> BenchmarkResumeResult:
    """Retry only unresolved system/case records in an existing benchmark."""

    directory = Path(benchmark_directory)
    metrics = _read_json_object(directory / "metrics.json")
    recorded_model = metrics.get("model")
    if recorded_model != model:
        raise ValueError(
            "resume requires the original benchmark model "
            f"{recorded_model!r}; current GEMINI_MODEL is {model!r}"
        )

    baseline_records = _read_json_records(directory / "baseline_results.json")
    traceback_records = _read_json_records(directory / "traceback_results.json")
    _validate_resume_records(baseline_records, traceback_records)

    baseline = Baseline(baseline_provider)
    traceback = TracebackWorkflow(traceback_provider)
    retried: list[Tuple[str, str]] = []
    changed_systems: set[str] = set()

    for system, records in (
        ("baseline", baseline_records),
        ("traceback", traceback_records),
    ):
        for record in records:
            if record.get("result") is not None or not record.get("error"):
                continue

            incident_id = str(record["incident_id"])
            incident = get_incident(incident_id)
            attempts = list(record.get("attempts") or ())
            if not attempts:
                attempts.append(_attempt_from_record(record, model, 1))

            started = perf_counter()
            prediction: BenchmarkPrediction | None = None
            retry_error: str | None = None
            try:
                if system == "baseline":
                    diagnosis = baseline.diagnose(incident)
                else:
                    investigation = traceback.investigate(incident)
            except Exception as error:  # Persist the new failure without aborting resume.
                retry_error = _safe_error(error)
            elapsed_ms = (perf_counter() - started) * 1000.0

            # Evaluation-only data remains inaccessible until the workflow attempt ends.
            expected = RootCauseCategory(get_ground_truth(incident_id).root_cause)
            if record.get("ground_truth") != expected.value:
                raise ValueError(
                    f"stored GroundTruth does not match current case {incident_id}"
                )
            if retry_error is None:
                prediction = (
                    _baseline_prediction(diagnosis, expected, elapsed_ms)
                    if system == "baseline"
                    else _traceback_prediction(investigation, expected, elapsed_ms)
                )

            attempts.append(
                _attempt(
                    number=len(attempts) + 1,
                    model=model,
                    prediction=prediction,
                    error=retry_error,
                )
            )
            record["attempts"] = attempts
            record["result"] = prediction.to_dict() if prediction else None
            record["error"] = retry_error
            retried.append((system, incident_id))
            changed_systems.add(system)

    result = _aggregate_records(model, baseline_records, traceback_records)
    if retried:
        from traceback_rca.export import update_benchmark

        update_benchmark(
            result,
            directory,
            baseline_records,
            traceback_records,
            retried,
            changed_systems,
        )
    return BenchmarkResumeResult(benchmark=result, retried=tuple(retried))


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
    baseline_completed = sum(case.baseline is not None for case in cases)
    traceback_completed = sum(case.traceback is not None for case in cases)
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
    completed_true_incidents = tuple(
        case for case in true_incidents if case.traceback is not None
    )
    return BenchmarkResult(
        model=model,
        cases=cases,
        baseline_accuracy=(
            round(baseline_correct / total, 4) if baseline_completed == total else None
        ),
        traceback_accuracy=(
            round(traceback_correct / total, 4)
            if traceback_completed == total
            else None
        ),
        baseline_correct=baseline_correct,
        traceback_correct=traceback_correct,
        baseline_completed_cases=baseline_completed,
        traceback_completed_cases=traceback_completed,
        baseline_provider_errors=sum(
            case.baseline is None and _is_provider_error(case.baseline_error)
            for case in cases
        ),
        traceback_provider_errors=sum(
            case.traceback is None and _is_provider_error(case.traceback_error)
            for case in cases
        ),
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
                case.traceback.experiments_used for case in completed_true_incidents
            )
            / len(completed_true_incidents)
            if completed_true_incidents
            else 0.0,
            4,
        ),
        replay_supported_diagnosis_count=supported_count,
        replay_supported_diagnosis_rate=round(
            supported_count / len(completed_true_incidents)
            if completed_true_incidents
            else 0.0,
            4,
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


def _aggregate_records(
    model: str,
    baseline_records: list[dict[str, Any]],
    traceback_records: list[dict[str, Any]],
) -> BenchmarkResult:
    cases = []
    for baseline_record, traceback_record in zip(
        baseline_records, traceback_records, strict=True
    ):
        cases.append(
            CaseBenchmarkResult(
                incident_id=str(baseline_record["incident_id"]),
                ground_truth=RootCauseCategory(str(baseline_record["ground_truth"])),
                baseline=_prediction_from_dict(baseline_record.get("result")),
                traceback=_prediction_from_dict(traceback_record.get("result")),
                baseline_error=baseline_record.get("error"),
                traceback_error=traceback_record.get("error"),
            )
        )
    return _aggregate(model, tuple(cases))


def _prediction_from_dict(value: Any) -> BenchmarkPrediction | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("benchmark prediction must be an object or null")
    return BenchmarkPrediction(
        predicted_root_cause=RootCauseCategory(value["predicted_root_cause"]),
        confidence=float(value["confidence"]),
        correct=bool(value["correct"]),
        experiments_used=int(value["experiments_used"]),
        latency_ms=float(value["latency_ms"]),
        summary=str(value["summary"]),
        final_assessment=value.get("final_assessment"),
        supported_hypotheses=tuple(value.get("supported_hypotheses") or ()),
        fails_to_support_hypotheses=tuple(
            value.get("fails_to_support_hypotheses") or ()
        ),
        inconclusive_hypotheses=tuple(value.get("inconclusive_hypotheses") or ()),
        replay_evidence=tuple(value.get("replay_evidence") or ()),
        input_tokens=value.get("input_tokens"),
        output_tokens=value.get("output_tokens"),
    )


def _validate_resume_records(
    baseline_records: list[dict[str, Any]],
    traceback_records: list[dict[str, Any]],
) -> None:
    if not baseline_records or len(baseline_records) != len(traceback_records):
        raise ValueError("benchmark result files must contain the same non-zero case count")
    baseline_ids = [record.get("incident_id") for record in baseline_records]
    traceback_ids = [record.get("incident_id") for record in traceback_records]
    if baseline_ids != traceback_ids:
        raise ValueError("baseline and Traceback result case order does not match")
    for baseline_record, traceback_record in zip(
        baseline_records, traceback_records, strict=True
    ):
        if baseline_record.get("ground_truth") != traceback_record.get("ground_truth"):
            raise ValueError("baseline and Traceback GroundTruth records do not match")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load benchmark file {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"benchmark file must contain an object: {path}")
    return value


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load benchmark file {path}") from error
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"benchmark file must contain a list of objects: {path}")
    return value


def _attempt_from_record(
    record: Mapping[str, Any], model: str, number: int
) -> dict[str, Any]:
    return _attempt(
        number=number,
        model=model,
        prediction=_prediction_from_dict(record.get("result")),
        error=record.get("error"),
    )


def _attempt(
    number: int,
    model: str,
    prediction: BenchmarkPrediction | None,
    error: str | None,
) -> dict[str, Any]:
    return {
        "attempt": number,
        "model": model,
        "status": (
            "success"
            if prediction is not None
            else "provider_error" if _is_provider_error(error) else "error"
        ),
        "error": error,
        "prediction": (
            prediction.predicted_root_cause.value if prediction is not None else None
        ),
    }


def _is_provider_error(error: str | None) -> bool:
    if not error:
        return False
    normalized = error.casefold()
    return any(
        marker in normalized
        for marker in (
            "status=",
            "provider",
            "clienterror",
            "servererror",
            "apierror",
        )
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
    parser.add_argument(
        "--resume",
        type=Path,
        help="existing benchmark directory to update in place",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="retry only unresolved system/case combinations in --resume",
    )
    args = parser.parse_args()
    if not args.confirm_real_api:
        parser.error("pass --confirm-real-api to explicitly authorize real Gemini calls")
    if args.resume is None and args.retry_failed:
        parser.error("--retry-failed requires --resume")
    if args.resume is not None and not args.retry_failed:
        parser.error("--resume requires --retry-failed")

    provider = GeminiProvider.from_environment()
    if args.resume is not None:
        resumed = resume_failed_benchmark(
            args.resume, provider, provider, model=provider.model
        )
        result = resumed.benchmark
        output_directory = args.resume
        for system, incident_id in resumed.retried:
            print(f"retried={system}/{display_incident_id(incident_id)}")
        if not resumed.retried:
            print("retried=none")
    else:
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
    _print_accuracy(
        "baseline",
        result.baseline_correct,
        result.baseline_completed_cases,
        len(result.cases),
        result.baseline_accuracy,
    )
    _print_accuracy(
        "traceback",
        result.traceback_correct,
        result.traceback_completed_cases,
        len(result.cases),
        result.traceback_accuracy,
    )
    print(f"results_saved_to={output_directory}")


def _print_accuracy(
    label: str, correct: int, completed: int, total: int, accuracy: float | None
) -> None:
    if accuracy is None:
        print(f"{label}_accuracy=INCOMPLETE ({completed}/{total} completed)")
    else:
        print(f"{label}_accuracy={correct}/{total} ({accuracy:.1%})")


if __name__ == "__main__":
    main()
