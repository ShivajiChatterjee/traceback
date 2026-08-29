"""Offline scorer comparing baseline and Traceback against hidden ground truth."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable, Tuple

from traceback_rca.baseline import Baseline
from traceback_rca.ground_truth import get_ground_truth
from traceback_rca.incidents import get_incident
from traceback_rca.providers import GeminiProvider, LLMProvider
from traceback_rca.taxonomy import RootCauseCategory
from traceback_rca.workflow import TracebackWorkflow


@dataclass(frozen=True, slots=True)
class BenchmarkPrediction:
    predicted_root_cause: RootCauseCategory
    confidence: float
    correct: bool
    experiments_used: int
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_root_cause": self.predicted_root_cause.value,
            "confidence": self.confidence,
            "correct": self.correct,
            "experiments_used": self.experiments_used,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass(frozen=True, slots=True)
class CaseBenchmarkResult:
    incident_id: str
    expected_root_cause: RootCauseCategory
    baseline: BenchmarkPrediction
    traceback: BenchmarkPrediction

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "expected_root_cause": self.expected_root_cause.value,
            "baseline": self.baseline.to_dict(),
            "traceback": self.traceback.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    cases: Tuple[CaseBenchmarkResult, ...]
    baseline_accuracy: float
    traceback_accuracy: float


class BenchmarkRunner:
    """Run predictions first, then retrieve hidden answers solely for scoring."""

    def __init__(
        self, baseline_provider: LLMProvider, traceback_provider: LLMProvider
    ) -> None:
        self._baseline = Baseline(baseline_provider)
        self._traceback = TracebackWorkflow(traceback_provider)

    def run(self, incident_ids: Iterable[str] = ("I01", "I03", "I10")) -> BenchmarkResult:
        case_results = []
        for incident_id in incident_ids:
            incident = get_incident(incident_id)

            baseline_started = perf_counter()
            baseline = self._baseline.diagnose(incident)
            baseline_latency = (perf_counter() - baseline_started) * 1000.0

            traceback_started = perf_counter()
            traceback_run = self._traceback.investigate(incident)
            traceback_latency = (perf_counter() - traceback_started) * 1000.0

            # Hidden evaluation data is accessed only after both predictions exist.
            expected = RootCauseCategory(get_ground_truth(incident_id).root_cause)
            case_results.append(
                CaseBenchmarkResult(
                    incident_id=incident_id,
                    expected_root_cause=expected,
                    baseline=BenchmarkPrediction(
                        predicted_root_cause=baseline.predicted_root_cause,
                        confidence=baseline.confidence,
                        correct=baseline.predicted_root_cause is expected,
                        experiments_used=0,
                        latency_ms=round(baseline_latency, 2),
                        input_tokens=baseline.input_tokens,
                        output_tokens=baseline.output_tokens,
                    ),
                    traceback=BenchmarkPrediction(
                        predicted_root_cause=(
                            traceback_run.report.predicted_root_cause
                        ),
                        confidence=traceback_run.report.confidence,
                        correct=(
                            traceback_run.report.predicted_root_cause is expected
                        ),
                        experiments_used=traceback_run.report.experiments_run,
                        latency_ms=round(traceback_latency, 2),
                        input_tokens=traceback_run.investigator.input_tokens,
                        output_tokens=traceback_run.investigator.output_tokens,
                    ),
                )
            )

        cases = tuple(case_results)
        if not cases:
            raise ValueError("benchmark requires at least one incident")
        return BenchmarkResult(
            cases=cases,
            baseline_accuracy=sum(case.baseline.correct for case in cases) / len(cases),
            traceback_accuracy=sum(case.traceback.correct for case in cases) / len(cases),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-real-api",
        action="store_true",
        help="required acknowledgement that this command makes multiple Gemini calls",
    )
    args = parser.parse_args()
    if not args.confirm_real_api:
        parser.error("pass --confirm-real-api to explicitly authorize real Gemini calls")

    provider = GeminiProvider.from_environment()
    result = BenchmarkRunner(provider, provider).run()
    for case in result.cases:
        print(
            f"{case.incident_id}: expected={case.expected_root_cause.value} "
            f"baseline={case.baseline.predicted_root_cause.value} "
            f"traceback={case.traceback.predicted_root_cause.value} "
            f"experiments={case.traceback.experiments_used}"
        )
    print(f"baseline_accuracy={result.baseline_accuracy:.4f}")
    print(f"traceback_accuracy={result.traceback_accuracy:.4f}")


if __name__ == "__main__":
    main()
