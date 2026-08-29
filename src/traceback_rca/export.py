"""Machine-readable and human-shareable result persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from traceback_rca.incidents import display_incident_id

if TYPE_CHECKING:
    from traceback_rca.benchmark import BenchmarkResult, CaseBenchmarkResult
    from traceback_rca.models import Incident
    from traceback_rca.workflow import InvestigationRun


def save_investigation(
    incident: "Incident",
    run: "InvestigationRun",
    results_root: str | Path = "results",
    timestamp: datetime | None = None,
) -> Path:
    """Persist a production-safe RCA report without hidden ground truth."""

    label = _timestamp_label(timestamp)
    directory = _unique_directory(
        Path(results_root) / "incidents" / f"{display_incident_id(incident.incident_id)}_{label}"
    )
    report_payload = run.report.to_dict()
    _write_json(directory / "rca_report.json", report_payload)
    _write_json(
        directory / "replay_evidence.json",
        [verification.replay.to_dict() for verification in run.verifications],
    )
    (directory / "rca_report.md").write_text(
        _incident_markdown(incident, run), encoding="utf-8"
    )
    return directory


def save_benchmark(
    result: "BenchmarkResult",
    results_root: str | Path = "results",
    timestamp: datetime | None = None,
) -> Path:
    """Persist complete scored benchmark evidence and a generated Markdown summary."""

    directory = _unique_directory(
        Path(results_root) / f"benchmark_{_timestamp_label(timestamp)}"
    )
    payload = result.to_dict()
    _write_json(
        directory / "metrics.json",
        {
            "model": payload["model"],
            "case_count": payload["case_count"],
            "primary_metrics": payload["primary_metrics"],
            "secondary_metrics": payload["secondary_metrics"],
        },
    )
    _write_json(
        directory / "baseline_results.json",
        [
            {
                "incident_id": case.incident_id,
                "ground_truth": case.ground_truth.value,
                "result": case.baseline.to_dict() if case.baseline else None,
                "error": case.baseline_error,
            }
            for case in result.cases
        ],
    )
    _write_json(
        directory / "traceback_results.json",
        [
            {
                "incident_id": case.incident_id,
                "ground_truth": case.ground_truth.value,
                "result": case.traceback.to_dict() if case.traceback else None,
                "error": case.traceback_error,
            }
            for case in result.cases
        ],
    )
    incidents_directory = directory / "incidents"
    incidents_directory.mkdir()
    for case in result.cases:
        case_directory = incidents_directory / display_incident_id(case.incident_id)
        case_directory.mkdir()
        _write_json(case_directory / "case_result.json", case.to_dict())
        _write_json(
            case_directory / "replay_evidence.json",
            list(case.traceback.replay_evidence) if case.traceback else [],
        )
    (directory / "summary.md").write_text(
        _benchmark_markdown(result), encoding="utf-8"
    )
    return directory


def _incident_markdown(incident: "Incident", run: "InvestigationRun") -> str:
    report = run.report
    changes = "\n".join(
        f"- `{change.field_name}`: `{change.before}` -> `{change.after}`"
        for change in incident.changes
    )
    hypotheses = (
        "\n".join(
            f"- `{hypothesis.root_cause.value}` (prior {hypothesis.prior_confidence:.2f})"
            for hypothesis in run.investigator.hypotheses
        )
        if run.investigator
        else "- None; deterministic detector found no material incident."
    )
    experiments = (
        "\n".join(
            f"- `{verification.experiment.experiment_id}`: "
            f"`{verification.experiment.hypothesis.root_cause.value}` -> "
            f"**{verification.outcome.value}**; {verification.rationale}"
            for verification in run.verifications
        )
        if run.verifications
        else "- No controlled experiment required."
    )
    return f"""# TRACEBACK INCIDENT REPORT

## Incident ID

{display_incident_id(incident.incident_id)} (`{incident.incident_id}`)

## Status

`{report.status.value}`

## OBSERVED - Regression

{_bullets(report.observed)}

## Affected Metrics

{_bullets(report.affected_metrics) if report.affected_metrics else '- None above detector thresholds.'}

## OBSERVED - Configuration Changes

{changes}

## INFERRED - Competing Hypotheses

{hypotheses}

## Controlled Experiments

{experiments}

## SUPPORTED / CHALLENGED BY REPLAY

{_bullets(report.replay_evidence)}

## Final RCA

`{report.predicted_root_cause.value}`

{report.final_assessment}

## Confidence

{report.confidence:.1%}

## RECOMMENDED - Not Executed

{report.recommended_action}

## Human Approval Required

`{str(report.human_approval_required).lower()}`
"""


def _benchmark_markdown(result: "BenchmarkResult") -> str:
    rows = []
    for case in result.cases:
        baseline_prediction = (
            case.baseline.predicted_root_cause.value if case.baseline else "ERROR"
        )
        baseline_correct = str(case.baseline.correct) if case.baseline else "False"
        traceback_prediction = (
            case.traceback.predicted_root_cause.value if case.traceback else "ERROR"
        )
        traceback_correct = str(case.traceback.correct) if case.traceback else "False"
        experiments = case.traceback.experiments_used if case.traceback else 0
        rows.append(
            f"| {display_incident_id(case.incident_id)} | {case.ground_truth.value} | "
            f"{baseline_prediction} | {baseline_correct} | {traceback_prediction} | "
            f"{traceback_correct} | {experiments} |"
        )
    i10 = next((case for case in result.cases if case.incident_id == "I10"), None)
    showcase = _showcase_markdown(i10) if i10 else "INC-10 was not included in this run."
    absolute_change = result.traceback_accuracy - result.baseline_accuracy
    return f"""# Traceback Benchmark Summary

Model: `{result.model}`

Cases: {len(result.cases)}

## Primary Metric

- Baseline RCA Accuracy: **{result.baseline_correct} / {len(result.cases)} = {result.baseline_accuracy:.1%}**
- Traceback RCA Accuracy: **{result.traceback_correct} / {len(result.cases)} = {result.traceback_accuracy:.1%}**
- Absolute change: **{absolute_change:+.1%}**

## Secondary Metrics

- Baseline healthy-case false positive: `{result.healthy_false_positive_baseline}`
- Traceback healthy-case false positive: `{result.healthy_false_positive_traceback}`
- Mean Traceback experiments per true incident: `{result.mean_traceback_experiments_per_true_incident:.4f}`
- Replay-supported diagnoses: `{result.replay_supported_diagnosis_count}` ({result.replay_supported_diagnosis_rate:.1%})
- Mean baseline latency: `{result.mean_baseline_latency_ms}` ms
- Mean Traceback latency: `{result.mean_traceback_latency_ms}` ms
- Baseline input/output tokens: `{result.baseline_input_tokens}` / `{result.baseline_output_tokens}`
- Traceback input/output tokens: `{result.traceback_input_tokens}` / `{result.traceback_output_tokens}`

## Per-case Results

| Incident | Ground Truth | Baseline Prediction | Baseline Correct | Traceback Prediction | Traceback Correct | Experiments |
|---|---|---|---:|---|---:|---:|
{chr(10).join(rows)}

## Showcase Incident - INC-10

{showcase}

This summary is generated from the recorded run. No LLM judge or invented benchmark
number is used.
"""


def _showcase_markdown(case: "CaseBenchmarkResult") -> str:
    if not case.traceback:
        return f"Traceback failed for this case: `{case.traceback_error}`"
    lines = []
    for replay in case.traceback.replay_evidence:
        hypothesis = replay.get("hypothesis", "unknown")
        quality_delta = replay.get("quality_delta", "unknown")
        outcome = replay.get("outcome", "unknown")
        lines.append(
            f"- `{hypothesis}`: aggregate delta `{quality_delta}`, outcome `{outcome}`"
        )
    lines.append(
        f"- Final supported cause: `{case.traceback.predicted_root_cause.value}`"
    )
    return "\n".join(lines)


def _timestamp_label(timestamp: datetime | None) -> str:
    value = timestamp or datetime.now(timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _unique_directory(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    candidate = target
    suffix = 1
    while candidate.exists():
        candidate = target.with_name(f"{target.name}_{suffix:02d}")
        suffix += 1
    candidate.mkdir()
    return candidate


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _bullets(values: tuple[str, ...]) -> str:
    return "\n".join(f"- {value}" for value in values)

