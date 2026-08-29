"""Offline ten-case benchmark scoring, fairness, failure, and persistence tests."""

from datetime import datetime, timezone
import json

import traceback_rca.benchmark as benchmark_module
from tests.scripted_cases import baseline_response, investigator_response
from traceback_rca.benchmark import BenchmarkRunner
from traceback_rca.export import save_benchmark, save_investigation
from traceback_rca.incidents import get_incident, list_incidents
from traceback_rca.providers import ScriptedProvider
from traceback_rca.workflow import TracebackWorkflow


INCIDENT_IDS = tuple(incident.incident_id for incident in list_incidents())
TRUE_INCIDENT_IDS = tuple(incident_id for incident_id in INCIDENT_IDS if incident_id != "I09")


def _scripted_benchmark() -> tuple[BenchmarkRunner, ScriptedProvider, ScriptedProvider]:
    baseline_provider = ScriptedProvider(
        [baseline_response(incident_id) for incident_id in INCIDENT_IDS]
    )
    traceback_provider = ScriptedProvider(
        [investigator_response(incident_id) for incident_id in TRUE_INCIDENT_IDS]
    )
    return (
        BenchmarkRunner(
            baseline_provider, traceback_provider, model="scripted-offline-test"
        ),
        baseline_provider,
        traceback_provider,
    )


def test_ten_case_benchmark_scores_only_after_each_pair_of_predictions(monkeypatch) -> None:
    runner, baseline_provider, traceback_provider = _scripted_benchmark()
    real_get_ground_truth = benchmark_module.get_ground_truth
    score_calls = 0

    def guarded_ground_truth(incident_id: str):
        nonlocal score_calls
        score_calls += 1
        processed = INCIDENT_IDS[:score_calls]
        assert baseline_provider.call_count == score_calls
        assert traceback_provider.call_count == sum(item != "I09" for item in processed)
        return real_get_ground_truth(incident_id)

    monkeypatch.setattr(benchmark_module, "get_ground_truth", guarded_ground_truth)
    result = runner.run()

    assert len(result.cases) == 10
    assert result.baseline_correct == 10
    assert result.traceback_correct == 10
    assert result.baseline_accuracy == 1.0
    assert result.traceback_accuracy == 1.0
    assert result.healthy_false_positive_baseline is False
    assert result.healthy_false_positive_traceback is False
    assert result.replay_supported_diagnosis_count == 9
    assert result.replay_supported_diagnosis_rate == 1.0
    assert result.mean_traceback_experiments_per_true_incident == 1.1111
    assert baseline_provider.call_count == 10
    assert traceback_provider.call_count == 9


def test_benchmark_collects_per_case_failures_instead_of_aborting() -> None:
    result = BenchmarkRunner(
        ScriptedProvider([]), ScriptedProvider([]), model="offline-errors"
    ).run(("I01", "I02"))

    assert len(result.cases) == 2
    assert result.baseline_accuracy == 0.0
    assert result.traceback_accuracy == 0.0
    assert all(case.baseline is None and case.baseline_error for case in result.cases)
    assert all(case.traceback is None and case.traceback_error for case in result.cases)


def test_benchmark_result_serialization_has_required_schema() -> None:
    result = _scripted_benchmark()[0].run()
    payload = result.to_dict()

    assert payload["model"] == "scripted-offline-test"
    assert payload["case_count"] == 10
    assert "baseline_root_cause_accuracy" in payload["primary_metrics"]
    assert "healthy_false_positive_traceback" in payload["secondary_metrics"]
    assert len(payload["cases"]) == 10
    assert payload["cases"][0]["baseline"]["summary"]
    assert "supported_hypotheses" in payload["cases"][0]["traceback"]
    assert "replay_evidence" in payload["cases"][0]["traceback"]
    assert payload["cases"][0]["traceback"]["final_assessment"]


def test_benchmark_export_writes_json_markdown_and_per_case_evidence(tmp_path) -> None:
    result = _scripted_benchmark()[0].run()
    directory = save_benchmark(
        result,
        results_root=tmp_path,
        timestamp=datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc),
    )

    assert directory.name == "benchmark_20260829_120000"
    assert (directory / "metrics.json").is_file()
    assert (directory / "baseline_results.json").is_file()
    assert (directory / "traceback_results.json").is_file()
    assert (directory / "summary.md").is_file()
    assert (directory / "incidents" / "INC-10" / "case_result.json").is_file()
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    summary = (directory / "summary.md").read_text(encoding="utf-8")
    assert metrics["case_count"] == 10
    assert "Baseline RCA Accuracy" in summary
    assert "Showcase Incident - INC-10" in summary


def test_incident_report_export_is_human_readable_and_ground_truth_free(tmp_path) -> None:
    incident = get_incident("I10")
    run = TracebackWorkflow(
        ScriptedProvider([investigator_response("I10")])
    ).investigate(incident)
    directory = save_investigation(
        incident,
        run,
        results_root=tmp_path,
        timestamp=datetime(2026, 8, 29, 12, 30, tzinfo=timezone.utc),
    )

    assert (directory / "rca_report.json").is_file()
    assert (directory / "rca_report.md").is_file()
    assert (directory / "replay_evidence.json").is_file()
    all_text = "\n".join(
        path.read_text(encoding="utf-8") for path in directory.iterdir() if path.is_file()
    )
    assert "OBSERVED" in all_text
    assert "SUPPORTED / CHALLENGED BY REPLAY" in all_text
    assert "ground_truth" not in all_text
    assert "GEMINI_API_KEY" not in all_text
    assert "test-only-secret" not in all_text
