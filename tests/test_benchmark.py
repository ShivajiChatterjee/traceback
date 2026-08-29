"""Offline ten-case benchmark scoring, fairness, failure, and persistence tests."""

from datetime import datetime, timezone
import json

import pytest

import traceback_rca.benchmark as benchmark_module
from tests.scripted_cases import baseline_response, investigator_response
from traceback_rca.benchmark import BenchmarkRunner, resume_failed_benchmark
from traceback_rca.export import save_benchmark, save_investigation
from traceback_rca.incidents import get_incident, list_incidents
from traceback_rca.providers import ProviderResponseError, ScriptedProvider
from traceback_rca.workflow import TracebackWorkflow


INCIDENT_IDS = tuple(incident.incident_id for incident in list_incidents())
TRUE_INCIDENT_IDS = tuple(incident_id for incident_id in INCIDENT_IDS if incident_id != "I09")


class OfflineFailingProvider:
    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, prompt, response_schema):
        self.call_count += 1
        raise ProviderResponseError("offline retry failure")


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


def _failed_i10_benchmark(tmp_path):
    baseline_provider = ScriptedProvider(
        [baseline_response(incident_id) for incident_id in INCIDENT_IDS]
    )
    traceback_provider = ScriptedProvider(
        [
            investigator_response(incident_id)
            for incident_id in TRUE_INCIDENT_IDS
            if incident_id != "I10"
        ]
    )
    result = BenchmarkRunner(
        baseline_provider, traceback_provider, model="scripted-offline-test"
    ).run()
    directory = save_benchmark(
        result,
        results_root=tmp_path,
        timestamp=datetime(2026, 8, 29, 17, 1, 18, tzinfo=timezone.utc),
    )

    provider_error = "ClientError (status=429): request or workflow failed"
    traceback_path = directory / "traceback_results.json"
    traceback_records = json.loads(traceback_path.read_text(encoding="utf-8"))
    traceback_records[-1]["error"] = provider_error
    traceback_path.write_text(
        json.dumps(traceback_records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    case_path = directory / "incidents" / "INC-10" / "case_result.json"
    case = json.loads(case_path.read_text(encoding="utf-8"))
    case["traceback_error"] = provider_error
    case_path.write_text(
        json.dumps(case, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return directory


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
    assert result.baseline_accuracy is None
    assert result.traceback_accuracy is None
    assert result.baseline_completed_cases == 0
    assert result.traceback_completed_cases == 0
    assert result.baseline_provider_errors == 2
    assert result.traceback_provider_errors == 2
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


def test_resume_retries_only_failed_case_and_recomputes_metrics(tmp_path) -> None:
    directory = _failed_i10_benchmark(tmp_path)
    baseline_before = (directory / "baseline_results.json").read_bytes()
    traceback_before = json.loads(
        (directory / "traceback_results.json").read_text(encoding="utf-8")
    )
    baseline_provider = ScriptedProvider([])
    traceback_provider = ScriptedProvider([investigator_response("I10")])

    resumed = resume_failed_benchmark(
        directory,
        baseline_provider,
        traceback_provider,
        model="scripted-offline-test",
    )

    assert resumed.retried == (("traceback", "I10"),)
    assert baseline_provider.call_count == 0
    assert traceback_provider.call_count == 1
    assert (directory / "baseline_results.json").read_bytes() == baseline_before
    traceback_after = json.loads(
        (directory / "traceback_results.json").read_text(encoding="utf-8")
    )
    assert traceback_after[:9] == traceback_before[:9]
    inc10 = traceback_after[-1]
    assert inc10["error"] is None
    assert inc10["result"]["predicted_root_cause"] == "retriever_top_k_regression"
    assert inc10["attempts"] == [
        {
            "attempt": 1,
            "error": "ClientError (status=429): request or workflow failed",
            "model": "scripted-offline-test",
            "prediction": None,
            "status": "provider_error",
        },
        {
            "attempt": 2,
            "error": None,
            "model": "scripted-offline-test",
            "prediction": "retriever_top_k_regression",
            "status": "success",
        },
    ]

    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    primary = metrics["primary_metrics"]
    assert primary["benchmark_complete"] is True
    assert primary["baseline_completed_cases"] == 10
    assert primary["traceback_completed_cases"] == 10
    assert primary["baseline_provider_errors"] == 0
    assert primary["traceback_provider_errors"] == 0
    assert primary["traceback_root_cause_accuracy"] == 1.0
    case = json.loads(
        (directory / "incidents" / "INC-10" / "case_result.json").read_text(
            encoding="utf-8"
        )
    )
    assert case["attempt_history"]["traceback"] == inc10["attempts"]
    assert case["traceback"]["correct"] is True
    assert json.loads(
        (directory / "incidents" / "INC-10" / "replay_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    assert "Status: **COMPLETE**" in (directory / "summary.md").read_text(
        encoding="utf-8"
    )


def test_failed_resume_retains_original_error_and_stays_incomplete(tmp_path) -> None:
    directory = _failed_i10_benchmark(tmp_path)
    baseline_provider = ScriptedProvider([])
    traceback_provider = OfflineFailingProvider()

    resumed = resume_failed_benchmark(
        directory,
        baseline_provider,
        traceback_provider,
        model="scripted-offline-test",
    )

    assert resumed.retried == (("traceback", "I10"),)
    assert baseline_provider.call_count == 0
    assert traceback_provider.call_count == 1
    inc10 = json.loads(
        (directory / "traceback_results.json").read_text(encoding="utf-8")
    )[-1]
    assert inc10["result"] is None
    assert inc10["error"].startswith("ProviderResponseError")
    assert inc10["attempts"][0]["error"] == (
        "ClientError (status=429): request or workflow failed"
    )
    assert inc10["attempts"][1]["status"] == "provider_error"
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    primary = metrics["primary_metrics"]
    assert primary["benchmark_complete"] is False
    assert primary["traceback_completed_cases"] == 9
    assert primary["traceback_provider_errors"] == 1
    assert primary["traceback_root_cause_accuracy"] is None
    summary = (directory / "summary.md").read_text(encoding="utf-8")
    assert "Status: **INCOMPLETE**" in summary
    assert "final accuracy withheld" in summary


def test_resume_rejects_model_change_before_any_retry(tmp_path) -> None:
    directory = _failed_i10_benchmark(tmp_path)
    baseline_provider = ScriptedProvider([])
    traceback_provider = ScriptedProvider([investigator_response("I10")])

    with pytest.raises(ValueError, match="original benchmark model"):
        resume_failed_benchmark(
            directory,
            baseline_provider,
            traceback_provider,
            model="different-model",
        )

    assert baseline_provider.call_count == 0
    assert traceback_provider.call_count == 0
