"""Offline three-case benchmark scoring with hidden-answer boundary checks."""

import traceback_rca.benchmark as benchmark_module
from traceback_rca.benchmark import BenchmarkRunner
from traceback_rca.providers import ScriptedProvider


def _baseline(root_cause: str) -> dict:
    return {
        "predicted_root_cause": root_cause,
        "confidence": 0.7,
        "summary": "Scripted unit-test prediction.",
        "supporting_evidence": ["visible incident evidence"],
    }


def _hypothesis(root_cause: str, field: str, value, confidence: float) -> dict:
    return {
        "root_cause": root_cause,
        "prior_confidence": confidence,
        "supporting_evidence": ["visible incident evidence"],
        "contradicting_evidence": ["requires replay"],
        "proposed_intervention": {"field": field, "value": value},
    }


def _investigation(*hypotheses: dict) -> dict:
    return {"hypotheses": list(hypotheses)}


def test_three_case_benchmark_scores_predictions_only_after_completion(monkeypatch) -> None:
    baseline_provider = ScriptedProvider(
        [
            _baseline("retriever_top_k_regression"),
            _baseline("prompt_regression"),
            _baseline("prompt_regression"),
        ]
    )
    traceback_provider = ScriptedProvider(
        [
            _investigation(
                _hypothesis("retriever_top_k_regression", "retriever_top_k", 8, 0.7),
                _hypothesis("prompt_regression", "prompt_profile", "stable_v1", 0.3),
            ),
            _investigation(
                _hypothesis("prompt_regression", "prompt_profile", "stable_v1", 0.7),
                _hypothesis("retriever_top_k_regression", "retriever_top_k", 8, 0.3),
            ),
            _investigation(
                _hypothesis("prompt_regression", "prompt_profile", "stable_v1", 0.7),
                _hypothesis("retriever_top_k_regression", "retriever_top_k", 8, 0.6),
            ),
        ]
    )
    real_get_ground_truth = benchmark_module.get_ground_truth
    score_calls = 0

    def guarded_ground_truth(incident_id: str):
        nonlocal score_calls
        score_calls += 1
        assert baseline_provider.call_count == score_calls
        assert traceback_provider.call_count == score_calls
        return real_get_ground_truth(incident_id)

    monkeypatch.setattr(benchmark_module, "get_ground_truth", guarded_ground_truth)
    result = BenchmarkRunner(baseline_provider, traceback_provider).run()

    assert len(result.cases) == 3
    assert result.baseline_accuracy == 2 / 3
    assert result.traceback_accuracy == 1.0
    assert [case.traceback.experiments_used for case in result.cases] == [1, 1, 2]

