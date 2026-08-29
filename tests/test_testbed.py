"""Tests for the fixed evaluation workload and deterministic scoring."""

import json

import pytest

from traceback_rca.dataset import EVALUATION_DATASET
from traceback_rca.evaluator import evaluate_configuration
from traceback_rca.ground_truth import get_ground_truth
from traceback_rca.incidents import get_incident, list_incidents
from traceback_rca.models import SystemConfiguration


def test_evaluation_dataset_is_small_fixed_and_complete() -> None:
    assert len(EVALUATION_DATASET) == 12
    assert len({case.query_id for case in EVALUATION_DATASET}) == 12
    assert all(len(case.expected_key_facts) == 2 for case in EVALUATION_DATASET)
    assert all(
        case.gold_document_id in case.candidate_ranking
        for case in EVALUATION_DATASET
    )


def test_identical_configuration_produces_identical_results() -> None:
    configuration = SystemConfiguration(
        prompt_profile="stable_v1", retriever_top_k=8
    )

    assert evaluate_configuration(configuration) == evaluate_configuration(configuration)


def test_healthy_configuration_retrieves_and_answers_all_cases() -> None:
    result = evaluate_configuration(
        SystemConfiguration(prompt_profile="stable_v1", retriever_top_k=8)
    )

    assert result.metrics.retrieval_relevance == 1.0
    assert result.metrics.groundedness == 1.0
    assert result.metrics.answer_quality == 1.0
    assert result.metrics.aggregate_quality == 1.0
    assert all(query.gold_document_retrieved for query in result.query_results)


def test_regressed_prompt_preserves_retrieval_but_harms_answering() -> None:
    stable = evaluate_configuration(
        SystemConfiguration(prompt_profile="stable_v1", retriever_top_k=8)
    )
    regressed = evaluate_configuration(
        SystemConfiguration(prompt_profile="regressed", retriever_top_k=8)
    )

    assert regressed.metrics.retrieval_relevance == stable.metrics.retrieval_relevance
    assert regressed.metrics.groundedness < stable.metrics.groundedness
    assert regressed.metrics.answer_quality < stable.metrics.answer_quality
    assert all(result.gold_document_retrieved for result in regressed.query_results)


def test_neutral_prompt_revision_has_equivalent_behavior() -> None:
    stable_v1 = evaluate_configuration(
        SystemConfiguration(prompt_profile="stable_v1", retriever_top_k=8)
    )
    stable_v1_1 = evaluate_configuration(
        SystemConfiguration(prompt_profile="stable_v1_1", retriever_top_k=8)
    )

    assert stable_v1.metrics == stable_v1_1.metrics


def test_unsupported_prompt_profile_fails_explicitly() -> None:
    configuration = SystemConfiguration(prompt_profile="unknown", retriever_top_k=8)

    with pytest.raises(ValueError, match="unsupported prompt_profile"):
        evaluate_configuration(configuration)


@pytest.mark.parametrize("incident", list_incidents())
def test_testbed_serialization_does_not_expose_incident_ground_truth(incident) -> None:
    evaluation = evaluate_configuration(incident.configuration_after)
    serialized = json.dumps(evaluation.to_dict())

    assert "ground_truth" not in serialized
    assert "root_cause" not in serialized
    assert get_ground_truth(incident.incident_id).root_cause not in serialized


@pytest.mark.parametrize("incident_id", ("I01", "I03", "I10"))
def test_incident_telemetry_is_generated_by_the_evaluator(incident_id: str) -> None:
    incident = get_incident(incident_id)
    before = evaluate_configuration(incident.configuration_before).metrics
    after = evaluate_configuration(incident.configuration_after).metrics

    assert incident.telemetry_before.retrieval_relevance == before.retrieval_relevance
    assert incident.telemetry_before.groundedness == before.groundedness
    assert incident.telemetry_before.answer_quality == before.answer_quality
    assert incident.telemetry_after.retrieval_relevance == after.retrieval_relevance
    assert incident.telemetry_after.groundedness == after.groundedness
    assert incident.telemetry_after.answer_quality == after.answer_quality
