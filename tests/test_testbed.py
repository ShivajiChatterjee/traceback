"""Deterministic dataset, fault semantics, and aggregate metric tests."""

import json

import pytest

from traceback_rca.dataset import EVALUATION_DATASET
from traceback_rca.evaluator import evaluate_configuration
from traceback_rca.ground_truth import get_ground_truth
from traceback_rca.incidents import get_incident, list_incidents
from traceback_rca.models import SystemConfiguration


def _metrics(incident_id: str, before: bool = False):
    incident = get_incident(incident_id)
    configuration = (
        incident.configuration_before if before else incident.configuration_after
    )
    return evaluate_configuration(configuration).metrics


def test_evaluation_dataset_is_fixed_readable_and_complete() -> None:
    assert len(EVALUATION_DATASET) == 12
    assert len({case.query_id for case in EVALUATION_DATASET}) == 12
    assert all(len(case.expected_key_facts) == 2 for case in EVALUATION_DATASET)
    assert all(len(case.candidate_ranking) == 10 for case in EVALUATION_DATASET)
    assert all(
        case.gold_document_id in case.candidate_ranking
        for case in EVALUATION_DATASET
    )


def test_identical_configuration_produces_identical_results() -> None:
    configuration = SystemConfiguration(
        prompt_profile="stable_v1", retriever_top_k=8
    )

    assert evaluate_configuration(configuration) == evaluate_configuration(configuration)


def test_healthy_configuration_has_complete_evidence_and_answers() -> None:
    result = evaluate_configuration(
        SystemConfiguration(prompt_profile="stable_v1", retriever_top_k=8)
    )

    assert result.metrics.retrieval_relevance == 1.0
    assert result.metrics.groundedness == 1.0
    assert result.metrics.answer_quality == 1.0
    assert result.metrics.aggregate_quality == 1.0
    assert result.metrics.context_inclusion_rate == 1.0
    assert result.metrics.fresh_evidence_rate == 1.0


def test_embedding_mismatch_changes_ranking_for_subset_not_freshness() -> None:
    after = _metrics("I02")

    assert after.retrieval_relevance == 0.6667
    assert after.answer_quality == 0.6667
    assert after.fresh_evidence_rate == 1.0
    result = evaluate_configuration(get_incident("I02").configuration_after)
    affected = [query for query in result.query_results if not query.gold_document_retrieved]
    assert {query.query_id for query in affected} == {"Q02", "Q06", "Q09", "Q12"}


def test_stale_index_removes_current_evidence_and_records_stale_documents() -> None:
    after = _metrics("I04")
    result = evaluate_configuration(get_incident("I04").configuration_after)

    assert after.retrieval_relevance == 0.75
    assert after.fresh_evidence_rate == 0.75
    assert after.stale_document_count == 3
    assert any(
        document.startswith("stale::")
        for query in result.query_results
        for document in query.candidate_document_ids
    )


def test_disabled_reranker_leaves_selected_gold_documents_below_top_k() -> None:
    healthy = evaluate_configuration(get_incident("I05").configuration_before)
    disabled = evaluate_configuration(get_incident("I05").configuration_after)

    assert disabled.metrics.retrieval_relevance == 0.75
    assert disabled.metrics.fresh_evidence_rate == 1.0
    healthy_q04 = next(item for item in healthy.query_results if item.query_id == "Q04")
    disabled_q04 = next(item for item in disabled.query_results if item.query_id == "Q04")
    assert healthy_q04.gold_rank_before_rerank == 9
    assert healthy_q04.gold_rank_after_rerank == 4
    assert disabled_q04.gold_rank_after_rerank == 9


def test_over_strict_guardrail_preserves_retrieval_but_blocks_valid_outputs() -> None:
    after = _metrics("I06")

    assert after.retrieval_relevance == 1.0
    assert after.groundedness == 1.0
    assert after.guardrail_rejection_rate == 0.3333
    assert after.usable_answer_rate == 0.6667
    assert after.blocked_answer_count == 4


def test_tool_latency_fault_preserves_all_content_quality() -> None:
    before = _metrics("I07", before=True)
    after = _metrics("I07")

    assert after.aggregate_quality == before.aggregate_quality == 1.0
    assert after.latency_ms - before.latency_ms == 840.0
    assert after.tool_latency_ms - before.tool_latency_ms == 840.0


def test_context_fault_preserves_retrieval_but_truncates_included_evidence() -> None:
    after = _metrics("I08")

    assert after.retrieval_relevance == 1.0
    assert after.context_inclusion_rate == 0.6667
    assert after.truncated_context_count == 4
    assert after.answer_quality == 0.6667


def test_prompt_fault_and_neutral_revisions_are_distinct() -> None:
    stable = _metrics("I03", before=True)
    regressed = _metrics("I03")
    neutral_i10 = evaluate_configuration(
        SystemConfiguration(prompt_profile="stable_v1_1", retriever_top_k=8)
    ).metrics
    canary = _metrics("I09")

    assert regressed.retrieval_relevance == stable.retrieval_relevance
    assert regressed.answer_quality == 0.5
    assert neutral_i10 == stable
    assert canary.answer_quality == 0.9583
    assert stable.aggregate_quality - canary.aggregate_quality == pytest.approx(0.0167)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("prompt_profile", "unknown"),
        ("embedding_profile", "unknown"),
        ("index_profile", "unknown"),
        ("guardrail_profile", "unknown"),
        ("tool_latency_profile", "unknown"),
        ("context_profile", "unknown"),
    ],
)
def test_unsupported_profile_fails_explicitly(field_name: str, bad_value: str) -> None:
    values = {"prompt_profile": "stable_v1", "retriever_top_k": 8}
    values[field_name] = bad_value
    configuration = SystemConfiguration(**values)

    with pytest.raises(ValueError, match=f"unsupported {field_name}"):
        evaluate_configuration(configuration)


@pytest.mark.parametrize("incident", list_incidents())
def test_testbed_serialization_does_not_expose_incident_ground_truth(incident) -> None:
    serialized = json.dumps(
        evaluate_configuration(incident.configuration_after).to_dict()
    )

    assert "ground_truth" not in serialized
    assert "root_cause" not in serialized
    assert get_ground_truth(incident.incident_id).root_cause not in serialized


@pytest.mark.parametrize("incident", list_incidents())
def test_incident_telemetry_is_generated_by_evaluator(incident) -> None:
    before = evaluate_configuration(incident.configuration_before).metrics
    after = evaluate_configuration(incident.configuration_after).metrics

    assert incident.telemetry_before.aggregate_quality == before.aggregate_quality
    assert incident.telemetry_before.p95_latency_ms == before.latency_ms
    assert incident.telemetry_after.retrieval_relevance == after.retrieval_relevance
    assert incident.telemetry_after.answer_quality == after.answer_quality
    assert incident.telemetry_after.guardrail_block_rate == after.guardrail_rejection_rate
    assert incident.telemetry_after.context_inclusion_rate == after.context_inclusion_rate
    assert incident.telemetry_after.fresh_evidence_rate == after.fresh_evidence_rate
