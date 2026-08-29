"""Generalized controlled replay evidence tests."""

import json
from dataclasses import fields

import pytest

from tests.scripted_cases import CASE_CAUSES
from traceback_rca.ground_truth import get_ground_truth
from traceback_rca.incidents import get_incident
from traceback_rca.models import ExperimentStatus, SystemConfiguration
from traceback_rca.replay import replay


TRUE_INCIDENTS = ("I01", "I02", "I03", "I04", "I05", "I06", "I07", "I08", "I10")


@pytest.mark.parametrize("incident_id", TRUE_INCIDENTS)
def test_restoring_causal_field_reproduces_healthy_metrics(incident_id: str) -> None:
    incident = get_incident(incident_id)
    _, field_name, before_value = CASE_CAUSES[incident_id]
    result = replay(incident, {field_name: before_value})

    assert result.replay_metrics == result.healthy_metrics
    assert result.experiment_status is ExperimentStatus.COMPLETED
    assert result.before_values[field_name] == before_value
    assert result.degraded_values[field_name] == getattr(
        incident.configuration_after, field_name
    )
    assert result.replay_values[field_name] == before_value


def test_replay_preserves_fault_specific_metric_deltas() -> None:
    assert replay(
        get_incident("I02"), {"embedding_profile": "aligned_v1"}
    ).delta_from_degraded.retrieval_relevance == 0.3333
    assert replay(
        get_incident("I04"), {"index_profile": "current_v2"}
    ).delta_from_degraded.fresh_evidence_rate == 0.25
    assert replay(
        get_incident("I06"), {"guardrail_profile": "balanced"}
    ).delta_from_degraded.usable_answer_rate == 0.3333
    assert replay(
        get_incident("I07"), {"tool_latency_profile": "healthy"}
    ).delta_from_degraded.latency_ms == -840.0
    assert replay(
        get_incident("I08"), {"context_profile": "standard"}
    ).delta_from_degraded.context_inclusion_rate == 0.3333


def test_i10_prompt_only_replay_is_neutral_but_top_k_replay_recovers() -> None:
    incident = get_incident("I10")
    prompt = replay(incident, {"prompt_profile": "stable_v1"})
    top_k = replay(incident, {"retriever_top_k": 8})

    assert prompt.delta_from_degraded.aggregate_quality == 0.0
    assert prompt.replay_configuration.retriever_top_k == 2
    assert top_k.delta_from_degraded.aggregate_quality == 0.8333
    assert top_k.replay_configuration.prompt_profile == "stable_v1_1"


@pytest.mark.parametrize("incident_id", TRUE_INCIDENTS)
def test_replay_changes_only_requested_field(incident_id: str) -> None:
    incident = get_incident(incident_id)
    _, field_name, before_value = CASE_CAUSES[incident_id]
    result = replay(incident, {field_name: before_value})
    after = incident.configuration_after.to_dict()
    replayed = result.replay_configuration.to_dict()
    changed_fields = {
        model_field.name
        for model_field in fields(SystemConfiguration)
        if after[model_field.name] != replayed[model_field.name]
    }

    assert changed_fields == {field_name}


def test_top_k_alias_is_normalized() -> None:
    result = replay(get_incident("I01"), {"top_k": 8})

    assert result.intervention == {"retriever_top_k": 8}
    assert result.replay_configuration.retriever_top_k == 8


@pytest.mark.parametrize("intervention", ({}, {"not_a_field": "value"}))
def test_invalid_intervention_is_rejected(intervention) -> None:
    with pytest.raises(ValueError):
        replay(get_incident("I01"), intervention)


@pytest.mark.parametrize("incident_id", TRUE_INCIDENTS)
def test_replay_serialization_is_rich_and_ground_truth_free(incident_id: str) -> None:
    incident = get_incident(incident_id)
    _, field_name, before_value = CASE_CAUSES[incident_id]
    payload = replay(incident, {field_name: before_value}).to_dict()
    serialized = json.dumps(payload)

    assert payload["experiment_status"] == "completed"
    assert "before_configuration" in payload
    assert "degraded_configuration" in payload
    assert "replay_configuration" in payload
    assert "delta_from_degraded" in payload
    assert "ground_truth" not in serialized
    assert "root_cause" not in serialized
    assert get_ground_truth(incident_id).root_cause not in serialized
