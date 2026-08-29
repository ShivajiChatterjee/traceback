"""Causal-behavior tests for controlled counterfactual replay."""

import json
from dataclasses import fields

import pytest

from traceback_rca.evaluator import (
    MATERIAL_QUALITY_IMPROVEMENT,
    evaluate_configuration,
)
from traceback_rca.ground_truth import get_ground_truth
from traceback_rca.incidents import get_incident
from traceback_rca.models import SystemConfiguration
from traceback_rca.replay import replay


def _quality(incident_id: str, before: bool) -> float:
    incident = get_incident(incident_id)
    configuration = (
        incident.configuration_before if before else incident.configuration_after
    )
    return evaluate_configuration(configuration).metrics.aggregate_quality


def test_i01_degrades_and_restoring_top_k_materially_recovers() -> None:
    incident = get_incident("I01")
    result = replay(incident, {"retriever_top_k": 8})

    assert _quality("I01", before=False) < _quality("I01", before=True)
    assert (
        result.delta_from_degraded.aggregate_quality
        >= MATERIAL_QUALITY_IMPROVEMENT
    )
    assert result.replay_metrics.aggregate_quality == _quality("I01", before=True)


def test_i03_has_stable_retrieval_and_prompt_restore_materially_recovers() -> None:
    incident = get_incident("I03")
    before = evaluate_configuration(incident.configuration_before).metrics
    after = evaluate_configuration(incident.configuration_after).metrics
    result = replay(incident, {"prompt_profile": "stable_v1"})

    assert abs(after.retrieval_relevance - before.retrieval_relevance) <= 0.01
    assert after.answer_quality < before.answer_quality
    assert (
        result.delta_from_degraded.aggregate_quality
        >= MATERIAL_QUALITY_IMPROVEMENT
    )


def test_i10_prompt_only_restore_does_not_materially_recover() -> None:
    incident = get_incident("I10")
    result = replay(incident, {"prompt_profile": "stable_v1"})

    assert _quality("I10", before=False) < _quality("I10", before=True)
    assert (
        abs(result.delta_from_degraded.aggregate_quality)
        < MATERIAL_QUALITY_IMPROVEMENT
    )
    assert result.replay_configuration.retriever_top_k == 2


def test_i10_top_k_only_restore_materially_recovers() -> None:
    incident = get_incident("I10")
    result = replay(incident, {"retriever_top_k": 8})

    assert (
        result.delta_from_degraded.aggregate_quality
        >= MATERIAL_QUALITY_IMPROVEMENT
    )
    assert result.replay_configuration.prompt_profile == "stable_v1_1"
    assert result.replay_metrics.aggregate_quality == _quality("I10", before=True)


@pytest.mark.parametrize(
    ("incident_id", "intervention"),
    [
        ("I01", {"retriever_top_k": 8}),
        ("I03", {"prompt_profile": "stable_v1"}),
        ("I10", {"prompt_profile": "stable_v1"}),
        ("I10", {"retriever_top_k": 8}),
    ],
)
def test_replay_changes_only_requested_fields(incident_id: str, intervention) -> None:
    incident = get_incident(incident_id)
    result = replay(incident, intervention)
    after = incident.configuration_after.to_dict()
    replayed = result.replay_configuration.to_dict()
    changed_fields = {
        field.name
        for field in fields(SystemConfiguration)
        if after[field.name] != replayed[field.name]
    }

    assert changed_fields == set(intervention)


def test_top_k_alias_is_normalized_without_changing_other_fields() -> None:
    incident = get_incident("I01")
    result = replay(incident, {"top_k": 8})

    assert result.intervention == {"retriever_top_k": 8}
    assert result.replay_configuration.retriever_top_k == 8
    assert (
        result.replay_configuration.prompt_profile
        == incident.configuration_after.prompt_profile
    )


@pytest.mark.parametrize("intervention", ({}, {"not_a_field": "value"}))
def test_invalid_intervention_is_rejected(intervention) -> None:
    with pytest.raises(ValueError):
        replay(get_incident("I01"), intervention)


@pytest.mark.parametrize("incident_id", ("I01", "I03", "I10"))
def test_replay_serialization_does_not_expose_ground_truth(incident_id: str) -> None:
    incident = get_incident(incident_id)
    change = incident.changes[0]
    result = replay(incident, {change.field_name: change.before})
    serialized = json.dumps(result.to_dict())

    assert "ground_truth" not in serialized
    assert "root_cause" not in serialized
    assert get_ground_truth(incident_id).root_cause not in serialized
