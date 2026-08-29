"""Validation for the synthetic incident benchmark definitions."""

import json
from dataclasses import fields

import pytest

from traceback_rca.ground_truth import get_ground_truth
from traceback_rca.incidents import get_incident, list_incidents
from traceback_rca.models import Incident


@pytest.mark.parametrize(
    ("incident_id", "expected_root_cause"),
    [
        ("I01", "retriever_top_k_regression"),
        ("I03", "prompt_regression"),
        ("I10", "retriever_top_k_regression"),
    ],
)
def test_incident_is_valid_and_has_separate_ground_truth(
    incident_id: str, expected_root_cause: str
) -> None:
    incident = get_incident(incident_id)
    truth = get_ground_truth(incident_id)

    assert incident.incident_id == incident_id
    assert incident.telemetry_after.answer_quality < incident.telemetry_before.answer_quality
    assert truth.incident_id == incident_id
    assert truth.root_cause == expected_root_cause


def test_incident_ids_are_unique() -> None:
    ids = [incident.incident_id for incident in list_incidents()]

    assert len(ids) == 3
    assert len(ids) == len(set(ids))
    assert set(ids) == {"I01", "I03", "I10"}


def test_i01_contains_only_intended_top_k_change() -> None:
    incident = get_incident("I01")

    assert incident.configuration_before.retriever_top_k == 8
    assert incident.configuration_after.retriever_top_k == 2
    assert (
        incident.configuration_before.prompt_profile
        == incident.configuration_after.prompt_profile
    )
    assert [change.field_name for change in incident.changes] == ["retriever_top_k"]


def test_i03_contains_only_intended_prompt_change() -> None:
    incident = get_incident("I03")

    assert incident.configuration_before.prompt_profile == "stable_v1"
    assert incident.configuration_after.prompt_profile == "regressed"
    assert (
        incident.configuration_before.retriever_top_k
        == incident.configuration_after.retriever_top_k
    )
    assert [change.field_name for change in incident.changes] == ["prompt_profile"]


def test_i10_contains_both_prompt_and_top_k_changes() -> None:
    incident = get_incident("I10")

    assert incident.configuration_before.prompt_profile == "stable_v1"
    assert incident.configuration_after.prompt_profile == "stable_v1_1"
    assert incident.configuration_before.retriever_top_k == 8
    assert incident.configuration_after.retriever_top_k == 2
    assert {change.field_name for change in incident.changes} == {
        "prompt_profile",
        "retriever_top_k",
    }


@pytest.mark.parametrize("incident", list_incidents())
def test_investigator_representation_cannot_reveal_ground_truth(incident: Incident) -> None:
    payload = incident.to_investigator_dict()
    serialized = json.dumps(payload)
    model_fields = {model_field.name for model_field in fields(Incident)}

    assert "ground_truth" not in model_fields
    assert "ground_truth" not in payload
    assert "root_cause" not in payload
    assert get_ground_truth(incident.incident_id).root_cause not in serialized


@pytest.mark.parametrize("incident", list_incidents())
def test_investigator_serialization_is_json_compatible(incident: Incident) -> None:
    payload = incident.to_investigator_dict()

    round_tripped = json.loads(json.dumps(payload))
    assert round_tripped["incident_id"] == incident.incident_id
    assert round_tripped["detected_at"].endswith("+00:00")
    assert isinstance(round_tripped["changes"], list)
