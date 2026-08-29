"""Validation for all ten investigator-visible benchmark incidents."""

import json
from dataclasses import fields

import pytest

from tests.scripted_cases import CASE_CAUSES
from traceback_rca.detector import IncidentDetector
from traceback_rca.ground_truth import get_ground_truth
from traceback_rca.incidents import display_incident_id, get_incident, list_incidents
from traceback_rca.models import Incident


@pytest.mark.parametrize("incident_id", tuple(CASE_CAUSES))
def test_incident_is_valid_with_separate_expected_ground_truth(incident_id: str) -> None:
    incident = get_incident(incident_id)
    truth = get_ground_truth(incident_id)

    assert incident.incident_id == incident_id
    assert incident.telemetry_before.sample_size == 12
    assert incident.telemetry_after.sample_size == 12
    assert truth.incident_id == incident_id
    assert truth.root_cause == CASE_CAUSES[incident_id][0]


def test_incident_ids_are_unique_complete_and_displayable() -> None:
    ids = [incident.incident_id for incident in list_incidents()]

    assert ids == [f"I{number:02d}" for number in range(1, 11)]
    assert len(ids) == len(set(ids)) == 10
    assert display_incident_id("I01") == "INC-01"
    assert display_incident_id("I10") == "INC-10"


@pytest.mark.parametrize(
    ("incident_id", "expected_fields"),
    [
        ("I01", {"retriever_top_k"}),
        ("I02", {"embedding_profile"}),
        ("I03", {"prompt_profile"}),
        ("I04", {"index_profile"}),
        ("I05", {"reranker_enabled"}),
        ("I06", {"guardrail_profile"}),
        ("I07", {"tool_latency_profile"}),
        ("I08", {"context_profile"}),
        ("I09", {"prompt_profile"}),
        ("I10", {"prompt_profile", "retriever_top_k"}),
    ],
)
def test_incident_contains_exact_intended_configuration_changes(
    incident_id: str, expected_fields: set[str]
) -> None:
    incident = get_incident(incident_id)

    assert {change.field_name for change in incident.changes} == expected_fields
    for change in incident.changes:
        assert getattr(incident.configuration_before, change.field_name) == change.before
        assert getattr(incident.configuration_after, change.field_name) == change.after


def test_i10_contains_neutral_prompt_and_harmful_top_k_changes() -> None:
    incident = get_incident("I10")

    assert incident.configuration_before.prompt_profile == "stable_v1"
    assert incident.configuration_after.prompt_profile == "stable_v1_1"
    assert incident.configuration_before.retriever_top_k == 8
    assert incident.configuration_after.retriever_top_k == 2


def test_i09_is_visible_but_not_a_material_incident() -> None:
    incident = get_incident("I09")
    detection = IncidentDetector().detect(incident)

    assert incident.configuration_before.prompt_profile == "stable_v1"
    assert incident.configuration_after.prompt_profile == "stable_canary"
    assert not detection.material_incident


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
    round_tripped = json.loads(json.dumps(incident.to_investigator_dict()))

    assert round_tripped["incident_id"] == incident.incident_id
    assert round_tripped["detected_at"].endswith("+00:00")
    assert isinstance(round_tripped["changes"], list)
    assert "aggregate_quality" in round_tripped["telemetry_after"]
    assert "tool_latency_ms" in round_tripped["telemetry_after"]
