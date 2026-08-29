"""LLM reasoning stage that proposes ranked competing hypotheses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

from traceback_rca.models import ConfigurationValue, Incident
from traceback_rca.providers import LLMProvider, LLMResponse
from traceback_rca.structured import (
    StructuredOutputError,
    parse_json_object,
    require_confidence,
    require_object_sequence,
    require_string,
    require_string_tuple,
)
from traceback_rca.taxonomy import RootCauseCategory, root_cause_labels


INVESTIGATOR_RESPONSE_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["hypotheses"],
    "properties": {
        "hypotheses": {
            "type": "array",
            "minItems": 2,
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "root_cause",
                    "prior_confidence",
                    "supporting_evidence",
                    "contradicting_evidence",
                    "proposed_intervention",
                ],
                "properties": {
                    "root_cause": {
                        "type": "string",
                        "enum": list(root_cause_labels()),
                    },
                    "prior_confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "supporting_evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "contradicting_evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "proposed_intervention": {
                        "anyOf": [
                            {"type": "null"},
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["field", "value"],
                                "properties": {
                                    "field": {"type": "string"},
                                    "value": {
                                        "anyOf": [
                                            {"type": "string"},
                                            {"type": "integer"},
                                            {"type": "number"},
                                            {"type": "boolean"},
                                        ]
                                    },
                                },
                            },
                        ]
                    },
                },
            },
        }
    },
}


@dataclass(frozen=True, slots=True)
class InterventionProposal:
    field: str
    value: ConfigurationValue


@dataclass(frozen=True, slots=True)
class InvestigationHypothesis:
    root_cause: RootCauseCategory
    prior_confidence: float
    supporting_evidence: Tuple[str, ...]
    contradicting_evidence: Tuple[str, ...]
    proposed_intervention: Optional[InterventionProposal]


@dataclass(frozen=True, slots=True)
class InvestigatorResult:
    incident_id: str
    hypotheses: Tuple[InvestigationHypothesis, ...]
    provider_latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None


class HypothesisInvestigator:
    """Ask one LLM request for a small ranked set of competing explanations."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def investigate(self, incident: Incident) -> InvestigatorResult:
        response = self._provider.generate(
            _investigator_prompt(incident), INVESTIGATOR_RESPONSE_SCHEMA
        )
        return _parse_investigator_response(incident.incident_id, response)


def _investigator_prompt(incident: Incident) -> str:
    visible_evidence = json.dumps(
        incident.to_investigator_dict(), sort_keys=True, indent=2
    )
    return (
        "You are Traceback's hypothesis investigator. Generate 2 to 4 ranked, "
        "competing explanations using only the investigator-visible incident below. "
        "Do not declare a verdict. For a testable configuration hypothesis, propose "
        "one counterfactual that restores that changed field to its visible before "
        "value. The currently executable fields are retriever_top_k, prompt_profile, "
        "embedding_profile, index_profile, reranker_enabled, guardrail_profile, "
        "tool_latency_profile, and context_profile. Use null when no safe "
        "counterfactual exists. You have no "
        "ground truth, replay results, tools, or hidden causal annotations.\n\n"
        "INVESTIGATOR-VISIBLE INCIDENT:\n"
        + visible_evidence
    )


def _parse_investigator_response(
    incident_id: str, response: LLMResponse
) -> InvestigatorResult:
    data = parse_json_object(response)
    raw_hypotheses = require_object_sequence(data, "hypotheses")
    if not 2 <= len(raw_hypotheses) <= 4:
        raise StructuredOutputError("hypotheses must contain between 2 and 4 items")

    hypotheses = tuple(_parse_hypothesis(item) for item in raw_hypotheses)
    labels = [hypothesis.root_cause for hypothesis in hypotheses]
    if len(labels) != len(set(labels)):
        raise StructuredOutputError("hypotheses must use distinct root-cause categories")

    return InvestigatorResult(
        incident_id=incident_id,
        hypotheses=hypotheses,
        provider_latency_ms=response.latency_ms,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )


def _parse_hypothesis(data: Mapping[str, Any]) -> InvestigationHypothesis:
    root_cause_label = require_string(data, "root_cause")
    try:
        root_cause = RootCauseCategory(root_cause_label)
    except ValueError as error:
        raise StructuredOutputError(
            f"root_cause is not in the allowed taxonomy: {root_cause_label}"
        ) from error

    proposed = data.get("proposed_intervention")
    intervention: Optional[InterventionProposal]
    if proposed is None:
        intervention = None
    elif isinstance(proposed, dict):
        field_name = require_string(proposed, "field")
        value = proposed.get("value")
        if value is None or not isinstance(value, (str, int, float, bool)):
            raise StructuredOutputError(
                "proposed_intervention.value must be a scalar configuration value"
            )
        intervention = InterventionProposal(field_name, value)
    else:
        raise StructuredOutputError(
            "proposed_intervention must be an object or null"
        )

    return InvestigationHypothesis(
        root_cause=root_cause,
        prior_confidence=require_confidence(data, "prior_confidence"),
        supporting_evidence=require_string_tuple(data, "supporting_evidence"),
        contradicting_evidence=require_string_tuple(data, "contradicting_evidence"),
        proposed_intervention=intervention,
    )
