"""Fair one-request, no-tools LLM baseline and its command-line entry point."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from traceback_rca.incidents import get_incident, list_incidents
from traceback_rca.models import Incident
from traceback_rca.providers import GeminiProvider, LLMProvider, LLMResponse
from traceback_rca.structured import (
    StructuredOutputError,
    parse_json_object,
    require_confidence,
    require_string,
    require_string_tuple,
)
from traceback_rca.taxonomy import RootCauseCategory, root_cause_labels


BASELINE_RESPONSE_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "predicted_root_cause",
        "confidence",
        "summary",
        "supporting_evidence",
    ],
    "properties": {
        "predicted_root_cause": {
            "type": "string",
            "enum": list(root_cause_labels()),
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "summary": {"type": "string"},
        "supporting_evidence": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}


@dataclass(frozen=True, slots=True)
class BaselineDiagnosis:
    """Validated single-request baseline prediction."""

    incident_id: str
    predicted_root_cause: RootCauseCategory
    confidence: float
    summary: str
    supporting_evidence: Tuple[str, ...]
    provider_latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "predicted_root_cause": self.predicted_root_cause.value,
            "confidence": self.confidence,
            "summary": self.summary,
            "supporting_evidence": list(self.supporting_evidence),
            "provider_latency_ms": self.provider_latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


class Baseline:
    """Make exactly one diagnosis request from initial investigator-visible evidence."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def diagnose(self, incident: Incident) -> BaselineDiagnosis:
        response = self._provider.generate(
            _baseline_prompt(incident), BASELINE_RESPONSE_SCHEMA
        )
        return _parse_baseline_response(incident.incident_id, response)


def _baseline_prompt(incident: Incident) -> str:
    visible_evidence = json.dumps(
        incident.to_investigator_dict(), sort_keys=True, indent=2
    )
    return (
        "You are the single-prompt baseline for an LLM/RAG incident diagnosis. "
        "Make one best diagnosis using only the investigator-visible evidence below. "
        "You have no tools, replay results, hidden annotations, or ground truth. "
        "Choose predicted_root_cause from the schema vocabulary and cite only facts "
        "present in the evidence.\n\nINVESTIGATOR-VISIBLE INCIDENT:\n"
        + visible_evidence
    )


def _parse_baseline_response(
    incident_id: str, response: LLMResponse
) -> BaselineDiagnosis:
    data = parse_json_object(response)
    root_cause_label = require_string(data, "predicted_root_cause")
    try:
        root_cause = RootCauseCategory(root_cause_label)
    except ValueError as error:
        raise StructuredOutputError(
            f"predicted_root_cause is not in the allowed taxonomy: {root_cause_label}"
        ) from error

    return BaselineDiagnosis(
        incident_id=incident_id,
        predicted_root_cause=root_cause,
        confidence=require_confidence(data),
        summary=require_string(data, "summary"),
        supporting_evidence=require_string_tuple(data, "supporting_evidence"),
        provider_latency_ms=response.latency_ms,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "incident_id", choices=tuple(incident.incident_id for incident in list_incidents())
    )
    args = parser.parse_args()
    diagnosis = Baseline(GeminiProvider.from_environment()).diagnose(
        get_incident(args.incident_id)
    )

    print(f"BASELINE DIAGNOSIS: {diagnosis.incident_id}")
    print(f"predicted_root_cause={diagnosis.predicted_root_cause.value}")
    print(f"confidence={diagnosis.confidence:.4f}")
    print(f"summary={diagnosis.summary}")
    print("supporting_evidence:")
    for evidence in diagnosis.supporting_evidence:
        print(f"- {evidence}")


if __name__ == "__main__":
    main()
