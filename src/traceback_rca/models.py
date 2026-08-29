"""Strongly typed domain models for Traceback.

The investigator-facing :class:`Incident` deliberately has no ground-truth field.
Evaluation-only answers live in the separate :class:`GroundTruth` model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Optional, Tuple, Union


ConfigurationValue = Union[str, int, float, bool]


def _require_aware_timestamp(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must include timezone information")


def _require_unit_interval(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")


class ExperimentStatus(str, Enum):
    """Lifecycle state for a future controlled replay."""

    PLANNED = "planned"
    COMPLETED = "completed"
    FAILED = "failed"


class RCAStatus(str, Enum):
    """Verification state of an RCA result."""

    INCONCLUSIVE = "inconclusive"
    FALSIFIED = "falsified"
    VERIFIED = "verified"


@dataclass(frozen=True, slots=True)
class SystemConfiguration:
    """A reproducible snapshot of relevant LLM/RAG configuration."""

    prompt_profile: str
    retriever_top_k: int
    embedding_profile: str = "aligned_v1"
    index_profile: str = "current_v2"
    reranker_enabled: bool = True
    guardrail_profile: str = "balanced"
    tool_latency_profile: str = "healthy"
    context_profile: str = "standard"
    model_name: str = "synthetic-llm-v1"

    def __post_init__(self) -> None:
        if not self.prompt_profile:
            raise ValueError("prompt_profile cannot be empty")
        if self.retriever_top_k < 1:
            raise ValueError("retriever_top_k must be at least 1")
        for name in (
            "embedding_profile",
            "index_profile",
            "guardrail_profile",
            "tool_latency_profile",
            "context_profile",
            "model_name",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TelemetrySnapshot:
    """Quality and operational telemetry aggregated at one point in time."""

    captured_at: datetime
    answer_quality: float
    retrieval_relevance: float
    groundedness: float
    aggregate_quality: float
    p95_latency_ms: float
    tool_latency_ms: float
    guardrail_block_rate: float
    usable_answer_rate: float
    context_inclusion_rate: float
    fresh_evidence_rate: float
    retrieved_evidence_count: int
    included_context_count: int
    truncated_context_count: int
    blocked_answer_count: int
    stale_document_count: int
    sample_size: int

    def __post_init__(self) -> None:
        _require_aware_timestamp(self.captured_at, "captured_at")
        _require_unit_interval(self.answer_quality, "answer_quality")
        _require_unit_interval(self.retrieval_relevance, "retrieval_relevance")
        _require_unit_interval(self.groundedness, "groundedness")
        _require_unit_interval(self.aggregate_quality, "aggregate_quality")
        _require_unit_interval(self.guardrail_block_rate, "guardrail_block_rate")
        _require_unit_interval(self.usable_answer_rate, "usable_answer_rate")
        _require_unit_interval(self.context_inclusion_rate, "context_inclusion_rate")
        _require_unit_interval(self.fresh_evidence_rate, "fresh_evidence_rate")
        if self.p95_latency_ms < 0 or self.tool_latency_ms < 0:
            raise ValueError("latency values cannot be negative")
        for name in (
            "retrieved_evidence_count",
            "included_context_count",
            "truncated_context_count",
            "blocked_answer_count",
            "stale_document_count",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.sample_size < 1:
            raise ValueError("sample_size must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["captured_at"] = self.captured_at.isoformat()
        return result


@dataclass(frozen=True, slots=True)
class ConfigurationChange:
    """One auditable deployment-time configuration change."""

    field_name: str
    before: ConfigurationValue
    after: ConfigurationValue
    changed_at: datetime
    deployment_id: str

    def __post_init__(self) -> None:
        if not self.field_name:
            raise ValueError("field_name cannot be empty")
        if self.before == self.after:
            raise ValueError("a configuration change must alter the value")
        _require_aware_timestamp(self.changed_at, "changed_at")
        if not self.deployment_id:
            raise ValueError("deployment_id cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["changed_at"] = self.changed_at.isoformat()
        return result


@dataclass(frozen=True, slots=True)
class Incident:
    """Investigator-visible incident data, intentionally excluding its answer."""

    incident_id: str
    title: str
    description: str
    detected_at: datetime
    telemetry_before: TelemetrySnapshot
    telemetry_after: TelemetrySnapshot
    configuration_before: SystemConfiguration
    configuration_after: SystemConfiguration
    changes: Tuple[ConfigurationChange, ...]

    def __post_init__(self) -> None:
        if not self.incident_id or not self.title:
            raise ValueError("incident_id and title cannot be empty")
        _require_aware_timestamp(self.detected_at, "detected_at")
        if self.telemetry_before.captured_at >= self.telemetry_after.captured_at:
            raise ValueError("before telemetry must precede after telemetry")
        if not self.changes:
            raise ValueError("an incident must include at least one change")

    def to_investigator_dict(self) -> dict[str, Any]:
        """Serialize only evidence that an investigator is allowed to observe."""

        return {
            "incident_id": self.incident_id,
            "title": self.title,
            "description": self.description,
            "detected_at": self.detected_at.isoformat(),
            "telemetry_before": self.telemetry_before.to_dict(),
            "telemetry_after": self.telemetry_after.to_dict(),
            "configuration_before": self.configuration_before.to_dict(),
            "configuration_after": self.configuration_after.to_dict(),
            "changes": [change.to_dict() for change in self.changes],
        }


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """Evaluation-only injected answer, never nested inside an Incident."""

    incident_id: str
    root_cause: str


@dataclass(frozen=True, slots=True)
class RootCauseHypothesis:
    """A future investigator hypothesis and its current evidence balance."""

    hypothesis_id: str
    incident_id: str
    suspected_root_cause: str
    rationale: str
    evidence_for: Tuple[str, ...] = ()
    evidence_against: Tuple[str, ...] = ()
    confidence: float = 0.0

    def __post_init__(self) -> None:
        _require_unit_interval(self.confidence, "confidence")


@dataclass(frozen=True, slots=True)
class ReplayExperiment:
    """The specification and outcome placeholder for a future replay."""

    experiment_id: str
    incident_id: str
    hypothesis_id: str
    configuration_overrides: Mapping[str, ConfigurationValue]
    status: ExperimentStatus = ExperimentStatus.PLANNED
    observed_metrics: Optional[Mapping[str, float]] = None


@dataclass(frozen=True, slots=True)
class RCAResult:
    """Evidence-backed RCA output that always requires human approval."""

    incident_id: str
    hypotheses: Tuple[RootCauseHypothesis, ...]
    replay_experiments: Tuple[ReplayExperiment, ...]
    status: RCAStatus = RCAStatus.INCONCLUSIVE
    root_cause: Optional[str] = None
    confidence: float = 0.0
    evidence_summary: Tuple[str, ...] = ()
    recommended_remediation: Optional[str] = None
    requires_human_approval: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        _require_unit_interval(self.confidence, "confidence")
