"""Safe boundary between LLM proposals and local synthetic replay execution."""

from dataclasses import dataclass
from typing import Tuple

from traceback_rca.investigator import InvestigationHypothesis, InterventionProposal
from traceback_rca.models import ConfigurationValue, Incident
from traceback_rca.taxonomy import RootCauseCategory


ALLOWED_INTERVENTION_FIELDS = frozenset(
    {
        "retriever_top_k",
        "prompt_profile",
        "embedding_profile",
        "index_profile",
        "reranker_enabled",
        "guardrail_profile",
        "tool_latency_profile",
        "context_profile",
    }
)
MAX_EXPERIMENTS = 3
_CATEGORY_INTERVENTION_FIELDS = {
    RootCauseCategory.RETRIEVER_TOP_K_REGRESSION: "retriever_top_k",
    RootCauseCategory.PROMPT_REGRESSION: "prompt_profile",
    RootCauseCategory.EMBEDDING_REGRESSION: "embedding_profile",
    RootCauseCategory.STALE_INDEX: "index_profile",
    RootCauseCategory.RERANKER_DISABLED: "reranker_enabled",
    RootCauseCategory.GUARDRAIL_REGRESSION: "guardrail_profile",
    RootCauseCategory.TOOL_LATENCY_REGRESSION: "tool_latency_profile",
    RootCauseCategory.CONTEXT_TRUNCATION: "context_profile",
}


class UnsafeInterventionError(ValueError):
    """Raised when an LLM proposal is not a known safe local rollback."""


@dataclass(frozen=True, slots=True)
class ValidatedIntervention:
    field: str
    value: ConfigurationValue

    def to_dict(self) -> dict[str, ConfigurationValue]:
        return {self.field: self.value}


@dataclass(frozen=True, slots=True)
class PlannedExperiment:
    experiment_id: str
    hypothesis: InvestigationHypothesis
    intervention: ValidatedIntervention


@dataclass(frozen=True, slots=True)
class RejectedIntervention:
    hypothesis: InvestigationHypothesis
    reason: str


@dataclass(frozen=True, slots=True)
class PlanningResult:
    experiments: Tuple[PlannedExperiment, ...]
    rejected: Tuple[RejectedIntervention, ...]


class SafeExperimentPlanner:
    """Accept only allowlisted restorations to incident-visible before values."""

    def __init__(self, max_experiments: int = MAX_EXPERIMENTS) -> None:
        if not 1 <= max_experiments <= MAX_EXPERIMENTS:
            raise ValueError(
                f"max_experiments must be between 1 and {MAX_EXPERIMENTS}"
            )
        self._max_experiments = max_experiments

    def validate(
        self, incident: Incident, proposal: InterventionProposal
    ) -> ValidatedIntervention:
        if proposal.field not in ALLOWED_INTERVENTION_FIELDS:
            raise UnsafeInterventionError(
                f"field {proposal.field!r} is not allowlisted for local replay"
            )

        matching_changes = tuple(
            change for change in incident.changes if change.field_name == proposal.field
        )
        if len(matching_changes) != 1:
            raise UnsafeInterventionError(
                f"field {proposal.field!r} is not one visible, unambiguous incident change"
            )
        change = matching_changes[0]
        configured_before = getattr(incident.configuration_before, proposal.field)
        configured_after = getattr(incident.configuration_after, proposal.field)
        if configured_before != change.before or configured_after != change.after:
            raise UnsafeInterventionError(
                f"incident change record for {proposal.field!r} is inconsistent"
            )
        if type(proposal.value) is not type(change.before) or proposal.value != change.before:
            raise UnsafeInterventionError(
                f"field {proposal.field!r} may only be restored to its known before value"
            )
        if proposal.value == configured_after:
            raise UnsafeInterventionError("intervention must change the degraded configuration")
        return ValidatedIntervention(proposal.field, proposal.value)

    def plan(
        self,
        incident: Incident,
        hypotheses: Tuple[InvestigationHypothesis, ...],
    ) -> PlanningResult:
        experiments: list[PlannedExperiment] = []
        rejected: list[RejectedIntervention] = []
        seen_interventions: set[tuple[str, ConfigurationValue]] = set()

        for hypothesis in hypotheses:
            proposal = hypothesis.proposed_intervention
            if proposal is None:
                continue
            try:
                expected_field = _CATEGORY_INTERVENTION_FIELDS.get(
                    hypothesis.root_cause
                )
                if expected_field != proposal.field:
                    raise UnsafeInterventionError(
                        f"{hypothesis.root_cause.value} cannot be tested by changing "
                        f"{proposal.field!r}"
                    )
                intervention = self.validate(incident, proposal)
                key = (intervention.field, intervention.value)
                if key in seen_interventions:
                    raise UnsafeInterventionError("duplicate intervention was not scheduled")
                seen_interventions.add(key)
            except UnsafeInterventionError as error:
                rejected.append(RejectedIntervention(hypothesis, str(error)))
                continue

            if len(experiments) >= self._max_experiments:
                rejected.append(
                    RejectedIntervention(
                        hypothesis,
                        f"experiment limit of {self._max_experiments} reached",
                    )
                )
                continue
            experiments.append(
                PlannedExperiment(
                    experiment_id=f"{incident.incident_id}-E{len(experiments) + 1}",
                    hypothesis=hypothesis,
                    intervention=intervention,
                )
            )

        return PlanningResult(tuple(experiments), tuple(rejected))
