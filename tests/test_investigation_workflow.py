"""Offline end-to-end tests for planning, replay verification, and reporting."""

import inspect
import json

import pytest

import traceback_rca.baseline as baseline_module
import traceback_rca.investigator as investigator_module
import traceback_rca.planner as planner_module
import traceback_rca.reporter as reporter_module
import traceback_rca.verifier as verifier_module
import traceback_rca.workflow as workflow_module
import traceback_rca
from traceback_rca.incidents import get_incident
from traceback_rca.investigator import (
    HypothesisInvestigator,
    InvestigationHypothesis,
    InterventionProposal,
)
from traceback_rca.planner import (
    SafeExperimentPlanner,
    UnsafeInterventionError,
)
from traceback_rca.providers import ScriptedProvider
from traceback_rca.reporter import ReportStatus
from traceback_rca.structured import StructuredOutputError
from traceback_rca.taxonomy import RootCauseCategory
from traceback_rca.verifier import VerificationOutcome
from traceback_rca.workflow import TracebackWorkflow


def _hypothesis(
    root_cause: str,
    field: str,
    value,
    confidence: float,
) -> dict:
    return {
        "root_cause": root_cause,
        "prior_confidence": confidence,
        "supporting_evidence": [f"visible change in {field}"],
        "contradicting_evidence": ["temporal association alone is not causal"],
        "proposed_intervention": {"field": field, "value": value},
    }


def _i10_response() -> dict:
    return {
        "hypotheses": [
            _hypothesis("prompt_regression", "prompt_profile", "stable_v1", 0.72),
            _hypothesis(
                "retriever_top_k_regression", "retriever_top_k", 8, 0.68
            ),
        ]
    }


def test_i10_workflow_uses_real_replay_evidence_to_overturn_prompt_hypothesis() -> None:
    provider = ScriptedProvider([_i10_response()])
    run = TracebackWorkflow(provider).investigate(get_incident("I10"))

    assert provider.call_count == 1
    assert {item.root_cause for item in run.investigator.hypotheses} == {
        RootCauseCategory.PROMPT_REGRESSION,
        RootCauseCategory.RETRIEVER_TOP_K_REGRESSION,
    }
    outcomes = {
        result.experiment.hypothesis.root_cause: result
        for result in run.verifications
    }
    assert outcomes[RootCauseCategory.PROMPT_REGRESSION].quality_delta == 0.0
    assert (
        outcomes[RootCauseCategory.PROMPT_REGRESSION].outcome
        is VerificationOutcome.FAILS_TO_SUPPORT
    )
    assert (
        outcomes[RootCauseCategory.RETRIEVER_TOP_K_REGRESSION].quality_delta
        == 0.8333
    )
    assert (
        outcomes[RootCauseCategory.RETRIEVER_TOP_K_REGRESSION].outcome
        is VerificationOutcome.SUPPORTED
    )
    assert (
        run.report.predicted_root_cause
        is RootCauseCategory.RETRIEVER_TOP_K_REGRESSION
    )
    assert run.report.status is ReportStatus.SUPPORTED_BY_REPLAY


def test_report_has_required_structure_and_human_approval() -> None:
    report = TracebackWorkflow(ScriptedProvider([_i10_response()])).investigate(
        get_incident("I10")
    ).report
    payload = report.to_dict()

    assert {
        "incident_id",
        "status",
        "affected_metrics",
        "predicted_root_cause",
        "confidence",
        "supporting_evidence",
        "contradicting_evidence",
        "experiments_run",
        "before_metrics",
        "degraded_metrics",
        "replay_results",
        "observed",
        "inferred",
        "replay_evidence",
        "final_assessment",
        "recommended_action",
        "human_approval_required",
    } <= payload.keys()
    assert payload["human_approval_required"] is True
    assert payload["experiments_run"] == 2
    assert "approved change" in payload["recommended_action"]
    serialized = json.dumps(payload)
    assert "ground_truth" not in serialized
    assert "expected_root_cause" not in serialized


@pytest.mark.parametrize(
    "proposal",
    [
        InterventionProposal("model_name", "arbitrary-model"),
        InterventionProposal("retriever_top_k", 999),
        InterventionProposal("retriever_top_k", "8"),
    ],
)
def test_unauthorized_or_unknown_intervention_is_rejected(proposal) -> None:
    with pytest.raises(UnsafeInterventionError):
        SafeExperimentPlanner().validate(get_incident("I10"), proposal)


def test_planner_enforces_experiment_limit() -> None:
    hypotheses = HypothesisInvestigator(ScriptedProvider([_i10_response()])).investigate(
        get_incident("I10")
    ).hypotheses
    planning = SafeExperimentPlanner(max_experiments=1).plan(
        get_incident("I10"), hypotheses
    )

    assert len(planning.experiments) == 1
    assert len(planning.rejected) == 1
    assert "experiment limit" in planning.rejected[0].reason


@pytest.mark.parametrize(
    "response",
    [
        "not-json",
        {"hypotheses": []},
        {"hypotheses": [_hypothesis("invalid", "prompt_profile", "stable_v1", 0.5),
                         _hypothesis("prompt_regression", "prompt_profile", "stable_v1", 0.5)]},
        {"hypotheses": [_hypothesis("prompt_regression", "prompt_profile", "stable_v1", 0.5),
                         _hypothesis("prompt_regression", "prompt_profile", "stable_v1", 0.4)]},
    ],
)
def test_investigator_surfaces_malformed_hypotheses(response) -> None:
    with pytest.raises(StructuredOutputError):
        HypothesisInvestigator(ScriptedProvider([response])).investigate(
            get_incident("I10")
        )


def test_production_workflow_modules_cannot_import_hidden_ground_truth() -> None:
    modules = (
        baseline_module,
        investigator_module,
        planner_module,
        verifier_module,
        reporter_module,
        workflow_module,
    )

    for module in modules:
        source = inspect.getsource(module)
        assert "get_ground_truth" not in source
        assert "traceback_rca.ground_truth" not in source
    assert not hasattr(traceback_rca, "get_ground_truth")
    assert not hasattr(traceback_rca, "GroundTruth")


def test_planner_records_unsafe_llm_proposal_without_executing_it() -> None:
    unsafe = InvestigationHypothesis(
        root_cause=RootCauseCategory.MODEL_REGRESSION,
        prior_confidence=0.4,
        supporting_evidence=(),
        contradicting_evidence=(),
        proposed_intervention=InterventionProposal("model_name", "malicious-value"),
    )
    planning = SafeExperimentPlanner().plan(get_incident("I10"), (unsafe,))

    assert planning.experiments == ()
    assert len(planning.rejected) == 1


def test_hypothesis_cannot_claim_support_from_an_unrelated_intervention() -> None:
    mismatched = InvestigationHypothesis(
        root_cause=RootCauseCategory.PROMPT_REGRESSION,
        prior_confidence=0.8,
        supporting_evidence=(),
        contradicting_evidence=(),
        proposed_intervention=InterventionProposal("retriever_top_k", 8),
    )
    planning = SafeExperimentPlanner().plan(get_incident("I10"), (mismatched,))

    assert planning.experiments == ()
    assert "cannot be tested" in planning.rejected[0].reason


def test_workflow_rejects_an_unbounded_experiment_limit() -> None:
    with pytest.raises(ValueError, match="max_experiments"):
        TracebackWorkflow(ScriptedProvider([]), max_experiments=4)
