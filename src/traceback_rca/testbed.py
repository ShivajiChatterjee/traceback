"""Deterministic production-like RAG fault-injection testbed."""

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Tuple

from traceback_rca.dataset import EVALUATION_DATASET, EvaluationCase
from traceback_rca.models import SystemConfiguration


@dataclass(frozen=True, slots=True)
class _PromptBehavior:
    expected_fact_fraction: float
    unsupported_fact_count: int


_PROMPT_BEHAVIORS: Mapping[str, _PromptBehavior] = {
    "stable_v1": _PromptBehavior(1.0, 0),
    "stable_v1_1": _PromptBehavior(1.0, 0),
    "stable_canary": _PromptBehavior(1.0, 0),
    "regressed": _PromptBehavior(0.5, 1),
}
_EMBEDDING_PROFILES = frozenset({"aligned_v1", "mismatched_v2"})
_INDEX_PROFILES = frozenset({"current_v2", "stale_v1"})
_GUARDRAIL_PROFILES = frozenset({"balanced", "over_strict"})
_TOOL_LATENCIES: Mapping[str, float] = {"healthy": 60.0, "slow": 900.0}
_CONTEXT_PROFILES = frozenset({"standard", "truncated"})

# Disjoint fault subsets make retrieval mechanisms diagnostically distinguishable.
_EMBEDDING_MISMATCH_QUERIES = frozenset({"Q02", "Q06", "Q09", "Q12"})
_RERANKER_DEPENDENT_QUERIES = frozenset({"Q04", "Q07", "Q10"})
_STALE_INDEX_QUERIES = frozenset({"Q03", "Q08", "Q11"})
_TRUNCATED_CONTEXT_QUERIES = frozenset({"Q03", "Q05", "Q08", "Q11"})
_OVER_STRICT_BLOCK_QUERIES = frozenset({"Q02", "Q05", "Q08", "Q11"})


@dataclass(frozen=True, slots=True)
class QueryEvaluation:
    """Deterministic retrieval, context, generation, guardrail, and latency evidence."""

    query_id: str
    candidate_document_ids: Tuple[str, ...]
    retrieved_document_ids: Tuple[str, ...]
    gold_rank_before_rerank: int | None
    gold_rank_after_rerank: int | None
    gold_document_retrieved: bool
    fresh_evidence_available: bool
    context_evidence_included: bool
    context_truncated: bool
    covered_key_facts: Tuple[str, ...]
    produced_fact_count: int
    supported_fact_count: int
    guardrail_blocked: bool
    usable_answer: bool
    retrieval_relevance: float
    groundedness: float
    answer_quality: float
    context_inclusion: float
    latency_ms: float
    tool_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SyntheticRAGTestbed:
    """Run the fixed workload through deterministic, composable fault semantics."""

    def __init__(
        self, evaluation_dataset: Tuple[EvaluationCase, ...] = EVALUATION_DATASET
    ) -> None:
        if not evaluation_dataset:
            raise ValueError("evaluation_dataset cannot be empty")
        self._evaluation_dataset = evaluation_dataset

    @property
    def evaluation_dataset(self) -> Tuple[EvaluationCase, ...]:
        return self._evaluation_dataset

    def run(self, configuration: SystemConfiguration) -> Tuple[QueryEvaluation, ...]:
        """Evaluate every fixed query for one validated system configuration."""

        prompt_behavior = self._validate_configuration(configuration)
        return tuple(
            self._evaluate_case(case, configuration, prompt_behavior)
            for case in self._evaluation_dataset
        )

    @staticmethod
    def _validate_configuration(
        configuration: SystemConfiguration,
    ) -> _PromptBehavior:
        supported_profiles = {
            "prompt_profile": _PROMPT_BEHAVIORS,
            "embedding_profile": _EMBEDDING_PROFILES,
            "index_profile": _INDEX_PROFILES,
            "guardrail_profile": _GUARDRAIL_PROFILES,
            "tool_latency_profile": _TOOL_LATENCIES,
            "context_profile": _CONTEXT_PROFILES,
        }
        for field_name, supported in supported_profiles.items():
            value = getattr(configuration, field_name)
            if value not in supported:
                choices = ", ".join(sorted(supported))
                raise ValueError(
                    f"unsupported {field_name} {value!r}; expected one of: {choices}"
                )
        return _PROMPT_BEHAVIORS[configuration.prompt_profile]

    @staticmethod
    def _evaluate_case(
        case: EvaluationCase,
        configuration: SystemConfiguration,
        prompt_behavior: _PromptBehavior,
    ) -> QueryEvaluation:
        candidates = list(case.candidate_ranking)

        if (
            configuration.embedding_profile == "mismatched_v2"
            and case.query_id in _EMBEDDING_MISMATCH_QUERIES
        ):
            _move_document(candidates, case.gold_document_id, len(candidates))

        fresh_evidence_available = not (
            configuration.index_profile == "stale_v1"
            and case.query_id in _STALE_INDEX_QUERIES
        )
        if not fresh_evidence_available:
            gold_position = candidates.index(case.gold_document_id)
            candidates[gold_position] = f"stale::{case.gold_document_id}"

        gold_rank_before = _rank(candidates, case.gold_document_id)
        post_rerank = list(candidates)
        if (
            configuration.reranker_enabled
            and case.query_id in _RERANKER_DEPENDENT_QUERIES
            and case.gold_document_id in post_rerank
        ):
            _move_document(post_rerank, case.gold_document_id, 4)
        gold_rank_after = _rank(post_rerank, case.gold_document_id)

        retrieved = tuple(post_rerank[: configuration.retriever_top_k])
        gold_retrieved = case.gold_document_id in retrieved
        context_truncated = (
            configuration.context_profile == "truncated"
            and case.query_id in _TRUNCATED_CONTEXT_QUERIES
            and gold_retrieved
        )
        context_included = gold_retrieved and not context_truncated

        if context_included:
            fact_fraction = prompt_behavior.expected_fact_fraction
            unsupported_count = prompt_behavior.unsupported_fact_count
            if configuration.prompt_profile == "stable_canary" and case.query_id == "Q12":
                fact_fraction = 0.5
                unsupported_count = 0
            covered_count = round(len(case.expected_key_facts) * fact_fraction)
            covered_facts = case.expected_key_facts[:covered_count]
            supported_count = len(covered_facts)
            produced_count = supported_count + unsupported_count
        else:
            covered_facts = ()
            supported_count = 0
            produced_count = 1

        guardrail_blocked = (
            configuration.guardrail_profile == "over_strict"
            and case.query_id in _OVER_STRICT_BLOCK_QUERIES
            and context_included
        )
        raw_answer_quality = len(covered_facts) / len(case.expected_key_facts)
        answer_quality = 0.0 if guardrail_blocked else raw_answer_quality
        groundedness = supported_count / produced_count if produced_count else 0.0
        tool_latency_ms = _TOOL_LATENCIES[configuration.tool_latency_profile]
        latency_ms = (
            120.0
            + (12.0 * len(retrieved))
            + (18.0 if configuration.reranker_enabled else 0.0)
            + tool_latency_ms
        )

        return QueryEvaluation(
            query_id=case.query_id,
            candidate_document_ids=tuple(candidates),
            retrieved_document_ids=retrieved,
            gold_rank_before_rerank=gold_rank_before,
            gold_rank_after_rerank=gold_rank_after,
            gold_document_retrieved=gold_retrieved,
            fresh_evidence_available=fresh_evidence_available,
            context_evidence_included=context_included,
            context_truncated=context_truncated,
            covered_key_facts=covered_facts,
            produced_fact_count=produced_count,
            supported_fact_count=supported_count,
            guardrail_blocked=guardrail_blocked,
            usable_answer=not guardrail_blocked,
            retrieval_relevance=float(gold_retrieved),
            groundedness=groundedness,
            answer_quality=answer_quality,
            context_inclusion=float(context_included),
            latency_ms=latency_ms,
            tool_latency_ms=tool_latency_ms,
        )


def _move_document(ranking: list[str], document_id: str, one_based_rank: int) -> None:
    ranking.remove(document_id)
    ranking.insert(one_based_rank - 1, document_id)


def _rank(ranking: list[str], document_id: str) -> int | None:
    try:
        return ranking.index(document_id) + 1
    except ValueError:
        return None

