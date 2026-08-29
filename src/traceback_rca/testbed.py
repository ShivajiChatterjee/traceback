"""A deterministic, production-like RAG fault-injection testbed."""

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Tuple

from traceback_rca.dataset import EVALUATION_DATASET, EvaluationCase
from traceback_rca.models import SystemConfiguration


@dataclass(frozen=True, slots=True)
class _PromptBehavior:
    expected_fact_fraction: float
    unsupported_fact_count: int


# I10's revision is deliberately neutral; only ``regressed`` injects a prompt fault.
_PROMPT_BEHAVIORS: Mapping[str, _PromptBehavior] = {
    "stable_v1": _PromptBehavior(1.0, 0),
    "stable_v1_1": _PromptBehavior(1.0, 0),
    "regressed": _PromptBehavior(0.5, 1),
}


@dataclass(frozen=True, slots=True)
class QueryEvaluation:
    """Deterministic evidence and scores for one evaluation query."""

    query_id: str
    retrieved_document_ids: Tuple[str, ...]
    gold_document_retrieved: bool
    covered_key_facts: Tuple[str, ...]
    produced_fact_count: int
    supported_fact_count: int
    retrieval_relevance: float
    groundedness: float
    answer_quality: float
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SyntheticRAGTestbed:
    """Run fixed cases through deterministic retrieval and prompt behavior rules."""

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
        """Evaluate every fixed query for one system configuration."""

        try:
            prompt_behavior = _PROMPT_BEHAVIORS[configuration.prompt_profile]
        except KeyError as error:
            supported = ", ".join(sorted(_PROMPT_BEHAVIORS))
            raise ValueError(
                f"unsupported prompt_profile {configuration.prompt_profile!r}; "
                f"expected one of: {supported}"
            ) from error

        return tuple(
            self._evaluate_case(case, configuration, prompt_behavior)
            for case in self._evaluation_dataset
        )

    @staticmethod
    def _evaluate_case(
        case: EvaluationCase,
        configuration: SystemConfiguration,
        prompt_behavior: _PromptBehavior,
    ) -> QueryEvaluation:
        retrieved = case.candidate_ranking[: configuration.retriever_top_k]
        gold_retrieved = case.gold_document_id in retrieved

        if gold_retrieved:
            covered_count = round(
                len(case.expected_key_facts) * prompt_behavior.expected_fact_fraction
            )
            covered_facts = case.expected_key_facts[:covered_count]
            supported_count = len(covered_facts)
            produced_count = supported_count + prompt_behavior.unsupported_fact_count
        else:
            covered_facts = ()
            supported_count = 0
            produced_count = 1

        retrieval_relevance = float(gold_retrieved)
        answer_quality = len(covered_facts) / len(case.expected_key_facts)
        groundedness = supported_count / produced_count if produced_count else 0.0
        latency_ms = (
            120.0
            + (12.0 * len(retrieved))
            + (18.0 if configuration.reranker_enabled else 0.0)
            + configuration.tool_latency_ms
        )

        return QueryEvaluation(
            query_id=case.query_id,
            retrieved_document_ids=retrieved,
            gold_document_retrieved=gold_retrieved,
            covered_key_facts=covered_facts,
            produced_fact_count=produced_count,
            supported_fact_count=supported_count,
            retrieval_relevance=retrieval_relevance,
            groundedness=groundedness,
            answer_quality=answer_quality,
            latency_ms=latency_ms,
        )

