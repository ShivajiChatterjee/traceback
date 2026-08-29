"""Allowed root-cause prediction vocabulary for structured RCA outputs."""

from enum import Enum
from typing import Tuple


class RootCauseCategory(str, Enum):
    """Current and planned synthetic benchmark root-cause categories."""

    RETRIEVER_TOP_K_REGRESSION = "retriever_top_k_regression"
    PROMPT_REGRESSION = "prompt_regression"
    NO_INCIDENT = "no_incident"
    EMBEDDING_REGRESSION = "embedding_regression"
    STALE_INDEX = "stale_index"
    RERANKER_DISABLED = "reranker_disabled"
    GUARDRAIL_REGRESSION = "guardrail_regression"
    TOOL_LATENCY_REGRESSION = "tool_latency_regression"
    CONTEXT_TRUNCATION = "context_truncation"
    INGESTION_FAILURE = "ingestion_failure"
    MODEL_REGRESSION = "model_regression"


def root_cause_labels() -> Tuple[str, ...]:
    """Return the prediction vocabulary without revealing any incident answer."""

    return tuple(category.value for category in RootCauseCategory)

