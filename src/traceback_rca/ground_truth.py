"""Offline benchmark-only injected answers.

Production-facing workflows must never import this module. The benchmark retrieves an
answer only after baseline and Traceback predictions are complete.
"""

from typing import Dict

from traceback_rca.models import GroundTruth


_GROUND_TRUTHS: Dict[str, GroundTruth] = {
    "I01": GroundTruth("I01", "retriever_top_k_regression"),
    "I02": GroundTruth("I02", "embedding_regression"),
    "I03": GroundTruth("I03", "prompt_regression"),
    "I04": GroundTruth("I04", "stale_index"),
    "I05": GroundTruth("I05", "reranker_disabled"),
    "I06": GroundTruth("I06", "guardrail_regression"),
    "I07": GroundTruth("I07", "tool_latency_regression"),
    "I08": GroundTruth("I08", "context_truncation"),
    "I09": GroundTruth("I09", "no_incident"),
    "I10": GroundTruth("I10", "retriever_top_k_regression"),
}


def get_ground_truth(incident_id: str) -> GroundTruth:
    """Return one hidden answer for offline scoring only."""

    return _GROUND_TRUTHS[incident_id]

