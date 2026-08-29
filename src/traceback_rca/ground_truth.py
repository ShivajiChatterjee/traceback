"""Offline benchmark-only injected answers.

Production-facing workflows must never import this module. The benchmark retrieves an
answer only after both baseline and Traceback predictions are complete.
"""

from typing import Dict

from traceback_rca.models import GroundTruth


_GROUND_TRUTHS: Dict[str, GroundTruth] = {
    "I01": GroundTruth("I01", "retriever_top_k_regression"),
    "I03": GroundTruth("I03", "prompt_regression"),
    "I10": GroundTruth("I10", "retriever_top_k_regression"),
}


def get_ground_truth(incident_id: str) -> GroundTruth:
    """Return one hidden answer for offline scoring only."""

    return _GROUND_TRUTHS[incident_id]

