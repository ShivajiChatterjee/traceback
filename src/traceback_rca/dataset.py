"""Fixed, local evaluation workload for the deterministic RAG testbed."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """A query with deterministic retrieval and answer-scoring expectations."""

    query_id: str
    query: str
    gold_document_id: str
    expected_key_facts: Tuple[str, ...]
    candidate_ranking: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.query_id or not self.query:
            raise ValueError("query_id and query cannot be empty")
        if not self.expected_key_facts:
            raise ValueError("expected_key_facts cannot be empty")
        if self.gold_document_id not in self.candidate_ranking:
            raise ValueError("gold document must appear in candidate_ranking")
        if len(set(self.candidate_ranking)) != len(self.candidate_ranking):
            raise ValueError("candidate_ranking cannot contain duplicate documents")


_DISTRACTORS = tuple(f"distractor-{number:02d}" for number in range(1, 9))


def _ranking(gold_document_id: str, gold_rank: int) -> Tuple[str, ...]:
    """Place a gold document at a readable, one-based deterministic rank."""

    if not 1 <= gold_rank <= 8:
        raise ValueError("gold_rank must be between 1 and 8")
    ranked = list(_DISTRACTORS[:7])
    ranked.insert(gold_rank - 1, gold_document_id)
    return tuple(ranked)


EVALUATION_DATASET: Tuple[EvaluationCase, ...] = (
    EvaluationCase(
        "Q01",
        "How long are audit logs retained?",
        "doc-audit-retention",
        ("audit logs are retained for 90 days", "archived logs are access controlled"),
        _ranking("doc-audit-retention", 1),
    ),
    EvaluationCase(
        "Q02",
        "When must service account keys be rotated?",
        "doc-key-rotation",
        ("keys rotate every 60 days", "emergency rotation follows suspected exposure"),
        _ranking("doc-key-rotation", 2),
    ),
    EvaluationCase(
        "Q03",
        "What is required before production access is granted?",
        "doc-production-access",
        ("manager approval is required", "security training must be current"),
        _ranking("doc-production-access", 3),
    ),
    EvaluationCase(
        "Q04",
        "What is the target for critical incident acknowledgement?",
        "doc-incident-response",
        ("critical incidents are acknowledged within 10 minutes", "an incident lead is assigned"),
        _ranking("doc-incident-response", 4),
    ),
    EvaluationCase(
        "Q05",
        "How are customer exports protected?",
        "doc-export-security",
        ("exports are encrypted", "download links expire after 24 hours"),
        _ranking("doc-export-security", 5),
    ),
    EvaluationCase(
        "Q06",
        "What checks are required for a production deployment?",
        "doc-deployment-policy",
        ("automated tests must pass", "a rollback plan is required"),
        _ranking("doc-deployment-policy", 6),
    ),
    EvaluationCase(
        "Q07",
        "How frequently are access permissions reviewed?",
        "doc-access-review",
        ("permissions are reviewed quarterly", "unused access is revoked"),
        _ranking("doc-access-review", 7),
    ),
    EvaluationCase(
        "Q08",
        "What is the backup recovery objective?",
        "doc-backup-policy",
        ("the recovery point objective is four hours", "restore tests run monthly"),
        _ranking("doc-backup-policy", 8),
    ),
    EvaluationCase(
        "Q09",
        "What data may be used in development environments?",
        "doc-development-data",
        ("production personal data is prohibited", "approved synthetic data may be used"),
        _ranking("doc-development-data", 3),
    ),
    EvaluationCase(
        "Q10",
        "When are critical vulnerabilities remediated?",
        "doc-vulnerability-policy",
        ("critical findings are fixed within seven days", "exceptions require security approval"),
        _ranking("doc-vulnerability-policy", 5),
    ),
    EvaluationCase(
        "Q11",
        "What happens when an employee leaves?",
        "doc-offboarding",
        ("accounts are disabled within four hours", "company devices are collected"),
        _ranking("doc-offboarding", 7),
    ),
    EvaluationCase(
        "Q12",
        "How should API rate-limit errors be handled?",
        "doc-rate-limits",
        ("clients use exponential backoff", "retry jitter is required"),
        _ranking("doc-rate-limits", 8),
    ),
)

