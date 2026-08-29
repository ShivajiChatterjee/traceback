"""Command-line demonstration of deterministic controlled replays."""

import argparse
from typing import Iterable

from traceback_rca.evaluator import QualityMetrics, evaluate_configuration
from traceback_rca.incidents import get_incident
from traceback_rca.models import ConfigurationChange, Incident
from traceback_rca.replay import replay


def _format_metrics(metrics: QualityMetrics) -> str:
    return (
        f"retrieval={metrics.retrieval_relevance:.4f} "
        f"groundedness={metrics.groundedness:.4f} "
        f"answer={metrics.answer_quality:.4f} "
        f"aggregate={metrics.aggregate_quality:.4f} "
        f"latency_ms={metrics.latency_ms:.2f}"
    )


def _print_configuration(incident: Incident, before: bool) -> None:
    configuration = (
        incident.configuration_before if before else incident.configuration_after
    )
    print(
        f"prompt_profile={configuration.prompt_profile} "
        f"retriever_top_k={configuration.retriever_top_k}"
    )


def run_demo(incident_id: str) -> None:
    incident = get_incident(incident_id)
    before = evaluate_configuration(incident.configuration_before).metrics
    after = evaluate_configuration(incident.configuration_after).metrics

    print(f"INCIDENT {incident.incident_id}: {incident.title}")
    print("\nHEALTHY BEFORE")
    _print_configuration(incident, before=True)
    print(_format_metrics(before))
    print("\nDEGRADED AFTER")
    _print_configuration(incident, before=False)
    print(_format_metrics(after))

    for number, change in enumerate(_replay_changes(incident), start=1):
        result = replay(incident, {change.field_name: change.before})
        print(f"\nCOUNTERFACTUAL {number}")
        print(f"restore {change.field_name}={change.before}")
        print(_format_metrics(result.replay_metrics))
        print(
            "aggregate_delta_from_degraded="
            f"{result.delta_from_degraded.aggregate_quality:+.4f}"
        )


def _replay_changes(incident: Incident) -> Iterable[ConfigurationChange]:
    return incident.changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("incident_id", choices=("I01", "I03", "I10"))
    args = parser.parse_args()
    run_demo(args.incident_id)


if __name__ == "__main__":
    main()

