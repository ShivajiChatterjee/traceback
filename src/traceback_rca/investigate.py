"""Run a real Gemini-backed Traceback investigation from the command line."""

import argparse

from traceback_rca.incidents import get_incident, list_incidents
from traceback_rca.detector import IncidentDetector
from traceback_rca.providers import GeminiProvider, ScriptedProvider
from traceback_rca.reporter import RCAReport
from traceback_rca.workflow import TracebackWorkflow


def _print_report(report: RCAReport) -> None:
    print(f"TRACEBACK RCA: {report.incident_id}")
    print(f"status={report.status.value}")
    print(f"predicted_root_cause={report.predicted_root_cause.value}")
    print(f"confidence={report.confidence:.4f}")
    print(f"affected_metrics={', '.join(report.affected_metrics)}")

    print("\nOBSERVED")
    for evidence in report.observed:
        print(f"- {evidence}")
    print("\nINFERRED")
    for evidence in report.inferred:
        print(f"- {evidence}")
    print("\nSUPPORTED / CHALLENGED BY REPLAY")
    for evidence in report.replay_evidence:
        print(f"- {evidence}")
    print("\nFINAL ASSESSMENT")
    print(report.final_assessment)
    print("\nRECOMMENDED - NOT EXECUTED")
    print(report.recommended_action)
    print(f"human_approval_required={str(report.human_approval_required).lower()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "incident_id", choices=tuple(incident.incident_id for incident in list_incidents())
    )
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    incident = get_incident(args.incident_id)
    provider = (
        GeminiProvider.from_environment()
        if IncidentDetector().detect(incident).material_incident
        else ScriptedProvider([])
    )
    run = TracebackWorkflow(provider).investigate(incident)
    _print_report(run.report)
    if args.save:
        from traceback_rca.export import save_investigation

        output_directory = save_investigation(
            incident, run, results_root=args.results_dir
        )
        print(f"report_saved_to={output_directory}")


if __name__ == "__main__":
    main()
