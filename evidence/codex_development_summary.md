# Coding-Agent Development Summary

Codex was used throughout the project to inspect the repository, implement scoped
milestones, write offline tests, run verification commands, and prepare submission
evidence. This document summarizes observable repository work; it is not a transcript
and contains no private chain-of-thought.

A raw Codex conversation export is not committed in this repository. Accordingly,
this summary relies on git history, source files, tests, milestone outputs, and frozen
result artifacts rather than invented dialogue.

## Milestone 1 - Typed Foundation

Objective: establish a small Python 3.10 package without external model calls.

Observable outputs:

- `traceback_rca` package, avoiding collision with Python's standard-library
  `traceback` module;
- typed configurations, telemetry, incidents, hypotheses, experiments, and results;
- initial INC-01, INC-03, and INC-10 definitions;
- isolated `ground_truth.py` and investigator-visible serialization tests;
- safe `.env.example` and `.gitignore`; and
- initial offline pytest coverage.

Design decision: keep GroundTruth evaluation-only and out of every
production-facing object.

## Milestone 2 - Deterministic Testbed

Objective: make suspected causes experimentally testable without Gemini.

Observable outputs:

- fixed 12-query dataset;
- deterministic RAG testbed and evaluator;
- before/degraded/replay metrics;
- controlled replay from the degraded configuration; and
- `python -m traceback_rca.demo`.

Verification: offline tests assert deterministic repetition, fault behavior, and
single-field replay isolation.

Design decision: use deterministic Python, not an LLM judge, for all scoring and
metric arithmetic.

## Milestone 3 - End-to-End Investigation

Objective: add Gemini reasoning around the deterministic experiment core.

Observable git evidence: commit `5eb317c` records the Gemini provider, scripted test
provider, fair baseline, Investigator, planner, verifier, reporter, workflow, CLI,
and their tests.

Verification: structured-output and malformed-response tests remained network-free.

Design decision: treat LLM hypotheses as untrusted until interventions pass planner
validation and deterministic replay evaluation.

## Milestone 4 - Ten-Case Benchmark

Objective: broaden fault coverage and produce reproducible scored evidence.

Observable git evidence: commit `e782737` adds the deterministic detector, ten-case
suite, expanded telemetry, metric-aware verification, no-incident handling,
benchmark persistence, and report export.

Verification: all cases run end to end with `ScriptedProvider`; INC-09 exits without
an Investigator call.

Design decision: replace aggregate-quality-only replay classification with
category-specific metric rules. The git diff between `5eb317c` and `e782737` records
this genuine rejected/replaced design.

## Frozen Benchmark

Objective: run the fair real comparison without changing cases after observing model
behavior.

Observable git evidence: commit `90ef523` records the benchmark directory and saved
INC-10 report.

Recorded outcome:

- baseline: 10/10 correct;
- Traceback: 10/10 correct after completion;
- accuracy difference: 0 percentage points; and
- Traceback: replay-supported evidence for 9/9 true incidents.

Design decision: report the accuracy tie honestly and frame verification—not another
classification—as the architectural contribution.

## Benchmark Retry Fix

Objective: recover the exact frozen run after a real HTTP 429 on
Traceback/INC-10.

Observable outputs:

- system/case-level resume and retry-failed support;
- same-model guard;
- retained attempt history;
- completion-aware accuracy and provider-error metrics; and
- offline tests for successful and failed retry behavior.

Verification: only Traceback/INC-10 was retried. The first 429 and second
`gemini-3.6-flash` success both remain in the frozen artifact.

Design decision: do not count a missing provider response as an incorrect RCA and do
not introduce cross-model fallback into the official benchmark.

## Milestone 5 - Submission Preparation

Objective: make the implementation, limitations, trajectories, reproducibility, and
video story reviewable without more real API calls.

Outputs:

- submission-quality README;
- frozen-summary interpretation and retry provenance;
- improvement changelog and real rejected-design evidence;
- INC-10 and INC-09 observable trajectories;
- standalone reproduction guide;
- sub-five-minute demo script; and
- submission checklist.

Design decision: prefer saved real artifacts plus deterministic replay for the demo,
avoiding a fresh 19-request benchmark during screen recording.
