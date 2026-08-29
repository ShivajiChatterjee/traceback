# Improvement Changelog

This changelog summarizes implemented repository states, tests, recorded artifacts,
and git history. It does not claim experiments or measurements that are absent from
the project evidence.

## Baseline / Initial Version

Added:

- typed incident, telemetry, configuration, hypothesis, experiment, and RCA models;
- isolated evaluation-only `GroundTruth`;
- initial INC-01, INC-03, and correlated-change INC-10 definitions; and
- investigator-visible serialization that excludes the hidden answer.

What was missing:

- executable counterfactual replay;
- deterministic verification;
- most of the final fault taxonomy;
- a healthy abstention case; and
- a real model-backed investigation workflow.

## Iteration 1 - Deterministic Testbed

Added:

- a fixed 12-query synthetic RAG workload;
- deterministic retrieval, answer, groundedness, quality, and latency metrics;
- fault profiles for the initial incidents;
- a replay engine that starts from the degraded configuration and restores only an
  explicitly selected field; and
- an offline replay demo.

Learning: deterministic fault injection makes the causal mechanism known to the
evaluation harness without exposing it to the investigator. It also lets the exact
same workload run before, after, and under intervention.

## Iteration 2 - Verification-First Workflow

Added:

- the official Gemini provider behind a small provider protocol;
- a network-free scripted provider for tests;
- the fair one-request baseline;
- structured competing hypotheses;
- the safe experiment planner;
- replay support/falsification classification;
- evidence-separated RCA reporting; and
- a mandatory human-approval boundary for corrective action.

INC-10 became the central discriminator: prompt rollback produced no meaningful
recovery, while top-k rollback restored the relevant metrics.

## Iteration 3 - Broader Benchmark

Added:

- ten cases covering top-k, embeddings, prompt, index freshness, reranking,
  guardrails, tool latency, context truncation, and no incident;
- metric-group incident detection;
- category-specific verification rules;
- no-incident early exit;
- benchmark and incident-report persistence; and
- offline end-to-end tests for all cases.

The final taxonomy and intervention set remained deliberately bounded.

## Rejected / Replaced Design - Aggregate-Quality-Only Verification

**Hypothesis/design:** the first verifier classified every replay from aggregate
answer-quality movement alone.

**What happened:** the benchmark expanded beyond answer-quality faults. A tool
latency incident can breach its SLO while answer quality stays stable; guardrail,
freshness, and context faults also have distinct diagnostic metrics.

**Evidence/problem:** git commit `5eb317c` contains the original verifier reading only
`delta_from_degraded.aggregate_quality`. Commit `e782737` replaces it with category
rules over retrieval, groundedness, answer, freshness, guardrail, latency, and
context metrics. Under the original design, a latency recovery could not be judged
from the metric that actually regressed.

**Decision:** reject one aggregate-quality rule for every incident category.

**Replacement:** deterministic metric-aware verification. Each category now requires
recovery in its relevant metrics and records both the measured deltas and decision
thresholds.

This is a documented design replacement, not a fabricated model experiment.

## Iteration 4 - Benchmark Reliability

The first real benchmark exposed an operational failure: Gemini returned HTTP 429
for Traceback/INC-10 before producing a prediction. The initial exporter represented
that missing prediction like an incorrect result and had no safe way to continue the
exact run.

Added:

- in-place `--resume --retry-failed` support;
- retry targeting at the system/case level;
- a guard requiring the original model;
- preserved attempt history;
- completion and unresolved-provider-error metrics; and
- incomplete reporting that withholds final accuracy instead of scoring an API
  failure as an incorrect RCA.

Only Traceback/INC-10 was retried. Attempt 1 remains recorded as HTTP 429; attempt 2
used `gemini-3.6-flash` and succeeded. No cross-model fallback was added to the
official benchmark.

This reliability change was discovered through actual evaluation, not invented for
the changelog.

## Final Measured Result

| System | Completed | Correct | Accuracy |
|---|---:|---:|---:|
| Baseline | 10 / 10 | 10 / 10 | 100% |
| Traceback | 10 / 10 | 10 / 10 | 100% |

The experiment did not show an accuracy improvement. It showed that a verification
workflow can preserve the classification accuracy of a strong prompt while adding
controlled evidence that supports one explanation and challenges another. The
trade-off was additional latency and API work.
