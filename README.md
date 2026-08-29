# Traceback

**Verification-first root-cause analysis for production LLM/RAG systems.**

Traceback is a hackathon project for AI, GenAI, ML, LLMOps, and platform engineers
responsible for an LLM/RAG application whose behavior has unexpectedly degraded.

## Problem and complete use case

LLM systems can continue returning HTTP 200 responses while retrieval relevance,
groundedness, answer usefulness, guardrail behavior, context construction, evidence
freshness, or latency becomes materially worse. Traditional crash and exception
monitoring may therefore report a healthy service during a serious silent regression.

Several configurations can also change together. Traceback does not promote the
first correlated change to a verdict. It detects affected metric groups, asks Gemini
for competing explanations, executes allowlisted counterfactual rollbacks in a local
testbed, deterministically evaluates metric recovery, and creates an evidence-backed
RCA recommendation for human approval.

## Architecture

```text
Investigator-visible incident telemetry
    |
Deterministic Incident Detector
    | no material incident -> no_incident report; no LLM or replay
    |
Gemini Hypothesis Investigator
    |
Safe Experiment Planner
    |
Deterministic Counterfactual Replay
    |
Metric-aware Verifier / Falsifier
    |
Evidence-backed RCA Reporter
    |
Human Approval
```

Gemini performs reasoning where alternatives must be proposed. Deterministic Python
owns configuration diffs, metrics, regression detection, intervention validation,
replay execution, threshold calculations, final evidence classification, benchmark
scoring, and hidden-answer comparison. The primary benchmark does not use an
LLM-as-judge.

## Gemini and provider boundary

The official `google-genai` SDK is isolated behind the small
`LLMProvider.generate(prompt, response_schema)` protocol. `GeminiProvider` supplies
JSON-schema-constrained real responses; `ScriptedProvider` supplies deterministic
network-free test responses. A malformed response produces an explicit structured
parsing failure rather than an invented prediction.

Real commands load `GEMINI_API_KEY` and `GEMINI_MODEL` from the environment or local
`.env`. No model ID is hard-coded. The API key is never printed or included in an
export, and `.env` is Git-ignored.

## Deterministic RAG fault-injection environment

The workload contains 12 fixed knowledge-assistant queries. Every query has two gold
answer facts, one current gold document, and ten deterministic candidate documents.
The testbed records candidate and post-rerank positions, retrieved evidence, current
evidence availability, context inclusion, fact coverage, guardrail decisions, and
tool/end-to-end latency.

Aggregate quality is:

```text
0.30 * retrieval_relevance
+ 0.30 * groundedness
+ 0.40 * answer_quality
```

This is a deterministic synthetic fault-injection environment created to provide
known causal ground truth. It does **not** claim to reproduce all semantic,
operational, or ranking behavior of a commercial production RAG stack.

## Ten-case incident benchmark

| Case | Hidden category | Injected mechanism | Primary visible metrics | Valid replay |
|---|---|---|---|---|
| INC-01 | `retriever_top_k_regression` | `top_k` 8 -> 2 excludes evidence | retrieval, answer, aggregate quality | restore `retriever_top_k=8` |
| INC-02 | `embedding_regression` | mismatch demotes four gold documents | retrieval, answer; freshness stays healthy | restore `embedding_profile=aligned_v1` |
| INC-03 | `prompt_regression` | prompt omits facts and adds unsupported content | groundedness and answer; retrieval stays healthy | restore `prompt_profile=stable_v1` |
| INC-04 | `stale_index` | current evidence is replaced for three queries | freshness, stale count, answer quality | restore `index_profile=current_v2` |
| INC-05 | `reranker_disabled` | three documents remain below `top_k=8` | post-rerank rank, retrieval, answer | restore `reranker_enabled=true` |
| INC-06 | `guardrail_regression` | four valid answers are blocked | rejection and usable-answer rates | restore `guardrail_profile=balanced` |
| INC-07 | `tool_latency_regression` | tool latency increases by 840 ms | tool and end-to-end latency; quality is stable | restore `tool_latency_profile=healthy` |
| INC-08 | `context_truncation` | four retrieved evidence items are omitted | context inclusion, truncation, answer quality | restore `context_profile=standard` |
| INC-09 | `no_incident` | canary causes small normal answer variation | all changes remain inside tolerance | no replay |
| INC-10 | `retriever_top_k_regression` | neutral prompt revision and harmful top-k change ship together | retrieval and quality | prompt replay fails; top-k replay recovers |

The benchmark deliberately mixes obvious, moderate, subtle, operational, healthy,
and correlated-change scenarios. Cases are defined from fault semantics, not adjusted
after seeing Gemini predictions.

## Deterministic incident detector

The detector operates only on investigator-visible before/after telemetry:

- Aggregate-quality drop: material at `>= 0.10`.
- Retrieval, groundedness, or answer-quality drop: material at `>= 0.15`.
- Guardrail rejection increase or usable-answer drop: material at `>= 0.15`.
- Context-inclusion drop: material at `>= 0.15`.
- Fresh-evidence drop: material at `>= 0.15`.
- Latency: material when the increase is at least `200 ms` and at least `30%`.

Affected metrics are grouped as quality, guardrail, performance, context, and
freshness. INC-09's aggregate drop is only `0.0167`, so the workflow returns
`no_incident` without asking Gemini, forcing a hypothesis, or running a replay.

## Fair single-request baseline

The baseline receives the exact initial investigator-visible serialization used by
Traceback: telemetry, before/after configuration, visible changes, deployment IDs,
and timestamps. It makes exactly one diagnosis request with no tools, replays,
verifier, hidden annotations, or ground truth.

Baseline and Traceback use the same `GEMINI_MODEL`, the same ten cases, and the same
initial incident evidence. The benchmark is not designed to make the baseline fail;
a correct baseline prediction is a valid successful outcome.

## Investigator and safe experiment planner

For a material incident, the Investigator proposes two to four ranked, distinct
hypotheses with prior confidence, supporting evidence, uncertainty, and an optional
rollback. LLM proposals are untrusted until validated.

The final allowlist is:

```text
retriever_top_k
prompt_profile
embedding_profile
index_profile
reranker_enabled
guardrail_profile
tool_latency_profile
context_profile
```

The field must be a visible incident change, the value and type must exactly match the
known before value, and the root-cause category must correspond to the intervention
field. Unknown fields, arbitrary values, duplicates, unrelated interventions, code
execution, and production mutation are rejected. At most three local experiments run.

## Controlled replay evidence

Replay always starts from the incident's after-configuration and modifies only the
explicit field. Results preserve:

- full before, degraded, and replay configurations;
- before, degraded, and replay values for the intervention;
- healthy, degraded, and replay metrics;
- replay-minus-degraded deltas for every metric;
- progress toward healthy metrics; and
- completed experiment status.

Replay produces evidence only. It does not claim a cause.

## Metric-aware Verifier / Falsifier

All relevant criteria for a category must meet their material thresholds for
`supported_by_replay`. When every relevant recovery remains within its non-material
bound, the result is `fails_to_support`; partial recovery is `inconclusive`.

| Category | Required recovery metrics | Material thresholds |
|---|---|---|
| top-k | retrieval relevance, aggregate quality | `0.15`, `0.15` |
| embedding | retrieval relevance, answer quality | `0.15`, `0.10` |
| prompt | answer quality, groundedness | `0.15`, `0.15` |
| stale index | fresh-evidence rate, answer quality | `0.15`, `0.10` |
| reranker | retrieval relevance, answer quality | `0.15`, `0.10` |
| guardrail | rejection decrease, usable-answer increase | `0.15`, `0.15` |
| tool latency | end-to-end and tool-latency decrease | `200 ms`, `200 ms` |
| context truncation | context inclusion, answer quality | `0.15`, `0.10` |

Quality/rate movements up to `0.05` and latency movements up to `50 ms` are treated as
non-material for falsification. These are deterministic engineering conventions for
the synthetic benchmark, not mathematical proof of causality.

## RCA Reporter and human approval

The report separates `OBSERVED`, `INFERRED`, `SUPPORTED / CHALLENGED BY REPLAY`, and
`RECOMMENDED - NOT EXECUTED`. True-incident remediation always requires human
approval. A healthy `no_incident` result recommends no action and sets
`human_approval_required=false` because there is no corrective change to approve.

Traceback never automatically modifies a production configuration.

## Ground-truth isolation

Injected answers live only in `ground_truth.py`, which is not exported from the
package API and is never imported by the detector, baseline, investigator, planner,
replay engine, verifier, reporter, workflow, or incident-report exporter. The offline
benchmark retrieves a hidden answer only after both prediction attempts for that case
have completed.

## Evaluation methodology

Primary metric:

```text
root_cause_accuracy = correct predictions / 10 cases
```

Secondary metrics include healthy-case false positives, experiments per true
incident, replay-supported diagnosis rate, latency, token usage when supplied by the
SDK, and per-case correctness. API cost is deliberately omitted because no locally
trustworthy pricing configuration is assumed.

The real runner records per-case errors and continues after a failed provider or
parse request. It does not silently invent missing predictions. The full real
benchmark is never started without `--confirm-real-api`.

## Saved benchmark and incident results

A real benchmark automatically creates:

```text
results/
  benchmark_YYYYMMDD_HHMMSS/
    metrics.json
    baseline_results.json
    traceback_results.json
    summary.md
    incidents/
      INC-01/
        case_result.json
        replay_evidence.json
      ...
      INC-10/
```

`summary.md` is generated from actual recorded values and includes accuracy,
secondary metrics, a per-case table, and the INC-10 prompt-versus-top-k showcase.

`investigate --save` creates a production-safe report:

```text
results/
  incidents/
    INC-10_YYYYMMDD_HHMMSS/
      rca_report.json
      rca_report.md
      replay_evidence.json
```

Production incident exports contain no hidden ground truth, credentials, API keys, or
private chain-of-thought.

## Setup and commands

Install into a Python 3.10 virtual environment:

```powershell
python -m pip install -e ".[dev]"
```

Create `.env` only if needed, then edit it locally:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

```dotenv
GEMINI_API_KEY=<your Gemini API key>
GEMINI_MODEL=<your selected Gemini model ID>
```

Run all offline tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Run the opt-in Gemini smoke test:

```powershell
.\.venv\Scripts\python.exe -m traceback_rca.smoke_gemini
```

Run one baseline diagnosis:

```powershell
.\.venv\Scripts\python.exe -m traceback_rca.baseline I07
```

Run one Traceback investigation, optionally saving the report:

```powershell
.\.venv\Scripts\python.exe -m traceback_rca.investigate I10
.\.venv\Scripts\python.exe -m traceback_rca.investigate I10 --save
```

Run a deterministic replay demo without Gemini:

```powershell
.\.venv\Scripts\python.exe -m traceback_rca.demo I07
```

Run and automatically save the full real benchmark only after explicit approval:

```powershell
.\.venv\Scripts\python.exe -m traceback_rca.benchmark --confirm-real-api
```

The architecture uses approximately 19 diagnosis/hypothesis requests for a successful
ten-case run: ten baseline requests and nine Traceback Investigator requests. INC-09
does not invoke the Investigator. SDK retries or failed calls can change the actual
HTTP request count.

## Improvement Changelog

### Baseline Foundation

- Typed domain models and investigator-visible incidents.
- Hidden evaluation-only ground truth.
- Initial I01, I03, and I10 cases.

### Iteration 1

- Deterministic 12-query RAG testbed and quality metrics.
- Controlled counterfactual replay.
- Explicit harmful and behaviorally neutral prompt profiles.

### Iteration 2

- Gemini provider and offline ScriptedProvider.
- Fair single-request baseline and competing hypotheses.
- Safe experiment planning, Verifier/Falsifier, and structured RCA report.
- Mandatory human approval for remediation.

### Iteration 3

- Complete ten-incident benchmark with varied fault semantics.
- Metric-specific incident detection and replay verification.
- Healthy/no-incident early exit.
- Expanded, category-bound replay interventions.
- Benchmark persistence and human-shareable incident report export.

No measured performance improvement is claimed until the real ten-case benchmark is
explicitly run and its generated results are reviewed.

## Not implemented

Traceback does not include automatic remediation, a frontend, FastAPI, LangChain,
LangGraph, LangSmith, Datadog, Kubernetes, deployment, a real vector database, or a
connection to a production environment. It does not claim to have invented
counterfactual debugging or to be the first RCA system.

