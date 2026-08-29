# Traceback

**Verification-first root-cause analysis for production LLM/RAG regressions.**

Traceback investigates silent quality and performance regressions, proposes
competing explanations, and tests them with safe counterfactual replay before it
recommends a human-approved remediation.

## Frozen result

The final recorded benchmark used `gemini-3.6-flash` on ten deterministic synthetic
incidents:

| System | Completed | Correct | RCA accuracy |
|---|---:|---:|---:|
| Strong one-request Gemini baseline | 10 / 10 | 10 / 10 | 100% |
| Traceback | 10 / 10 | 10 / 10 | 100% |

**Accuracy change: 0 percentage points.** Traceback did not improve raw RCA accuracy
on this benchmark. Its measurable difference was verification: it ran controlled
replays, challenged competing explanations, and attached replay-supported evidence
to the final diagnosis. That evidence required additional latency and API work.

See the [frozen benchmark summary](results/benchmark_20260829_170118/summary.md) and
[improvement changelog](IMPROVEMENT_CHANGELOG.md).

## Problem

HTTP 200 does not mean an LLM/RAG system is healthy. Retrieval relevance,
groundedness, answer quality, freshness, guardrail behavior, context construction,
or latency can regress while the service continues returning valid responses.

Possible causes include retrieval configuration, embedding changes, a stale index,
a prompt regression, a disabled reranker, guardrail changes, slow tools, and context
truncation. Several changes can ship together. Telemetry can show correlation, but
it does not by itself establish which change caused the regression.

## Intended user

Traceback is for AI and GenAI engineers, ML platform engineers, and LLMOps/AI
reliability engineers responsible for diagnosing production-like LLM/RAG behavior.

## Core idea

**Production telemetry should generate hypotheses, not verdicts.**

```text
Detect
  -> Hypothesize
  -> Plan a safe intervention
  -> Counterfactual replay
  -> Falsify or support
  -> Evidence-backed RCA
  -> Human-approved remediation recommendation
```

The LLM proposes plausible alternatives. Deterministic Python owns metric
calculation, change validation, replay execution, verification thresholds, scoring,
and hidden-answer comparison.

## Architecture

```text
Investigator-visible telemetry and configuration history
                         |
                         v
               Deterministic Detector
                  |              |
        no incident|              |material incident
                  v              v
          Abstaining report   Gemini Investigator
                                   |
                         competing hypotheses
                                   v
                         Safe Experiment Planner
                                   |
                     allowlisted before-value rollback
                                   v
                       Synthetic Replay Environment
                                   |
                         metric recovery evidence
                                   v
                         Verifier / Falsifier
                                   |
                                   v
                       Evidence-backed Reporter
                                   |
                       Human Approval Boundary
```

- **Incident Detector:** identifies material metric changes and affected metric
  groups without an LLM.
- **Investigator:** asks Gemini for two to four structured competing hypotheses from
  investigator-visible evidence only.
- **Experiment Planner:** validates category-bound, allowlisted rollbacks to known
  before-values; at most three experiments run.
- **Synthetic Replay Environment:** starts from the degraded configuration, changes
  only the requested field, and reruns the fixed workload.
- **Verifier/Falsifier:** classifies recovery with category-specific deterministic
  metric rules.
- **Reporter:** separates observed facts, inferred hypotheses, replay evidence, and
  recommendations.
- **Human Approval Boundary:** never executes a consequential production change.

## Fair baseline

The baseline makes one Gemini diagnosis request. It receives the same model, same
ten cases, and exact initial investigator-visible evidence as Traceback. It has no
replay, tools, verifier, hidden annotations, or GroundTruth. The benchmark was not
designed to make the baseline fail; its 10/10 result is a valid outcome.

## Deterministic synthetic testbed

The testbed runs a fixed 12-query knowledge-assistant workload. Each query has two
gold facts, a current gold document, and deterministic candidate rankings. Fault
injection changes typed configuration fields, and the same workload runs before,
after, and during replay.

Metrics include retrieval relevance, groundedness, answer quality, aggregate
quality, freshness, guardrail rejection, usable-answer rate, context inclusion, tool
latency, and end-to-end latency. Aggregate quality is deterministic:

```text
0.30 * retrieval_relevance
+ 0.30 * groundedness
+ 0.40 * answer_quality
```

This environment provides known hidden causal GroundTruth. It is not claimed to
reproduce all behavior of a commercial RAG system or a real staging environment.

## Incident benchmark

| Case | GroundTruth category | Injected mechanism | Valid replay |
|---|---|---|---|
| INC-01 | `retriever_top_k_regression` | top-k `8 -> 2` excludes evidence | restore `retriever_top_k=8` |
| INC-02 | `embedding_regression` | embedding mismatch demotes gold documents | restore `embedding_profile=aligned_v1` |
| INC-03 | `prompt_regression` | prompt omits facts and adds unsupported content | restore `prompt_profile=stable_v1` |
| INC-04 | `stale_index` | current evidence is replaced | restore `index_profile=current_v2` |
| INC-05 | `reranker_disabled` | relevant documents remain below top-k | restore `reranker_enabled=true` |
| INC-06 | `guardrail_regression` | valid answers are blocked | restore `guardrail_profile=balanced` |
| INC-07 | `tool_latency_regression` | tool latency rises while quality stays stable | restore `tool_latency_profile=healthy` |
| INC-08 | `context_truncation` | retrieved evidence is omitted from context | restore `context_profile=standard` |
| INC-09 | `no_incident` | normal canary variation stays within tolerance | no replay |
| INC-10 | `retriever_top_k_regression` | neutral prompt and harmful top-k change together | prompt rollback fails; top-k rollback recovers |

Cases were defined from fault semantics and frozen before the final benchmark.

## Safe replay

Replay accepts only these fields:

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

The field must be a visible incident change, its value and type must exactly match
the known before-value, and the hypothesis category must match the intervention.
Unknown fields, arbitrary values, duplicate interventions, code execution, and
production mutation are rejected.

## Verification rules

All required metrics for a category must reach their material-recovery thresholds
for `supported_by_replay`. If all remain within non-material bounds, the hypothesis
`fails_to_support`; partial recovery is `inconclusive`.

| Category | Required recovery metrics | Thresholds |
|---|---|---:|
| top-k | retrieval relevance, aggregate quality | `0.15`, `0.15` |
| embedding | retrieval relevance, answer quality | `0.15`, `0.10` |
| prompt | answer quality, groundedness | `0.15`, `0.15` |
| stale index | fresh-evidence rate, answer quality | `0.15`, `0.10` |
| reranker | retrieval relevance, answer quality | `0.15`, `0.10` |
| guardrail | rejection decrease, usable-answer increase | `0.15`, `0.15` |
| tool latency | end-to-end and tool-latency decrease | `200 ms`, `200 ms` |
| context truncation | context inclusion, answer quality | `0.15`, `0.10` |

These are deterministic conventions for this testbed, not mathematical proof of
causality.

## Benchmark methodology

The primary metric is exact root-cause accuracy against hidden GroundTruth. Secondary
metrics are healthy-case false positives, experiment count, replay-supported
diagnoses, latency, token usage, and completion/provider errors. No LLM judge scores
the benchmark. GroundTruth is read only after each system's prediction attempt has
finished.

### GroundTruth isolation

Injected answers live in `ground_truth.py`. The detector, baseline, Investigator,
planner, replay engine, verifier, reporter, workflow, and production incident
exporter do not import it. Only the offline benchmark evaluator compares predictions
with hidden answers, after the relevant workflow attempt completes.

## Results

The final recorded metrics are:

| Metric | Baseline | Traceback |
|---|---:|---:|
| RCA accuracy | 100% | 100% |
| Healthy false positive | false | false |
| Mean latency | 7649.73 ms | 11087.59 ms |
| Input / output tokens | 10483 / 1649 | 9988 / 2691 |
| Completed cases | 10 / 10 | 10 / 10 |
| Unresolved provider errors | 0 | 0 |

Traceback averaged `1.1111` experiments per true incident, and all `9 / 9` true
incident diagnoses were replay-supported. These values come only from
`results/benchmark_20260829_170118`; no cost figure is invented.

The first Traceback attempt for INC-10 received HTTP 429. Resume support retried only
that failed combination using the same `gemini-3.6-flash` model and preserved both
attempts. Provider failure was treated as incomplete work, not an incorrect RCA.

## Showcase: INC-10

INC-10 changed both `prompt_profile` and `retriever_top_k`; aggregate quality fell
from `1.0000` to `0.1667`. Gemini proposed top-k regression (prior `0.60`) and prompt
regression (prior `0.40`).

- Restoring the prompt produced `+0.0000` answer-quality and groundedness recovery.
  The prompt explanation was challenged by replay.
- Restoring top-k produced `+0.8333` retrieval-relevance and aggregate-quality
  recovery. The top-k explanation was supported by replay.

Both baseline and Traceback correctly predicted `retriever_top_k_regression`. The
difference is that Traceback tested both explanations before recommending action.

## Healthy case: INC-09

INC-09's aggregate-quality drop was `0.0167`, inside the detector tolerance. Traceback
returned `no_incident` with no Gemini Investigator call, hypotheses, replay,
remediation, or human approval requirement. This demonstrates abstention and avoids
unnecessary investigation.

## Key finding

> A strong LLM prompt can be an excellent incident classifier. Agentic complexity is
> justified not merely by producing another diagnosis, but when it can generate
> evidence that challenges or supports that diagnosis.

Agentic investigation is not automatically better than a strong prompt. Verification
becomes valuable when the cost of a confident but causally unsupported diagnosis is
high.

## Safety

- Replay is local and synthetic.
- GroundTruth is isolated from the production-facing workflow.
- Credentials and hidden chain-of-thought are never exported.
- Real API commands are explicit; the full benchmark requires
  `--confirm-real-api`.
- Traceback recommends remediation but never executes it.
- Consequential remediation requires human approval.

## Reproduction

Full setup and reviewer workflow are in [REPRODUCE.md](REPRODUCE.md). From an existing
Windows PowerShell environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m traceback_rca.demo I10
```

For real-API commands only, copy `.env.example` to the ignored `.env` and fill in
`GEMINI_API_KEY` and `GEMINI_MODEL` locally:

```powershell
Copy-Item .env.example .env
```

The deterministic demo makes no API call. It demonstrates the testbed interventions,
not fresh Gemini reasoning. The saved real Gemini-backed report can be viewed with:

```powershell
Get-Content results\incidents\INC-10_20260829_170553\rca_report.md
```

Opt-in real commands:

```powershell
.\.venv\Scripts\python.exe -m traceback_rca.smoke_gemini
.\.venv\Scripts\python.exe -m traceback_rca.baseline I10
.\.venv\Scripts\python.exe -m traceback_rca.investigate I10 --save
.\.venv\Scripts\python.exe -m traceback_rca.benchmark --confirm-real-api
.\.venv\Scripts\python.exe -m traceback_rca.benchmark --resume results\benchmark_20260829_170118 --retry-failed --confirm-real-api
```

The resume command is provenance-preserving recovery, not cross-model fallback. Do
not run it against the completed frozen benchmark unless a new unresolved failure
actually exists.

## Outputs

```text
results/
  benchmark_20260829_170118/
    metrics.json
    baseline_results.json
    traceback_results.json
    summary.md
    incidents/INC-01 ... INC-10/
      case_result.json
      replay_evidence.json
  incidents/INC-10_20260829_170553/
    rca_report.json
    rca_report.md
    replay_evidence.json

evidence/
  trajectories/
    INC-10_traceback.md
    INC-09_no_incident.md
  codex_development_summary.md
  demo_script.md
```

## Limitations

- Synthetic benchmark with only ten cases.
- Fixed fault taxonomy and deterministic workload.
- Replay approximates staging experimentation; it is not production causality proof.
- No claim of commercial production equivalence.
- Gemini behavior can vary across calls and model versions.
- Verification adds latency, token usage, and API work.
- No repeated stochastic trials or confidence calibration.
- No autonomous remediation.

## Future work

- Ingest OpenTelemetry, LangSmith, or Langfuse traces.
- Replay real application snapshots in a staging sandbox.
- Evaluate a larger incident corpus and repeated stochastic trials.
- Calibrate confidence and experiment selection.
- Add model fallback with explicit provenance outside the official same-model
  benchmark.
- Collect feedback from incident-response engineers.

## Submission material

- [Improvement changelog](IMPROVEMENT_CHANGELOG.md)
- [Reproduction guide](REPRODUCE.md)
- [INC-10 trajectory](evidence/trajectories/INC-10_traceback.md)
- [INC-09 trajectory](evidence/trajectories/INC-09_no_incident.md)
- [Coding-agent development summary](evidence/codex_development_summary.md)
- [Demo script](evidence/demo_script.md)
- [Submission checklist](SUBMISSION_CHECKLIST.md)
