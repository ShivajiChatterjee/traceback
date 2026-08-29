# Representative Traceback Trajectory: INC-10

This is an observable application workflow trajectory assembled from the saved real
incident report and frozen benchmark artifacts. It contains structured inputs,
outputs, tool actions, and measured results only—no private chain-of-thought.

Sources:

- `results/incidents/INC-10_20260829_170553/rca_report.json`
- `results/benchmark_20260829_170118/incidents/INC-10/case_result.json`
- `results/benchmark_20260829_170118/incidents/INC-10/replay_evidence.json`

## 1. Incident received

```yaml
incident_id: I10
visible_changes:
  prompt_profile: stable_v1 -> stable_v1_1
  retriever_top_k: 8 -> 2
healthy_aggregate_quality: 1.0000
degraded_aggregate_quality: 0.1667
```

## 2. Deterministic detector result

```yaml
material_incident: true
severity: high
degradation_types:
  - quality
  - context
affected_metrics:
  - aggregate_quality
  - retrieval_relevance
  - groundedness
  - answer_quality
  - context_inclusion_rate
observed_drops:
  aggregate_quality: 0.8333
  retrieval_relevance: 0.8333
  groundedness: 0.8333
  answer_quality: 0.8333
  context_inclusion_rate: 0.8333
```

## 3. Structured Gemini hypotheses

The saved report records two competing hypotheses:

```yaml
- category: retriever_top_k_regression
  prior_confidence: 0.60
- category: prompt_regression
  prior_confidence: 0.40
```

These are hypotheses, not verdicts.

## 4. Planned intervention: restore top-k

```yaml
experiment_id: I10-E1
hypothesis: retriever_top_k_regression
intervention:
  retriever_top_k: 8
safety_check:
  allowlisted_field: true
  restores_known_before_value: true
  changes_only_requested_field: true
```

## 5. Replay and verifier result: top-k

The revised prompt remained active while top-k was restored.

| Metric | Degraded | Replay | Recovery | Threshold |
|---|---:|---:|---:|---:|
| Retrieval relevance | 0.1667 | 1.0000 | +0.8333 | 0.1500 |
| Aggregate quality | 0.1667 | 1.0000 | +0.8333 | 0.1500 |

```yaml
outcome: supported_by_replay
support_score: 1.0
```

## 6. Planned intervention: restore prompt

```yaml
experiment_id: I10-E2
hypothesis: prompt_regression
intervention:
  prompt_profile: stable_v1
safety_check:
  allowlisted_field: true
  restores_known_before_value: true
  changes_only_requested_field: true
```

## 7. Replay and verifier result: prompt

The harmful top-k value remained active while the prompt was restored.

| Metric | Degraded | Replay | Recovery | Threshold |
|---|---:|---:|---:|---:|
| Answer quality | 0.1667 | 0.1667 | +0.0000 | 0.1500 |
| Groundedness | 0.1667 | 0.1667 | +0.0000 | 0.1500 |

```yaml
outcome: fails_to_support
support_score: 0.0
```

## 8. Final RCA

```yaml
predicted_root_cause: retriever_top_k_regression
status: supported_by_replay
experiments_run: 2
recommended_action: >-
  Restore retriever_top_k to the previous value in an approved change,
  then run the complete regression suite.
human_approval_required: true
```

The prompt explanation was challenged by replay; the top-k explanation was
supported. The workflow recommended a change but did not execute one.

## 9. Benchmark context and retry provenance

The fair baseline also predicted `retriever_top_k_regression` correctly. Traceback's
distinction was testing the alternatives, not correcting a baseline failure.

The first frozen-benchmark Traceback/INC-10 attempt ended before a prediction with
HTTP 429. Only that system/case combination was resumed:

```yaml
- attempt: 1
  model: gemini-3.6-flash
  status: provider_error
  error: "ClientError (status=429): request or workflow failed"
  prediction: null
- attempt: 2
  model: gemini-3.6-flash
  status: success
  prediction: retriever_top_k_regression
```

The failed attempt remains part of the provenance.
