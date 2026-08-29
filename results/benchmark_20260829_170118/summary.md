# Traceback Benchmark Summary

Model: `gemini-3.6-flash`

Cases: 10

Status: **COMPLETE**

## Primary Metric

- Baseline RCA Accuracy: **10 / 10 = 100.0%**
- Traceback RCA Accuracy: **10 / 10 = 100.0%**
- Accuracy change: **0 percentage points**
- Baseline completed cases: `10` / `10`
- Traceback completed cases: `10` / `10`
- Baseline provider errors: `0`
- Traceback provider errors: `0`

Traceback did **not** improve raw RCA accuracy on this synthetic benchmark. The
strong one-request Gemini baseline and Traceback both classified every completed
case correctly. Traceback's measurable architectural difference is verification:
it tests competing explanations with controlled counterfactual replay and records
which explanations the replay supports or challenges.

## Secondary Metrics

| Metric | Baseline | Traceback |
|---|---:|---:|
| Healthy-case false positive | `False` | `False` |
| Mean latency | `7649.73 ms` | `11087.59 ms` |
| Input tokens | `10483` | `9988` |
| Output tokens | `1649` | `2691` |
| Completed cases | `10 / 10` | `10 / 10` |
| Unresolved provider errors | `0` | `0` |

Traceback ran a mean of `1.1111` experiments per true incident. Its final diagnosis
was supported by replay in `9 / 9` true incidents (`100.0%`). These are recorded-run
metrics only; no pricing estimate or unrecorded measurement is implied.

## Per-case Results

| Incident | Ground Truth | Baseline Prediction | Baseline Correct | Traceback Prediction | Traceback Correct | Experiments |
|---|---|---|---:|---|---:|---:|
| INC-01 | retriever_top_k_regression | retriever_top_k_regression | True | retriever_top_k_regression | True | 1 |
| INC-02 | embedding_regression | embedding_regression | True | embedding_regression | True | 1 |
| INC-03 | prompt_regression | prompt_regression | True | prompt_regression | True | 1 |
| INC-04 | stale_index | stale_index | True | stale_index | True | 1 |
| INC-05 | reranker_disabled | reranker_disabled | True | reranker_disabled | True | 1 |
| INC-06 | guardrail_regression | guardrail_regression | True | guardrail_regression | True | 1 |
| INC-07 | tool_latency_regression | tool_latency_regression | True | tool_latency_regression | True | 1 |
| INC-08 | context_truncation | context_truncation | True | context_truncation | True | 1 |
| INC-09 | no_incident | no_incident | True | no_incident | True | 0 |
| INC-10 | retriever_top_k_regression | retriever_top_k_regression | True | retriever_top_k_regression | True | 2 |

## Showcase Incident - INC-10

Two configuration changes were visible when aggregate quality fell from `1.0000`
to `0.1667`:

- `prompt_profile`: `stable_v1` -> `stable_v1_1`
- `retriever_top_k`: `8` -> `2`

The baseline correctly predicted `retriever_top_k_regression`. Traceback reached the
same diagnosis, but also tested both competing explanations:

- Prompt rollback restored `prompt_profile=stable_v1` while keeping `top_k=2`.
  Answer quality and groundedness recovery were both `+0.0000`, so replay
  **failed to support** the prompt explanation.
- Top-k rollback restored `retriever_top_k=8` while keeping the revised prompt.
  Retrieval relevance and aggregate-quality recovery were both `+0.8333`, so replay
  **supported** the top-k explanation.

Final RCA: `retriever_top_k_regression`.

### Retry provenance

- Attempt 1: model `gemini-3.6-flash`; status `provider_error`; recorded error
  `ClientError (status=429): request or workflow failed`; no prediction.
- Attempt 2: model `gemini-3.6-flash`; status `success`; prediction
  `retriever_top_k_regression`.

Only the failed Traceback/INC-10 combination was retried. The same model, incident,
prompt, workflow, planner, replay, verifier, GroundTruth, and benchmark semantics
were retained.

## Healthy-case Evidence - INC-09

INC-09 contained only normal metric variation. The deterministic detector did not
cross a material-incident threshold, so Traceback returned `no_incident` without a
Gemini Investigator call, hypothesis generation, replay, remediation, or human
approval requirement.

## Interpretation

Agentic investigation is not automatically better than a strong prompt. When
observational evidence is sufficient, a capable LLM may already identify the correct
culprit. Verification becomes valuable when the cost of a confident but causally
unsupported diagnosis is high. Here it added stronger evidence at the cost of
additional latency and API work, not higher raw accuracy.

This summary is generated from the frozen recorded run. No LLM judge, invented
benchmark number, or hidden chain-of-thought is used.
