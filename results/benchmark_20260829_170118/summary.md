# Traceback Benchmark Summary

Model: `gemini-3.6-flash`

Cases: 10

Status: **COMPLETE**

## Primary Metric

- Baseline RCA Accuracy: **10 / 10 = 100.0%**
- Traceback RCA Accuracy: **10 / 10 = 100.0%**
- Absolute change: **+0.0%**
- Baseline completed cases: `10` / `10`
- Traceback completed cases: `10` / `10`
- Baseline provider errors: `0`
- Traceback provider errors: `0`

## Secondary Metrics

- Baseline healthy-case false positive: `False`
- Traceback healthy-case false positive: `False`
- Mean Traceback experiments per true incident: `1.1111`
- Replay-supported diagnoses: `9` (100.0%)
- Mean baseline latency: `7649.73` ms
- Mean Traceback latency: `11087.59` ms
- Baseline input/output tokens: `10483` / `1649`
- Traceback input/output tokens: `9988` / `2691`

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

- `retriever_top_k_regression`: aggregate delta `0.8333`, outcome `supported_by_replay`
- `prompt_regression`: aggregate delta `0.0`, outcome `fails_to_support`
- Final supported cause: `retriever_top_k_regression`

This summary is generated from the recorded run. No LLM judge or invented benchmark
number is used.
