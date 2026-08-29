# TRACEBACK INCIDENT REPORT

## Incident ID

INC-10 (`I10`)

## Status

`supported_by_replay`

## OBSERVED - Regression

- aggregate_quality dropped by 0.8333
- retrieval_relevance dropped by 0.8333
- groundedness dropped by 0.8333
- answer_quality dropped by 0.8333
- context_inclusion_rate dropped by 0.8333
- prompt_profile changed from 'stable_v1' to 'stable_v1_1'
- retriever_top_k changed from 8 to 2

## Affected Metrics

- aggregate_quality
- retrieval_relevance
- groundedness
- answer_quality
- context_inclusion_rate

## OBSERVED - Configuration Changes

- `prompt_profile`: `stable_v1` -> `stable_v1_1`
- `retriever_top_k`: `8` -> `2`

## INFERRED - Competing Hypotheses

- `retriever_top_k_regression` (prior 0.60)
- `prompt_regression` (prior 0.40)

## Controlled Experiments

- `I10-E1`: `retriever_top_k_regression` -> **supported_by_replay**; quality replay recovery: retrieval_relevance=+0.8333 (threshold 0.1500), aggregate_quality=+0.8333 (threshold 0.1500)
- `I10-E2`: `prompt_regression` -> **fails_to_support**; quality replay recovery: answer_quality=+0.0000 (threshold 0.1500), groundedness=+0.0000 (threshold 0.1500)

## SUPPORTED / CHALLENGED BY REPLAY

- retriever_top_k_regression: supported_by_replay; quality replay recovery: retrieval_relevance=+0.8333 (threshold 0.1500), aggregate_quality=+0.8333 (threshold 0.1500)
- prompt_regression: fails_to_support; quality replay recovery: answer_quality=+0.0000 (threshold 0.1500), groundedness=+0.0000 (threshold 0.1500)

## Final RCA

`retriever_top_k_regression`

The retriever_top_k_regression hypothesis has strong causal support from controlled local replay: quality replay recovery: retrieval_relevance=+0.8333 (threshold 0.1500), aggregate_quality=+0.8333 (threshold 0.1500).

## Confidence

80.0%

## RECOMMENDED - Not Executed

Restore retriever_top_k to the previous value in an approved change, then run the complete regression suite.

## Human Approval Required

`true`
