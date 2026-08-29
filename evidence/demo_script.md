# Traceback Demo Script

Target runtime: **4:35-4:50**. Use the saved real report and deterministic demo; do
not depend on a fresh full Gemini benchmark.

## 0:00-0:35 - Problem and baseline

**Screen:** README title and problem.

**Say:**

“LLM and RAG systems can return HTTP 200 while retrieval, groundedness, guardrails,
or latency silently regress. When several configurations change together, telemetry
shows correlation—not necessarily cause. Traceback is verification-first RCA: it
turns telemetry into hypotheses, then tests them before recommending action. We also
compare it fairly with one strong Gemini request using the same visible evidence.”

## 0:35-1:05 - Architecture

**Screen:** README architecture diagram.

**Say:**

“A deterministic detector identifies affected metric groups. Gemini proposes
competing structured hypotheses. A safe planner permits only known before-value
rollbacks. The local replay engine reruns a fixed workload, and deterministic,
metric-aware rules support, challenge, or leave each hypothesis inconclusive. The
report recommends remediation, but a human must approve any consequential change.”

## 1:05-2:45 - INC-10 demonstration

**Screen:** run:

```powershell
.\.venv\Scripts\python.exe -m traceback_rca.demo I10
```

**Say:**

“INC-10 shipped two changes: the prompt moved from `stable_v1` to a behaviorally
neutral `stable_v1_1`, and top-k fell from eight to two. Aggregate quality dropped
from one to 0.1667. The live command is deterministic replay, so it carries no API
risk during this video.”

**Screen:** show
`results\incidents\INC-10_20260829_170553\rca_report.md`.

**Say:**

“The saved real Gemini run proposed top-k and prompt regressions. Restoring the
prompt while leaving top-k at two produced zero answer-quality and groundedness
recovery, challenging the prompt explanation. Restoring top-k while keeping the new
prompt produced 0.8333 recovery in retrieval relevance and aggregate quality,
supporting top-k. The final RCA is `retriever_top_k_regression`; remediation is only
recommended, never executed.”

“The baseline also identified top-k correctly. Traceback's distinction is that it
tested both explanations—it did not rescue a failed baseline.”

## 2:45-3:20 - Frozen benchmark

**Screen:**
`results\benchmark_20260829_170118\summary.md`.

**Say:**

“On the frozen ten-case benchmark, the strong Gemini baseline scored ten out of ten,
and Traceback also scored ten out of ten: zero percentage-point accuracy change.
Traceback replay-supported all nine true-incident diagnoses, but required more
latency and API work. The honest result is a verification trade-off, not an accuracy
win.”

## 3:20-3:50 - INC-09 abstention

**Screen:**
`evidence\trajectories\INC-09_no_incident.md`.

**Say:**

“INC-09 has only normal variation. The deterministic detector stays below its
material threshold, returns `no_incident`, and makes no Gemini Investigator call,
generates no hypotheses, runs no replay, recommends no remediation, and requires no
approval. Verification-first also means knowing when not to investigate.”

## 3:50-4:20 - Improvement evidence

**Screen:** `IMPROVEMENT_CHANGELOG.md`.

**Say:**

“Two real iterations matter. First, aggregate-quality-only verification was replaced
with category-specific rules because latency and guardrail failures cannot be judged
from answer quality alone. Second, the real benchmark hit HTTP 429 on Traceback
INC-10. Resume support preserved that failure, enforced the same model, retried only
the missing combination, and stopped treating provider failure as an incorrect RCA.”

## 4:20-4:45 - Hot take and limitations

**Screen:** README key finding and limitations.

**Say:**

“Traceback's result was not that agents always diagnose better. A strong LLM already
classified every frozen case correctly. The value of the workflow is that it can
challenge its own diagnosis with controlled interventions before recommending
action. This is a small deterministic synthetic benchmark, not production causality
proof; richer staging replay and a larger incident corpus are future work.”

## Recording checklist

- Keep the terminal font large and hide `.env`.
- Run only the deterministic `demo I10` command live.
- Use saved artifacts for Gemini and benchmark evidence.
- Do not show API credentials or claim an accuracy improvement.
- Stop by 4:50 to leave margin under the five-minute limit.
