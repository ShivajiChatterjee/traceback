# Traceback

**Verification-first root-cause analysis for production LLM/RAG systems.**

Traceback is a hackathon project for AI, GenAI, ML, LLMOps, and platform engineers
responsible for production LLM/RAG applications whose quality has unexpectedly
degraded.

## The problem

LLM applications can keep returning HTTP 200 responses while answer quality,
retrieval relevance, groundedness, latency, or guardrail behavior becomes worse.
Traditional error monitoring may still see a healthy service because there is no
exception, crash, or failed request. Several prompt, retrieval, model, index, or
guardrail changes can also occur together, making correlation an unsafe substitute
for causation.

## Verification-first architecture

Production telemetry generates hypotheses, not verdicts. Traceback challenges
plausible diagnoses with controlled local experiments before reporting causal
support:

```text
Investigator-visible incident
    |
Gemini Hypothesis Investigator
    |
Safe Experiment Planner
    |
Deterministic Counterfactual Replay
    |
Verifier / Falsifier
    |
Evidence-backed RCA Reporter
    |
Human Approval
```

Gemini performs the reasoning-heavy hypothesis step. Deterministic Python owns
configuration diffs, metrics, intervention validation, replay execution, thresholds,
final evidence classification, benchmark scoring, and ground-truth comparison. No
LLM-as-judge is used for the primary metric.

## Gemini and provider boundary

Traceback uses the official
[`google-genai`](https://googleapis.github.io/python-genai/) SDK through a minimal
`LLMProvider.generate(prompt, response_schema)` protocol. `GeminiProvider` supplies
JSON-schema-constrained real responses; `ScriptedProvider` supplies deterministic
offline responses for tests. Application workflow code therefore does not depend
directly on Gemini client objects.

No model ID is hard-coded. Real runs load `GEMINI_API_KEY` and `GEMINI_MODEL` from the
environment or a local `.env`. The API key is never printed or included in reports,
and `.env` is Git-ignored.

## Single-request baseline

The fair baseline receives exactly the incident's initial investigator serialization:
telemetry, before/after configuration, visible changes, deployment IDs, and
timestamps. It makes one request using the same configured Gemini model and returns a
validated root-cause category, confidence, summary, and supporting evidence.

The baseline receives no replay results, tools, verifier, hidden causal annotations,
or ground truth. Invalid JSON or schema values produce an explicit parsing error;
Traceback does not invent fallback predictions.

## Investigation workflow

### Hypothesis Investigator

The Investigator makes one structured request for two to four ranked, distinct
hypotheses. Each includes a taxonomy category, prior confidence, supporting and
contradicting evidence, and an optional proposed counterfactual. The taxonomy is an
allowed prediction vocabulary, not an incident answer.

### Safe Experiment Planner

LLM output is treated as an untrusted proposal. For this milestone, only
`retriever_top_k` and `prompt_profile` may be replayed. A proposal is executable only
when the field is a visible incident change and its value exactly restores the known
before value with the correct type. Unknown fields, arbitrary values, duplicates,
and production mutations are rejected. At most three local experiments can run.

### Verifier / Falsifier

The verifier executes the existing deterministic replay engine and classifies the
measured aggregate-quality delta:

- `delta >= 0.15`: supported by replay;
- `delta <= 0.05`: fails to support the hypothesis;
- between those bounds: inconclusive.

These labels mean experimental support within the synthetic testbed, not mathematical
proof of causality. Gemini does not calculate the deltas or choose the classification.

### RCA Reporter and human approval

The reporter selects the strongest replay-supported hypothesis and creates a
structured result containing affected metrics, prediction and confidence, supporting
and contradicting evidence, experiments, before/degraded/replay metrics, final
assessment, and a recommended action.

Human-readable output separates `OBSERVED`, `INFERRED`, `SUPPORTED / CHALLENGED BY
REPLAY`, and `RECOMMENDED - NOT EXECUTED`. Every report has
`human_approval_required=true`; Traceback never changes production configuration.

## Deterministic RAG testbed

The local testbed uses 12 fixed knowledge-assistant queries, deterministic candidate
rankings, one gold evidence document per query, and two expected answer facts. There
is no network access, sampling, model inference, or LLM-as-judge in evaluation or
replay.

- Retrieval relevance is whether the gold document appears in the retrieved prefix.
- Answer quality is the fraction of expected facts covered.
- Groundedness is the fraction of produced facts supported by gold evidence.
- Aggregate quality is `0.30 * retrieval + 0.30 * groundedness + 0.40 * answer`.

`stable_v1` and `stable_v1_1` are behaviorally equivalent. The `regressed` prompt
profile covers one of two expected facts and produces one unsupported fact. Low
`retriever_top_k` excludes required documents. Other configuration fields are domain
preparation for future cases; this harness does not claim to reproduce a commercial
RAG platform.

## Ground-truth isolation and benchmarking

Injected answers live only in `ground_truth.py`, which is not exported by the package
API and is never imported by baseline, investigator, planner, verifier, reporter, or
workflow modules. The offline `BenchmarkRunner` completes both predictions before it
retrieves the hidden answer for scoring.

The runner supports I01, I03, and I10 and records per-case correctness, confidence,
Traceback experiment count, elapsed latency, and aggregate root-cause accuracy. Unit
tests use scripted providers. No real benchmark result is claimed here.

## Setup and commands

Create or activate a Python 3.10 virtual environment and install the project:

```powershell
python -m pip install -e ".[dev]"
```

Create a local `.env` from the blank committed template if it does not already exist:

```powershell
Copy-Item .env.example .env
```

Then edit `.env` locally and set both values:

```dotenv
GEMINI_API_KEY=<your Gemini API key>
GEMINI_MODEL=<the Gemini model ID you selected>
```

Do not commit `.env`. A real command fails clearly when either setting is blank.

Run the network-free unit suite:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Run the opt-in minimal real Gemini smoke test:

```powershell
.\.venv\Scripts\python.exe -m traceback_rca.smoke_gemini
```

Run one real baseline diagnosis or Traceback investigation:

```powershell
.\.venv\Scripts\python.exe -m traceback_rca.baseline I10
.\.venv\Scripts\python.exe -m traceback_rca.investigate I10
```

The real three-case benchmark makes multiple API calls and requires an explicit
acknowledgement flag:

```powershell
.\.venv\Scripts\python.exe -m traceback_rca.benchmark --confirm-real-api
```

The deterministic Milestone 2 replay demonstration remains available:

```powershell
.\.venv\Scripts\python.exe -m traceback_rca.demo I10
```

## Improvement Changelog

### Baseline foundation

- Typed incidents and domain contracts.
- Investigator-visible data separated from hidden ground truth.

### Iteration 1

- Deterministic RAG testbed with fixed evaluation data.
- Reproducible quality metrics and controlled replay infrastructure.
- Explicit harmful and behaviorally neutral prompt profiles.

### Iteration 2

- Gemini hypothesis investigator behind a provider abstraction.
- Fair, structured single-LLM baseline.
- Safe experiment planning and replay-based verifier/falsifier.
- Evidence-backed RCA report with mandatory human approval.
- Offline three-case benchmark infrastructure.

## Not implemented yet

This milestone does not add the remaining seven incidents, an LLM baseline benchmark
result, automatic remediation, a frontend, FastAPI, deployment, LangChain, LangGraph,
LangSmith, Datadog, a real vector database, or other external production integration.

