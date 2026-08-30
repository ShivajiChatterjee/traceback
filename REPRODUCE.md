# Reproducing Traceback

This guide separates offline deterministic commands from opt-in Gemini commands. No
real benchmark run is required to inspect the frozen evidence.

## 1. Open or clone the repository

```powershell
git clone <https://github.com/ShivajiChatterjee/traceback>
Set-Location traceback
```

If the repository is already open, run all commands from its root.

## 2. Create a Python 3.10 virtual environment

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
```

Platform-neutral alternative:

```text
python3.10 -m venv .venv
```

## 3. Install the package and development dependencies

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

macOS/Linux equivalent:

```text
.venv/bin/python -m pip install -e ".[dev]"
```

## 4. Configure Gemini only for real-API commands

Copy the safe template and edit the new local file:

```powershell
Copy-Item .env.example .env
```

```dotenv
GEMINI_API_KEY=<your key>
GEMINI_MODEL=<your model ID>
```

`.env` is Git-ignored. Never commit or paste its contents into logs or reports.

## 5. Run all offline tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Tests use `ScriptedProvider` and make no network calls.

## 6. Run the deterministic INC-10 demo

```powershell
.\.venv\Scripts\python.exe -m traceback_rca.demo I10
```

This reruns deterministic testbed interventions. It is not a fresh Gemini trajectory.
The output shows that prompt restoration leaves quality degraded while top-k
restoration recovers it.

## 7. Inspect the frozen real evidence without an API call

```powershell
Get-Content results\benchmark_20260829_170118\summary.md
Get-Content results\incidents\INC-10_20260829_170553\rca_report.md
```

## 8. Run the opt-in Gemini smoke test

```powershell
.\.venv\Scripts\python.exe -m traceback_rca.smoke_gemini
```

This makes a real request using `GEMINI_MODEL`.

## 9. Run one baseline diagnosis

```powershell
.\.venv\Scripts\python.exe -m traceback_rca.baseline I10
```

The baseline makes one Gemini request and has no replay tools.

## 10. Run one Traceback investigation

```powershell
.\.venv\Scripts\python.exe -m traceback_rca.investigate I10
.\.venv\Scripts\python.exe -m traceback_rca.investigate I10 --save
```

`--save` writes a production-safe report under `results/incidents/`.

INC-09 is an offline no-incident path because the deterministic detector exits before
constructing a Gemini provider request:

```powershell
.\.venv\Scripts\python.exe -m traceback_rca.investigate I09
```

## 11. Run a new full benchmark only with explicit authorization

```powershell
.\.venv\Scripts\python.exe -m traceback_rca.benchmark --confirm-real-api
```

A successful ten-case run uses approximately ten baseline and nine Investigator
requests; provider retries can alter the underlying HTTP count. It saves a new
timestamped directory and does not overwrite the frozen result.

## 12. Resume unresolved provider failures

```powershell
.\.venv\Scripts\python.exe -m traceback_rca.benchmark --resume results\benchmark_YYYYMMDD_HHMMSS --retry-failed --confirm-real-api
```

Resume retries only unresolved system/case combinations, preserves prior attempts,
and rejects a different current model. The official benchmark does not use
cross-model fallback.

The frozen `results\benchmark_20260829_170118` benchmark is already complete; do not
rerun it merely for reproduction.

## 13. Optional integrity checks

```powershell
.\.venv\Scripts\python.exe -m compileall -q src tests
.\.venv\Scripts\python.exe -m pip check
git diff --check
git check-ignore -v .env
```

## Output locations

```text
results/
  benchmark_YYYYMMDD_HHMMSS/
    metrics.json
    baseline_results.json
    traceback_results.json
    summary.md
    incidents/INC-01 ... INC-10/
  incidents/INC-10_YYYYMMDD_HHMMSS/
    rca_report.json
    rca_report.md
    replay_evidence.json
```
