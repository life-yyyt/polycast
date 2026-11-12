# polycast

[English](README.md) | [中文](docs/README.zh-CN.md)

Multi-agent forecasting with calibrated probabilities and a full audit trail.

A panel of independent analysts reads a retrieved evidence set, produces its own
probability for a resolvable question, gets challenged by a red-team pass, and
is then aggregated into a single calibrated number with an interval, a
disagreement measure, and a trace of how it was reached.

Everything runs offline against a deterministic stub provider, so the repo
clones and runs without an API key.

## Why it is built this way

Most "AI forecaster" demos stop at generating a number. The number is the least
interesting part: what matters is whether it is calibrated, whether you can
audit it later, and whether the pipeline degrades gracefully when the model
disagrees with itself. This project treats those as the primary deliverables.

- **Independent analysts.** Panel members never see each other's answers.
  Sequential debate with shared context collapses disagreement and produces
  comfortable, overconfident forecasts.
- **A red team after aggregation inputs exist.** The critic argues against a
  concrete number and the panel spread, not a strawman. Its shift is bounded to
  ±0.3 so a single pass cannot swing the outcome.
- **Calibration is measured, not claimed.** Brier score, log loss, reliability
  bins, and skill against the base rate, with Platt recalibration fitted on a
  disjoint split. A forecaster that is well ranked but badly overconfident shows
  up as a reliability curve that bows away from the diagonal.
- **Everything is traced.** Each run writes an append-only JSONL trace: retrieved
  documents and their scores, each analyst's reasoning, the red team's
  objections, and the aggregation arithmetic.
- **Intervals reflect disagreement.** The width is driven by panel spread and
  the red-team shift, not a fixed constant. A panel that agrees for correlated
  reasons produces a narrow interval with the shift doing the widening.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

polycast forecast \
  --text "Will the central bank cut the policy rate at the next scheduled meeting?" \
  --criteria "Resolved yes if the published decision confirms a reduction." \
  --resolve-by 2026-12-18

polycast serve --port 8000
polycast eval --dataset evals/questions.json
polycast simulate --agents 40 --truth 0.7
```

Shared options (`--corpus`, `--db`, `--audit-dir`, `--model`) are accepted
before or after the subcommand, so `polycast --db app.db serve` and
`polycast serve --db app.db` are the same command.

## Pipeline

```
question
  -> retrieval (TF-IDF, recency-weighted)
  -> [base_rate | bull | bear]        independent, run concurrently
  -> red_team                         challenges the panel spread
  -> aggregation                      confidence-weighted, trimmed, shifted
  -> forecast + interval + disagreement
  -> sqlite (forecast) + jsonl (trace)
```

## Calibration

`polycast eval` scores the pipeline against resolved questions, fits Platt
scaling on a training split, and reports the held-out split before and after
recalibration. The bundled dataset is 22 real, publicly resolvable central bank
decisions from 2024 and 2025.

```json
{
  "scaler": { "a": 2.4013, "b": 1.0007 },
  "test_raw":          { "n": 7, "brier": 0.2555, "log_loss": 0.7043, "base_rate_brier": 0.2041 },
  "test_recalibrated": { "n": 7, "brier": 0.2235, "log_loss": 0.6528, "base_rate_brier": 0.2041 }
}
```

Read this carefully rather than skimming the numbers. The offline stub is a
lexical model with no world knowledge, so its raw forecasts on monetary policy
questions carry no information and its Brier score sits at the base rate. What
the numbers do demonstrate is the recalibration path: pushing probabilities
toward the base rate improves log loss from 0.704 to 0.653 and Brier from
0.256 to 0.224 on held-out questions. That is the harness working. Point
`POLYCAST_MODEL` at a hosted model and the same command measures that model.

Two properties of the harness are deliberate and tested:

- The train/test split is shuffled with a fixed seed. A sequential split
  produces a pathological training set whenever the file is ordered by
  outcome, and a recalibration fitted on it then extrapolates hard into the
  test set.
- Platt scaling refuses to fit when the training split contains a single
  outcome class, because the intercept it learns is unbounded there.

## HTTP API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | liveness |
| `POST` | `/forecast` | run one forecast |
| `GET` | `/forecast/{id}` | fetch a stored forecast |
| `GET` | `/forecasts` | list recent forecasts |
| `GET` | `/metrics` | counts, mean disagreement, active provider |
| `POST` | `/simulate` | agent-based belief simulation |

```bash
curl -X POST http://127.0.0.1:8000/forecast \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Will unemployment claims decline for a fifth consecutive week?",
    "resolution_criteria": "Resolved yes if the published count is below the prior week.",
    "resolve_by": "2026-12-31T00:00:00Z",
    "reference_class": "weekly continuing claims series"
  }'
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `POLYCAST_MODEL` | unset | model id; unset selects the offline stub |
| `POLYCAST_API_KEY` | unset | bearer token for the hosted provider |
| `POLYCAST_BASE_URL` | `https://api.openai.com/v1` | any OpenAI-compatible endpoint |
| `POLYCAST_CORPUS` | `data/corpus.json` | evidence corpus |
| `POLYCAST_DB` | `polycast.db` | sqlite path for stored forecasts |
| `POLYCAST_AUDIT_DIR` | `runs` | directory for JSONL traces |

`POLYCAST_BASE_URL` also points at local runtimes such as Ollama or vLLM, so
the pipeline can run fully on-device.

## Simulation

`polycast simulate` runs an agent-based model of belief formation: agents
hold private noisy signals, update toward a neighbourhood mean, and are
seeded from random priors. It answers questions a single run cannot: how many
independent analysts are needed before a crowd converges, and where social
influence sends it. The interesting result is that higher social weight
produces faster consensus and a less accurate one, because a crowd that leans
on its neighbours converges on their answer rather than on the signal. Results
are reproducible for a given seed.

## Project layout

```
src/polycast/
  schemas.py      typed contracts shared across the pipeline and the API
  providers.py    provider protocol, deterministic offline stub, OpenAI-compatible client
  retrieval.py    TF-IDF index with recency weighting
  agents.py       analyst personas, prompt construction, aggregation
  pipeline.py     orchestration: retrieve -> fan out -> red team -> aggregate -> persist
  calibration.py  Brier, log loss, reliability bins, Platt scaling
  evaluation.py   resolved-question harness with train/test split
  simulation.py   agent-based belief simulation
  audit.py        JSONL audit trail and sqlite store
  server.py       FastAPI application
  cli.py          command line entry point
data/corpus.json  sample evidence corpus
evals/            resolved questions and the eval runner
tests/            unit, pipeline, simulation and API tests
docs/             design notes and the Chinese translation of this README
```

## Testing

```bash
pytest
```

The suite covers the scoring functions against hand-computed values, the
aggregation rules (trimming, red-team shift, interval bounds), the pipeline
end to end including the audit trace, the API contract, the reproducibility
of the simulation, and the two ways a recalibration can silently break.

## Development

| Command | What it does |
| --- | --- |
| `make install` | editable install with the dev extras |
| `make test` | the pytest suite |
| `make lint` | ruff over `src` and `tests` |
| `make eval` | score the pipeline against the resolved questions |
| `make serve` | run the API with reload enabled |
| `make clean` | remove run artefacts and caches |

## Limitations

Worth being explicit about, since the project claims more than it delivers:

- The bundled corpus is twelve synthetic documents. Retrieval quality on real
  corpora is untested here.
- Aggregation weights are hand-set. A proper implementation would learn the
  per-persona weights on a resolved history.
- The offline stub is a lexical model with no world knowledge. It exercises the
  machinery, not the forecasting, and it has no predictive power on the bundled
  policy questions.
- The evaluation dataset is 22 real decisions. That is enough to exercise the
  harness and not enough to measure a forecaster: the standard error on a Brier
  score at this sample size is far larger than the differences being compared.
- The API is single-process and synchronous on SQLite. It is adequate for
  read-mostly internal use; a queue and a real database come before scale.
- There is no authentication or rate limiting on the API.

## License

MIT
