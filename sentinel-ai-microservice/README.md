# Sentinel — Cloud-Deployed AI Microservice

> Unsupervised anomaly detection for pharmaceutical equipment telemetry, packaged as a containerised, autoscaling, observable microservice with a full CI/CD pipeline.

[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?logo=github-actions&logoColor=white)](.github/workflows/ci-cd.yml)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi&logoColor=white)](#)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](#)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?logo=kubernetes&logoColor=white)](#)
[![Tests](https://img.shields.io/badge/tests-18_passing-success)](#testing)

**Live demo:** _<add deployed URL here>_
**Screenshot / demo GIF:** _<add `/metrics` dashboard + Swagger UI screenshot here>_

---

## Why this project

This is **AI/ML Project 10** — the one that takes a trained model and *ships it*. It productionises the [Project 8 anomaly detector](#portfolio-context) into a service that an SRE could actually run: health probes, resource limits, horizontal autoscaling, Prometheus metrics, a hardened non-root container, and a push-to-deploy pipeline.

The model itself is deliberately lightweight — the engineering value on display here is **the path from model to running cloud service**, not the model's novelty. That is the gap most junior AI portfolios never cross.

The domain — flagging abnormal readings from cleanroom / utility equipment (WFI pumps, HVAC, compressors) — comes straight from 8+ years of GMP manufacturing engineering, so the example is real, not a toy.

---

## What it does

`POST` a batch of sensor readings → get an anomaly score, a boolean flag, and a severity label per reading. Optionally `POST` a single reading to `/explain` for a plain-language maintenance narrative (deterministic by default; LLM-enriched via Claude when a key is configured).

```jsonc
// POST /api/v1/predict
{
  "readings": [
    { "equipment_id": "WFI-PUMP-09", "temperature": 65, "pressure": 12,
      "vibration": 28, "humidity": 92, "flow_rate": 15 }
  ]
}

// 200 OK
{
  "success": true,
  "data": {
    "model_version": "iforest-v1-c0.02",
    "count": 1,
    "anomalies": 1,
    "results": [
      { "equipment_id": "WFI-PUMP-09", "anomaly_score": 0.1904,
        "is_anomaly": true, "severity": "medium" }
    ]
  },
  "message": "scored"
}
```

---

## Architecture

```
                         ┌───────────────────────────────────────────┐
                         │            Kubernetes (namespace: sentinel)│
                         │                                            │
   client ── Ingress ───▶│  Service (ClusterIP) ──▶ Deployment (2–5)  │
                         │                            │  pods (HPA)   │
                         │                            ▼               │
                         │                   ┌──────────────────┐     │
                         │                   │  FastAPI + Gunicorn│    │
                         │                   │  /healthz /readyz  │    │
                         │                   │  /api/v1/predict   │    │
                         │                   │  /api/v1/explain   │    │
                         │                   │  /metrics ─────────┼──▶ Prometheus
                         │                   │  IsolationForest   │    │
                         │                   │  (baked artifact)  │    │
                         │                   └──────────────────┘     │
                         │      ConfigMap (config) + Secret (API key)  │
                         └───────────────────────────────────────────┘

   GitHub push ─▶ Actions: ruff + pytest ─▶ build image ─▶ GHCR ─▶ kubectl rollout
```

**Request flow:** middleware records latency/count metrics → router validates input against typed Pydantic schemas (out-of-range or unknown fields rejected at the edge with `422`) → the shared in-memory `AnomalyDetector` (loaded once at startup, attached to `app.state`) scores the batch → response is wrapped in the standard `{ success, data, message }` envelope.

---

## Tech stack

| Layer | Choice | Why |
|-------|--------|-----|
| API | FastAPI + Pydantic v2 | Typed request/response validation, automatic OpenAPI docs at `/docs` |
| Server | Gunicorn + Uvicorn workers | Process supervision + async concurrency in one |
| Model | scikit-learn `IsolationForest` | Unsupervised, label-free, tiny artifact, millisecond cold start |
| Metrics | prometheus-client | Operational + model-behaviour observability via `/metrics` |
| LLM (optional) | Anthropic Claude (lazy import) | Richer explanations without becoming a hard dependency |
| Container | Multi-stage Docker, non-root | Small, low-attack-surface runtime image |
| Orchestration | Kubernetes (Deployment, Service, HPA, Ingress) | Probes, autoscaling, config/secret separation |
| CI/CD | GitHub Actions → GHCR → kubectl | Lint, test, build, push, gated deploy |

---

## API reference

| Method | Path | Purpose | Success |
|--------|------|---------|---------|
| `GET` | `/` | Service metadata | `200` |
| `GET` | `/healthz` | Liveness probe (process alive) | `200` |
| `GET` | `/readyz` | Readiness probe (model loaded) | `200` / `503` |
| `GET` | `/metrics` | Prometheus exposition | `200` |
| `POST` | `/api/v1/predict` | Score a batch (1–500 readings) | `200` / `422` |
| `POST` | `/api/v1/explain` | Score + explain one reading | `200` / `422` |

Interactive docs are served at **`/docs`** (Swagger UI) and **`/redoc`** when the app is running.

### Exposed metrics

| Metric | Type | Meaning |
|--------|------|---------|
| `http_requests_total` | counter | Requests by method, route, status |
| `http_request_duration_seconds` | histogram | Request latency |
| `predictions_total` | counter | Individual readings scored |
| `anomalies_detected_total` | counter | Readings flagged anomalous |
| `prediction_duration_seconds` | histogram | Batch scoring latency |
| `model_loaded` | gauge | `1` when the model is ready |

> **Drift signal:** a rising `anomalies_detected_total / predictions_total` ratio over time is an early indicator that live data has drifted away from the training envelope — the monitoring hook an MLOps reviewer looks for.

---

## Quickstart

### 1. Local (Python)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
make run            # uvicorn with reload on :8000
make test           # 18 tests
make lint           # ruff
```

The model trains itself on first startup if no artifact exists, so there is nothing else to set up.

### 2. Docker

```bash
make docker-build   # multi-stage build, model baked in
make docker-run     # serves on :8000
```

### 3. Docker Compose (with monitoring)

```bash
make compose-up                 # API only
make compose-monitoring         # API + Prometheus on :9090
```

### 4. Kubernetes

```bash
# (optional) create the secret for LLM explanations
kubectl create namespace sentinel
kubectl -n sentinel create secret generic sentinel-secrets \
  --from-literal=ANTHROPIC_API_KEY=sk-ant-...

make k8s-deploy                 # namespace, config, service, hpa, deployment, ingress
kubectl -n sentinel get pods
kubectl -n sentinel port-forward svc/sentinel-ai 8000:80
```

Works against any cluster, including local `minikube` or `kind` for the demo.

### Try it

```bash
curl -X POST localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"readings":[{"equipment_id":"WFI-PUMP-09","temperature":65,"pressure":12,"vibration":28,"humidity":92,"flow_rate":15}]}'
```

---

## Configuration

All config is environment-driven (one `Settings` object, read identically locally, in Docker, and in K8s).

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `MODEL_PATH` | `app/artifacts/model.joblib` | Where the artifact is loaded/saved |
| `CONTAMINATION` | `0.02` | Expected anomaly fraction at training time |
| `RANDOM_SEED` | `42` | Reproducible training |
| `ENABLE_LLM_EXPLANATIONS` | `false` | Turn on the Claude-backed `/explain` path |
| `ANTHROPIC_API_KEY` | _(unset)_ | Required only when LLM explanations are enabled |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | Cheapest model — explanations are short |

LLM enrichment degrades gracefully: if the key is missing, the call fails, or the package is unavailable, `/explain` silently returns the deterministic rule-based narrative instead. The response shape is identical either way.

---

## Testing

```bash
make test
# 18 passed
```

Coverage spans the model class (train / predict / save / load / determinism), every HTTP endpoint, input validation (`422` on empty batch, out-of-range values, unknown fields), the readiness gate, and the rule-based explanation fallback. The suite is **fully hermetic** — LLM is forced off and a temp model path is used, so there is no network call, no API key, and no artifact left behind. The same `ruff check` + `pytest` gate runs in CI on every push and PR.

---

## CI/CD

`.github/workflows/ci-cd.yml` runs three jobs:

1. **test** — `ruff check` + `pytest --cov` on every push and PR.
2. **build-and-push** — on `main`, builds the image with layer caching and pushes to **GHCR** tagged with both the commit SHA and `latest`.
3. **deploy** — on `main`, gated behind a `production` environment; applies the manifests and rolls the deployment to the new SHA-tagged image.

> The deploy job needs two repo settings before it runs for real: a `production` environment and a base64-encoded `KUBE_CONFIG` secret. Until those exist, jobs 1–2 run and job 3 is skipped — the pipeline is still green.

---

## Project structure

```
sentinel-ai-microservice/
├── app/
│   ├── main.py            # app factory, lifespan, middleware, /metrics
│   ├── config.py          # pydantic-settings config
│   ├── schemas.py         # typed request/response + {success,data,message}
│   ├── model.py           # IsolationForest: train / persist / load / predict
│   ├── explain.py         # rule-based + optional Claude explanations
│   ├── metrics.py         # Prometheus metric definitions
│   └── routers/
│       ├── health.py      # /healthz, /readyz
│       └── predict.py     # /api/v1/predict, /api/v1/explain
├── scripts/train_model.py # bakes the artifact at build time
├── tests/                 # 18 pytest tests (hermetic)
├── k8s/                   # namespace, configmap, secret, deployment, service, hpa, ingress
├── monitoring/            # prometheus scrape config
├── .github/workflows/     # ci-cd pipeline
├── Dockerfile             # multi-stage, non-root, healthcheck
├── docker-compose.yml     # local + optional prometheus profile
├── Makefile               # dev shortcuts
└── pyproject.toml         # ruff + pytest config
```

---

## Interview talking points

- **"Ships models, not just builds them."** The repo demonstrates the whole lifecycle: train → serialise → containerise → deploy → observe → autoscale → CI/CD. This is the line between a notebook and a service.
- **Liveness vs readiness.** `/healthz` answers "is the process alive?" (restart if not); `/readyz` answers "is the model loaded?" (gate traffic until yes). A `startupProbe` covers slow first loads so liveness doesn't kill a healthy-but-loading pod. Conflating these is a common production bug.
- **Security-hardened container.** Non-root user, dropped capabilities, `readOnlyRootFilesystem` with a writable `/tmp` emptyDir, `allowPrivilegeEscalation: false` — what hardened clusters require.
- **Observability built for drift.** Metrics aren't just request counts; the anomaly-rate ratio is a deliberate data-drift signal an MLOps team would alert on.
- **Graceful degradation.** The LLM path is optional and lazy-imported, so the core service never has a hard dependency on it or a hidden per-prediction cost — directly relevant to cost-aware production AI.
- **Config once, run anywhere.** A single typed `Settings` object reads from `.env`, Docker env, or a K8s ConfigMap + Secret with zero code changes between environments.
- **Bounded metric cardinality.** Request metrics are labelled by the route *template*, not the raw path, so the time series count stays finite — a Prometheus gotcha that bites people in production.

---

## Portfolio context

This service productionises the model from **Project 8 — Anomaly Detection on Equipment Sensor Data** (the unsupervised-learning entry in the AI/ML track), turning it from a trained model into a deployable cloud microservice. It targets the JD's preferred skills: **Docker, Kubernetes, CI/CD, and cloud deployment**.

---

## License

MIT — see `LICENSE` _(add file before publishing if desired)_.
