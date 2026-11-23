# llm-gateway

A small OpenAI-compatible LLM gateway. It sits between agents/services and model providers and handles
**routing, fallback, exact-match caching, cost accounting per key and team, budgets and metrics**.
Clients keep using the OpenAI SDK and only change `base_url`.

This is a **portfolio/demo-scale project**, not a production gateway. Providers are stubs
(fake models with configurable latency, price and failures) plus one real free provider: a local
Gemma 3 12B in LM Studio. No paid API is called anywhere, including tests.

## Architecture

```
  client (OpenAI SDK / curl / agent)
     │  POST /v1/chat/completions   Authorization: Bearer <key>   X-Task-Type: code
     ▼
┌──────────────────────────── llm-gateway (FastAPI) ─────────────────────────────┐
│ 1 auth ── key -> name, team, budget                                             │
│ 2 budget ── spent(key) from journal >= budget ?  ──────────────► 402            │
│ 3 exact cache ── sha256(model, task, params, messages) hit? ───► answer, $0     │
│ 4 router ── model=auto: first matching rule (task / prompt length / budget left)│
│             model=<route>: that chain   model=<provider>: that provider only    │
│ 5 fallback ── chain [A, B, C]: 429 / 5xx / timeout / unreachable -> next,       │
│               at most max_attempts; 4xx stops at once                           │
│ 6 accounting ── tokens x price table -> cost_usd                                │
│ 7 journal (requests table) + Prometheus metrics + x_gateway block in response   │
└────────┬───────────────────────┬───────────────────────────┬────────────────────┘
         ▼                       ▼                           ▼
   stub-smart / stub-fast   asus-gemma (LM Studio,      Postgres: requests
   stub-cheap / stub-flaky  OpenAI-compatible API,      GET /metrics  GET /stats
   (fake, offline)          http://your-lm-studio-host:1234)
```

Code map:

| file | what |
|---|---|
| `gateway/main.py` | FastAPI app, request flow, OpenAI-shaped responses |
| `gateway/router.py` | `choose_route` (rules) and `run_chain` (fallback) |
| `gateway/providers.py` | `StubProvider`, `OpenAIProvider`, cost formula |
| `gateway/cache.py` | exact-match LRU cache with TTL |
| `gateway/store.py` | `requests` journal (SQLAlchemy async: asyncpg / aiosqlite), `/stats` aggregates |
| `gateway/metrics.py` | Prometheus counters and latency histogram |
| `config/gateway.yaml` | providers, routes, routing rules, demo keys |

## Run

With Docker (app + Postgres):

```bash
docker compose up --build
./scripts/smoke.sh              # models, routing, cache hit, fallback, stats, metrics
```

Without Docker (SQLite journal):

```bash
uv venv --python 3.12 .venv && uv pip install -e ".[dev]"
.venv/bin/uvicorn --factory gateway.main:create_app --port 8080
```

Env: `GATEWAY_CONFIG` (default `config/gateway.yaml`), `DATABASE_URL`
(default `sqlite+aiosqlite:///./gateway.db`, Postgres: `postgresql+asyncpg://user:pass@host:5432/db`).

Tests (stubs only, no network, no secrets):

```bash
.venv/bin/pytest                                          # SQLite
TEST_DATABASE_URL=postgresql+asyncpg://... .venv/bin/pytest   # same suite on Postgres
LLM_GATEWAY_LIVE=1 .venv/bin/pytest -m live               # real LM Studio on ASUS
```

## Example

```bash
curl -s localhost:8080/v1/chat/completions \
  -H 'Authorization: Bearer sk-demo-games' \
  -H 'Content-Type: application/json' \
  -H 'X-Task-Type: code' \
  -d '{"model":"auto","messages":[{"role":"user","content":"write a function"}]}'
```

```json
{
  "object": "chat.completion",
  "model": "stub-smart-v1",
  "choices": [{"index": 0, "finish_reason": "stop",
               "message": {"role": "assistant", "content": "[stub-smart] stub answer to: write a function"}}],
  "usage": {"prompt_tokens": 4, "completion_tokens": 11, "total_tokens": 15},
  "x_gateway": {"route": "smart", "rule": "code", "provider": "stub-smart",
                "cache_hit": false, "cost_usd": 0.000177, "latency_ms": 402.8,
                "attempts": [{"provider": "stub-smart", "ok": true, "latency_ms": 401.9}]}
}
```

Response headers repeat the essentials: `X-Gateway-Provider`, `X-Gateway-Route`, `X-Gateway-Cache`,
`X-Gateway-Cost-USD`. The same request again returns `"cache_hit": true, "cost_usd": 0.0`.

Fallback, as seen with LM Studio on ASUS switched off (`task: private`):

```json
"attempts": [{"provider": "asus-gemma", "ok": false, "error": "unavailable", "status": 503, "latency_ms": 3069.1},
             {"provider": "stub-cheap", "ok": true, "latency_ms": 61.8}]
```

With the OpenAI SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-demo-games")
client.chat.completions.create(model="auto", messages=[...], extra_headers={"X-Task-Type": "classify"})
```

## Routing rules (`config/gateway.yaml`)

`model: "auto"` walks the rules top to bottom; the first rule whose conditions all hold wins.
Task type comes from header `X-Task-Type`, body field `task`, or `metadata.task`.
Prompt length is estimated as chars / 4.

| # | rule | condition | route | chain (fallback order) |
|---|---|---|---|---|
| 1 | `budget-nearly-spent` | key has < $0.50 budget left | `cheap` | stub-cheap → stub-fast |
| 2 | `private-data` | task = `private` | `local` | asus-gemma → stub-cheap |
| 3 | `code` | task = `code` | `smart` | stub-smart → stub-fast |
| 4 | `long-prompt` | prompt ≥ 2000 tokens | `long` | stub-smart → stub-fast |
| 5 | `classify` | task = `classify` | `cheap` | stub-cheap → stub-fast |
| – | default | nothing matched | `general` | stub-fast → stub-cheap |

Also: `model: "<route>"` uses that route's chain directly; `model: "<provider>"` calls one provider with
no fallback; `flaky-demo` route shows fallback (its first provider always answers 429).
`max_attempts: 3` caps providers tried per request. `GET /v1/models` lists `auto`, routes and providers.

Budgets: per key, in USD, computed from the journal (`sum(cost_usd)`). When spent ≥ budget the gateway
answers **402** `budget_exceeded` with key, team, spent and budget in the message. Keys without
`budget_usd` are unlimited.

## Observability

- `GET /metrics` (Prometheus): `gateway_requests_total{route,provider,status,cache}`,
  `gateway_cache_hits_total`, `gateway_cache_misses_total`, `gateway_provider_errors_total{provider,kind}`,
  `gateway_fallbacks_total{route}`, `gateway_budget_rejections_total{team}`,
  `gateway_cost_usd_total{team,provider}`, `gateway_tokens_total{team,direction}`,
  `gateway_request_latency_seconds` (histogram by provider and cache).
- `GET /stats` (JSON): requests, cost, tokens, cache hits, avg latency by key, team and provider.
- Postgres table `requests`: ts, key_name, team, model, route, rule, provider, attempts, prompt/completion
  tokens, cost_usd, cache_hit, status, error, latency_ms. Every request is logged, including 402/404/5xx.
  The secret key itself is never stored, only its name.

## What is tested

41 pytest cases on stubs (+1 live test against ASUS, skipped by default): `/v1/models` and auth, OpenAI response shape, every routing rule and explicit
routes, fallback on 429 / 500 / 503 / timeout, recovery after N failures, all-providers-failed,
`max_attempts`, failed requests not cached, cache hit = $0 and one provider call, cache miss on changed
params, budget 402, budget rule switching to the cheap route, journal rows and `/stats` per team,
cost = price table, Prometheus output, config validation, OpenAI-compatible provider error mapping
(mocked transport). CI (`.github/workflows/tests.yml`) runs the suite on SQLite and on a Postgres service.

## Not done (honest scope)

- **Streaming** (`stream: true`) — rejected with 400.
- **SSO, RBAC, key management API** — keys are static demo entries in YAML; no admin UI, no rotation.
- **Real paid provider keys** — no OpenAI/Anthropic/Gemini adapters wired up; only stubs and local LM Studio.
  `OpenAIProvider` supports `api_key_env` for any OpenAI-compatible upstream, but it was not used with a real key.
- **Semantic cache** — exact match only; cache is in-process (not shared between replicas, lost on restart).
- **Token counting** — estimate (chars / 4) for routing and stubs; real usage comes from the upstream when it reports it.
- **Budgets** — per key only, no team budgets, no time windows (monthly reset); check-then-spend is not
  atomic, so concurrent requests can overshoot slightly.
- **Migrations** — `create_all` on start, no Alembic. No rate limiting per key, no retries with backoff on
  the same provider, no circuit breaker, no quality evaluation of answers.
- Scale: single process, tested on a laptop, not load-tested.
