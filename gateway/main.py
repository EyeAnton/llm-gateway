"""FastAPI app: OpenAI-compatible /v1/chat/completions and /v1/models, plus /metrics and /stats.

Request flow:
  auth (Bearer key) -> budget check -> exact cache -> routing rule -> fallback chain
  -> cost accounting -> journal (DB) + Prometheus metrics -> OpenAI-shaped response
"""
from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict

from .cache import GENERATION_PARAMS, ExactCache, cache_key
from .config import GatewayConfig, KeyCfg, load_config
from .metrics import Metrics
from .providers import Provider, ProviderError, build_providers, cost_usd, estimate_tokens, messages_text
from .router import ChainFailed, UnknownModel, choose_route, run_chain


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "auto"
    messages: list[dict[str, Any]]
    stream: bool = False
    task: str | None = None  # task type for routing; header X-Task-Type wins
    team: str | None = None  # overrides the key's team for accounting
    metadata: dict[str, Any] | None = None

    def generation_params(self) -> dict[str, Any]:
        extra = self.model_extra or {}
        return {k: extra[k] for k in GENERATION_PARAMS if k in extra}


def error(status: int, type_: str, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse({"error": {"type": type_, "message": message, **extra}}, status_code=status)


class Gateway:
    def __init__(self, cfg: GatewayConfig, database_url: str, providers: dict[str, Provider] | None = None):
        from .store import Store

        self.cfg = cfg
        self.providers = providers or build_providers(cfg.providers)
        self.cache = ExactCache(cfg.cache_ttl_s, cfg.cache_max_entries)
        self.metrics = Metrics()
        self.store = Store(database_url)

    def auth(self, authorization: str | None) -> KeyCfg | None:
        if not authorization or not authorization.lower().startswith("bearer "):
            return None
        return self.cfg.keys.get(authorization.split(" ", 1)[1].strip())


def create_app(
    config: GatewayConfig | str | dict | None = None,
    database_url: str | None = None,
    providers: dict[str, Provider] | None = None,
) -> FastAPI:
    cfg = config if isinstance(config, GatewayConfig) else load_config(config)
    db_url = database_url or os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./gateway.db")
    gw = Gateway(cfg, db_url, providers)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await gw.store.init()
        yield
        await gw.store.close()

    app = FastAPI(title="llm-gateway", version="0.1.0", lifespan=lifespan)
    app.state.gateway = gw

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.get("/v1/models")
    async def models(authorization: str | None = Header(None)):
        if gw.auth(authorization) is None:
            return error(401, "invalid_api_key", "Missing or unknown API key")
        data = [{"id": "auto", "object": "model", "owned_by": "llm-gateway",
                 "description": "routed by rules in config"}]
        data += [{"id": name, "object": "model", "owned_by": "llm-gateway",
                  "description": "route: " + " -> ".join(chain)} for name, chain in cfg.routes.items()]
        data += [{"id": name, "object": "model", "owned_by": p.type,
                  "description": f"provider {p.model}, no fallback"} for name, p in cfg.providers.items()]
        return {"object": "list", "data": data}

    @app.get("/metrics")
    async def metrics():
        return Response(gw.metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/stats")
    async def stats():
        out = await gw.store.stats()
        out["cache_entries"] = len(gw.cache)
        return out

    @app.post("/v1/chat/completions")
    async def chat(
        body: ChatRequest,
        authorization: str | None = Header(None),
        x_task_type: str | None = Header(None),
    ):
        t0 = time.perf_counter()
        key = gw.auth(authorization)
        if key is None:
            return error(401, "invalid_api_key", "Missing or unknown API key")
        if body.stream:
            return error(400, "unsupported", "stream=true is not supported yet")

        team = body.team or key.team
        task = x_task_type or body.task or (body.metadata or {}).get("task")
        params = body.generation_params()
        m = gw.metrics

        async def journal(status: int, **f: Any) -> None:
            await gw.store.log(key_name=key.name, team=team, model=body.model, status=status,
                               latency_ms=(time.perf_counter() - t0) * 1000, **f)

        # 1. budget
        budget_left = None
        if key.budget_usd is not None:
            spent = await gw.store.spent_usd(key.name)
            budget_left = key.budget_usd - spent
            if budget_left <= 0:
                m.budget_rejections.labels(team).inc()
                m.requests.labels("-", "-", "402", "miss").inc()
                await journal(402, error="budget_exceeded")
                return error(402, "budget_exceeded",
                             f"Key '{key.name}' (team {team}) has spent ${spent:.4f} of its "
                             f"${key.budget_usd:.2f} budget. Ask the platform team to raise it.",
                             spent_usd=round(spent, 6), budget_usd=key.budget_usd)

        # 2. exact cache
        ckey = cache_key(body.model, task, body.messages, params)
        cached = gw.cache.get(ckey)
        if cached is not None:
            m.cache_hits.inc()
            latency = time.perf_counter() - t0
            m.requests.labels(cached["route"], cached["provider"], "200", "hit").inc()
            m.latency.labels(cached["provider"], "hit").observe(latency)
            await journal(200, route=cached["route"], rule=cached["rule"], provider=cached["provider"],
                          attempts=0, prompt_tokens=0, completion_tokens=0, cost_usd=0.0, cache_hit=True)
            return respond(cached, cache_hit=True, cost=0.0, latency=latency, attempts=[])
        m.cache_misses.inc()

        # 3. routing
        prompt_tokens = estimate_tokens(messages_text(body.messages))
        try:
            decision = choose_route(cfg, body.model, task, prompt_tokens, budget_left)
        except UnknownModel:
            await journal(404, error="model_not_found")
            return error(404, "model_not_found",
                         f"Unknown model {body.model!r}. See GET /v1/models.")

        # 4. fallback chain
        def on_error(provider: str, err: ProviderError) -> None:
            m.provider_errors.labels(provider, err.kind).inc()

        try:
            provider, comp, attempts = await run_chain(
                decision.chain, gw.providers, body.messages, params, cfg.max_attempts, on_error)
        except ChainFailed as e:
            last = e.last_error
            status = 429 if last and last.kind == "rate_limit" else 502
            if last and not last.retryable:
                status = last.status
            m.requests.labels(decision.route, "-", str(status), "miss").inc()
            await journal(status, route=decision.route, rule=decision.rule, attempts=len(e.attempts),
                          error=(last.message if last else "no provider")[:256])
            return error(status, "upstream_failed",
                         f"All providers failed for route {decision.route!r}: "
                         f"{last.message if last else 'none tried'}",
                         attempts=[a.as_dict() for a in e.attempts])

        # 5. accounting
        cost = cost_usd(cfg.providers[provider], comp.prompt_tokens, comp.completion_tokens)
        latency = time.perf_counter() - t0
        if len(attempts) > 1:
            m.fallbacks.labels(decision.route).inc()
        m.requests.labels(decision.route, provider, "200", "miss").inc()
        m.latency.labels(provider, "miss").observe(latency)
        m.cost.labels(team, provider).inc(cost)
        m.tokens.labels(team, "prompt").inc(comp.prompt_tokens)
        m.tokens.labels(team, "completion").inc(comp.completion_tokens)
        await journal(200, route=decision.route, rule=decision.rule, provider=provider,
                      attempts=len(attempts), prompt_tokens=comp.prompt_tokens,
                      completion_tokens=comp.completion_tokens, cost_usd=cost, cache_hit=False)

        entry = {"route": decision.route, "rule": decision.rule, "provider": provider,
                 "model": comp.model, "content": comp.content, "finish_reason": comp.finish_reason,
                 "prompt_tokens": comp.prompt_tokens, "completion_tokens": comp.completion_tokens}
        gw.cache.put(ckey, entry)
        return respond(entry, cache_hit=False, cost=cost, latency=latency,
                       attempts=[a.as_dict() for a in attempts])

    def respond(entry: dict[str, Any], cache_hit: bool, cost: float, latency: float,
                attempts: list[dict[str, Any]]) -> JSONResponse:
        payload = {
            "id": "chatcmpl-" + uuid.uuid4().hex[:24],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": entry["model"],
            "choices": [{"index": 0, "finish_reason": entry["finish_reason"],
                         "message": {"role": "assistant", "content": entry["content"]}}],
            "usage": {"prompt_tokens": entry["prompt_tokens"],
                      "completion_tokens": entry["completion_tokens"],
                      "total_tokens": entry["prompt_tokens"] + entry["completion_tokens"]},
            "x_gateway": {"route": entry["route"], "rule": entry["rule"], "provider": entry["provider"],
                          "cache_hit": cache_hit, "cost_usd": cost,
                          "latency_ms": round(latency * 1000, 1), "attempts": attempts},
        }
        headers = {"X-Gateway-Provider": entry["provider"], "X-Gateway-Route": entry["route"],
                   "X-Gateway-Cache": "hit" if cache_hit else "miss", "X-Gateway-Cost-USD": f"{cost:.8f}"}
        return JSONResponse(payload, headers=headers)

    return app
# run: uvicorn --factory gateway.main:create_app  (config from GATEWAY_CONFIG, DB from DATABASE_URL)
