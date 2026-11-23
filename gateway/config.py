"""Gateway configuration: providers, routes, routing rules, API keys.

Loaded from YAML (config/gateway.yaml) or a plain dict (tests).
Validated on load: every route must reference known providers,
every rule must reference a known route.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

VALID_FAILURES = {None, "429", "500", "503", "timeout"}


@dataclass
class ProviderCfg:
    name: str
    type: str  # "stub" | "openai"
    model: str  # upstream model id
    price_in_per_1k: float = 0.0  # USD per 1k prompt tokens
    price_out_per_1k: float = 0.0  # USD per 1k completion tokens
    timeout_s: float = 30.0
    # openai-compatible upstream
    base_url: str | None = None
    api_key_env: str | None = None  # name of env var with the key, never the key itself
    connect_timeout_s: float = 3.0  # fail fast when the host is down (e.g. ASUS asleep)
    # stub behaviour
    latency_ms: float = 0.0
    fail_with: str | None = None  # "429" | "500" | "503" | "timeout"
    fail_first_n: int | None = None  # fail only the first N calls, then recover


@dataclass
class Rule:
    name: str
    route: str
    task: str | None = None
    min_prompt_tokens: int | None = None
    max_prompt_tokens: int | None = None
    budget_left_below_usd: float | None = None

    def matches(self, task: str | None, prompt_tokens: int, budget_left: float | None) -> bool:
        if self.task is not None and task != self.task:
            return False
        if self.min_prompt_tokens is not None and prompt_tokens < self.min_prompt_tokens:
            return False
        if self.max_prompt_tokens is not None and prompt_tokens > self.max_prompt_tokens:
            return False
        if self.budget_left_below_usd is not None:
            if budget_left is None or budget_left >= self.budget_left_below_usd:
                return False
        return True


@dataclass
class KeyCfg:
    key: str
    name: str
    team: str
    budget_usd: float | None = None  # None = unlimited


@dataclass
class GatewayConfig:
    providers: dict[str, ProviderCfg]
    routes: dict[str, list[str]]
    rules: list[Rule]
    default_route: str
    max_attempts: int = 3
    keys: dict[str, KeyCfg] = field(default_factory=dict)
    cache_ttl_s: float = 3600.0
    cache_max_entries: int = 10_000


def _provider(name: str, raw: dict[str, Any]) -> ProviderCfg:
    raw = dict(raw)
    if raw.get("fail_with") is not None:
        raw["fail_with"] = str(raw["fail_with"])
    cfg = ProviderCfg(name=name, **raw)
    if cfg.type not in ("stub", "openai"):
        raise ValueError(f"provider {name}: unknown type {cfg.type!r}")
    if cfg.fail_with not in VALID_FAILURES:
        raise ValueError(f"provider {name}: fail_with must be one of 429/500/503/timeout")
    if cfg.type == "openai" and not cfg.base_url:
        raise ValueError(f"provider {name}: base_url is required for type openai")
    return cfg


def load_config(src: str | Path | dict[str, Any] | None = None) -> GatewayConfig:
    if src is None:
        src = os.environ.get("GATEWAY_CONFIG", "config/gateway.yaml")
    if isinstance(src, (str, Path)):
        raw = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
    else:
        raw = src

    providers = {n: _provider(n, p) for n, p in (raw.get("providers") or {}).items()}
    routes = {n: list(chain) for n, chain in (raw.get("routes") or {}).items()}
    routing = raw.get("routing") or {}
    rules = [Rule(**r) for r in routing.get("rules") or []]
    keys = {
        k["key"]: KeyCfg(key=k["key"], name=k["name"], team=k["team"], budget_usd=k.get("budget_usd"))
        for k in raw.get("keys") or []
    }
    cache = raw.get("cache") or {}
    cfg = GatewayConfig(
        providers=providers,
        routes=routes,
        rules=rules,
        default_route=routing.get("default_route", next(iter(routes), "")),
        max_attempts=int(routing.get("max_attempts", 3)),
        keys=keys,
        cache_ttl_s=float(cache.get("ttl_s", 3600)),
        cache_max_entries=int(cache.get("max_entries", 10_000)),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: GatewayConfig) -> None:
    if not cfg.routes:
        raise ValueError("config: at least one route is required")
    for name, chain in cfg.routes.items():
        if not chain:
            raise ValueError(f"route {name}: empty provider chain")
        for p in chain:
            if p not in cfg.providers:
                raise ValueError(f"route {name}: unknown provider {p!r}")
        if name in cfg.providers:
            raise ValueError(f"route {name}: name clashes with a provider")
    if cfg.default_route not in cfg.routes:
        raise ValueError(f"default_route {cfg.default_route!r} is not a route")
    for r in cfg.rules:
        if r.route not in cfg.routes:
            raise ValueError(f"rule {r.name}: unknown route {r.route!r}")
    if cfg.max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
