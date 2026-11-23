"""Routing (which provider chain serves a request) and fallback (walking the chain)."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from .config import GatewayConfig
from .providers import Completion, Provider, ProviderError


class UnknownModel(Exception):
    pass


@dataclass
class Decision:
    route: str  # route name, or provider name for a direct call
    rule: str  # which rule fired: rule name | "explicit-route" | "explicit-provider" | "default"
    chain: list[str]


def choose_route(
    cfg: GatewayConfig,
    model: str,
    task: str | None,
    prompt_tokens: int,
    budget_left: float | None,
) -> Decision:
    """model="auto" -> first matching rule (top to bottom) or default route.
    model=<route>    -> that route's chain.
    model=<provider> -> that provider only, no fallback."""
    if model in cfg.routes:
        return Decision(model, "explicit-route", cfg.routes[model])
    if model in cfg.providers:
        return Decision(model, "explicit-provider", [model])
    if model != "auto":
        raise UnknownModel(model)
    for rule in cfg.rules:
        if rule.matches(task, prompt_tokens, budget_left):
            return Decision(rule.route, rule.name, cfg.routes[rule.route])
    return Decision(cfg.default_route, "default", cfg.routes[cfg.default_route])


@dataclass
class Attempt:
    provider: str
    ok: bool
    latency_ms: float
    error_kind: str | None = None
    status: int | None = None

    def as_dict(self) -> dict[str, Any]:
        d = {"provider": self.provider, "ok": self.ok, "latency_ms": round(self.latency_ms, 1)}
        if not self.ok:
            d.update(error=self.error_kind, status=self.status)
        return d


class ChainFailed(Exception):
    def __init__(self, attempts: list[Attempt], last_error: ProviderError | None):
        super().__init__("all providers failed")
        self.attempts = attempts
        self.last_error = last_error


async def run_chain(
    chain: list[str],
    providers: dict[str, Provider],
    messages: list[dict[str, Any]],
    params: dict[str, Any],
    max_attempts: int,
    on_error=None,
) -> tuple[str, Completion, list[Attempt]]:
    """Try providers in order. Retryable errors (429, 5xx, timeout, unreachable) move on to the
    next provider; at most `max_attempts` providers are tried. Non-retryable errors stop at once."""
    attempts: list[Attempt] = []
    last: ProviderError | None = None
    for name in chain[:max_attempts]:
        p = providers[name]
        t0 = time.perf_counter()
        try:
            comp = await asyncio.wait_for(p.complete(messages, params), timeout=p.cfg.timeout_s)
        except asyncio.TimeoutError:
            err = ProviderError("timeout", 504, f"{name}: no answer in {p.cfg.timeout_s}s")
        except ProviderError as e:
            err = e
        else:
            attempts.append(Attempt(name, True, (time.perf_counter() - t0) * 1000))
            return name, comp, attempts
        attempts.append(Attempt(name, False, (time.perf_counter() - t0) * 1000, err.kind, err.status))
        if on_error:
            on_error(name, err)
        last = err
        if not err.retryable:
            break
    raise ChainFailed(attempts, last)
