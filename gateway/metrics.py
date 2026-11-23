"""Prometheus metrics. Each app instance owns its registry (keeps tests independent)."""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.requests = Counter(
            "gateway_requests_total", "Chat requests by outcome",
            ["route", "provider", "status", "cache"], registry=r)
        self.cache_hits = Counter("gateway_cache_hits_total", "Exact cache hits", registry=r)
        self.cache_misses = Counter("gateway_cache_misses_total", "Exact cache misses", registry=r)
        self.provider_errors = Counter(
            "gateway_provider_errors_total", "Upstream failures (each one triggers fallback)",
            ["provider", "kind"], registry=r)
        self.fallbacks = Counter(
            "gateway_fallbacks_total", "Requests served by a non-first provider in the chain",
            ["route"], registry=r)
        self.budget_rejections = Counter(
            "gateway_budget_rejections_total", "Requests refused because the key's budget is spent",
            ["team"], registry=r)
        self.cost = Counter("gateway_cost_usd_total", "Spend in USD", ["team", "provider"], registry=r)
        self.tokens = Counter("gateway_tokens_total", "Tokens", ["team", "direction"], registry=r)
        self.latency = Histogram(
            "gateway_request_latency_seconds", "End-to-end request latency",
            ["provider", "cache"], registry=r,
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60))

    def render(self) -> bytes:
        return generate_latest(self.registry)
