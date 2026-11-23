"""LLM providers.

StubProvider    - fake model with configurable latency, price and failures (429/5xx/timeout).
OpenAIProvider  - any OpenAI-compatible /chat/completions upstream (LM Studio, vLLM, Ollama ...).
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from .config import ProviderCfg


class ProviderError(Exception):
    """Upstream failure. `retryable` errors move the request to the next provider."""

    RETRYABLE = {"rate_limit", "server_error", "timeout", "unavailable"}

    def __init__(self, kind: str, status: int, message: str):
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.message = message

    @property
    def retryable(self) -> bool:
        return self.kind in self.RETRYABLE


@dataclass
class Completion:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str = "stop"


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token). Good enough for routing and stub usage."""
    return max(1, len(text) // 4)


def messages_text(messages: list[dict[str, Any]]) -> str:
    parts = []
    for m in messages:
        c = m.get("content", "")
        parts.append(c if isinstance(c, str) else json.dumps(c, ensure_ascii=False))
    return "\n".join(parts)


def cost_usd(cfg: ProviderCfg, prompt_tokens: int, completion_tokens: int) -> float:
    return round(
        prompt_tokens / 1000 * cfg.price_in_per_1k + completion_tokens / 1000 * cfg.price_out_per_1k,
        8,
    )


class Provider(Protocol):
    cfg: ProviderCfg

    async def complete(self, messages: list[dict[str, Any]], params: dict[str, Any]) -> Completion: ...


class StubProvider:
    def __init__(self, cfg: ProviderCfg):
        self.cfg = cfg
        self.calls = 0

    async def complete(self, messages: list[dict[str, Any]], params: dict[str, Any]) -> Completion:
        self.calls += 1
        if self.cfg.latency_ms:
            await asyncio.sleep(self.cfg.latency_ms / 1000)

        failing = self.cfg.fail_with and (
            self.cfg.fail_first_n is None or self.calls <= self.cfg.fail_first_n
        )
        if failing:
            kind = self.cfg.fail_with
            if kind == "timeout":
                await asyncio.sleep(self.cfg.timeout_s + 5)  # caller's wait_for cuts this off
            if kind == "429":
                raise ProviderError("rate_limit", 429, f"{self.cfg.name}: rate limited (stub)")
            raise ProviderError("server_error", int(kind), f"{self.cfg.name}: upstream {kind} (stub)")

        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), {})
        text = f"[{self.cfg.name}] stub answer to: {messages_text([last_user])[:80]}"
        max_tokens = params.get("max_tokens")
        completion_tokens = estimate_tokens(text)
        if max_tokens:
            completion_tokens = min(completion_tokens, int(max_tokens))
        return Completion(
            content=text,
            model=self.cfg.model,
            prompt_tokens=estimate_tokens(messages_text(messages)),
            completion_tokens=completion_tokens,
        )


class OpenAIProvider:
    def __init__(self, cfg: ProviderCfg, transport: httpx.AsyncBaseTransport | None = None):
        self.cfg = cfg
        self._transport = transport

    async def complete(self, messages: list[dict[str, Any]], params: dict[str, Any]) -> Completion:
        headers = {}
        if self.cfg.api_key_env and os.environ.get(self.cfg.api_key_env):
            headers["Authorization"] = f"Bearer {os.environ[self.cfg.api_key_env]}"
        body = {"model": self.cfg.model, "messages": messages, **params, "stream": False}
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=self.cfg.timeout_s, transport=self._transport) as client:
                r = await client.post(url, json=body, headers=headers)
        except httpx.TimeoutException as e:
            raise ProviderError("timeout", 504, f"{self.cfg.name}: timeout ({e.__class__.__name__})") from e
        except httpx.TransportError as e:
            raise ProviderError("unavailable", 503, f"{self.cfg.name}: unreachable ({e.__class__.__name__})") from e

        if r.status_code == 429:
            raise ProviderError("rate_limit", 429, f"{self.cfg.name}: rate limited")
        if r.status_code >= 500:
            raise ProviderError("server_error", r.status_code, f"{self.cfg.name}: upstream {r.status_code}")
        if r.status_code >= 400:
            raise ProviderError("bad_request", r.status_code, f"{self.cfg.name}: {r.text[:200]}")

        data = r.json()
        choice = (data.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        return Completion(
            content=content,
            model=data.get("model") or self.cfg.model,
            prompt_tokens=int(usage.get("prompt_tokens") or estimate_tokens(messages_text(messages))),
            completion_tokens=int(usage.get("completion_tokens") or estimate_tokens(content)),
            finish_reason=choice.get("finish_reason") or "stop",
        )


def build_providers(cfgs: dict[str, ProviderCfg]) -> dict[str, Provider]:
    out: dict[str, Provider] = {}
    for name, cfg in cfgs.items():
        out[name] = StubProvider(cfg) if cfg.type == "stub" else OpenAIProvider(cfg)
    return out
