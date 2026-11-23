"""Exact-match response cache (in-process LRU with TTL).

Key = sha256 of (requested model, task, generation params, messages). Any difference
in any of them is a miss. A hit is served for $0 and never touches a provider.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any

# body fields that change the answer and therefore belong in the key
GENERATION_PARAMS = (
    "temperature", "top_p", "max_tokens", "stop", "presence_penalty",
    "frequency_penalty", "seed", "response_format", "tools", "tool_choice", "n",
)


def cache_key(model: str, task: str | None, messages: list[dict[str, Any]], params: dict[str, Any]) -> str:
    payload = {"model": model, "task": task, "messages": messages, "params": params}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


class ExactCache:
    def __init__(self, ttl_s: float = 3600, max_entries: int = 10_000):
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self._data: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()

    def get(self, key: str) -> dict[str, Any] | None:
        item = self._data.get(key)
        if item is None:
            return None
        expires, value = item
        if expires < time.monotonic():
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return value

    def put(self, key: str, value: dict[str, Any]) -> None:
        self._data[key] = (time.monotonic() + self.ttl_s, value)
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)
