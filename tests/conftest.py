import copy
from typing import Any

import pytest
from fastapi.testclient import TestClient

from gateway.main import create_app

KEY = "sk-test-games"
POOR_KEY = "sk-test-poor"

BASE_CONFIG: dict[str, Any] = {
    "providers": {
        "smart": {"type": "stub", "model": "smart-v1", "price_in_per_1k": 0.003, "price_out_per_1k": 0.015},
        "fast": {"type": "stub", "model": "fast-v1", "price_in_per_1k": 0.0005, "price_out_per_1k": 0.0015},
        "cheap": {"type": "stub", "model": "cheap-v1", "price_in_per_1k": 0.0001, "price_out_per_1k": 0.0002},
        "local": {"type": "stub", "model": "local-v1"},
    },
    "routes": {
        "r-smart": ["smart", "fast"],
        "r-general": ["fast", "cheap"],
        "r-cheap": ["cheap", "fast"],
        "r-long": ["smart", "fast"],
        "r-local": ["local", "cheap"],
    },
    "routing": {
        "default_route": "r-general",
        "max_attempts": 3,
        "rules": [
            {"name": "budget-low", "budget_left_below_usd": 0.5, "route": "r-cheap"},
            {"name": "private", "task": "private", "route": "r-local"},
            {"name": "code", "task": "code", "route": "r-smart"},
            {"name": "long", "min_prompt_tokens": 500, "route": "r-long"},
        ],
    },
    "keys": [
        {"key": KEY, "name": "games-dev", "team": "games", "budget_usd": 100.0},
        {"key": POOR_KEY, "name": "poor-bot", "team": "analytics", "budget_usd": 0.0001},
        {"key": "sk-test-unlimited", "name": "admin", "team": "platform"},
    ],
}


def auth(key: str = KEY) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def chat_body(text: str = "hello", model: str = "auto", **extra: Any) -> dict[str, Any]:
    return {"model": model, "messages": [{"role": "user", "content": text}], **extra}


@pytest.fixture
def make_client(tmp_path):
    """make_client(patch=fn) -> TestClient. `patch(cfg_dict)` may edit a deep copy of BASE_CONFIG."""
    clients = []

    def _make(patch=None) -> TestClient:
        cfg = copy.deepcopy(BASE_CONFIG)
        if patch:
            patch(cfg)
        db = tmp_path / f"gw{len(clients)}.db"
        app = create_app(cfg, database_url=f"sqlite+aiosqlite:///{db}")
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def client(make_client):
    return make_client()
