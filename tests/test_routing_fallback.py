"""Routing rules, fallback chain, config validation, OpenAI-compatible provider mapping."""
import asyncio
import os

import httpx
import pytest

from conftest import BASE_CONFIG, auth, chat_body
from gateway.config import load_config
from gateway.providers import OpenAIProvider, ProviderError
from gateway.router import choose_route


def gw(resp):
    return resp.json()["x_gateway"]


# ---------- routing (unit) ----------

@pytest.fixture
def cfg():
    return load_config(BASE_CONFIG)


@pytest.mark.parametrize("task,tokens,budget_left,route,rule", [
    (None, 10, 50.0, "r-general", "default"),
    ("code", 10, 50.0, "r-smart", "code"),
    ("private", 10, 50.0, "r-local", "private"),
    (None, 900, 50.0, "r-long", "long"),
    ("code", 10, 0.1, "r-cheap", "budget-low"),  # budget rule is first, beats task
    (None, 10, None, "r-general", "default"),  # unlimited key never hits budget rule
])
def test_rules_first_match_wins(cfg, task, tokens, budget_left, route, rule):
    d = choose_route(cfg, "auto", task, tokens, budget_left)
    assert (d.route, d.rule) == (route, rule)
    assert d.chain == cfg.routes[route]


def test_explicit_route_and_provider(cfg):
    assert choose_route(cfg, "r-smart", "private", 10, 50).rule == "explicit-route"
    d = choose_route(cfg, "cheap", None, 10, 50)
    assert (d.rule, d.chain) == ("explicit-provider", ["cheap"])


# ---------- routing (through HTTP) ----------

def test_task_from_header_field_and_metadata(client):
    r = client.post("/v1/chat/completions", json=chat_body("a"), headers={**auth(), "X-Task-Type": "code"})
    assert (gw(r)["route"], gw(r)["provider"]) == ("r-smart", "smart")
    r = client.post("/v1/chat/completions", json=chat_body("b", task="private"), headers=auth())
    assert gw(r)["provider"] == "local"
    r = client.post("/v1/chat/completions", json=chat_body("c", metadata={"task": "code"}), headers=auth())
    assert gw(r)["rule"] == "code"


def test_long_prompt_goes_to_long_route(client):
    r = client.post("/v1/chat/completions", json=chat_body("word " * 1000), headers=auth())
    assert (gw(r)["route"], gw(r)["rule"]) == ("r-long", "long")


def test_budget_rule_switches_to_cheap_route(make_client):
    def patch(c):
        c["keys"].append({"key": "sk-low", "name": "low", "team": "t", "budget_usd": 0.4})
    client = make_client(patch)
    r = client.post("/v1/chat/completions", json=chat_body("x", task="code"), headers=auth("sk-low"))
    assert (gw(r)["rule"], gw(r)["provider"]) == ("budget-low", "cheap")


# ---------- fallback ----------

@pytest.mark.parametrize("failure,kind", [("429", "rate_limit"), ("500", "server_error"), ("503", "server_error")])
def test_fallback_on_provider_error(make_client, failure, kind):
    client = make_client(lambda c: c["providers"]["fast"].update(fail_with=failure))
    r = client.post("/v1/chat/completions", json=chat_body(), headers=auth())
    assert r.status_code == 200
    g = gw(r)
    assert g["provider"] == "cheap"
    assert r.headers["X-Gateway-Provider"] == "cheap"
    assert [a["provider"] for a in g["attempts"]] == ["fast", "cheap"]
    assert g["attempts"][0] == {**g["attempts"][0], "ok": False, "error": kind, "status": int(failure)}
    metrics = client.get("/metrics").text
    assert f'gateway_provider_errors_total{{kind="{kind}",provider="fast"}} 1.0' in metrics
    assert 'gateway_fallbacks_total{route="r-general"} 1.0' in metrics


def test_fallback_on_timeout(make_client):
    def patch(c):
        c["providers"]["fast"].update(fail_with="timeout", timeout_s=0.05)
    client = make_client(patch)
    r = client.post("/v1/chat/completions", json=chat_body(), headers=auth())
    assert r.status_code == 200
    assert gw(r)["attempts"][0]["error"] == "timeout"
    assert gw(r)["provider"] == "cheap"


def test_provider_recovers_after_first_failures(make_client):
    client = make_client(lambda c: c["providers"]["fast"].update(fail_with="500", fail_first_n=1))
    assert gw(client.post("/v1/chat/completions", json=chat_body("1"), headers=auth()))["provider"] == "cheap"
    assert gw(client.post("/v1/chat/completions", json=chat_body("2"), headers=auth()))["provider"] == "fast"


def test_all_providers_fail_returns_error_with_attempts(make_client):
    def patch(c):
        c["providers"]["fast"]["fail_with"] = "500"
        c["providers"]["cheap"]["fail_with"] = "429"
    client = make_client(patch)
    r = client.post("/v1/chat/completions", json=chat_body(), headers=auth())
    assert r.status_code == 429  # last error was a rate limit
    err = r.json()["error"]
    assert err["type"] == "upstream_failed"
    assert [a["provider"] for a in err["attempts"]] == ["fast", "cheap"]
    log = client.portal.call(client.app.state.gateway.store.rows)
    assert log[0].status == 429 and log[0].attempts == 2


def test_max_attempts_limits_the_chain(make_client):
    def patch(c):
        c["routes"]["r-general"] = ["fast", "cheap", "smart"]
        c["routing"]["max_attempts"] = 2
        c["providers"]["fast"]["fail_with"] = "500"
        c["providers"]["cheap"]["fail_with"] = "500"
    client = make_client(patch)
    r = client.post("/v1/chat/completions", json=chat_body(), headers=auth())
    assert r.status_code == 502
    assert len(r.json()["error"]["attempts"]) == 2
    assert client.app.state.gateway.providers["smart"].calls == 0


def test_failed_request_is_not_cached(make_client):
    client = make_client(lambda c: c["providers"]["fast"].update(fail_with="500", fail_first_n=1))
    client.post("/v1/chat/completions", json=chat_body(model="fast"), headers=auth())  # fails, no fallback
    r = client.post("/v1/chat/completions", json=chat_body(model="fast"), headers=auth())
    assert r.status_code == 200 and gw(r)["cache_hit"] is False


# ---------- config validation ----------

def test_config_rejects_unknown_provider_in_route():
    bad = {**BASE_CONFIG, "routes": {"x": ["ghost"]}, "routing": {"default_route": "x"}}
    with pytest.raises(ValueError, match="unknown provider"):
        load_config(bad)


def test_config_rejects_rule_with_unknown_route():
    bad = {**BASE_CONFIG, "routing": {"default_route": "r-general", "rules": [{"name": "z", "route": "nope"}]}}
    with pytest.raises(ValueError, match="unknown route"):
        load_config(bad)


def test_shipped_config_is_valid():
    cfg = load_config("config/gateway.yaml")
    assert "asus-gemma" in cfg.providers and cfg.providers["asus-gemma"].type == "openai"


# ---------- OpenAI-compatible provider (mocked transport, no network) ----------

def _provider(handler):
    cfg = load_config("config/gateway.yaml").providers["asus-gemma"]
    return OpenAIProvider(cfg, transport=httpx.MockTransport(handler))


def test_openai_provider_parses_answer():
    def handler(req):
        assert req.url.path == "/v1/chat/completions"
        return httpx.Response(200, json={
            "model": "google/gemma-3-12b",
            "choices": [{"message": {"role": "assistant", "content": "Ок."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        })
    c = asyncio.run(_provider(handler).complete([{"role": "user", "content": "hi"}], {}))
    assert (c.content, c.prompt_tokens, c.completion_tokens) == ("Ок.", 12, 3)


@pytest.mark.parametrize("status,kind", [(429, "rate_limit"), (500, "server_error"), (400, "bad_request")])
def test_openai_provider_maps_errors(status, kind):
    p = _provider(lambda req: httpx.Response(status, text="err"))
    with pytest.raises(ProviderError) as e:
        asyncio.run(p.complete([{"role": "user", "content": "hi"}], {}))
    assert e.value.kind == kind
    assert e.value.retryable == (kind != "bad_request")


def test_openai_provider_unreachable_is_retryable():
    def handler(req):
        raise httpx.ConnectError("asus asleep")
    with pytest.raises(ProviderError) as e:
        asyncio.run(_provider(handler).complete([{"role": "user", "content": "hi"}], {}))
    assert e.value.kind == "unavailable" and e.value.retryable


# ---------- live: real LM Studio on ASUS (opt-in) ----------

@pytest.mark.live
@pytest.mark.skipif(os.environ.get("LLM_GATEWAY_LIVE") != "1", reason="set LLM_GATEWAY_LIVE=1 to call LM Studio on ASUS")
def test_live_asus_gemma():
    cfg = load_config("config/gateway.yaml").providers["asus-gemma"]
    c = asyncio.run(OpenAIProvider(cfg).complete([{"role": "user", "content": "Ответь одним словом: Ок."}], {"max_tokens": 10}))
    assert c.content.strip()
