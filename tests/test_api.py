"""Endpoints, auth, cache, budget, journal, metrics — everything on stub providers."""
from conftest import KEY, POOR_KEY, auth, chat_body


def gw(resp):
    return resp.json()["x_gateway"]


# ---------- /v1/models and auth ----------

def test_models_lists_auto_routes_and_providers(client):
    r = client.get("/v1/models", headers=auth())
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    ids = {m["id"] for m in body["data"]}
    assert {"auto", "r-smart", "r-general", "smart", "fast", "cheap"} <= ids
    assert all(m["object"] == "model" for m in body["data"])


def test_models_requires_key(client):
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers=auth("sk-nope")).status_code == 401


def test_chat_requires_key(client):
    r = client.post("/v1/chat/completions", json=chat_body())
    assert r.status_code == 401
    assert r.json()["error"]["type"] == "invalid_api_key"


def test_chat_response_is_openai_shaped(client):
    r = client.post("/v1/chat/completions", json=chat_body("hi"), headers=auth())
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    assert r.headers["X-Gateway-Provider"] == "fast"


def test_stream_is_rejected_explicitly(client):
    r = client.post("/v1/chat/completions", json=chat_body(stream=True), headers=auth())
    assert r.status_code == 400


def test_unknown_model_404(client):
    r = client.post("/v1/chat/completions", json=chat_body(model="gpt-99"), headers=auth())
    assert r.status_code == 404
    assert r.json()["error"]["type"] == "model_not_found"


# ---------- cache ----------

def test_second_identical_request_is_cache_hit_and_free(client):
    first = client.post("/v1/chat/completions", json=chat_body("same question"), headers=auth())
    second = client.post("/v1/chat/completions", json=chat_body("same question"), headers=auth())
    assert gw(first)["cache_hit"] is False
    assert gw(first)["cost_usd"] > 0
    assert gw(second)["cache_hit"] is True
    assert gw(second)["cost_usd"] == 0.0
    assert second.headers["X-Gateway-Cache"] == "hit"
    assert second.json()["choices"][0]["message"]["content"] == first.json()["choices"][0]["message"]["content"]
    # provider was called once
    assert client.app.state.gateway.providers["fast"].calls == 1
    metrics = client.get("/metrics").text
    assert "gateway_cache_hits_total 1.0" in metrics
    assert "gateway_cache_misses_total 1.0" in metrics


def test_different_params_are_cache_miss(client):
    client.post("/v1/chat/completions", json=chat_body("q", temperature=0.1), headers=auth())
    r = client.post("/v1/chat/completions", json=chat_body("q", temperature=0.9), headers=auth())
    assert gw(r)["cache_hit"] is False
    r = client.post("/v1/chat/completions", json=chat_body("q", temperature=0.1, task="code"), headers=auth())
    assert gw(r)["cache_hit"] is False


# ---------- budget ----------

def test_budget_exceeded_returns_402(client):
    first = client.post("/v1/chat/completions", json=chat_body("x" * 400, model="r-smart"), headers=auth(POOR_KEY))
    assert first.status_code == 200  # first request spends more than the tiny budget
    r = client.post("/v1/chat/completions", json=chat_body("another"), headers=auth(POOR_KEY))
    assert r.status_code == 402
    err = r.json()["error"]
    assert err["type"] == "budget_exceeded"
    assert "poor-bot" in err["message"] and "budget" in err["message"]
    assert err["spent_usd"] >= err["budget_usd"]
    assert "gateway_budget_rejections_total" in client.get("/metrics").text


def test_unlimited_key_has_no_budget(client):
    r = client.post("/v1/chat/completions", json=chat_body(), headers=auth("sk-test-unlimited"))
    assert r.status_code == 200


# ---------- journal / accounting ----------

def rows(client):
    store = client.app.state.gateway.store
    return client.portal.call(store.rows)


def test_journal_records_tokens_cost_team_and_cache(client):
    client.post("/v1/chat/completions", json=chat_body("log me"), headers=auth())
    client.post("/v1/chat/completions", json=chat_body("log me"), headers=auth())
    client.post("/v1/chat/completions", json=chat_body("for analytics", team="analytics"), headers=auth())
    log = list(reversed(rows(client)))
    assert len(log) == 3
    miss, hit, other = log
    assert (miss.key_name, miss.team, miss.provider, miss.status, miss.cache_hit) == ("games-dev", "games", "fast", 200, False)
    assert miss.prompt_tokens > 0 and miss.completion_tokens > 0 and miss.cost_usd > 0
    assert miss.latency_ms >= 0 and miss.route == "r-general" and miss.rule == "default"
    assert hit.cache_hit is True and hit.cost_usd == 0.0
    assert other.team == "analytics"

    stats = client.get("/stats").json()
    assert stats["by_team"]["games"]["requests"] == 2
    assert stats["by_team"]["games"]["cache_hits"] == 1
    assert stats["by_team"]["games"]["cost_usd"] == round(miss.cost_usd, 8)
    assert stats["by_key"]["games-dev"]["requests"] == 3


def test_cost_matches_price_table(client):
    r = client.post("/v1/chat/completions", json=chat_body("price check", model="smart"), headers=auth())
    u = r.json()["usage"]
    expected = round(u["prompt_tokens"] / 1000 * 0.003 + u["completion_tokens"] / 1000 * 0.015, 8)
    assert gw(r)["cost_usd"] == expected


def test_rejections_are_journaled(client):
    client.post("/v1/chat/completions", json=chat_body(model="nope"), headers=auth())
    assert rows(client)[0].status == 404


# ---------- metrics ----------

def test_metrics_prometheus_format(client):
    client.post("/v1/chat/completions", json=chat_body("m"), headers=auth())
    text = client.get("/metrics").text
    assert 'gateway_requests_total{cache="miss",provider="fast",route="r-general",status="200"} 1.0' in text
    assert "gateway_request_latency_seconds_bucket" in text
    assert 'gateway_cost_usd_total{provider="fast",team="games"}' in text
    assert 'gateway_tokens_total{direction="prompt",team="games"}' in text
