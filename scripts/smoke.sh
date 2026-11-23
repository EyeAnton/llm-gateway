#!/usr/bin/env bash
# Smoke test against a running gateway (default http://localhost:8080) with the demo config.
set -euo pipefail
URL="${GATEWAY_URL:-http://localhost:8080}"
KEY="${GATEWAY_KEY:-sk-demo-games}"

say() { printf '\n== %s\n' "$1"; }
chat() {
  curl -sf "$URL/v1/chat/completions" -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' "$@" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); g=d["x_gateway"]; print(json.dumps({k: g[k] for k in ("route","rule","provider","cache_hit","cost_usd","latency_ms")}), "|", d["choices"][0]["message"]["content"][:60])'
}

say "models";         curl -sf "$URL/v1/models" -H "Authorization: Bearer $KEY" | python3 -c 'import json,sys; print([m["id"] for m in json.load(sys.stdin)["data"]])'
say "default route";  chat -d '{"model":"auto","messages":[{"role":"user","content":"smoke hello"}]}'
say "same again (cache)"; chat -d '{"model":"auto","messages":[{"role":"user","content":"smoke hello"}]}'
say "task=code";      chat -H 'X-Task-Type: code' -d '{"model":"auto","messages":[{"role":"user","content":"write a function"}]}'
say "fallback demo";  chat -d '{"model":"flaky-demo","messages":[{"role":"user","content":"flaky?"}]}'
say "stats";          curl -sf "$URL/stats" | python3 -m json.tool | head -30
say "metrics";        curl -sf "$URL/metrics" | grep -E '^gateway_(requests|cache_hits|fallbacks|cost)' | head -12
