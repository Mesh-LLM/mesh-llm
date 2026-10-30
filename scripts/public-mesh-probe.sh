#!/usr/bin/env bash
# Pingdom-style probe for the public mesh console.
#
# Checks, in order:
#   1. GET  /health                      -> HTTP 200
#   2. GET  /v1/models                   -> HTTP 200 with at least one model
#   3. POST /v1/chat/completions (auto)  -> HTTP 200 and a non-empty reply
#
# Step 3 is the one that matters: /health reports "ok" whenever the console
# has peers, even while every completion fails, so only a real completion
# proves public inference works.
#
# Usage: public-mesh-probe.sh [base_url]
# Env:   MESH_PROBE_BASE_URL  (default https://public.meshllm.cloud)
#        MESH_PROBE_MODEL     (default auto)
#        MESH_PROBE_ATTEMPTS  (default 2; completion retries before failing)
#        MESH_PROBE_TIMEOUT   (default 90; seconds per request)
#        MESH_PROBE_OUT       (optional; write a JSON result here)
#
# Exit 0 when every check passes, 1 when any check fails, 2 on usage errors.
# Always prints one JSON line: {"state":"ok"|"failed","detail":...,...}.
set -euo pipefail

base_url="${1:-${MESH_PROBE_BASE_URL:-https://public.meshllm.cloud}}"
base_url="${base_url%/}"
model="${MESH_PROBE_MODEL:-auto}"
attempts="${MESH_PROBE_ATTEMPTS:-2}"
timeout_secs="${MESH_PROBE_TIMEOUT:-90}"
out_path="${MESH_PROBE_OUT:-}"

for tool in curl jq; do
  command -v "$tool" >/dev/null 2>&1 || { echo "missing dependency: $tool" >&2; exit 2; }
done

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

state="ok"
detail=""
health_code=""
models_code=""
model_count=0
completion_code=""
completion_ms=""
completion_attempts=0
reply_chars=0

# seconds_to_ms "1.234567" -> 1234. curl reports %{time_total} in seconds;
# this avoids float arithmetic and any interpreter dependency.
seconds_to_ms() {
  local t="${1/,/.}" whole frac
  whole="${t%%.*}"
  frac="${t#"$whole"}"
  frac="${frac#.}000"
  [[ "$whole" =~ ^[0-9]+$ ]] || whole=0
  [[ "$frac" =~ ^[0-9]+$ ]] || frac=000
  echo $((10#$whole * 1000 + 10#${frac:0:3}))
}

# http GET|POST path [body] -> prints HTTP code ("000" on transport failure),
# response body saved to $workdir/body, curl stderr to $workdir/curl.err,
# request wall time in milliseconds (curl's time_total) in $workdir/ms.
http() {
  local method="$1" path="$2" body="${3:-}" out code secs rc=0
  local -a args=(-sS -m "$timeout_secs" -o "$workdir/body" -w '%{http_code} %{time_total}' -X "$method" "$base_url$path")
  if [[ -n "$body" ]]; then
    args+=(-H 'content-type: application/json' --data "$body")
  fi
  : >"$workdir/body"
  out="$(curl "${args[@]}" 2>"$workdir/curl.err")" || rc=$?
  code="${out%% *}"
  secs="${out#* }"
  [[ "$secs" == "$out" ]] && secs=0
  seconds_to_ms "$secs" >"$workdir/ms"
  if [[ $rc -ne 0 || -z "$code" ]]; then
    code="000"
  fi
  echo "$code"
}

fail() {
  state="failed"
  detail="$1"
}

# 1. /health
health_code="$(http GET /health)"
if [[ "$health_code" != "200" ]]; then
  fail "GET /health -> HTTP ${health_code} $(head -c 200 "$workdir/curl.err" 2>/dev/null || true)"
fi

# 2. /v1/models
if [[ "$state" == "ok" ]]; then
  models_code="$(http GET /v1/models)"
  if [[ "$models_code" != "200" ]]; then
    fail "GET /v1/models -> HTTP ${models_code}"
  else
    model_count="$(jq -r '[.data[]? | select(.id != "mesh" and .id != "auto")] | length' "$workdir/body" 2>/dev/null || echo 0)"
    if [[ "$model_count" -lt 1 ]]; then
      fail "GET /v1/models -> 200 but no models advertised"
    fi
  fi
fi

# 3. real completion through the mesh
if [[ "$state" == "ok" ]]; then
  request="$(jq -cn --arg model "$model" '{
    model: $model,
    messages: [{role: "user", content: "Reply with exactly the word OK."}],
    max_tokens: 128,
    temperature: 0,
    stream: false
  }')"
  for ((i = 1; i <= attempts; i++)); do
    completion_attempts=$i
    completion_code="$(http POST /v1/chat/completions "$request")"
    completion_ms="$(cat "$workdir/ms")"
    if [[ "$completion_code" == "200" ]]; then
      # Accept content or reasoning_content: thinking models can spend the
      # whole budget reasoning. Either proves tokens came back from a worker.
      reply_chars="$(jq -r '
        (.choices[0].message.content // "") + (.choices[0].message.reasoning_content // "")
        | length' "$workdir/body" 2>/dev/null || echo 0)"
      api_error="$(jq -r '.error.code // .error.message // empty' "$workdir/body" 2>/dev/null || true)"
      if [[ -n "$api_error" ]]; then
        fail "POST /v1/chat/completions model=${model} -> HTTP 200 with error: ${api_error}"
      elif [[ "$reply_chars" -lt 1 ]]; then
        fail "POST /v1/chat/completions model=${model} -> HTTP 200 but empty reply"
      else
        state="ok"
        detail="POST /v1/chat/completions model=${model} -> 200 in ${completion_ms} ms (${reply_chars} chars, attempt ${i}/${attempts}); ${model_count} models advertised"
        break
      fi
    else
      api_error="$(jq -r '.error.code // .error.message // empty' "$workdir/body" 2>/dev/null || true)"
      [[ -z "$api_error" ]] && api_error="$(head -c 160 "$workdir/body" 2>/dev/null | tr '\n' ' ' || true)"
      [[ -z "$api_error" && -s "$workdir/curl.err" ]] && api_error="$(head -c 160 "$workdir/curl.err" | tr '\n' ' ')"
      fail "POST /v1/chat/completions model=${model} -> HTTP ${completion_code} (${api_error:-no body}) after ${completion_ms} ms, attempt ${i}/${attempts}"
    fi
    if [[ $i -lt $attempts ]]; then
      sleep 10
    fi
  done
fi

# Belt and braces: whatever path we took, no 200 completion means failed.
if [[ "$completion_code" != "200" ]]; then
  state="failed"
  [[ -z "$detail" ]] && detail="no successful completion (last HTTP ${completion_code:-none})"
fi

result="$(jq -cn \
  --arg state "$state" \
  --arg detail "$detail" \
  --arg base_url "$base_url" \
  --arg model "$model" \
  --arg health_code "$health_code" \
  --arg models_code "$models_code" \
  --argjson model_count "${model_count:-0}" \
  --arg completion_code "$completion_code" \
  --argjson completion_ms "${completion_ms:-0}" \
  --argjson completion_attempts "$completion_attempts" \
  --argjson reply_chars "${reply_chars:-0}" \
  --arg checked_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{state: $state, detail: $detail, base_url: $base_url, model: $model,
    health_code: $health_code, models_code: $models_code, model_count: $model_count,
    completion_code: $completion_code, completion_ms: $completion_ms,
    completion_attempts: $completion_attempts, reply_chars: $reply_chars,
    checked_at: $checked_at}')"

echo "$result"
if [[ -n "$out_path" ]]; then
  printf '%s\n' "$result" >"$out_path"
fi
[[ "$state" == "ok" ]]
