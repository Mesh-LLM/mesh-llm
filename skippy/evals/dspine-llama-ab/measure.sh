#!/usr/bin/env bash
# Drives one benchmark arm against a running llama-server and prints one
# DSPINE_AB JSON summary line per arm.
#
# Usage: measure.sh <port> <arm-name> <prompts.jsonl> [limit]
#
# Sequentially sends each prompt to /completion (greedy, 256 tokens), records
# per-request decode tok/s and draft acceptance from the response timings, then
# prints the aggregate as a single machine-readable line on stdout.
set -euo pipefail

port="$1"; arm="$2"; prompts_path="$3"; limit="${4:-0}"
n_predict=256

complete() { # complete <prompt> <n_predict> -> response json on stdout
  jq -n --arg p "$1" --argjson n "$2" \
    '{prompt: $p, n_predict: $n, temperature: 0, cache_prompt: false}' \
  | curl -sf --max-time 1800 -H 'Content-Type: application/json' -d @- \
    "http://127.0.0.1:$port/completion"
}

complete "Hello" 8 >/dev/null # warmup

rows="[]"; tot_tokens=0; tot_draft=0; tot_accepted=0; tot_ms=0
i=0
n_total=$(wc -l < "$prompts_path")
[ "$limit" -gt 0 ] && [ "$limit" -lt "$n_total" ] && n_total=$limit
t0=$SECONDS
while IFS= read -r line; do
  i=$((i + 1))
  [ "$limit" -gt 0 ] && [ "$i" -gt "$limit" ] && break
  id=$(jq -r '.id' <<<"$line")
  category=$(jq -r '.category // "?"' <<<"$line")
  prompt=$(jq -r '.prompt' <<<"$line")
  resp=$(complete "$prompt" "$n_predict")
  n=$(jq '.timings.predicted_n' <<<"$resp")
  ms=$(jq '.timings.predicted_ms' <<<"$resp")
  draft=$(jq '.timings.draft_n // 0' <<<"$resp")
  acc=$(jq '.timings.draft_n_accepted // 0' <<<"$resp")
  tot_tokens=$((tot_tokens + n)); tot_draft=$((tot_draft + draft)); tot_accepted=$((tot_accepted + acc))
  tot_ms=$(jq -n --argjson a "$tot_ms" --argjson b "$ms" '$a + $b')
  rows=$(jq -n --argjson rows "$rows" --arg id "$id" --arg category "$category" \
    --argjson n "$n" --argjson ms "$ms" --argjson draft "$draft" --argjson acc "$acc" \
    '$rows + [{id: $id, category: $category, tokens: $n, ms: ($ms * 10 | round / 10),
               tok_s: (if $ms > 0 then (1000 * $n / $ms * 100 | round / 100) else null end),
               draft: $draft, accepted: $acc}]')
  toks=$(jq -n --argjson n "$n" --argjson ms "$ms" '1000 * $n / $ms')
  extra=""; [ "$draft" -gt 0 ] && extra=" draft=$draft accepted=$acc"
  printf '  [%d/%d] %s %.1f tok/s%s\n' "$i" "$n_total" "$id" "$toks" "$extra"
done < "$prompts_path"

jq -rn --arg arm "$arm" --argjson rows "$rows" --argjson tt "$tot_tokens" \
  --argjson tm "$tot_ms" --argjson td "$tot_draft" --argjson ta "$tot_accepted" \
  --argjson wall "$((SECONDS - t0))" \
  '"DSPINE_AB " + ({arm: $arm, prompts: ($rows | length), wall_s: $wall, tokens: $tt,
    tok_s: (if $tm > 0 then (1000 * $tt / $tm * 100 | round / 100) else null end),
    draft_n: $td, draft_n_accepted: $ta,
    acceptance_pct: (if $td > 0 then (100 * $ta / $td * 10 | round / 10) else null end),
    per_prompt: $rows} | @json)'
