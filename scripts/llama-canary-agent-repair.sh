#!/usr/bin/env bash
set -euo pipefail

# llama-upstream canary agent repair (issue #1434; wired into
# llama-upstream-canary.yml for both patch-queue apply failures and family
# battery failures).
#
# Usage: llama-canary-agent-repair.sh <mode> [upstream-sha]
#   mode: patch-queue  - prepare-llama.sh failed to apply the patch queue
#                        onto the new upstream; the agent rebases the queue.
#   mode: battery      - the patch queue applied but the family battery
#                        failed; the agent fixes the root cause.
#   upstream-sha: 40-hex llama.cpp commit. When omitted, resolves master
#                  via git ls-remote. "latest" is also accepted.
#
# Drives a non-interactive `opencode` agent (model:
# zai-coding-plan/glm-5.3-flash by default) to repair, then the wrapper
# itself re-runs the certification battery. If it fails, each failure gets
# its own opencode repair turn (with the battery failure summary in the
# prompt) followed by a recertify, up to CANARY_REPAIR_MAX_TURNS (default 2)
# repair turns. The script only succeeds when the battery actually passes
# on this runner.
#
# PR guarantee: whatever the outcome, the wrapper (not the agent) ensures a
# repair PR exists on $BRANCH and posts a status comment describing the work
# done and, on failure, what the agent is stuck on and needs human help
# with. The PR description is written by an agent turn (upstream changes,
# patch-queue evolution, risks) with a deterministic fallback body.
#
# Credentials: pushes/PRs use $CANARY_REPAIR_TOKEN (fine-grained PAT with
# Contents+PR write; the canary job itself stays contents: read). The agent
# needs OPENCODE_API_KEY/NEMOTRON_API_KEY or an `opencode auth login`
# profile on the runner.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODE="${1:?usage: llama-canary-agent-repair.sh <patch-queue|battery> [upstream-sha]}"
case "$MODE" in
  patch-queue | battery) ;;
  *)
    echo "unknown repair mode: $MODE (expected patch-queue or battery)" >&2
    exit 1
    ;;
esac

UPSTREAM_SHA="${2:-latest}"
if [[ "$UPSTREAM_SHA" == "latest" || -z "$UPSTREAM_SHA" ]]; then
  UPSTREAM_SHA="$(git ls-remote https://github.com/ggml-org/llama.cpp.git master | awk '{print $1}')"
fi
if [[ ! "$UPSTREAM_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "refusing to repair against a non-40-hex upstream SHA: $UPSTREAM_SHA" >&2
  exit 1
fi

OLD_SHA="$(tr -d '[:space:]' < "$ROOT/third_party/llama.cpp/upstream.txt")"
AGENT_MODEL="${CANARY_AGENT_MODEL:-zai-coding-plan/glm-5.3-flash}"
MAX_REPAIR_TURNS="${CANARY_REPAIR_MAX_TURNS:-2}"
BRANCH="llama-canary/patch-queue-fix"
BATTERY_LOG="$ROOT/.deps/llama-canary-repair-battery.log"

mkdir -p "$ROOT/.deps"
echo "$UPSTREAM_SHA" > "$ROOT/.deps/llama-canary-target-sha"

if ! command -v opencode >/dev/null 2>&1; then
  echo "opencode CLI not found on runner; install opencode-ai on the family-certify image" >&2
  exit 1
fi
# Agent credentials: either an explicit API key env var, or an opencode CLI
# that has been logged in on the runner (`opencode auth login`), which
# `opencode run` picks up from its own auth store.
if [[ -z "${OPENCODE_API_KEY:-}" && -z "${NEMOTRON_API_KEY:-}" ]]; then
  if [[ ! -s "${HOME}/.local/share/opencode/auth.json" ]] && ! opencode auth list 2>/dev/null | grep -Eq '[1-9][0-9]* credentials'; then
    echo "no agent credentials: set OPENCODE_API_KEY/NEMOTRON_API_KEY or run 'opencode auth login' on the runner" >&2
    exit 1
  fi
fi
# The canary job itself is read-only; the repair branch push and PR need the
# dedicated fine-grained token.
if [[ -z "${CANARY_REPAIR_TOKEN:-}" ]]; then
  echo "CANARY_REPAIR_TOKEN is not set; cannot push the repair branch or open the repair PR" >&2
  exit 1
fi
export GH_TOKEN="$CANARY_REPAIR_TOKEN"

cd "$ROOT"

# The agent reuses the open repair PR on $BRANCH if one exists, so repeated
# canary failures amend a single PR instead of stacking duplicates. Surface
# the current PR number (if any) in the prompt so it does not have to guess.
EXISTING_PR=""
if command -v gh >/dev/null 2>&1; then
  EXISTING_PR="$(gh pr list --head "$BRANCH" --state open --json number --jq '.[0].number' || true)"
fi

agent_turn() {
  # Non-fatal: a crashed agent turn must not skip PR reporting.
  local prompt="$1"
  opencode run --model "$AGENT_MODEL" "$prompt" \
    || echo "warning: opencode turn exited non-zero" >&2
}

battery_summary() {
  # Last 80 lines of the most recent battery log, enough to name the failing
  # family/split lanes without flooding the agent prompt.
  local log="$1"
  tail -n 80 "$log"
}

current_pr() {
  gh pr list --head "$BRANCH" --state open --json number --jq '.[0].number' 2>/dev/null || true
}

ensure_pr() {
  # The agent is asked to open the PR, but the wrapper guarantees it: if no
  # open PR exists on $BRANCH, push the branch and create one. If the branch
  # has no diff against the base (agent produced nothing), fall back to an
  # issue so the outcome is still visible to humans.
  local pr title body
  pr="$(current_pr)"
  if [[ -n "$pr" ]]; then
    printf '%s\n' "$pr"
    return 0
  fi
  title="fix(llama): rebase patch queue onto upstream ${UPSTREAM_SHA:0:10}"
  body="Automated canary repair PR for the llama.cpp patch queue at upstream ${UPSTREAM_SHA}."
  if git push -u origin "$BRANCH" 2>/dev/null \
     && pr="$(gh pr create --head "$BRANCH" --title "$title" --body "$body" 2>/dev/null \
              | grep -oE '[0-9]+$')"; then
    printf '%s\n' "$pr"
    return 0
  fi
  gh issue create --title "llama canary repair needs human assistance (upstream ${UPSTREAM_SHA:0:10})" \
    --body "The canary repair loop could not open a PR on \`$BRANCH\` (branch missing or no diff). See the canary run log for the repair-loop outcome." \
    | grep -oE '[0-9]+$' || true
  return 0
}

pr_comment() {
  # Post a status comment on the repair PR; never fails the loop.
  local body="$1" resource
  resource="$(current_pr)"
  [[ -n "$resource" ]] || resource="$(ensure_pr)"
  [[ -n "$resource" ]] || return 0
  # ensure_pr returns an issue number when no PR exists; use the right command.
  if gh pr view "$resource" >/dev/null 2>&1; then
    gh pr comment "$resource" --body "$body" >/dev/null 2>&1 || true
  else
    gh issue comment "$resource" --body "$body" >/dev/null 2>&1 || true
  fi
}

write_pr_body() {
  # One agent turn writes the repair PR description: key upstream changes
  # between the old pin and the repair target, how the patch queue evolved,
  # risks, and validation. Falls back to a deterministic body so a PR always
  # carries a meaningful description.
  local pr body_file
  pr="$(current_pr)"
  [[ -n "$pr" ]] || return 0
  body_file="$ROOT/.deps/llama-canary-pr-body.md"
  agent_turn "$(printf 'Write the description for repair PR #%s.\nAnalyze the llama.cpp changes between %s (old pinned upstream) and %s\n(repair target), summarize the key upstream changes, explain how the patch\nqueue in third_party/llama.cpp/patches/ evolved in this repair (per patch:\nwhat conflicted and how it was resolved), and identify risks for reviewers\n(including any ABI impact and any lane that is newly failing or excluded).\nWrite the finished Markdown description to %s using your file tools. Do not\nedit any other file.' \
    "$pr" "${OLD_SHA:0:10}" "${UPSTREAM_SHA:0:10}" "$body_file")"
  if [[ ! -s "$body_file" ]]; then
    {
      echo "Automated canary repair at upstream ${UPSTREAM_SHA}."
      echo
      echo "- Old pinned upstream: ${OLD_SHA}"
      echo "- Repair target upstream: ${UPSTREAM_SHA}"
      echo "- Mode: ${MODE}"
      echo
      echo "The agent-written upstream/queue analysis was unavailable; reviewers"
      echo "should diff the patch queue against main directly."
    } > "$body_file"
  fi
  gh pr edit "$pr" --body-file "$body_file" >/dev/null 2>&1 || true
}

run_battery() {
  # Runs the certification battery; prints the summary line and returns the
  # battery exit code. Log path is echoed for the caller.
  scripts/build-llama.sh || return 1
  if scripts/skippy-family-battery.sh >"$BATTERY_LOG" 2>&1; then
    tail -n 2 "$BATTERY_LOG"
    return 0
  fi
  tail -n 2 "$BATTERY_LOG"
  return 1
}

repair_followup_prompt() {
  # Shared prompt for every "battery failed again" agent turn after the first.
  printf 'The family certification battery failed after the patch-queue repair
at upstream %s (attempt %s of %s). You are still on branch %s.

Read ci/llama-canary/agent-repair-prompt.md and the repo skills it names, then
fix the root cause — do not weaken a failing lane. If a model is genuinely
broken by upstream, fix our patches or flag it in the PR body. The failing
battery output (tail):

%s

Re-run scripts/skippy-family-battery.sh --skip-build yourself to confirm your
fix, commit to %s, and push to the PR.' \
    "$UPSTREAM_SHA" "$1" "$MAX_REPAIR_TURNS" "$BRANCH" "$(battery_summary "$BATTERY_LOG")" "$BRANCH"
}

if [[ "$MODE" == "patch-queue" ]]; then
  agent_turn "$(printf 'The canary failed to apply the llama.cpp patch queue at upstream %s.
Read ci/llama-canary/agent-repair-prompt.md in this repo and follow it exactly.
When done, ensure your work is committed on branch %s and a PR exists (reuse
open PR %s on that branch if listed, otherwise create one). Report the PR URL.' \
    "$UPSTREAM_SHA" "$BRANCH" "${EXISTING_PR:-none}")"

  echo "agent repair turn finished; verifying queue applies..."
  if ! scripts/prepare-llama.sh "$UPSTREAM_SHA"; then
    write_pr_body
    pr_comment "$(printf '**Repair stuck — needs human assistance.** The patch queue still does not apply at upstream %s after the agent repair turn (see the canary run log for the failing patch). The agent work so far is on this branch.' \
      "$UPSTREAM_SHA")"
    exit 1
  fi
else
  # battery mode: the queue already applies on this runner; start from a
  # certification attempt so the first agent turn carries real failure output.
  echo "battery mode: running initial certification attempt..."
  run_battery || true
fi

# Certify → repair → recertify loop. The wrapper — not the agent — decides
# when certification passes, so a lane failure can never be talked past.
for turn in $(seq 1 "$MAX_REPAIR_TURNS"); do
  echo "certification attempt $turn..."
  if run_battery; then
    echo "family battery passed; repair complete"
    write_pr_body
    pr_comment "$(printf '**Family battery passed** after the agent repair at upstream %s.\nAll certification lanes green on the family-certify runner.' \
      "$UPSTREAM_SHA")"
    exit 0
  fi
  echo "family battery failed on repair turn $turn; handing failures to the agent"
  agent_turn "$(repair_followup_prompt "$turn")"
  echo "agent repair turn $turn finished; verifying queue applies..."
  if ! scripts/prepare-llama.sh "$UPSTREAM_SHA"; then
    write_pr_body
    pr_comment "$(printf '**Repair stuck — needs human assistance.** The patch queue regressed or still does not apply at upstream %s after repair turn %s/%s. The agent work is on this branch; see the canary run log for the failing patch.' \
      "$UPSTREAM_SHA" "$turn" "$MAX_REPAIR_TURNS")"
    exit 1
  fi
done

echo "final certification attempt..."
if run_battery; then
  echo "family battery passed; repair complete"
  write_pr_body
  pr_comment "$(printf '**Family battery passed** after the agent repair at upstream %s.\nAll certification lanes green on the family-certify runner.' \
    "$UPSTREAM_SHA")"
  exit 0
fi

write_pr_body
# The final status comment embeds a fenced battery tail; the literal
# backticks are intentional Markdown, not command substitution.
# shellcheck disable=SC2016
pr_comment "$(printf '**Repair stuck — needs human assistance.** The family battery is still failing after %s agent repair turns at upstream %s. The agent work is on this branch; the failing battery output (tail):\n\n```\n%s\n```' \
  "$MAX_REPAIR_TURNS" "$UPSTREAM_SHA" "$(battery_summary "$BATTERY_LOG")")"
echo "family battery still failing after $MAX_REPAIR_TURNS agent repair turns" >&2
exit 1
