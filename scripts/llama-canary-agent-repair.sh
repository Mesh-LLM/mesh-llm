#!/usr/bin/env bash
set -euo pipefail

# llama-upstream canary patch-queue repair (issue #1434).
#
# Invoked by llama-upstream-canary.yml when `scripts/prepare-llama.sh` fails
# to apply the third_party/llama.cpp/patches queue onto the new upstream pin.
# Drives a non-interactive `opencode` agent (model: Nemotron 3 Ultra Free by
# default) to: rebase the patch queue, rebuild, run the family battery, and
# open a PR — reusing the already-open canary repair PR if one exists.
#
# Repair loop: after the agent's first turn the wrapper itself runs the
# certification battery. If it fails, each failure gets its own opencode
# repair turn (with the battery failure summary in the prompt) followed by a
# recertify, up to CANARY_REPAIR_MAX_TURNS (default 2) repair turns. The
# script only succeeds when the battery actually passes on this runner.
#
# PR guarantee: whatever the outcome, the wrapper (not the agent) ensures a
# repair PR exists on $BRANCH and posts a status comment describing the work
# done and, on failure, what the agent is stuck on and needs human help with.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

UPSTREAM_SHA="${1:?usage: llama-canary-agent-repair.sh <new-upstream-sha>}"
AGENT_MODEL="${CANARY_AGENT_MODEL:-Nemotron 3 Ultra Free}"
MAX_REPAIR_TURNS="${CANARY_REPAIR_MAX_TURNS:-2}"
BRANCH="llama-canary/patch-queue-fix"

mkdir -p "$ROOT/.deps"
echo "$UPSTREAM_SHA" > "$ROOT/.deps/llama-canary-target-sha"

if ! command -v opencode >/dev/null 2>&1; then
  echo "opencode CLI not found on runner; install opencode-ai on the family-certify image" >&2
  exit 1
fi
# Credentials: either an explicit API key env var, or an opencode CLI that has
# been logged in on the runner (`opencode auth login`), which `opencode run`
# picks up from its own auth store.
if [[ -z "${OPENCODE_API_KEY:-}" && -z "${NEMOTRON_API_KEY:-}" ]]; then
  if [[ ! -s "${HOME}/.local/share/opencode/auth.json" ]] && ! opencode auth list 2>/dev/null | grep -Eq '[1-9][0-9]* credentials'; then
    echo "no agent credentials: set OPENCODE_API_KEY/NEMOTRON_API_KEY or run 'opencode auth login' on the runner" >&2
    exit 1
  fi
fi

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

run_battery() {
  # Runs the certification battery; prints the summary line and returns the
  # battery exit code. Log path is echoed for the caller.
  local log=".deps/llama-canary-repair-battery.log"
  scripts/build-llama.sh || return 1
  if scripts/skippy-family-battery.sh >"$log" 2>&1; then
    tail -n 2 "$log"
    return 0
  fi
  tail -n 2 "$log"
  return 1
}

agent_turn "$(printf 'The canary failed to apply the llama.cpp patch queue at upstream %s.
Read ci/llama-canary/agent-repair-prompt.md in this repo and follow it exactly.
When done, ensure your work is committed on branch %s and a PR exists (reuse
open PR %s on that branch if listed, otherwise create one). Report the PR URL.' \
    "$UPSTREAM_SHA" "$BRANCH" "${EXISTING_PR:-none}")"

echo "agent repair turn finished; verifying queue applies..."
if ! scripts/prepare-llama.sh "$UPSTREAM_SHA"; then
  pr_comment "$(printf '**Repair stuck — needs human assistance.** The patch queue still does not apply at upstream %s after the agent repair turn (see the canary run log for the failing patch). The agent work so far is on this branch.' \
    "$UPSTREAM_SHA")"
  exit 1
fi

# Certify → repair → recertify loop. The wrapper — not the agent — decides
# when certification passes, so a lane failure can never be talked past.
BATTERY_LOG=".deps/llama-canary-repair-battery.log"
for turn in $(seq 1 "$MAX_REPAIR_TURNS"); do
  echo "certification attempt $turn..."
  if run_battery; then
    echo "family battery passed; repair complete"
    pr_comment "$(printf '**Family battery passed** after the agent repair at upstream %s.\nAll certification lanes green on the family-certify runner.' \
      "$UPSTREAM_SHA")"
    exit 0
  fi
  echo "family battery failed on repair turn $turn; handing failures to the agent"
  agent_turn "$(printf 'The family certification battery failed after the patch-queue repair
at upstream %s (attempt %s of %s). You are still on branch %s.

Read ci/llama-canary/agent-repair-prompt.md and the repo skills it names, then
fix the root cause — do not weaken a failing lane. If a model is genuinely
broken by upstream, fix our patches or flag it in the PR body. The failing
battery output (tail):

%s

Re-run scripts/skippy-family-battery.sh --skip-build yourself to confirm your
fix, commit to %s, and push to the PR.' \
    "$UPSTREAM_SHA" "$turn" "$MAX_REPAIR_TURNS" "$BRANCH" "$(battery_summary "$BATTERY_LOG")" "$BRANCH")"
  echo "agent repair turn $turn finished; verifying queue applies..."
  if ! scripts/prepare-llama.sh "$UPSTREAM_SHA"; then
    pr_comment "$(printf '**Repair stuck — needs human assistance.** The patch queue regressed or still does not apply at upstream %s after repair turn %s/%s. The agent work is on this branch; see the canary run log for the failing patch.' \
      "$UPSTREAM_SHA" "$turn" "$MAX_REPAIR_TURNS")"
    exit 1
  fi
done

echo "final certification attempt..."
if run_battery; then
  echo "family battery passed; repair complete"
  pr_comment "$(printf '**Family battery passed** after the agent repair at upstream %s.\nAll certification lanes green on the family-certify runner.' \
    "$UPSTREAM_SHA")"
  exit 0
fi

pr_comment "$(printf '**Repair stuck — needs human assistance.** The family battery is still failing after %s agent repair turns at upstream %s. The agent work is on this branch; the failing battery output (tail):\n\n```\n%s\n```' \
  "$MAX_REPAIR_TURNS" "$UPSTREAM_SHA" "$(battery_summary "$BATTERY_LOG")")"
echo "family battery still failing after $MAX_REPAIR_TURNS agent repair turns" >&2
exit 1
