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
#   upstream-sha: 40-hex llama.cpp commit, preferentially passed via the
#                  UPSTREAM_SHA_INPUT environment variable (callers must
#                  never interpolate untrusted dispatch values into shell).
#                  When omitted, resolves master via git ls-remote. "latest"
#                  is also accepted. Battery-mode evidence: when the workflow
#                  tees its battery log to $BATTERY_LOG, it is reused instead
#                  of re-running the battery before the first repair turn.
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
# with. The PR description is written by an agent turn that runs BEFORE
# certification (upstream changes, patch-queue evolution, risks) with a
# deterministic fallback body; no agent turn ever runs after a green battery.
#
# Credential split: the agent never sees a GitHub token — CANARY_REPAIR_TOKEN
# is stripped from its environment, and only the deterministic wrapper
# performs git pushes, PR creation, PR edits, and comments with the token
# scoped to individual commands. The wrapper — never the agent — commits the
# certified tree, pushes it, and verifies the repair PR head equals the
# certified commit before reporting success.
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

UPSTREAM_SHA="${2:-${UPSTREAM_SHA_INPUT:-latest}}"
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
# dedicated fine-grained token. The token is never exported into the process
# environment — every GitHub mutation is routed through gh_repair(), which
# scopes it to that single command — so agent turns (and anything they spawn)
# never inherit repository-write credentials.
if [[ -z "${CANARY_REPAIR_TOKEN:-}" ]]; then
  echo "CANARY_REPAIR_TOKEN is not set; cannot push the repair branch or open the repair PR" >&2
  exit 1
fi

gh_repair() {
  # Deterministic GitHub mutations only. The write PAT is scoped to this one
  # command; it is deliberately absent from the ambient environment.
  GH_TOKEN="$CANARY_REPAIR_TOKEN" "$@"
}

cd "$ROOT"

# Run-scope the persistent-runner artifacts before anything else can fail and
# leave a previous run's state behind. The PR-body file is always cleared. The
# battery evidence log is cleared only in patch-queue mode: in battery mode it
# holds THIS run's workflow battery output (teed by the workflow immediately
# before invoking this script) and must survive to seed the first repair turn.
if [[ "$MODE" == "patch-queue" ]]; then
  rm -f "$BATTERY_LOG"
fi
rm -f "$ROOT/.deps/llama-canary-pr-body.md"

# The agent reuses the open repair PR on $BRANCH if one exists, so repeated
# canary failures amend a single PR instead of stacking duplicates. Surface
# the current PR number (if any) in the prompt so it does not have to guess.
# Every GitHub call — read or write — carries the repair token: the workflow
# job never exports GH_TOKEN/GITHUB_TOKEN and checks out with
# persist-credentials disabled, so an unauthenticated gh would silently fail.
EXISTING_PR=""
if command -v gh >/dev/null 2>&1; then
  EXISTING_PR="$(gh_repair gh pr list --head "$BRANCH" --state open --json number --jq '.[0].number' || true)"
fi

agent_turn() {
  # Non-fatal: a crashed agent turn must not skip PR reporting. The model
  # runs without any GitHub token: push/PR/comment mutations are wrapper-only.
  local prompt="$1"
  env -u GH_TOKEN -u GITHUB_TOKEN -u CANARY_REPAIR_TOKEN \
    opencode run --model "$AGENT_MODEL" "$prompt" \
    || echo "warning: opencode turn exited non-zero" >&2
}

battery_summary() {
  # Last 80 lines of the most recent battery evidence — either this run's
  # workflow-teed log (battery mode) or the wrapper's own certification run —
  # enough to name the failing family/split lanes without flooding the agent
  # prompt.
  local log="$1"
  if [[ ! -s "$log" ]]; then
    echo "(no battery output captured; see the canary run log)"
    return 0
  fi
  tail -n 80 "$log"
}

current_pr() {
  gh_repair gh pr list --head "$BRANCH" --state open --json number --jq '.[0].number' 2>/dev/null || true
}

# Repair remote: the write PAT is embedded in the push URL (never echoed) and
# exists only for the lifetime of the single git push command.
repair_remote() {
  echo "https://x-access-token:${CANARY_REPAIR_TOKEN}@github.com/${GITHUB_REPOSITORY:?GITHUB_REPOSITORY not set}.git"
}

redact_token() {
  # Strip the repair token from anything that might reach the log.
  sed "s/${CANARY_REPAIR_TOKEN}/***redacted***/g"
}

publish_repair_branch() {
  # The wrapper — never the agent — owns commit and push. Puts the current
  # working tree (agent's repaired queue) on $BRANCH and records the exact
  # certified SHA. Force-push is intentional: the agent rebases the patch
  # queue, so non-fast-forward updates are the normal case on this
  # wrapper-owned branch.
  git checkout -B "$BRANCH"
  if [[ -n "$(git status --porcelain)" ]]; then
    git add -A
    git commit -m "fix(llama): canary repair at upstream ${UPSTREAM_SHA:0:10}" \
      -m "Automated llama.cpp canary repair (mode: ${MODE}). Certified by the family battery on the family-certify runner."
  fi
  CERTIFIED_SHA="$(git rev-parse HEAD)"
  git push "$(repair_remote)" "+HEAD:refs/heads/${BRANCH}" 2> >(redact_token >&2)
}

ensure_pr() {
  # Wrapper-owned PR guarantee: if no open PR exists on $BRANCH (the branch
  # was just pushed by publish_repair_branch), create one. If the branch has
  # no diff against the base (agent produced nothing), fall back to an issue
  # so the outcome is still visible to humans.
  local pr title body
  pr="$(current_pr)"
  if [[ -n "$pr" ]]; then
    printf '%s\n' "$pr"
    return 0
  fi
  title="fix(llama): rebase patch queue onto upstream ${UPSTREAM_SHA:0:10}"
  body="Automated canary repair PR for the llama.cpp patch queue at upstream ${UPSTREAM_SHA}."
  if ! git diff --quiet origin/main..."$BRANCH" 2>/dev/null; then
    if pr="$(gh_repair gh pr create --head "$BRANCH" --title "$title" --body "$body" 2>/dev/null \
             | grep -oE '[0-9]+$')"; then
      printf '%s\n' "$pr"
      return 0
    fi
  fi
  gh_repair gh issue create --title "llama canary repair needs human assistance (upstream ${UPSTREAM_SHA:0:10})" \
    --body "The canary repair loop could not open a PR on \`$BRANCH\` (branch missing or no diff). See the canary run log for the repair-loop outcome." \
    | grep -oE '[0-9]+$' || true
  return 0
}

verify_pr_head_is_certified() {
  # The green battery must be bound to the bytes on the repair PR: the remote
  # PR head must equal the commit the wrapper pushed after certification.
  local pr remote_head attempt
  pr="$(current_pr)"
  if [[ -z "$pr" ]]; then
    echo "no repair PR to verify" >&2
    return 1
  fi
  remote_head="$(gh_repair gh pr view "$pr" --json headRefOid --jq .headRefOid 2>/dev/null || true)"
  for attempt in 1 2 3; do
    if [[ "$remote_head" == "${CERTIFIED_SHA:?}" ]]; then
      return 0
    fi
    sleep "$attempt"
    remote_head="$(gh_repair gh pr view "$pr" --json headRefOid --jq .headRefOid 2>/dev/null || true)"
  done
  echo "repair PR #${pr} head (${remote_head:-none}) does not match the certified commit ${CERTIFIED_SHA}" >&2
  return 1
}

pr_comment() {
  # Post a status comment on the repair PR; never fails the loop.
  local body="$1" resource
  resource="$(current_pr)"
  [[ -n "$resource" ]] || resource="$(ensure_pr)"
  [[ -n "$resource" ]] || return 0
  # ensure_pr returns an issue number when no PR exists; use the right command.
  if gh_repair gh pr view "$resource" >/dev/null 2>&1; then
    gh_repair gh pr comment "$resource" --body "$body" >/dev/null 2>&1 || true
  else
    gh_repair gh issue comment "$resource" --body "$body" >/dev/null 2>&1 || true
  fi
}

report_success() {
  # Green-battery closeout: no agent turn runs after this point. The wrapper
  # publishes the certified tree, writes the status comment, and verifies the
  # PR head binding before declaring success.
  publish_repair_branch
  apply_pr_body
  # The literal backticks around the certified SHA are Markdown, not command
  # substitution.
  # shellcheck disable=SC2016
  pr_comment "$(printf '**Family battery passed** after the agent repair at upstream %s.\nAll certification lanes green on the family-certify runner; certified commit: `%s`.' \
    "$UPSTREAM_SHA" "${CERTIFIED_SHA:?}")"
  verify_pr_head_is_certified
}

draft_pr_body() {
  # One agent turn drafts the repair PR description BEFORE certification: key
  # upstream changes between the old pin and the repair target, how the patch
  # queue evolved, risks, and validation. Runs strictly before any battery
  # attempt it describes; a failed or empty turn falls back to the
  # deterministic body in apply_pr_body.
  local pr body_file
  pr="$(current_pr)"
  [[ -n "$pr" ]] || return 0
  body_file="$ROOT/.deps/llama-canary-pr-body.md"
  agent_turn "$(printf 'Write the description for repair PR #%s.\nAnalyze the llama.cpp changes between %s (old pinned upstream) and %s\n(repair target), summarize the key upstream changes, explain how the patch\nqueue in third_party/llama.cpp/patches/ evolved in this repair (per patch:\nwhat conflicted and how it was resolved), and identify risks for reviewers\n(including any ABI impact and any lane that is newly failing or excluded).\nWrite the finished Markdown description to %s using your file tools. Do not\nedit any other file. Note: you have no GitHub credentials; the wrapper owns\nall pushes and PR updates.' \
    "$pr" "${OLD_SHA:0:10}" "${UPSTREAM_SHA:0:10}" "$body_file")"
}

apply_pr_body() {
  # Publish the PR description from the pre-certification agent draft (or the
  # deterministic fallback). No agent involvement: this may run after a green
  # battery, so it must be token-only and deterministic.
  local pr body_file
  pr="$(current_pr)"
  [[ -n "$pr" ]] || return 0
  body_file="$ROOT/.deps/llama-canary-pr-body.md"
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
  gh_repair gh pr edit "$pr" --body-file "$body_file" >/dev/null 2>&1 || true
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
  # Shared prompt for every repair turn. In battery mode turn 1 this is
  # seeded directly from the workflow's teed failure evidence (no battery
  # re-run first); later turns carry the wrapper's own certification output.
  # The agent has no GitHub credentials; the wrapper commits, pushes, and
  # updates the PR.
  printf 'The family certification battery failed after the patch-queue repair
at upstream %s (attempt %s of %s). You are working in this repository checkout.

Read ci/llama-canary/agent-repair-prompt.md and the repo skills it names, then
fix the root cause — do not weaken a failing lane. If a model is genuinely
broken by upstream, fix our patches or flag it in the PR body. The failing
battery output (tail):

%s

Re-run scripts/skippy-family-battery.sh --skip-build yourself to confirm your
fix, and leave your work in the working tree or on local commits — the wrapper
will commit, push, and update the repair PR.' \
    "$UPSTREAM_SHA" "$1" "$MAX_REPAIR_TURNS" "$(battery_summary "$BATTERY_LOG")"
}

publish_work_in_progress() {
  # Put the agent's current work on the repair PR early, so even a stuck run
  # leaves reviewable bytes behind. Best-effort: failures here do not stop
  # the repair loop.
  publish_repair_branch || echo "warning: could not publish repair branch" >&2
  ensure_pr >/dev/null
  apply_pr_body
}

if [[ "$MODE" == "patch-queue" ]]; then
  agent_turn "$(printf 'The canary failed to apply the llama.cpp patch queue at upstream %s.
Read ci/llama-canary/agent-repair-prompt.md in this repo and follow it exactly.
Commit your work locally when done. You have no GitHub credentials — the
wrapper that invoked you owns all pushes and PR updates. Reuse open PR %s on
branch %s if listed.' \
    "$UPSTREAM_SHA" "${EXISTING_PR:-none}" "$BRANCH")"

  echo "agent repair turn finished; verifying queue applies..."
  if ! scripts/prepare-llama.sh "$UPSTREAM_SHA"; then
    publish_work_in_progress
    pr_comment "$(printf '**Repair stuck — needs human assistance.** The patch queue still does not apply at upstream %s after the agent repair turn (see the canary run log for the failing patch). The agent work so far is on this branch.' \
      "$UPSTREAM_SHA")"
    exit 1
  fi
else
  # battery mode: the queue already applies and the workflow's own battery
  # step just failed on this runner. Its evidence log (teed to
  # $BATTERY_LOG by the workflow) seeds the first repair turn, so no build
  # or battery run is repeated before the agent gets the failure output.
  if [[ ! -s "$BATTERY_LOG" ]]; then
    echo "battery mode: no workflow battery evidence at $BATTERY_LOG; running one diagnostic battery attempt..." >&2
    run_battery || true
  else
    echo "battery mode: reusing workflow battery evidence from $BATTERY_LOG"
  fi
fi

# Publish the agent's repair work and its PR before certification, so the
# PR-description agent turn (draft_pr_body) also runs strictly before any
# certification attempt — no agent turn ever runs after a green battery.
publish_work_in_progress
draft_pr_body
apply_pr_body

# Certify → repair → recertify loop. The wrapper — not the agent — decides
# when certification passes, so a lane failure can never be talked past.
# In battery mode the workflow's own battery step already failed on this
# runner (or the diagnostic attempt above did): iteration 1 is the repair
# turn seeded from that evidence, never another full build+battery run
# before the agent gets a chance to fix anything.
attempt=0
while (( attempt < MAX_REPAIR_TURNS )); do
  attempt=$((attempt + 1))
  if [[ "$MODE" == "battery" && "$attempt" -eq 1 ]]; then
    echo "battery mode: repair turn 1 seeded from the workflow battery failure evidence"
  else
    echo "certification attempt $attempt..."
    if run_battery; then
      echo "family battery passed; repair complete"
      report_success
      exit 0
    fi
  fi
  echo "family battery failed on repair turn $attempt; handing failures to the agent"
  agent_turn "$(repair_followup_prompt "$attempt")"
  echo "agent repair turn $attempt finished; verifying queue applies..."
  if ! scripts/prepare-llama.sh "$UPSTREAM_SHA"; then
    publish_work_in_progress
    pr_comment "$(printf '**Repair stuck — needs human assistance.** The patch queue regressed or still does not apply at upstream %s after repair turn %s/%s. The agent work is on this branch; see the canary run log for the failing patch.' \
      "$UPSTREAM_SHA" "$attempt" "$MAX_REPAIR_TURNS")"
    exit 1
  fi
done

echo "final certification attempt..."
if run_battery; then
  echo "family battery passed; repair complete"
  report_success
  exit 0
fi

publish_work_in_progress
# The final status comment embeds a fenced battery tail; the literal
# backticks are intentional Markdown, not command substitution.
# shellcheck disable=SC2016
pr_comment "$(printf '**Repair stuck — needs human assistance.** The family battery is still failing after %s agent repair turns at upstream %s. The agent work is on this branch; the failing battery output (tail):\n\n```\n%s\n```' \
  "$MAX_REPAIR_TURNS" "$UPSTREAM_SHA" "$(battery_summary "$BATTERY_LOG")")"
echo "family battery still failing after $MAX_REPAIR_TURNS agent repair turns" >&2
exit 1
