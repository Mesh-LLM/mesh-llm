#!/usr/bin/env bash
set -euo pipefail

# llama-upstream canary patch-queue repair (issue #1434).
#
# Invoked by llama-upstream-canary.yml when `scripts/prepare-llama.sh` fails
# to apply the third_party/llama.cpp/patches queue onto the new upstream pin.
# Drives a non-interactive `opencode` agent (model: Nemotron 3 Ultra Free by
# default) to: rebase the patch queue, rebuild, run the family battery, and
# open a PR — reusing the already-open canary repair PR if one exists.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

UPSTREAM_SHA="${1:?usage: llama-canary-agent-repair.sh <new-upstream-sha>}"
AGENT_MODEL="${CANARY_AGENT_MODEL:-Nemotron 3 Ultra Free}"
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

opencode run --model "$AGENT_MODEL" \
  "$(printf 'The canary failed to apply the llama.cpp patch queue at upstream %s.
Read ci/llama-canary/agent-repair-prompt.md in this repo and follow it exactly.
When done, ensure your work is committed on branch %s and a PR exists (reuse
open PR %s on that branch if listed, otherwise create one). Report the PR URL.' \
    "$UPSTREAM_SHA" "$BRANCH" "${EXISTING_PR:-none}")"

echo "agent repair turn finished; verifying queue applies..."
scripts/prepare-llama.sh "$UPSTREAM_SHA"
