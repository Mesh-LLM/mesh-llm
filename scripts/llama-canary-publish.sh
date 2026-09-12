#!/usr/bin/env bash
set -euo pipefail

# Publish a locally committed llama.cpp canary candidate after the repair
# harness has independently certified its exact tree. This script is invoked
# only from the success-gated, token-bearing workflow step.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BRANCH="${CANARY_BRANCH:?CANARY_BRANCH is required}"
CERTIFIED_SHA="${CANARY_CERTIFIED_SHA:?CANARY_CERTIFIED_SHA is required}"
PR_BODY="${CANARY_PR_BODY:?CANARY_PR_BODY is required}"
BUNDLE="${CANARY_BUNDLE:?CANARY_BUNDLE is required}"
REPOSITORY="${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
TOKEN="${CANARY_REPAIR_TOKEN:?CANARY_REPAIR_TOKEN is required}"

if [[ ! "$BRANCH" =~ ^llama-canary/repair-[0-9A-Za-z._-]+-[0-9]+-[0-9a-f]{10}$ ]]; then
  echo "refusing to publish unexpected canary branch: $BRANCH" >&2
  exit 1
fi
if [[ ! "$CERTIFIED_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "refusing to publish a non-40-hex certified SHA" >&2
  exit 1
fi
if [[ ! "$REPOSITORY" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
  echo "refusing to publish to malformed repository identity" >&2
  exit 1
fi
if [[ ! -s "$PR_BODY" ]]; then
  echo "certified canary PR body is missing or empty: $PR_BODY" >&2
  exit 1
fi
if [[ ! -s "$BUNDLE" ]]; then
  echo "certified canary bundle is missing or empty: $BUNDLE" >&2
  exit 1
fi
git bundle verify "$BUNDLE" >/dev/null
bundle_head="$(git bundle list-heads "$BUNDLE" "refs/heads/${BRANCH}" | awk '{print $1}')"
if [[ "$bundle_head" != "$CERTIFIED_SHA" ]]; then
  echo "candidate bundle does not advertise the certified commit on $BRANCH" >&2
  exit 1
fi
git fetch "$BUNDLE" "refs/heads/${BRANCH}"
git checkout --detach FETCH_HEAD
if [[ "$(git rev-parse HEAD)" != "$CERTIFIED_SHA" ]]; then
  echo "bundle checkout HEAD does not match the certified canary commit" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "certified canary checkout is not clean" >&2
  exit 1
fi

ASKPASS="$(mktemp "${RUNNER_TEMP:-/tmp}/llama-canary-askpass.XXXXXX")"
cat > "$ASKPASS" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
  Username*) printf '%s\n' 'x-access-token' ;;
  Password*) printf '%s\n' "${CANARY_REPAIR_TOKEN:?}" ;;
  *) exit 1 ;;
esac
EOF
chmod 700 "$ASKPASS"


gh_repair() {
  GH_TOKEN="$TOKEN" "$@"
}

redact_token() {
  python3 -c 'import os, sys; token = os.environ["CANARY_REPAIR_TOKEN"]; sys.stdout.write(sys.stdin.read().replace(token, "***redacted***"))'
}

encoded_branch="${BRANCH//\//%2F}"
branch_pushed=0
published=0
# Called from cleanup_before_pr, which is itself invoked by traps.
# shellcheck disable=SC2329
cleanup_exact_remote_branch() {
  local remote_head
  remote_head="$(gh_repair gh api "repos/${REPOSITORY}/git/ref/heads/${encoded_branch}" --jq .object.sha 2>/dev/null || true)"
  if [[ "$remote_head" == "$CERTIFIED_SHA" ]]; then
    gh_repair gh api --method DELETE \
      "repos/${REPOSITORY}/git/refs/heads%2F${encoded_branch}" >/dev/null 2>&1 || \
      echo "warning: could not remove unpublished canary branch $BRANCH" >&2
  elif [[ -n "$remote_head" ]]; then
    echo "warning: refusing to remove $BRANCH because its remote head changed" >&2
  fi
}

find_exact_ready_pr() {
  gh_repair gh pr list --repo "$REPOSITORY" --head "$BRANCH" --state open \
    --json url,isDraft,headRefOid \
    --jq ".[] | select(.headRefOid == \"$CERTIFIED_SHA\" and .isDraft == false) | .url" \
    2>/dev/null | head -n 1
}

# Invoked indirectly by the signal/exit traps below.
# shellcheck disable=SC2329
cleanup_before_pr() {
  local status="$1" exact_pr
  trap - EXIT INT TERM
  if (( branch_pushed == 1 && published == 0 )); then
    exact_pr="$(find_exact_ready_pr || true)"
    if [[ -n "$exact_pr" ]]; then
      echo "certified canary PR exists despite an interrupted response: $exact_pr"
      rm -f "$ASKPASS"
      exit 0
    fi
    cleanup_exact_remote_branch
  fi
  rm -f "$ASKPASS"
  exit "$status"
}
trap 'cleanup_before_pr $?' EXIT
trap 'cleanup_before_pr 130' INT
trap 'cleanup_before_pr 143' TERM

if existing="$(gh_repair gh pr list --repo "$REPOSITORY" --head "$BRANCH" \
    --state all --json number --jq '.[0].number' 2>/dev/null)" && [[ -n "$existing" ]]; then
  echo "refusing to reuse existing PR #${existing} for run-unique branch $BRANCH" >&2
  exit 1
fi

if ! GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
    git push "https://github.com/${REPOSITORY}.git" \
    "HEAD:refs/heads/${BRANCH}" 2> >(redact_token >&2); then
  echo "could not push certified canary branch $BRANCH" >&2
  exit 1
fi
branch_pushed=1

remote_head="$(gh_repair gh api "repos/${REPOSITORY}/git/ref/heads/${encoded_branch}" --jq .object.sha)"
if [[ "$remote_head" != "$CERTIFIED_SHA" ]]; then
  echo "pushed canary branch does not match the certified commit" >&2
  exit 1
fi

title="fix(llama): certify upstream $(tr -d '[:space:]' < third_party/llama.cpp/upstream.txt | cut -c1-10)"
if ! pr_url="$(gh_repair gh pr create --repo "$REPOSITORY" --base main --head "$BRANCH" \
    --title "$title" --body-file "$PR_BODY" 2> >(redact_token >&2))"; then
  pr_url="$(find_exact_ready_pr || true)"
  if [[ -z "$pr_url" ]]; then
    echo "could not create or reconcile the certified canary PR" >&2
    exit 1
  fi
  echo "reconciled certified canary PR after an ambiguous create response: $pr_url"
fi

# PR creation is the final external mutation. From this point the job reports
# success even if optional summary output cannot be written, so it never claims
# publication failed after a ready PR exists.
published=1
trap - EXIT INT TERM
rm -f "$ASKPASS"
set +e
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  printf 'pr_url=%s\n' "$pr_url" >> "$GITHUB_OUTPUT"
fi
echo "published certified canary PR: ${pr_url}; head=${CERTIFIED_SHA}"
exit 0
