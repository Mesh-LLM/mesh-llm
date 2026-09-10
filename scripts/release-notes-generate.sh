#!/usr/bin/env bash
# Regroup a published release's notes into Keep a Changelog sections.
#
# The deterministic pass classifies every entry it can from the canonical
# Conventional Commits subjects and always produces a publishable body. An
# optional agent pass then reviews that plan and may reclassify entries the
# commit metadata could not place. The agent is best-effort by construction:
# if it is missing, unauthenticated, out of quota, slow, or wrong, the
# deterministic notes are published and this script still succeeds.
#
# Required: RELEASE_TAG, RELEASE_NOTES_BASE. Optional: GITHUB_REPOSITORY,
# AGENT_MODEL, AGENT_REVIEW_BUDGET_SECONDS, AGENT_PROBE_SECONDS, DRY_RUN.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
REPO="${GITHUB_REPOSITORY:-Mesh-LLM/mesh-llm}"
TAG="${RELEASE_TAG:?RELEASE_TAG is required}"
BASE="${RELEASE_NOTES_BASE:?RELEASE_NOTES_BASE is required}"
AGENT_MODEL="${AGENT_MODEL:-}"
AGENT_REVIEW_BUDGET_SECONDS="${AGENT_REVIEW_BUDGET_SECONDS:-900}"
AGENT_PROBE_SECONDS="${AGENT_PROBE_SECONDS:-60}"
DRY_RUN="${DRY_RUN:-false}"

WORKDIR="${RELEASE_NOTES_WORKDIR:-$(mktemp -d)}"
mkdir -p "$WORKDIR"
echo "release-notes: tag=$TAG base=$BASE workdir=$WORKDIR"

version="${TAG#v}"
date_utc="$(date -u +%Y-%m-%d)"

# ── Deterministic pass ────────────────────────────────────────────────
# A failure here is a defect in our own tooling, so it fails the job loudly
# and publishes nothing.

gh release view "$TAG" --repo "$REPO" --json body -q .body > "$WORKDIR/body.md"
cp "$WORKDIR/body.md" "$WORKDIR/body.backup.md"

if ! grep -q '^\* .*\/pull\/[0-9]\+' "$WORKDIR/body.md"; then
  echo "release-notes: no PR entries in the published body; nothing to regroup"
  exit 0
fi

python3 "$ROOT/scripts/release-notes-classify.py" \
  --body "$WORKDIR/body.md" \
  --range "$BASE..$TAG" \
  --version "$version" \
  --date "$date_utc" \
  --repo-root "$ROOT" \
  --out "$WORKDIR/plan.deterministic.json"

python3 "$ROOT/scripts/release-notes-regroup.py" \
  --body "$WORKDIR/body.md" \
  --plan "$WORKDIR/plan.deterministic.json" \
  --out "$WORKDIR/notes.deterministic.md"

# Gate: the rendered body must carry exactly the PRs GitHub published. The
# renderer strips the conventional type prefix from a subject, so the entry
# lines are not byte-identical; the set of referenced pull requests is the
# invariant, and author credit is copied through untouched.
entry_prs() {
  grep -o 'pull/[0-9]\+' "$1" | sort
}

verify_entries() {
  local candidate="$1"
  diff <(entry_prs "$WORKDIR/body.backup.md") \
       <(entry_prs "$candidate") > "$WORKDIR/entry-diff.txt"
}

if ! verify_entries "$WORKDIR/notes.deterministic.md"; then
  echo "release-notes: deterministic render changed the entry set; refusing to publish" >&2
  cat "$WORKDIR/entry-diff.txt" >&2
  exit 1
fi

chosen="$WORKDIR/notes.deterministic.md"
source_label="deterministic"

# ── Optional agent pass ───────────────────────────────────────────────
# Every failure below is non-fatal: log the reason and keep `chosen`.

agent_reachable() {
  if [[ -z "$AGENT_MODEL" ]]; then
    echo "release-notes: AGENT_MODEL unset; skipping agent review"
    return 1
  fi
  if ! command -v opencode >/dev/null 2>&1; then
    echo "release-notes: opencode not installed on this runner; skipping agent review"
    return 1
  fi
  if [[ -z "${OPENCODE_API_KEY:-}" ]] && ! opencode auth list >/dev/null 2>&1; then
    echo "release-notes: no agent credentials; skipping agent review"
    return 1
  fi
  # Cheap liveness probe: quota exhaustion and provider outages surface here
  # rather than halfway through a long review turn.
  if ! timeout "$AGENT_PROBE_SECONDS" env -u GH_TOKEN -u GITHUB_TOKEN \
      opencode run --auto --model "$AGENT_MODEL" \
      'Reply with the single word: ready' > "$WORKDIR/probe.log" 2>&1; then
    echo "release-notes: agent probe failed (unreachable, out of quota, or timed out); skipping agent review"
    tail -5 "$WORKDIR/probe.log" || true
    return 1
  fi
  echo "release-notes: agent reachable; running review pass"
  return 0
}

agent_review() {
  python3 "$ROOT/scripts/release-notes-regroup.py" \
    --body "$WORKDIR/body.md" --list > "$WORKDIR/entries.tsv" 2>/dev/null

  local prompt
  prompt="$(cat <<PROMPT
Read $ROOT/.agents/skills/release-notes/SKILL.md and follow it.

A deterministic pass has already classified this release from Conventional
Commits metadata. Your job is to review and improve that plan, not to redo it.

- Current plan: $WORKDIR/plan.deterministic.json
- All entries, one "<pr>\t<subject>" per line: $WORKDIR/entries.tsv

Focus on what commit metadata cannot decide:
- entries in the "Other changes" section, which had no conventional subject;
- entries whose type is wrong for the change, above all security-relevant
  fixes that were committed as a plain fix and belong in Security;
- section sub-headings that do not read well for this release.

Write the improved plan to $WORKDIR/plan.agent.json in the same schema. Every
PR number in the current plan must appear exactly once in yours. Do not edit
any other file, do not run git or gh, and do not publish anything.
PROMPT
)"

  if ! timeout "$AGENT_REVIEW_BUDGET_SECONDS" env -u GH_TOKEN -u GITHUB_TOKEN \
      opencode run --auto --model "$AGENT_MODEL" "$prompt" \
      > "$WORKDIR/agent.log" 2>&1; then
    echo "release-notes: agent review turn failed or exceeded its budget; keeping deterministic notes"
    tail -20 "$WORKDIR/agent.log" || true
    return 1
  fi
  if [[ ! -s "$WORKDIR/plan.agent.json" ]]; then
    echo "release-notes: agent produced no plan; keeping deterministic notes"
    return 1
  fi
  # Version and date come from the deterministic plan, never the agent's, and
  # the renderer rejects any section or heading text the agent invents.
  if ! python3 "$ROOT/scripts/release-notes-regroup.py" \
      --body "$WORKDIR/body.md" --plan "$WORKDIR/plan.agent.json" \
      --metadata-from "$WORKDIR/plan.deterministic.json" \
      --out "$WORKDIR/notes.agent.md"; then
    echo "release-notes: agent plan failed validation; keeping deterministic notes"
    return 1
  fi
  if ! verify_entries "$WORKDIR/notes.agent.md"; then
    echo "release-notes: agent render changed the entry set; keeping deterministic notes" >&2
    cat "$WORKDIR/entry-diff.txt" >&2
    return 1
  fi
  return 0
}

if agent_reachable && agent_review; then
  chosen="$WORKDIR/notes.agent.md"
  source_label="deterministic + agent review"
fi

echo "release-notes: publishing $source_label notes"

# ── Publish ───────────────────────────────────────────────────────────

if [[ "$DRY_RUN" == "true" ]]; then
  echo "release-notes: DRY_RUN=true; not editing the release"
  echo "release-notes: would publish $chosen"
  exit 0
fi

# Editing a published release always requires an explicit approval signal.
# There is no implicit "any CI context is trusted" bypass: the release job
# declares RELEASE_NOTES_APPROVED in its own reviewable definition.
if [[ "${RELEASE_NOTES_APPROVED:-}" != "true" ]]; then
  echo "release-notes: refusing to edit a published release without approval" >&2
  echo "release-notes: preview with DRY_RUN=true, or publish with RELEASE_NOTES_APPROVED=true" >&2
  exit 1
fi

gh release edit "$TAG" --repo "$REPO" --notes-file "$chosen"

gh release view "$TAG" --repo "$REPO" --json body -q .body > "$WORKDIR/live.md"
if ! diff <(entry_prs "$WORKDIR/body.backup.md") <(entry_prs "$WORKDIR/live.md"); then
  echo "release-notes: published body does not carry the original entry set" >&2
  echo "release-notes: restore with: gh release edit $TAG --repo $REPO --notes-file $WORKDIR/body.backup.md" >&2
  exit 1
fi

echo "release-notes: published $source_label notes for $TAG"
