#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Fails if any tracked file carries a private key or an access token, or if
# the tree tracks a file that never belongs in this repository: Python (the
# plugin is Rust only), demo or red-team material, key or token files, node
# data (ledgers, databases, data directories), a NOTICE file.
#
# Usage: scripts/no-keys.sh [dir]    (default: the plugin's own directory)
#
# Only files tracked under the directory are read: in the mesh-llm checkout
# that is the plugin's directory, not the whole repository.
#
# It prints file:line and the kind of secret it matched, never the matched
# text, so a hit in CI does not republish the secret in the log. For an
# excluded file it prints the path and the kind.
#
# Public key material is not matched: key ids (`signing_key_id`,
# `key_id`), public keys and digests are 64-hex values too, so the hex rule
# only fires for private-key field names.
set -eu

root=${1:-$(dirname "$0")/..}
cd "$root"

# kind<TAB>extended regex
rules='private-key-block	-----BEGIN ([A-Z0-9]+ )*PRIVATE KEY( BLOCK)?-----
github-token	(gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})
crates-io-token	cio[A-Za-z0-9]{32}
hugging-face-token	hf_[A-Za-z0-9]{34,}
aws-access-key	AKIA[0-9A-Z]{16}
private-key-hex	(private_key|secret_key|signing_key|privkey|seed|secret)["'"'"']?[[:space:]]*[:=][[:space:]]*["'"'"']?[0-9a-fA-F]{64}'

found=0
tab=$(printf '\t')
files=$(mktemp)
trap 'rm -f "$files"' EXIT
git ls-files -z > "$files"

while IFS="$tab" read -r kind pattern; do
  hits=$(xargs -0 grep -I -n -E -o -- "$pattern" < "$files" 2>/dev/null | cut -d: -f1,2 || true)
  if [ -n "$hits" ]; then
    found=1
    printf '%s\n' "$hits" | while IFS= read -r where; do
      printf 'no-keys: %s: %s\n' "$where" "$kind" >&2
    done
  fi
done <<EOF
$rules
EOF

# kind<TAB>extended regex over the tracked path (case-insensitive)
paths='python-file	\.py[cdow]?$|(^|/)__pycache__/
demo-material	(^|/)(demo\.sh|demo\.md)$
red-team-material	(^|/)[^/]*red-?team[^/]*(/|$)
notice-file	(^|/)NOTICE(\.[a-z]+)?$
key-file	\.(pem|key|p8|p12|pfx|jwk)$|(^|/)id_(ed25519|rsa|ecdsa)[^/]*$
token-file	\.token$|(^|/)\.env(\.[^/]*)?$
node-data	\.(jsonl|sqlite3?|db)$|(^|/)(capsule-data|ledger)/'

excluded=0
while IFS="$tab" read -r kind pattern; do
  hits=$(tr '\0' '\n' < "$files" | grep -i -E -- "$pattern" || true)
  if [ -n "$hits" ]; then
    excluded=1
    printf '%s\n' "$hits" | while IFS= read -r where; do
      printf 'no-keys: %s: excluded file (%s)\n' "$where" "$kind" >&2
    done
  fi
done <<EOF
$paths
EOF

if [ "$found" -ne 0 ]; then
  echo "no-keys: private key or token material found (locations above)" >&2
fi
if [ "$excluded" -ne 0 ]; then
  echo "no-keys: files that never belong in this repository are tracked (above)" >&2
fi
if [ "$found" -ne 0 ] || [ "$excluded" -ne 0 ]; then
  exit 1
fi
echo "no-keys: clean"
