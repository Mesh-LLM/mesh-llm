#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "usage: $0 <backend> <case-id> <command...>" >&2
    exit 2
fi

BACKEND="$1"
CASE_ID="$2"
shift 2

ROOT="${VALIDATION_RESULTS_ROOT:-$PWD/MLX_VALIDATION_RESULTS}"
STAMP="${VALIDATION_RESULTS_STAMP:-$(date +%Y%m%d-%H%M%S)}"
CASE_DIR="$ROOT/$STAMP/$CASE_ID"

mkdir -p "$CASE_DIR"
export VALIDATION_CASE_DIR="$CASE_DIR"

printf '%s\n' "$BACKEND" > "$CASE_DIR/backend.txt"
printf '%s\n' "$CASE_ID" > "$CASE_DIR/case.txt"
printf '%s\n' "$PWD" > "$CASE_DIR/cwd.txt"
printf '%q ' "$@" > "$CASE_DIR/command.sh"
printf '\n' >> "$CASE_DIR/command.sh"

STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cat > "$CASE_DIR/state.json" <<EOF
{
  "backend": "$BACKEND",
  "case_id": "$CASE_ID",
  "status": "running",
  "started_at": "$STARTED_AT"
}
EOF

set +e
"$@" > >(tee "$CASE_DIR/stdout.log") 2> >(tee "$CASE_DIR/stderr.log" >&2)
STATUS=$?
set -e

printf '%s\n' "$STATUS" > "$CASE_DIR/exit_code.txt"

if [ "$BACKEND" = "gguf" ] && [ -f /tmp/mesh-llm-ci-gguf.log ]; then
    cp -f /tmp/mesh-llm-ci-gguf.log "$CASE_DIR/mesh.log"
fi

if [ "$BACKEND" = "mlx" ] && [ -f /tmp/mesh-llm-ci-mlx.log ]; then
    cp -f /tmp/mesh-llm-ci-mlx.log "$CASE_DIR/mesh.log"
fi

cat > "$CASE_DIR/meta.json" <<EOF
{
  "backend": "$BACKEND",
  "case_id": "$CASE_ID",
  "exit_code": $STATUS
}
EOF

FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cat > "$CASE_DIR/state.json" <<EOF
{
  "backend": "$BACKEND",
  "case_id": "$CASE_ID",
  "status": "completed",
  "started_at": "$STARTED_AT",
  "finished_at": "$FINISHED_AT",
  "exit_code": $STATUS
}
EOF

exit "$STATUS"
