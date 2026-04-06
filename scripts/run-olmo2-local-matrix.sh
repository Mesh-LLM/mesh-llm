#!/usr/bin/env bash

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  scripts/run-olmo2-local-matrix.sh <OLMO2_GGUF_PATH> [--suite <exact|behavior|all>] [--skip-build]
  [--root <RESULTS_ROOT>] [--stamp <STAMP>] [--max-prompts N] [--max-tokens N] [--wait-seconds N]
EOF
  exit 0
fi

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  scripts/run-olmo2-local-matrix.sh <OLMO2_GGUF_PATH> [--suite <exact|behavior|all>] [--skip-build]
  [--root <RESULTS_ROOT>] [--stamp <STAMP>] [--max-prompts N] [--max-tokens N] [--wait-seconds N]
EOF
  exit 1
fi

OLMO2_GGUF_PATH="$1"
shift

if [[ ! -f "$OLMO2_GGUF_PATH" ]]; then
  echo "OLMO2 model file not found: $OLMO2_GGUF_PATH" >&2
  exit 2
fi

if [[ "$OLMO2_GGUF_PATH" != *.gguf ]]; then
  echo "OLMO2 model file must be a .gguf path." >&2
  exit 2
fi

SUITE="all"
ROOT="${MLX_VALIDATION_ROOT:-MLX_VALIDATION_RESULTS}"
STAMP="$(date +%Y%m%d-%H%M%S)"
SKIP_BUILD=0
MAX_PROMPTS=0
MAX_TOKENS=0
WAIT_SECONDS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)
      SUITE="$2"
      shift 2
      ;;
    --root)
      ROOT="$2"
      shift 2
      ;;
    --stamp)
      STAMP="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --max-prompts)
      MAX_PROMPTS="$2"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --wait-seconds)
      WAIT_SECONDS="$2"
      shift 2
      ;;
    --help|-h)
      cat <<'EOF'
Usage:
  scripts/run-olmo2-local-matrix.sh <OLMO2_GGUF_PATH> [--suite <exact|behavior|all>] [--skip-build]
  [--root <RESULTS_ROOT>] [--stamp <STAMP>] [--max-prompts N] [--max-tokens N] [--wait-seconds N]
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$SUITE" != "exact" && "$SUITE" != "behavior" && "$SUITE" != "all" ]]; then
  echo "Invalid --suite value: $SUITE (expected exact|behavior|all)." >&2
  exit 2
fi

if [[ "${OLMO2_GGUF_PATH}" != /* ]]; then
  OLMO2_GGUF_PATH="$(cd "$(dirname "$OLMO2_GGUF_PATH")" && pwd)/$(basename "$OLMO2_GGUF_PATH")"
fi

TMP_MATRIX="$(mktemp /tmp/mesh-llm-olmo2-matrix-XXXXXX.json)"
trap 'rm -f "$TMP_MATRIX"' EXIT

python3 - "$OLMO2_GGUF_PATH" "$TMP_MATRIX" <<'PY'
import json
import sys
from pathlib import Path

src = Path("testdata/validation/matrix.json").read_text(encoding="utf-8")
matrix = json.loads(src)
olmo2_path = Path(sys.argv[1]).expanduser().resolve().as_posix()
tmp_matrix = Path(sys.argv[2])

models = [m for m in matrix["models"] if m.get("id") != "olmo2"]
models.append(
    {
        "id": "olmo2",
        "label": "olmo2-7b-instruct",
        "expectation_class": "weak-but-stable",
        "notes": "Local-only OLMO2 harness entry; not part of CI source-of-truth matrix.",
        "gguf": {
            "exact_case_id": "olmo2-local-gguf-exact",
            "behavior_case_id": "olmo2-local-gguf-behavior",
            "model_ref": olmo2_path,
        },
    }
)
matrix["models"] = models
tmp_matrix.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

ARGS=(--suite "$SUITE" --backend gguf --cases "olmo2" --matrix "$TMP_MATRIX" --root "$ROOT" --stamp "$STAMP")

if [[ "$SKIP_BUILD" == "1" ]]; then
  ARGS+=(--skip-build)
fi

if [[ "$MAX_PROMPTS" != "0" ]]; then
  ARGS+=(--max-prompts "$MAX_PROMPTS")
fi

if [[ "$MAX_TOKENS" != "0" ]]; then
  ARGS+=(--max-tokens "$MAX_TOKENS")
fi

if [[ "$WAIT_SECONDS" != "0" ]]; then
  ARGS+=(--wait-seconds "$WAIT_SECONDS")
fi

python3 scripts/run-validation-matrix.py "${ARGS[@]}"
