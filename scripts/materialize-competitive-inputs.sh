#!/usr/bin/env bash
# materialize-competitive-inputs.sh — materialize every pinned competitive-benchmark
# input from Hugging Face into the HF cache.
#
# All downloads land in the huggingface_hub cache (default $HF_HOME/hub, or
# $HF_HUB_CACHE / HF_HUB_OFFLINE=1 as set in the environment) and are verified
# against the digests pinned in evals/skippy-competitive-benchmark.json before
# being linked into a runnable input tree under <root>:
#
#   <root>/models/<model-key>/            — pinned GGUF file
#   <root>/tokenizers/<model-key>/        — llama-benchy tokenizer directory
#   <root>/vllm-configs/<model-key>/      — pinned HF config.json (vLLM GGUF plugin)
#   <root>/thoughtworks/sessions.parquet  — pinned dataset file
#   <root>/thoughtworks/manifest.json     — deterministic prompt manifest
#
# Tokenizer provenance (all exports verified byte-identical to the pinned
# tokenizer_sha256 directory digests):
#   llama32-dense        — tokenizer files byte-equal to unsloth/Llama-3.2-1B-Instruct
#                          @ 5a8abab4a5d6f164389b1079fb721cfab8d7126c (the pinned
#                          vllm_hf_config revision)
#   granite-h1-hybrid    — full snapshot of ibm-granite/granite-4.0-h-1b @
#                          d18cca4c121edb87d022116d281ce212c9136f57 minus
#                          README.md (whose bytes are not part of the pin)
#   deepseek-v2-moe      — transformers v5 AutoTokenizer.from_pretrained(repo,
#                          revision=<pinned vllm_hf_config revision>).save_pretrained(
#                          legacy_format=False); the pinned 7.5MB tokenizer.json is
#                          this re-export, not any HF repo head
#   falcon-h1-recurrent  — same transformers v5 export as deepseek
#
# Usage:
#   scripts/materialize-competitive-inputs.sh [--root DIR] [--models a,b,..]
#       [--skip-dataset] [--skip-tokenizers] [--skip-vllm-configs]
#
# Options:
#   --root DIR     runnable input tree (default: $PWD/bench-inputs); downloads
#                  always land in the HF cache regardless
#   --models LIST  comma-separated model keys to materialize (default: all four)
#   --skip-*       skip a materialization group
#
# Environment:
#   HF_TOKEN / HUGGING_FACE_HUB_TOKEN — optional; required only for gated repos
#   TRANSFORMERS_VERBOSITY=error      — silences the "PyTorch not found" banner
#
# Exit codes: 0 success; 1 usage/preflight/digest mismatch.
set -euo pipefail

ROOT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${MESH_COMPETITIVE_CONFIG:-$ROOT_SCRIPT_DIR/../evals/skippy-competitive-benchmark.json}"
PROMPT_GENERATOR="$ROOT_SCRIPT_DIR/../evals/skippy-agentic-prompt-manifest.py"

ROOT="${MESH_COMPETITIVE_INPUT_ROOT:-$PWD/bench-inputs}"
MODELS="${MESH_COMPETITIVE_MODELS:-}"
SKIP_DATASET=0
SKIP_TOKENIZERS=0
SKIP_VLLM_CONFIGS=0
PYTHON_BIN="${MESH_COMPETITIVE_PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --skip-dataset) SKIP_DATASET=1; shift ;;
    --skip-tokenizers) SKIP_TOKENIZERS=1; shift ;;
    --skip-vllm-configs) SKIP_VLLM_CONFIGS=1; shift ;;
    -h|--help) sed -n '2,49p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "error: unknown option: $1" >&2; exit 1 ;;
  esac
done

command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "error: $PYTHON_BIN not found" >&2; exit 1; }
command -v jq >/dev/null 2>&1 || { echo "error: jq not found" >&2; exit 1; }
command -v hf >/dev/null 2>&1 || { echo "error: hf CLI not found (pip install \"huggingface_hub[cli]\")" >&2; exit 1; }

[[ -f "$CONFIG" ]] || { echo "error: config not found: $CONFIG" >&2; exit 1; }
[[ -f "$PROMPT_GENERATOR" ]] || { echo "error: prompt generator not found: $PROMPT_GENERATOR" >&2; exit 1; }

# Pin the model list from the config (single source of truth) unless overridden.
if [[ -z "$MODELS" ]]; then
  MODELS="$(jq -r '[.models[].key] | join(",")' "$CONFIG")"
fi

model_field() { # <key> <jq filter over .models[] | select(.key == key)
  jq -r --arg key "$1" ".models[] | select(.key == \$key) | $2" "$CONFIG"
}

file_sha256() { # portable sha256 of a file (Linux sha256sum / macOS shasum)
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

resolve_path() { # portable `readlink -f` (BSD/macOS readlink has no -f)
  "$PYTHON_BIN" - "$1" <<'PY'
import pathlib, sys
print(pathlib.Path(sys.argv[1]).resolve())
PY
}

fail_count=0

echo "=== Materialize competitive inputs ==="
echo "  config:      $CONFIG"
echo "  input root:  $ROOT"
echo "  models:      $MODELS"
echo "  hf cache:    ${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
echo ""

# ---------------------------------------------------------------------------
# 1. Pinned GGUF models — hf download into the cache, then verify + link.
# ---------------------------------------------------------------------------
IFS=',' read -ra MODEL_KEYS <<< "$MODELS"
for key in "${MODEL_KEYS[@]}"; do
  key="$(echo "$key" | xargs)"
  [[ -n "$key" ]] || continue
  repo="$(model_field "$key" '.repo')"
  revision="$(model_field "$key" '.revision')"
  filename="$(model_field "$key" '.filename')"
  sha="$(model_field "$key" '.sha256')"
  [[ "$repo" != "null" ]] || { echo "error: unknown model key: $key" >&2; fail_count=$((fail_count+1)); continue; }

  echo "--- model $key: $repo @$revision"
  # No --local-dir: the blob lands in the HF cache; -q prints the path only.
  cached="$(hf download -q "$repo" "$filename" --revision "$revision" | tail -1)"
  actual="$("$PYTHON_BIN" - "$cached" <<'PY'
import hashlib, sys
digest = hashlib.sha256()
with open(sys.argv[1], "rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
)"
  if [[ "$actual" != "$sha" ]]; then
    echo "error: $key GGUF SHA-256 mismatch: expected=$sha actual=$actual" >&2
    fail_count=$((fail_count+1))
    continue
  fi
  mkdir -p "$ROOT/models/$key"
  ln -sfn "$(resolve_path "$cached")" "$ROOT/models/$key/$filename"
  echo "    ok: $filename ($(du -h "$(resolve_path "$cached")" | cut -f1), sha256 verified)"
done

# ---------------------------------------------------------------------------
# 2. llama-benchy tokenizer directories.
#    Derived inputs: transformers v5 re-exports (deepseek/falcon) verified
#    byte-identical to the pinned digests; llama32 files byte-equal to the
#    pinned unsloth snapshot; granite pinned snapshot minus README.md.
#    AutoTokenizer pulls the source files through the HF cache.
# ---------------------------------------------------------------------------
if [[ "$SKIP_TOKENIZERS" -eq 0 ]]; then
  echo "--- tokenizers (transformers v5 export through the HF cache)"
  TOKENIZER_EXPORT="$(mktemp -d)"
  if ! TRANSFORMERS_VERBOSITY=error "$PYTHON_BIN" - "$CONFIG" "$ROOT" "$TOKENIZER_EXPORT" "$MODELS" <<'PY'
import json
import pathlib
import shutil
import sys

config = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
root = pathlib.Path(sys.argv[2])
export_root = pathlib.Path(sys.argv[3])
wanted = {key.strip() for key in sys.argv[4].split(",") if key.strip()}

def directory_sha256(path: pathlib.Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise RuntimeError(f"directory contains no files: {path}")
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        inner = hashlib.sha256()
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                inner.update(chunk)
        digest.update(bytes.fromhex(inner.hexdigest()))
    return digest.hexdigest()

failures = []
for model in config["models"]:
    key = model["key"]
    if key not in wanted:
        continue
    pin = model["vllm_hf_config"]
    export = export_root / key
    if key == "granite-h1-hybrid":
        # Pinned tokenizer directory is the full HF snapshot minus README.md.
        from huggingface_hub import snapshot_download
        snapshot = pathlib.Path(snapshot_download(
            repo_id=pin["repo"],
            revision=pin["revision"],
        ))
        export.mkdir(parents=True, exist_ok=True)
        for candidate in sorted(p for p in snapshot.rglob("*") if p.is_file()):
            if candidate.name == "README.md":
                continue
            shutil.copy2(candidate, export / candidate.name)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(pin["repo"], revision=pin["revision"])
        tokenizer.save_pretrained(export, legacy_format=False)
        # The pin contains exactly these three files for non-granite keys.
        allowed = {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}
        for stray in export.iterdir():
            if stray.name not in allowed:
                stray.unlink()
    actual = directory_sha256(export)
    expected = model["tokenizer_sha256"]
    if actual != expected:
        failures.append((key, expected, actual))
        print(f"    MISMATCH {key}: expected={expected} actual={actual}", file=sys.stderr)
        continue
    target = root / "tokenizers" / key
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(export, target)
    print(f"    ok: {key} (directory sha256 verified)")
if failures:
    sys.exit(1)
PY
  then
    echo "error: tokenizer materialization failed (see mismatches above)" >&2
    fail_count=$((fail_count+1))
  fi
  rm -rf "$TOKENIZER_EXPORT"
else
  echo "--- tokenizers: skipped"
fi

# ---------------------------------------------------------------------------
# 3. Per-model vLLM HF configs — pinned config.json via hf download.
# ---------------------------------------------------------------------------
if [[ "$SKIP_VLLM_CONFIGS" -eq 0 ]]; then
  for key in "${MODEL_KEYS[@]}"; do
    key="$(echo "$key" | xargs)"
    [[ -n "$key" ]] || continue
    repo="$(model_field "$key" '.vllm_hf_config.repo')"
    revision="$(model_field "$key" '.vllm_hf_config.revision')"
    sha="$(model_field "$key" '.vllm_hf_config.sha256')"
    [[ "$repo" != "null" ]] || continue
    echo "--- vllm config $key: $repo @$revision"
    cached="$(hf download -q "$repo" "config.json" --revision "$revision" | tail -1)"
    actual="$(file_sha256 "$cached")"
    if [[ "$actual" != "$sha" ]]; then
      echo "error: $key config.json SHA-256 mismatch: expected=$sha actual=$actual" >&2
      fail_count=$((fail_count+1))
      continue
    fi
    mkdir -p "$ROOT/vllm-configs/$key"
    ln -sfn "$(resolve_path "$cached")" "$ROOT/vllm-configs/$key/config.json"
    echo "    ok: config.json (sha256 verified)"
  done
fi

# ---------------------------------------------------------------------------
# 4. Thoughtworks dataset + deterministic prompt manifest.
# ---------------------------------------------------------------------------
if [[ "$SKIP_DATASET" -eq 0 ]]; then
  ds_repo="$(jq -r '.thoughtworks.dataset.repo' "$CONFIG")"
  ds_rev="$(jq -r '.thoughtworks.dataset.revision' "$CONFIG")"
  ds_file="$(jq -r '.thoughtworks.dataset.filename' "$CONFIG")"
  ds_sha="$(jq -r '.thoughtworks.dataset.sha256' "$CONFIG")"
  echo "--- dataset: $ds_repo @$ds_rev"
  cached="$(hf download -q "$ds_repo" "$ds_file" --repo-type dataset --revision "$ds_rev" | tail -1)"
  actual="$(file_sha256 "$cached")"
  if [[ "$actual" != "$ds_sha" ]]; then
    echo "error: dataset SHA-256 mismatch: expected=$ds_sha actual=$actual" >&2
    fail_count=$((fail_count+1))
  else
    mkdir -p "$ROOT/thoughtworks"
    ln -sfn "$(resolve_path "$cached")" "$ROOT/thoughtworks/$ds_file"
    echo "    ok: $ds_file (sha256 verified)"
    sources=()
    while IFS= read -r source; do
      sources+=(--source-dataset "$source")
    done < <(jq -r '.thoughtworks.selection.sources[]' "$CONFIG")
    "$PYTHON_BIN" "$PROMPT_GENERATOR" \
      --dataset-file "$ROOT/thoughtworks/$ds_file" \
      --dataset-revision "$ds_rev" \
      --output "$ROOT/thoughtworks/manifest.json" \
      --families "$(jq -r '.thoughtworks.selection.families' "$CONFIG")" \
      --requests-per-family "$(jq -r '.thoughtworks.selection.requests_per_family' "$CONFIG")" \
      --min-isl "$(jq -r '.thoughtworks.selection.min_isl' "$CONFIG")" \
      --max-isl "$(jq -r '.thoughtworks.selection.max_isl_exclusive' "$CONFIG")" \
      --min-turns "$(jq -r '.thoughtworks.selection.min_turns' "$CONFIG")" \
      "${sources[@]}"
    want_manifest="$(jq -r '.thoughtworks.selection.manifest_sha256' "$CONFIG")"
    got_manifest="$(file_sha256 "$ROOT/thoughtworks/manifest.json")"
    if [[ "$got_manifest" != "$want_manifest" ]]; then
      echo "error: manifest SHA-256 mismatch: expected=$want_manifest actual=$got_manifest" >&2
      fail_count=$((fail_count+1))
    else
      echo "    ok: manifest.json (sha256 verified)"
    fi
  fi
fi

echo ""
if [[ "$fail_count" -gt 0 ]]; then
  echo "FAILED: $fail_count input group(s) failed verification" >&2
  exit 1
fi
echo "All pinned inputs materialized into the HF cache and verified."
echo "Runnable input tree: $ROOT"
