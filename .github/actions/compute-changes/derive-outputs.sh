#!/usr/bin/env bash
# Derive compute-changes outputs from the bounded files/crates inputs.
set -e -o pipefail

# Read the JSON output from affected-crates.sh
CRATES_JSON=$(cat /tmp/crates_output.json)

# Extract fields from JSON
AFFECTED_CRATES=$(echo "$CRATES_JSON" | jq -c '.affected // []')
TEST_CRATES=$(echo "$CRATES_JSON" | jq -c '.test_crates // []')
BATCHES=$(echo "$CRATES_JSON" | jq -c '.batches // []')
ALL_RUST=$(echo "$CRATES_JSON" | jq -r '.all_rust // false')
UI_CHANGED=$(echo "$CRATES_JSON" | jq -r '.ui_changed // false')
WEBSITE_CHANGED=$(echo "$CRATES_JSON" | jq -r '.website_changed // false')

if [[ "$ALL_RUST" == "true" ]]; then
  CLIPPY_BATCHES=$(bash scripts/plan-clippy-batches.sh --all)
else
  CLIPPY_BATCHES=$(bash scripts/plan-clippy-batches.sh --crates-json "$AFFECTED_CRATES")
fi

if [[ "$EVENT_NAME" != "pull_request" ]] || [[ "$ALL_RUST" == "true" ]]; then
  TEST_BATCHES=$(bash scripts/plan-test-batches.sh --all --bins 4)
else
  TEST_BATCHES=$(bash scripts/plan-test-batches.sh --crates-json "$AFFECTED_CRATES" --bins 4)
fi

FORCE_ALL="false"
if grep -qx "__force_all__" /tmp/changed_files.txt; then
  FORCE_ALL="true"
fi

# Read changed files for docs_only and rust_changed logic
CHANGED_FILES=$(cat /tmp/changed_files.txt | grep -v "^__force_all__$" || true)

RUNNER_CONTRACT_REQUIRED="false"
if [[ "$EVENT_NAME" == "workflow_dispatch" ]]; then
  RUNNER_CONTRACT_REQUIRED="true"
elif [[ -n "$CHANGED_FILES" ]]; then
  RUNNER_CONTRACT_INPUTS=$(echo "$CHANGED_FILES" | grep -E '(^\.github/cache-version\.txt$|^\.github/actionlint\.yaml$|^\.github/actions/(capture-sccache-stats|configure-sccache-gha|restore-sccache-seed|resolve-native-toolchain-epoch|select-ci-runners)/|^\.github/workflows/(cache-warm-sccache|ci|ci-control|ci-.*-(lane|slice)|depot-canary|main_[a-z]+|native-sdk-artifact|pr_[a-z]+|release|sdk-smoke|static-abi-artifact|swift-sdk-artifact)\.yml$)' || true)
  if [[ -n "$RUNNER_CONTRACT_INPUTS" ]]; then
    RUNNER_CONTRACT_REQUIRED="true"
  fi
fi

# Determine docs_only: true if all_rust=false, UI/website are unchanged,
# and all files match authored docs patterns.
DOCS_ONLY="false"
if [[ "$ALL_RUST" == "false" ]] && [[ "$UI_CHANGED" == "false" ]] && [[ "$WEBSITE_CHANGED" == "false" ]]; then
  # Check if all changed files match docs patterns (*.md or docs/**)
  if [[ -n "$CHANGED_FILES" ]]; then
    NON_DOCS=$(echo "$CHANGED_FILES" | grep -v -E '(\.md$|^docs/)' || true)
    if [[ -z "$NON_DOCS" ]]; then
      DOCS_ONLY="true"
    fi
  fi
fi

# Determine rust_changed: true if all_rust=true OR affected_crates is non-empty
RUST_CHANGED="false"
if [[ "$ALL_RUST" == "true" ]]; then
  RUST_CHANGED="true"
elif [[ $(echo "$AFFECTED_CRATES" | jq 'length') -gt 0 ]]; then
  RUST_CHANGED="true"
fi

# CLI surface definitions are public documentation inputs. Keep this
# limited to Clap/parser sources, not the React console UI or command
# handler internals, so website docs sync is precise and explainable.
CLI_SURFACE_CHANGED="false"
if [[ -n "$CHANGED_FILES" ]]; then
  CLI_SURFACE_INPUTS=$(echo "$CHANGED_FILES" | grep -E '^crates/mesh-llm-cli/src/(parser|models|runtime|benchmark)\.rs$' || true)
  if [[ -n "$CLI_SURFACE_INPUTS" ]]; then
    CLI_SURFACE_CHANGED="true"
  fi
fi

WEBSITE_DOCS_CHANGED="false"
if [[ -n "$CHANGED_FILES" ]]; then
  WEBSITE_DOC_INPUTS=$(echo "$CHANGED_FILES" | grep -E '^website/src/(docs/pages/|_includes/)' || true)
  if [[ -n "$WEBSITE_DOC_INPUTS" ]]; then
    WEBSITE_DOCS_CHANGED="true"
  fi
fi

# Shared Just recipe classification for the backend-input checks below.
JUSTFILE_RECIPE_AWK='
  function recipe_name(line, parts) {
    if (line ~ /^[[:alnum:]_-]/ &&
        line !~ /^(export|set)[[:space:]]/ &&
        line !~ /^[[:alnum:]_-]+[[:space:]]*:=/ &&
        line ~ /:/) {
      split(line, parts, /[[:space:]:]/)
      return parts[1]
    }
    return ""
  }

  function is_backend_recipe(name) {
    return name ~ /^(with-lld|build|build-dev|build-mac|build-linux|build-runtime|release-host-build|release-host-build-windows|release-runtime-build|release-build|release-build-[[:alnum:]-]+|llama-prepare|llama-prepare-latest|llama-build|skippy-quantize-standalone-build|skippy-quantize-standalone-release-build|bundle|release-bundle|release-bundle-[[:alnum:]-]+)$/
  }

  FNR == 1 { backend = 0 }
'

justfile_backend_recipe_lines() {
  local justfile_path="$1"
  awk "$JUSTFILE_RECIPE_AWK"'
    /^\[/ {
      backend = 0
      pending_attribute_lines[++pending_attribute_count] = NR
      next
    }
    {
      name = recipe_name($0)
      if (name != "") {
        backend = is_backend_recipe(name)
        if (backend) {
          for (pending_index = 1; pending_index <= pending_attribute_count; pending_index++) {
            print pending_attribute_lines[pending_index]
          }
        }
        delete pending_attribute_lines
        pending_attribute_count = 0
      } else if ($0 !~ /^[[:space:]]/) {
        backend = 0
        delete pending_attribute_lines
        pending_attribute_count = 0
      }
      if (backend) {
        print NR
      }
    }
  ' "$justfile_path"
}

# Top-level assignments are global Just inputs shared across the whole
# import graph, so a backend recipe can change meaning without any
# backend recipe line changing (for example `mesh_bin`, which `bundle`
# consumes). Collect every identifier appearing in a backend recipe
# header or body at one revision so assignment edits to those names are
# classified as backend inputs. Over-collection is intentional: it fails
# toward running backend lanes.
justfile_backend_recipe_tokens() {
  local sha="$1"
  local sources_dir source_path
  local source_paths=()
  sources_dir="$(mktemp -d)"
  while IFS= read -r source_path; do
    [[ -n "$source_path" ]] || continue
    mkdir -p "$sources_dir/$(dirname "$source_path")" || return 1
    git show "$sha:$source_path" > "$sources_dir/$source_path" || return 1
    source_paths+=("$sources_dir/$source_path")
  done < <(git ls-tree -r --name-only "$sha" | grep -E '^Justfile$|\.just$' || true)
  [[ ${#source_paths[@]} -gt 0 ]] || return 1
  awk "$JUSTFILE_RECIPE_AWK"'
    {
      name = recipe_name($0)
      if (name != "") {
        backend = is_backend_recipe(name)
      } else if ($0 !~ /^[[:space:]]/) {
        backend = 0
      }
      if (!backend) {
        next
      }
      line = $0
      while (match(line, /[[:alpha:]_][[:alnum:]_]*/)) {
        print substr(line, RSTART, RLENGTH)
        line = substr(line, RSTART + RLENGTH)
      }
    }
  ' "${source_paths[@]}" | sort -u
}

justfile_backend_input_lines() {
  local justfile_path="$1"
  local tokens_file="$2"
  awk -v tokens_file="$tokens_file" '
    BEGIN {
      while ((getline token < tokens_file) > 0) {
        referenced[token] = 1
      }
    }
    /^(export[[:space:]]+)?[[:alnum:]_-]+[[:space:]]*:=/ {
      assignment = $0
      sub(/^export[[:space:]]+/, "", assignment)
      match(assignment, /^[[:alnum:]_-]+/)
      if (referenced[substr(assignment, RSTART, RLENGTH)]) {
        print NR
      }
    }
  ' "$justfile_path"
}

justfile_has_recipe() {
  local justfile_path="$1"
  awk "$JUSTFILE_RECIPE_AWK"'
    recipe_name($0) != "" { found = 1 }
    END { exit found ? 0 : 1 }
  ' "$justfile_path"
}

justfile_changed_line_ranges() {
  awk '
    function emit_range(spec, label, parts, start, count, i) {
      gsub(/^[-+]/, "", spec)
      split(spec, parts, ",")
      start = parts[1]
      count = (parts[2] == "" ? 1 : parts[2])
      for (i = 0; i < count; i++) {
        print label, start + i
      }
    }

    /^@@ / {
      emit_range($2, "old")
      emit_range($3, "new")
    }
  '
}

justfile_changed_import() {
  awk '
    /^@@ / { in_hunk = 1; next }
    in_hunk && /^[+-]/ {
      line = substr($0, 2)
      if (line ~ /^import[?]?[[:space:]]+/) {
        found = 1
      }
    }
    END { exit found ? 0 : 1 }
  '
}

changed_range_touches_lines() {
  local backend_lines_file="$1"
  local changed_lines_file="$2"
  local side="$3"
  awk -v side="$side" '
    NR == FNR { backend[$1] = 1; next }
    $1 == side && backend[$2] { found = 1 }
    END { exit found ? 0 : 1 }
  ' "$backend_lines_file" "$changed_lines_file"
}

# Recipe-source edits are not automatically native build inputs. Inspect
# changed hunk ranges so website recipes can stay light while every line
# inside native build, ABI, release, and bundle recipes exercises backend
# lanes. Root import-graph changes and unreadable imported sources fail
# open because they can change which recipe definitions are effective.
BACKEND_RECIPE_CHANGED="false"
JUSTFILE_SOURCE_BASE_SHA="$BASE_SHA"
if [[ "$EVENT_NAME" == "pull_request" ]]; then
  JUSTFILE_DIFF_RANGE=("$BASE_SHA...$HEAD_SHA")
  if ! JUSTFILE_SOURCE_BASE_SHA=$(git merge-base "$BASE_SHA" "$HEAD_SHA"); then
    BACKEND_RECIPE_CHANGED="true"
  fi
else
  JUSTFILE_DIFF_RANGE=("$BASE_SHA" "$HEAD_SHA")
fi

JUSTFILE_SOURCES=$(echo "$CHANGED_FILES" | grep -E '^Justfile$|^just/.+\.just$' || true)
JUSTFILE_BACKEND_TOKENS_BASE="$(mktemp)"
JUSTFILE_BACKEND_TOKENS_HEAD="$(mktemp)"
if [[ -n "$JUSTFILE_SOURCES" ]]; then
  if ! justfile_backend_recipe_tokens "$JUSTFILE_SOURCE_BASE_SHA" > "$JUSTFILE_BACKEND_TOKENS_BASE" ||
      ! justfile_backend_recipe_tokens "$HEAD_SHA" > "$JUSTFILE_BACKEND_TOKENS_HEAD"; then
    BACKEND_RECIPE_CHANGED="true"
  fi
fi

while IFS= read -r JUSTFILE_SOURCE; do
  [[ -n "$JUSTFILE_SOURCE" ]] || continue
  [[ "$BACKEND_RECIPE_CHANGED" == "false" ]] || break

  JUSTFILE_SOURCE_BASE="$(mktemp)"
  JUSTFILE_SOURCE_HEAD="$(mktemp)"
  JUSTFILE_SOURCE_CHANGED_LINES="$(mktemp)"
  JUSTFILE_SOURCE_BACKEND_LINES_BASE="$(mktemp)"
  JUSTFILE_SOURCE_BACKEND_LINES_HEAD="$(mktemp)"
  JUSTFILE_SOURCE_BASE_AVAILABLE="false"
  JUSTFILE_SOURCE_HEAD_AVAILABLE="false"

  if ! JUSTFILE_SOURCE_STATUS=$(git diff --name-status --no-renames "${JUSTFILE_DIFF_RANGE[@]}" -- "$JUSTFILE_SOURCE"); then
    BACKEND_RECIPE_CHANGED="true"
    break
  fi
  case "$JUSTFILE_SOURCE_STATUS" in
    A$'\t'*) JUSTFILE_SOURCE_EXPECT_BASE="false"; JUSTFILE_SOURCE_EXPECT_HEAD="true" ;;
    D$'\t'*) JUSTFILE_SOURCE_EXPECT_BASE="true"; JUSTFILE_SOURCE_EXPECT_HEAD="false" ;;
    M$'\t'*) JUSTFILE_SOURCE_EXPECT_BASE="true"; JUSTFILE_SOURCE_EXPECT_HEAD="true" ;;
    *) BACKEND_RECIPE_CHANGED="true"; break ;;
  esac

  if [[ "$JUSTFILE_SOURCE_EXPECT_BASE" == "true" ]]; then
    if ! git cat-file -e "$JUSTFILE_SOURCE_BASE_SHA:$JUSTFILE_SOURCE" 2>/dev/null ||
        ! git show "$JUSTFILE_SOURCE_BASE_SHA:$JUSTFILE_SOURCE" > "$JUSTFILE_SOURCE_BASE" 2>/dev/null ||
        ! justfile_has_recipe "$JUSTFILE_SOURCE_BASE"; then
      BACKEND_RECIPE_CHANGED="true"
      break
    fi
    JUSTFILE_SOURCE_BASE_AVAILABLE="true"
    if ! justfile_backend_recipe_lines "$JUSTFILE_SOURCE_BASE" > "$JUSTFILE_SOURCE_BACKEND_LINES_BASE" ||
        ! justfile_backend_input_lines "$JUSTFILE_SOURCE_BASE" "$JUSTFILE_BACKEND_TOKENS_BASE" \
          >> "$JUSTFILE_SOURCE_BACKEND_LINES_BASE"; then
      BACKEND_RECIPE_CHANGED="true"
      break
    fi
  fi

  if [[ "$JUSTFILE_SOURCE_EXPECT_HEAD" == "true" ]]; then
    if ! git cat-file -e "$HEAD_SHA:$JUSTFILE_SOURCE" 2>/dev/null ||
        ! git show "$HEAD_SHA:$JUSTFILE_SOURCE" > "$JUSTFILE_SOURCE_HEAD" 2>/dev/null ||
        ! justfile_has_recipe "$JUSTFILE_SOURCE_HEAD"; then
      BACKEND_RECIPE_CHANGED="true"
      break
    fi
    JUSTFILE_SOURCE_HEAD_AVAILABLE="true"
    if ! justfile_backend_recipe_lines "$JUSTFILE_SOURCE_HEAD" > "$JUSTFILE_SOURCE_BACKEND_LINES_HEAD" ||
        ! justfile_backend_input_lines "$JUSTFILE_SOURCE_HEAD" "$JUSTFILE_BACKEND_TOKENS_HEAD" \
          >> "$JUSTFILE_SOURCE_BACKEND_LINES_HEAD"; then
      BACKEND_RECIPE_CHANGED="true"
      break
    fi
  fi

  if [[ "$JUSTFILE_SOURCE_BASE_AVAILABLE" == "false" && "$JUSTFILE_SOURCE_HEAD_AVAILABLE" == "false" ]]; then
    BACKEND_RECIPE_CHANGED="true"
    break
  fi
  if ! JUSTFILE_SOURCE_DIFF=$(git diff -U0 "${JUSTFILE_DIFF_RANGE[@]}" -- "$JUSTFILE_SOURCE"); then
    BACKEND_RECIPE_CHANGED="true"
    break
  fi
  if printf '%s\n' "$JUSTFILE_SOURCE_DIFF" | justfile_changed_import; then
    BACKEND_RECIPE_CHANGED="true"
    break
  fi
  printf '%s\n' "$JUSTFILE_SOURCE_DIFF" | justfile_changed_line_ranges > "$JUSTFILE_SOURCE_CHANGED_LINES"
  if [[ ! -s "$JUSTFILE_SOURCE_CHANGED_LINES" ]]; then
    BACKEND_RECIPE_CHANGED="true"
    break
  fi

  if [[ "$JUSTFILE_SOURCE_HEAD_AVAILABLE" == "true" ]] &&
      changed_range_touches_lines "$JUSTFILE_SOURCE_BACKEND_LINES_HEAD" "$JUSTFILE_SOURCE_CHANGED_LINES" new; then
    BACKEND_RECIPE_CHANGED="true"
  elif [[ "$JUSTFILE_SOURCE_BASE_AVAILABLE" == "true" ]] &&
      changed_range_touches_lines "$JUSTFILE_SOURCE_BACKEND_LINES_BASE" "$JUSTFILE_SOURCE_CHANGED_LINES" old; then
    BACKEND_RECIPE_CHANGED="true"
  fi
done <<< "$JUSTFILE_SOURCES"

# Backend/platform lanes rebuild only for all-rust escalations or files
# that can alter the native ABI/backend build products. Keep broad
# orchestration files like Justfile out of this path unless changed
# hunks touch native build/release recipes; cache keys still include
# them when backend jobs run for concrete build inputs.
BACKEND_CHANGED="false"
if [[ "$ALL_RUST" == "true" ]]; then
  BACKEND_CHANGED="true"
elif [[ -n "$CHANGED_FILES" ]]; then
  BACKEND_INPUTS=$(echo "$CHANGED_FILES" | grep -E '(^third_party/llama\.cpp/|^crates/skippy-ffi/|^scripts/(build-llama|prepare-llama|build-linux|build-linux-rocm|build-mac|build-windows|install-windows-sdk|build-host|build-release|package-release|package-native-runtime|verify-native-runtime-package|verify-checksum-sidecar|safe-extract-tar|compose-product-bundle|ci-compose-product-input|ci-client-readiness-smoke)\.|^\.github/actions/(prepare-host-input|prepare-windows-host-input|prepare-native-runtime-input|compose-product-input|resolve-native-toolchain-epoch|restore-smoke-inputs|restore-windows-abi-cache|save-and-verify-actions-cache|setup-windows-rocm-sdk)/|^\.github/workflows/(ci|main_[a-z]+|pr_[a-z]+|release|sdk-smoke|smoke)\.yml$|^\.github/cache-version\.txt$)' || true)
  if [[ -n "$BACKEND_INPUTS" ]] || [[ "$BACKEND_RECIPE_CHANGED" == "true" ]]; then
    BACKEND_CHANGED="true"
  fi
fi

WINDOWS_CPU_BUILD_REQUIRED="false"
WINDOWS_GPU_BUILD_REQUIRED="false"
if [[ "$FORCE_ALL" == "true" ]]; then
  WINDOWS_CPU_BUILD_REQUIRED="true"
  WINDOWS_GPU_BUILD_REQUIRED="true"
elif [[ -n "$CHANGED_FILES" ]]; then
  WINDOWS_CPU_INPUTS=$(echo "$CHANGED_FILES" | grep -E '(^crates/mesh-llm-release-footer/|^crates/mesh-llm-nodejs/|^crates/skippy-ffi/|^scripts/(build-windows|package-release)\.ps1$|^scripts/verify-host-dependencies\.py$|^scripts/(package-native-runtime|verify-native-runtime-package|verify-checksum-sidecar|safe-extract-tar|compose-product-bundle|ci-compose-product-input|ci-client-readiness-smoke)\.|^third_party/llama\.cpp/|^Cargo\.toml$|^Cargo\.lock$|^\.github/cache-version\.txt$|^\.github/workflows/(ci|main_[a-z]+|pr_[a-z]+|release|windows-warm-caches)\.yml$|^\.github/actions/(compute-changes/|prepare-windows-host-input/|prepare-native-runtime-input/|compose-product-input/|resolve-native-toolchain-epoch/|restore-windows-abi-cache/|save-and-verify-actions-cache/))' || true)
  WINDOWS_GPU_INPUTS=$(echo "$CHANGED_FILES" | grep -E '(^crates/skippy-ffi/|^scripts/(build-windows|install-windows-sdk|package-release)\.ps1$|^scripts/verify-host-dependencies\.py$|^scripts/(package-native-runtime|verify-native-runtime-package|verify-checksum-sidecar|safe-extract-tar|compose-product-bundle|ci-compose-product-input|ci-client-readiness-smoke)\.|^scripts/windows-native-runtime-deps\.py$|^scripts/tests/test_windows_native_runtime_deps\.py$|^third_party/llama\.cpp/|^\.github/cache-version\.txt$|^\.github/workflows/(ci|main_[a-z]+|pr_[a-z]+|release|windows-warm-caches)\.yml$|^\.github/actions/(compute-changes/|prepare-windows-host-input/|prepare-native-runtime-input/|compose-product-input/|resolve-native-toolchain-epoch/|restore-windows-abi-cache/|save-and-verify-actions-cache/|setup-windows-rocm-sdk/))' || true)
  if echo "$CHANGED_FILES" | grep -Eq '^\.github/workflows/(main|pr)_[a-z]+\.yml$'; then
    WINDOWS_CPU_INPUTS="ci-entry-workflow"
    WINDOWS_GPU_INPUTS="ci-entry-workflow"
  fi
  if [[ -n "$WINDOWS_CPU_INPUTS" ]] || [[ "$BACKEND_RECIPE_CHANGED" == "true" ]]; then
    WINDOWS_CPU_BUILD_REQUIRED="true"
  fi
  if [[ -n "$WINDOWS_GPU_INPUTS" ]] || [[ "$BACKEND_RECIPE_CHANGED" == "true" ]]; then
    WINDOWS_GPU_BUILD_REQUIRED="true"
  fi
fi

# SDK smokes are consumer tests: run for workflow dispatch, direct SDK
# files, or when affected crate analysis reaches the SDK/API crates.
SDK_SMOKE_REQUIRED="false"
if [[ "$EVENT_NAME" == "workflow_dispatch" ]]; then
  SDK_SMOKE_REQUIRED="true"
elif [[ -n "$CHANGED_FILES" ]]; then
  DIRECT_SDK_INPUTS=$(echo "$CHANGED_FILES" | grep -E '(^sdk/|^Package\.swift$|^scripts/ci-(rust|kotlin|swift)-sdk-smoke\.sh$|^scripts/ci-prepare-native-runtime\.sh$|^scripts/ci-sdk-fixture\.sh$|^scripts/(check-sdk-contract|package-sdk-console-assets|restore-native-sdk-input|restore-static-abi-input|verify-sdk-console-assets|verify-swift-privacy-manifest|verify-swift-release-artifact|prepare-llama|build-llama)\.sh$|^scripts/(package-native-sdk|package-native-sdk-crate|verify-native-sdk-package|verify-checksum-sidecar|verify-static-abi-build-stamp|safe-extract-(tar|zip)|verify-swift-xcframework)\.(sh|py)$|^\.github/actions/(compute-changes|prepare-native-sdk-input|prepare-static-abi-input|resolve-native-toolchain-epoch|restore-smoke-inputs)/|^\.github/workflows/(ci|main_[a-z]+|native-sdk-artifact|pr_[a-z]+|release|sdk-smoke|static-abi-artifact|swift-sdk-artifact)\.yml$)' || true)
  if echo "$CHANGED_FILES" | grep -Eq '^\.github/workflows/(main|pr)_[a-z]+\.yml$'; then
    DIRECT_SDK_INPUTS="ci-entry-workflow"
  fi
  if [[ -n "$DIRECT_SDK_INPUTS" ]]; then
    SDK_SMOKE_REQUIRED="true"
  elif echo "$AFFECTED_CRATES" | jq -e 'index("mesh-llm-client") or index("mesh-llm-api-client") or index("mesh-llm-api-server") or index("mesh-llm-config") or index("mesh-llm-console-server") or index("mesh-llm-ffi") or index("mesh-llm-native-runtime") or index("mesh-llm-protocol") or index("mesh-llm-routing") or index("mesh-llm-types")' >/dev/null; then
    SDK_SMOKE_REQUIRED="true"
  fi
fi

# Inference artifacts are needed for runtime-facing changes, SDK smoke
# tests, backend/native inputs, or embedded React console changes. Do
# not build mesh-llm artifacts just because Rust tooling such as xtask
# changed; the quality slice's targeted fmt/Clippy jobs cover those crates.
INFERENCE_ARTIFACT_REQUIRED="false"
if [[ "$ALL_RUST" == "true" ]] || [[ "$UI_CHANGED" == "true" ]] || [[ "$BACKEND_CHANGED" == "true" ]] || [[ "$SDK_SMOKE_REQUIRED" == "true" ]]; then
  INFERENCE_ARTIFACT_REQUIRED="true"
elif echo "$AFFECTED_CRATES" | jq -e 'index("mesh-llm") or index("mesh-llm-host-runtime") or index("mesh-llm-client") or index("openai-frontend") or index("skippy-server") or index("skippy-runtime") or index("model-artifact")' >/dev/null; then
  INFERENCE_ARTIFACT_REQUIRED="true"
fi

LINUX_TEST_GROUPS_JSON='[]'
add_linux_test_group() {
  local group="$1"
  local cache_key="$2"
  LINUX_TEST_GROUPS_JSON=$(jq -c --arg group "$group" --arg cache_key "$cache_key" '. + [{group: $group, cache_key: $cache_key}]' <<<"$LINUX_TEST_GROUPS_JSON")
}

if [[ "$EVENT_NAME" == "workflow_dispatch" ]] || [[ "$ALL_RUST" == "true" ]]; then
  add_linux_test_group protocol linux-tests-protocol
  add_linux_test_group skippy-smoke linux-tests-skippy-smoke
else
  if echo "$AFFECTED_CRATES" | jq -e 'index("mesh-llm") or index("mesh-llm-protocol")' >/dev/null; then
    add_linux_test_group protocol linux-tests-protocol
  fi
  if [[ "$INFERENCE_ARTIFACT_REQUIRED" == "true" ]]; then
    add_linux_test_group skippy-smoke linux-tests-skippy-smoke
  fi
fi

# Set outputs using GITHUB_OUTPUT
{
  echo "changed_files<<EOF"
  cat /tmp/changed_files.txt | grep -v "^__force_all__$" || true
  echo "EOF"
  echo "affected_crates=$AFFECTED_CRATES"
  echo "test_crates=$TEST_CRATES"
  echo "batches_json=$BATCHES"
  echo "test_batches_json=$TEST_BATCHES"
  echo "clippy_batches_json=$CLIPPY_BATCHES"
  echo "all_rust=$ALL_RUST"
  echo "ui_changed=$UI_CHANGED"
  echo "website_changed=$WEBSITE_CHANGED"
  echo "website_docs_changed=$WEBSITE_DOCS_CHANGED"
  echo "cli_surface_changed=$CLI_SURFACE_CHANGED"
  echo "docs_only=$DOCS_ONLY"
  echo "rust_changed=$RUST_CHANGED"
  echo "backend_changed=$BACKEND_CHANGED"
  echo "inference_artifact_required=$INFERENCE_ARTIFACT_REQUIRED"
  echo "backend_recipe_changed=$BACKEND_RECIPE_CHANGED"
  echo "windows_cpu_build_required=$WINDOWS_CPU_BUILD_REQUIRED"
  echo "windows_gpu_build_required=$WINDOWS_GPU_BUILD_REQUIRED"
  echo "sdk_smoke_required=$SDK_SMOKE_REQUIRED"
  echo "runner_contract_required=$RUNNER_CONTRACT_REQUIRED"
  echo "linux_test_groups_json=$LINUX_TEST_GROUPS_JSON"
} >> "$GITHUB_OUTPUT"
