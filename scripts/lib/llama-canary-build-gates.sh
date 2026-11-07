#!/usr/bin/env bash
# Shared controller adapters; typed repair-check owns the native build and smoke plan.

run_for() {
  local label="$1" seconds="$2"
  shift 2
  local transaction_root input executable result timeout_parent
  local automation=()
  if [[ "$HARNESS_MODE" == repair || "$HARNESS_MODE" == verify ]]; then
    repair_workload_controller_unchanged || return 125
    # The controller freezes this array before invoking the adapter.
    # shellcheck disable=SC2154
    automation=("${repair_workload_automation[@]}")
    timeout_parent="${RUNNER_TEMP:-/tmp}"
  else
    automation=("${MESH_LLM_AUTOMATION_BIN:?}")
    timeout_parent="${RUNNER_TEMP:?}"
  fi
  executable="$(command -v "$1")" || return 125
  if [[ "$executable" != /* && "$executable" == */* ]]; then
    executable="$PWD/$executable"
  fi
  if [[ "$executable" != /* ]]; then
    echo "$label requires an absolute executable" >&2
    return 125
  fi
  transaction_root="$(mktemp -d "$timeout_parent/canary-timeout.XXXXXXXX")" || return 125
  input="$transaction_root/input.json"
  shift
  jq -n --arg label "$label" --argjson seconds "$seconds" --arg cwd "$PWD" \
    --arg executable "$executable" --args \
    '{label:$label,seconds:$seconds,cwd:$cwd,executable:$executable,arguments:$ARGS.positional}' \
    -- "$@" > "$input" || { rm -rf "$transaction_root"; return 125; }
  if "${automation[@]}" automation canary-timeout --input "$input"; then
    result=0
  else
    result=$?
  fi
  rm -rf "$transaction_root" || return 125
  if [[ "$HARNESS_MODE" == repair || "$HARNESS_MODE" == verify ]]; then
    repair_workload_controller_unchanged || return 125
  fi
  return "$result"
}

run_prepare() {
  local prepared_upstream
  : > "$PREPARE_LOG"
  echo "trusted candidate gate: prepare" | tee -a "$PREPARE_LOG"
  write_repair_pin >>"$PREPARE_LOG" 2>&1 || return 1
  verify_repair_pin >>"$PREPARE_LOG" 2>&1 || return 1
  run_verification_logged "apply llama.cpp patch queue" "$PREPARE_LOG" \
    scripts/prepare-llama.sh pinned || return 1
  prepared_upstream="$(tr -d '[:space:]' < "$ROOT/.deps/llama.cpp/.mesh-llm-upstream-sha")"
  if [[ "$prepared_upstream" != "$UPSTREAM_SHA" ]]; then
    echo "prepared upstream is $prepared_upstream, expected $UPSTREAM_SHA" | tee -a "$PREPARE_LOG" >&2
    return 1
  fi
}

run_full_build() {
  : > "$BUILD_LOG"
  echo "trusted candidate gate: build" | tee -a "$BUILD_LOG"
  # shellcheck source=scripts/lib/llama-canary-native-build.sh
  source "$TRUSTED_ROOT/scripts/lib/llama-canary-native-build.sh"
  run_canary_native_build || return $?
  if [[ "$HARNESS_MODE" == *-build ]]; then
    # shellcheck disable=SC2016
    run_verification_logged "build transferable multimodal test executable" "$BUILD_LOG" \
      bash -c 'cargo test --locked -p skippy-serving --lib --no-default-features --no-run --message-format=json > "$1"' \
      build-mm "$STATE_DIR/mm-build.jsonl" || return $?
  fi
}

run_certification() {
  if [[ "$HARNESS_MODE" != repair && "$HARNESS_MODE" != verify ]]; then
    echo "local certification requires repair or verify mode" >&2
    return 2
  fi
  local setting workload_settings
  local workload_env=()
  workload_settings="$(bash scripts/skippy-workload-oracles-build.sh --print-env "${LLAMA_STAGE_BUILD_DIR:?}-workloads")" || return 1
  if [[ -z "$workload_settings" ]]; then
    echo "workload producer returned no certification environment" >&2
    return 1
  fi
  while IFS= read -r setting; do
    workload_env+=("$setting")
  done <<< "$workload_settings"
  : > "$CERTIFY_LOG"
  echo "trusted candidate gate: certify" | tee -a "$CERTIFY_LOG"
  if [[ "$HARNESS_MODE" == repair ]]; then
    repair_source_inspection local-parity-inventory "" "$CERTIFY_LOG" || return 1
  elif [[ "$HARNESS_MODE" == verify ]]; then
    verification_source_inspection verification-parity-inventory "$CERTIFY_LOG" || return 1
  fi
  repair_family_plan 1 "$CERTIFY_LOG" || return 1
  run_verification_logged "full supported-family certification" "$CERTIFY_LOG" env \
    FAMILY_BATTERY_RUN_ID="$FAMILY_BATTERY_RUN_ID" \
    "${workload_env[@]}" \
    scripts/skippy-family-battery.sh --skip-build --plan "$PLAN_PATH"
}

run_early_native_certification() {
  local setting workload_settings mm_test_bin
  local workload_env=()
  # A cached executable is usable only when its recorded source tree (and
  # therefore pin), native stamp, and every handed-off binary still match.
  repair_workload_controller_unchanged || return 1
  verification_candidate_unchanged || return 1
  run_verification_logged "verify exact workload producer" "$CERTIFY_LOG" \
    "${repair_workload_automation[@]}" automation canary-receipts workload-manifest verify \
      "$ROOT" "${LLAMA_STAGE_BUILD_DIR:?}-workloads/cargo/debug/skippy" \
      "${LLAMA_STAGE_BUILD_DIR:?}-workloads/native" \
      "${LLAMA_STAGE_BUILD_DIR:?}-workloads/producer.json" || return 1
  verification_candidate_unchanged || return 1
  repair_workload_controller_unchanged || return 1
  workload_settings="$(bash scripts/skippy-workload-oracles-build.sh --print-env "${LLAMA_STAGE_BUILD_DIR:?}-workloads")" || return 1
  [[ -n "$workload_settings" ]] || return 1
  while IFS= read -r setting; do
    workload_env+=("$setting")
  done <<< "$workload_settings"
  mm_test_bin="$(jq -rs '[.[] | select(.reason == "compiler-artifact" and .profile.test == true and .target.name == "skippy_serving" and .executable != null) | .executable] | unique | if length == 1 then .[0] else error("expected one exact multimodal test executable") end' "$STATE_DIR/mm-build.jsonl")" || return 1
  [[ -x "$mm_test_bin" ]] || return 1
  # One exact candidate on the producer backend, with representatives for split parity,
  # recurrent and MoE replay, both encode-only startup classes, T5 ordering,
  # and an actual image response. The full roster remains the promotion gate.
  run_verification_logged "early real-model native certification" "$CERTIFY_LOG" env \
    FAMILY_BATTERY_RUN_ID="${FAMILY_BATTERY_RUN_ID}-early" \
    FAMILY_BATTERY_MM_TEST_BIN="$mm_test_bin" \
    "${workload_env[@]}" \
    scripts/skippy-family-battery.sh --skip-build \
      --families llama,mamba2,deepseek2,nomic-bert-embedding,jina-bert-v2-rerank,t5-encoder-decoder,qwen3-vl
}

run_candidate_gates() {
  local roster_mode="${1:-verify}"
  if [[ "$roster_mode" != "verify" && "$roster_mode" != "refresh" ]]; then
    echo "invalid candidate-gate roster mode: $roster_mode" >&2
    return 2
  fi
  : > "$PREPARE_LOG"
  : > "$MANIFEST_POLICY_LOG"
  : > "$BUILD_LOG"
  : > "$CERTIFY_LOG"
  rm -f "$STATE_DIR/last-check.json"
  run_prepare || return 1
  if [[ "$roster_mode" == "refresh" ]]; then
    # Preparation writes the candidate upstream pin. Generate only after that
    # transition so the roster recipe matches the runtime about to be built.
    # Independent verification uses the default read-only mode below.
    write_split_certification_roster || return 1
  fi
  # The prepared pin supplies GGML layout data. Validate immutable cache and
  # tensor descriptors before compilation; placement uses conservative file sizes.
  repair_family_plan 256 "$CERTIFY_LOG" || return 1
  repair_family_plan_step "$CERTIFY_LOG" "${repair_workload_automation[@]}" \
    automation family-battery-policy --cache-descriptors "$ROOT" \
    "$ROOT/ci/llama-canary/family-certified.json" "$PLAN_PATH" "${HF_CACHE:?}" || return 1
  validate_agent_manifest_changes || return 1
  if [[ "$HARNESS_MODE" == *-build ]]; then
    run_verification_logged "validate family plan before compilation" "$CERTIFY_LOG" \
      "${MESH_LLM_AUTOMATION_BIN:?}" --repo-root "$ROOT" ci family-plan --shard-count 256 \
        --output "$PLAN_PATH" || return 1
  fi
  run_full_build || return $?
  if [[ "$HARNESS_MODE" == *-build ]]; then
    run_early_native_certification || return 1
    controller_parity_inventory
  else
    run_certification
  fi
}
