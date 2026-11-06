#!/usr/bin/env bash
# Controller adapter for the shared local/CI repair-check command.
run_canary_native_build() {
  local seconds
  local automation=("${MESH_LLM_AUTOMATION_BIN:-${repair_workload_controller:-}}")
  [[ -n "${automation[0]}" ]] || return 125
  seconds="$(remaining_verification_seconds)" || return 124
  run_verification_logged "shared full upgrade checks" "$BUILD_LOG" \
    "${automation[@]}" automation canary-receipts repair-check \
      --root "$ROOT" --controller-root "$TRUSTED_ROOT" \
      --state-dir "$STATE_DIR" --build-dir "$LLAMA_STAGE_BUILD_DIR" \
      --backend "${LLAMA_STAGE_BACKEND:?}" --phase full --seconds "$seconds"
}
