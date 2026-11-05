#!/usr/bin/env bash
# Platform adapter for the trusted canary's native build and generated patch gate.

run_canary_native_build() {
  local platform archive arches jinja_python
  local native_shell=(bash)
  local native_build=(scripts/build-llama.sh --build-only)
  jinja_python="${SKIPPY_WORKLOAD_SDK_PYTHON:-python3}"
  platform="$(uname -s)"
  case "$platform" in
    Darwin)
      native_shell=(arch -arm64 bash)
      native_build+=(-DCMAKE_OSX_ARCHITECTURES=arm64 -DGGML_METAL_EMBED_LIBRARY=ON)
      ;;
    Linux) ;;
    *)
      echo "unsupported canary native build platform: $platform" | tee -a "$BUILD_LOG" >&2
      return 1
      ;;
  esac
  run_verification_logged "complete patched llama.cpp build" "$BUILD_LOG" env \
    LLAMA_STAGE_UPSTREAM_TESTS=ON UV_PROJECT_ENVIRONMENT="$STATE_DIR/jinja-env" \
    uv run --locked --no-python-downloads --python "$jinja_python" --project "$JINJA_PROJECT" -- \
    "${native_shell[@]}" "${native_build[@]}" || return 1
  if [[ "$platform" == Darwin ]]; then
    archive="$LLAMA_STAGE_BUILD_DIR/src/libllama.a"
    arches="$(lipo -archs "$archive" 2>/dev/null || true)"
    if [[ "$arches" != arm64 ]]; then
      echo "candidate native archive must be arm64, got: ${arches:-missing}" | tee -a "$BUILD_LOG" >&2
      return 1
    fi
  fi
  run_verification_logged "generated model-family patch check" "$BUILD_LOG" \
    env UV_PROJECT_ENVIRONMENT="$STATE_DIR/jinja-env" \
    uv run --locked --no-python-downloads --python "$jinja_python" --project "$JINJA_PROJECT" -- \
    "${native_shell[@]}" scripts/check-skippy-generated-family-patch.sh --full-native-suite || return 1
}
