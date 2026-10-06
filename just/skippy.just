# Build the complete standalone Skippy product for local iteration.
[unix]
skippy backend="" cuda_arch="" rocm_arch="":
    @skippy/scripts/build-development-product.sh --backend "{{ backend }}" --cuda-arch "{{ cuda_arch }}" --rocm-arch "{{ rocm_arch }}"

[windows]
skippy backend="" cuda_arch="" rocm_arch="":
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1 -Backend "{{ backend }}" -CudaArch "{{ cuda_arch }}" -RocmArch "{{ rocm_arch }}" -SkippyOnly

# Build only the standalone server host; supply a separately packaged native
# runtime through --runtime-bundle when running a serving command.
[unix]
skippy-cli-build:
    cargo build --locked -p skippy-cli --features dynamic-native-runtime

[windows]
skippy-cli-build:
    @cargo build --locked -p skippy-cli --bin skippy --features dynamic-native-runtime

[unix]
skippy-cli-release-build:
    cargo build --release --locked -p skippy-cli --features dynamic-native-runtime

[windows]
skippy-cli-release-build:
    @cargo build --release --locked -p skippy-cli --bin skippy --features dynamic-native-runtime

# Build the complete Skippy release product for the current platform.
[unix]
skippy-release backend="" target="":
    @just release-runtime-build "{{ backend }}" "{{ target }}"
    @just skippy-cli-release-build

[windows]
skippy-release backend="":
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1 -Backend "{{ backend }}" -BuildProfile release -SkippyOnly

# Build the native rewriter fixture with an explicitly selected pinned LLVM.
[unix]
skippy-rewriter-build:
    #!/usr/bin/env bash
    set -euo pipefail
    prefix="${SKIPPY_REWRITER_LLVM_PREFIX:-}"
    if [[ -z "$prefix" || ! -x "$prefix/bin/clang" || ! -f "$prefix/lib/cmake/llvm/LLVMConfig.cmake" || ! -f "$prefix/lib/cmake/clang/ClangConfig.cmake" ]]; then
      printf '%s\n' 'SKIPPY_REWRITER_LLVM_PREFIX must name an existing pinned LLVM with clang and LLVM/Clang CMake packages' >&2
      exit 1
    fi
    if ! command -v ninja >/dev/null 2>&1; then
      printf '%s\n' 'Ninja is required for skippy-rewriter-build' >&2
      exit 1
    fi
    arch_args=()
    if [[ "$(uname -s)" == Darwin ]]; then
      native_arch="$(uname -m)"
      if [[ "$native_arch" == x86_64 && "$(sysctl -n hw.optional.arm64 2>/dev/null || true)" == 1 ]]; then
        exec arch -arm64 just skippy-rewriter-build
      fi
      arch_args=(-DCMAKE_OSX_ARCHITECTURES="$native_arch")
    fi
    cmake -S skippy/scripts/tools/skippy-stage-rewriter -B .scratch/skippy-stage-rewriter-build -G Ninja \
      -DBUILD_TESTING=ON "${arch_args[@]}" \
      -DLLVM_DIR="$prefix/lib/cmake/llvm" -DClang_DIR="$prefix/lib/cmake/clang"
    cmake --build .scratch/skippy-stage-rewriter-build

# Build the skippy benchmark/debug telemetry collector.
[unix]
metrics-server-build:
    just with-lld cargo build -p metrics-server

[windows]
metrics-server-build:
    @just with-lld cargo build -p metrics-server

# Build the binaries copied into the Skippy WAN Docker lab image.
[linux]
skippy-wan-lab-build-bins:
    just release-runtime-build cpu
    just skippy-cli-release-build
    python3 scripts/verify-host-dependencies.py target/release/skippy --report dist/skippy-host-imports.json
    just with-lld cargo build --release --locked -p metrics-server

# Build the resumable GGUF conversion/quantization replacement CLI.
[unix]
skippy-quantize-build:
    just with-lld cargo build -p skippy-quantize

[windows]
skippy-quantize-build:
    @just with-lld cargo build -p skippy-quantize

# Build the release binary used in HF conversion/quantization job images.
[unix]
skippy-quantize-release-build:
    just with-lld cargo build --release --locked -p skippy-quantize

[windows]
skippy-quantize-release-build:
    @just with-lld cargo build --release --locked -p skippy-quantize

# Build skippy-quantize as a standalone quantization binary with the pinned
# Skippy native ABI linked into the executable.
[unix]
skippy-quantize-standalone-build backend="cpu":
    scripts/prepare-llama.sh pinned
    LLAMA_STAGE_BACKEND="{{ backend }}" LLAMA_STAGE_LINK_MODE=static scripts/build-llama.sh
    LLAMA_STAGE_BACKEND="{{ backend }}" LLAMA_STAGE_LINK_MODE=static just with-lld cargo build -p skippy-quantize --no-default-features

[unix]
skippy-quantize-standalone-release-build backend="cpu":
    scripts/prepare-llama.sh pinned
    LLAMA_STAGE_BACKEND="{{ backend }}" LLAMA_STAGE_LINK_MODE=static scripts/build-llama.sh
    LLAMA_STAGE_BACKEND="{{ backend }}" LLAMA_STAGE_LINK_MODE=static just with-lld cargo build --release --locked -p skippy-quantize --no-default-features

# Build isolated CPU workload certification producers (not a shipping runtime).
[unix]
skippy-workload-oracles-build output:
    bash scripts/skippy-workload-oracles-build.sh "{{ output }}"

# Build and execute the opt-in private native Skippy application probes.
[unix]
skippy-native-tests backend="cpu":
    scripts/prepare-llama.sh pinned
    LLAMA_STAGE_BACKEND="{{ backend }}" LLAMA_STAGE_LINK_MODE=static LLAMA_STAGE_BUILD_TESTS=ON scripts/build-llama.sh

# Replay every retained and current native Skippy gate without broad upstream tests.
[unix]
skippy-native-full-replay backend="cpu":
    scripts/prepare-llama.sh pinned
    LLAMA_STAGE_BACKEND="{{ backend }}" LLAMA_STAGE_LINK_MODE=static LLAMA_STAGE_BUILD_DIR="$PWD/.deps/llama-build/build-stage-full-replay-static-{{ backend }}" LLAMA_STAGE_FULL_REPLAY=ON LLAMA_STAGE_FORCE_BUILD=1 scripts/build-llama.sh

# Generate a reproducible benchmark corpus for skippy bench tooling.
bench-corpus tier="smoke" *ARGS="":
    uv run --offline --locked --no-sync --no-python-downloads --project "{{justfile_directory()}}/ci/agentic-replay-nightly" python -I "{{justfile_directory()}}/scripts/generate-bench-corpus.py" "{{ tier }}" {{ ARGS }}

# Run skippy family certification checks.
family-certify *ARGS:
    just with-lld scripts/family-certify.sh {{ ARGS }}

# Run target/draft speculative compatibility checks.
spec-bench target draft *ARGS:
    just with-lld env LLAMA_STAGE_BUILD_DIR=".deps/llama-build/build-stage-abi-static" cargo build -p llama-spec-bench
    LLAMA_STAGE_BUILD_DIR=".deps/llama-build/build-stage-abi-static" target/debug/llama-spec-bench --target-model-path "{{ target }}" --draft-model-path "{{ draft }}" {{ ARGS }}

# Smoke a standalone skippy OpenAI frontend stage.
skippy-openai-smoke *ARGS:
    just with-lld scripts/skippy-openai-smoke.sh {{ ARGS }}

# Build the OpenAI benchmark server against an explicitly selected native ABI.
# Nightly runners provide LLAMA_STAGE_BUILD_DIR from their pre-baked image.
competitive-benchmark-build:
    just with-lld cargo build --release --locked -p skippy-cli

# Run the skippy benchmark/debug telemetry collector.
metrics-server db="/tmp/mesh-metrics.duckdb" http_addr="127.0.0.1:18080" otlp_addr="127.0.0.1:14317" *ARGS="": metrics-server-build
    target/debug/metrics-server serve --db "{{ db }}" --http-addr "{{ http_addr }}" --otlp-grpc-addr "{{ otlp_addr }}" {{ ARGS }}

# Native publisher helper for the isolated layer-package job.
snapshot-promoter-release-build:
    just with-lld cargo build --release --locked -p skippy-model-package --bin promote-layer-package-snapshot

# Isolated supplied-artifact HF publisher; never linked into xtask.
model-publisher-release-build:
    just with-lld cargo build --release --locked -p skippy-model-package --bin model-package-publish

# Read-only HF source metadata/package-root helper for the isolated job.
layer-job-helper-release-build:
    just with-lld cargo build --release --locked -p skippy-model-package --bin model-package-layer-job

# Native immutable acquisition/export; supplied semantic references and derived pins required.
competitive-inputs-helper-build:
    just with-lld cargo build --locked -p skippy-model-package --bin model-package-competitive-inputs

competitive-inputs-prefetch request helper helper_sha256 reader reader_sha256 evidence timeout="3600":
    cargo xtool automation replay-matrix competitive-inputs-prefetch --request "{{ request }}" --helper "{{ helper }}" --helper-sha256 "{{ helper_sha256 }}" --evidence-directory "{{ evidence }}" --timeout-seconds "{{ timeout }}" --reader "{{ reader }}" --reader-sha256 "{{ reader_sha256 }}"

# Private finite full-chain positive; xtask must be freshly built from this same source.
competitive-inputs-prefetch-fixture xtask:
    COMPETITIVE_FIXTURE_XTASK_BIN="{{ xtask }}" just with-lld cargo test --locked -p skippy-model-package --lib competitive_acquisition::full_frontend::actual_full_native_prefetch_acquires_four_families_exports_and_generates_manifest_and_readerless_skips -- --exact --ignored --nocapture

# Native pinned checkpoint staging helper; no model/native build is included.
mtp-checkpoint-helper-build:
    just with-lld cargo build --locked -p skippy-model-package --bin model-package-mtp-checkpoint

# Typed immutable checkpoint/bootstrap/native composition/publication owner.
hf-skippy-mtp-compose-default input output:
    cargo xtool automation hf-certify compose-default --input {{input}} --output-directory {{output}}

# Isolated native generic conversion Jobs facade; explicit submit authority remains required.
generic-conversion-jobs-release-build:
    just with-lld cargo build --release --locked -p skippy-model-package --bin model-package-generic-jobs
