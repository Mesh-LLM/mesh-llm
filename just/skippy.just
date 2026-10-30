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
    build_dir="${SKIPPY_REWRITER_BUILD_DIR:-.scratch/skippy-stage-rewriter-build}"
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
    cmake -S skippy/scripts/tools/skippy-stage-rewriter -B "$build_dir" -G Ninja \
      -DBUILD_TESTING=ON "${arch_args[@]}" \
      -DLLVM_DIR="$prefix/lib/cmake/llvm" -DClang_DIR="$prefix/lib/cmake/clang"
    cmake --build "$build_dir"

# Run the native C++ fixture against the same explicitly selected build directory.
[unix]
skippy-rewriter-test: skippy-rewriter-build
    #!/usr/bin/env bash
    set -euo pipefail
    build_dir="${SKIPPY_REWRITER_BUILD_DIR:-.scratch/skippy-stage-rewriter-build}"
    ctest --test-dir "$build_dir" --output-on-failure --no-tests=error \
      -R '^skippy-stage-rewriter-fixture-native$' --timeout 120

# Build the skippy benchmark/debug telemetry collector.
[unix]
metrics-server-build:
    just with-lld cargo build -p metrics-server

[windows]
metrics-server-build:
    @just with-lld cargo build -p metrics-server

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

# Build the grammar experiment against immutable baseline and candidate queues.
[unix]
grammar-sampling-build baseline:
    scripts/prepare-llama.sh pinned
    LLAMA_WORKDIR="$PWD/.deps/grammar-baseline-source" LLAMA_UPSTREAM_URL="$PWD/.deps/llama.cpp" "{{ baseline }}/scripts/prepare-llama.sh" pinned
    cmake -S skippy/evals/grammar-sampling -B .deps/grammar-A -G Ninja -DLLAMA_ROOT="$PWD/.deps/grammar-baseline-source"
    cmake --build .deps/grammar-A --parallel 8 --target grammar-sampling
    just grammar-sampling-candidate-build

# Reuse the frozen reference worker; rebuild only the new ABI candidate.
[unix]
grammar-sampling-candidate-build:
    scripts/prepare-llama.sh pinned
    cmake -S skippy/evals/grammar-sampling -B .deps/grammar-C -G Ninja -DLLAMA_ROOT="$PWD/.deps/llama.cpp" -DSKIPPY_EVAL_COMMON_SAMPLER=ON
    cmake --build .deps/grammar-C --parallel 8 --target grammar-sampling grammar-controller grammar-fresh-controller grammar-audit grammar-verification skippy-sampling-history skippy-runtime-events-test skippy-model-fixture-generator test-skippy-verify-checkpoint-retirement test-skippy-sampling-suppress test-grammar-trigger-patterns test-grammar-parser test-grammar-integration

[unix]
grammar-sampling-tests:
    ctest --test-dir .deps/grammar-C/native --output-on-failure --timeout 900 -R '^(skippy_sampling_history_|skippy_runtime_event_native_model_load_production|test-skippy-verify-checkpoint-retirement|test-skippy-sampling-suppress|test-grammar-trigger-patterns|test-grammar-parser|test-grammar-integration)'

# Prove the runtime has no upward dependency on llama-common.
[unix]
grammar-sampling-shared-build:
    cmake -S .deps/llama.cpp -B .deps/grammar-shared -G Ninja -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_STAGE_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DGGML_METAL=OFF -DGGML_ACCELERATE=ON
    cmake --build .deps/grammar-shared --parallel 8 --target llama

# Build only the diagnostic controller; its worker binaries remain frozen.
[unix]
grammar-process-control-build vendor:
    mkdir -p .deps/grammar-process-control
    c++ -std=c++17 -O3 -Wall -Wextra -Werror -I "{{ vendor }}" skippy/evals/grammar-sampling/process_control.cpp -o .deps/grammar-process-control/controller

# Build the opt-in native MTP developer probes against the local runtime.
[unix]
skippy-native-mtp-probes-build:
    bash skippy/evals/native-mtp-parity/build.sh

# Replay every retained and current native Skippy gate without broad upstream tests.
[unix]
skippy-native-full-replay backend="cpu":
    scripts/prepare-llama.sh pinned
    LLAMA_STAGE_BACKEND="{{ backend }}" LLAMA_STAGE_LINK_MODE=static LLAMA_STAGE_BUILD_DIR="$PWD/.deps/llama-build/build-stage-full-replay-static-{{ backend }}" LLAMA_STAGE_FULL_REPLAY=ON LLAMA_STAGE_FORCE_BUILD=1 scripts/build-llama.sh

# Generate a pinned native corpus; use a fresh --out-root to regenerate an existing tier.
bench-corpus tier="smoke" *ARGS="":
    just with-lld cargo run --locked -p trajectory-reader --features corpus-input --bin trajectory-reader -- corpus "{{ tier }}" {{ ARGS }}

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
    cargo xtool automation hf-certify compose-default --input "{{input}}" --output-directory "{{output}}"

# Isolated native generic conversion Jobs facade; explicit submit authority remains required.
generic-conversion-jobs-release-build:
    just with-lld cargo build --release --locked -p skippy-model-package --bin model-package-generic-jobs

# Read a local legacy package cache through the native product owner, with no ABI calls.
[private]
[unix]
[positional-arguments]
skippy-layer-package-inspect *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    just with-lld cargo build --quiet --locked -p skippy-package-builder --features skippy-runtime/dynamic-native-runtime
    exec cargo run --quiet --locked -p skippy-package-builder --features skippy-runtime/dynamic-native-runtime -- inspect-layer-package "$@"

# Parse package references through the source-built product owner.
[private]
[unix]
[positional-arguments]
skippy-package-reference REFERENCE:
    #!/usr/bin/env bash
    set -euo pipefail
    just with-lld cargo build --quiet --locked -p skippy-package-builder --features skippy-runtime/dynamic-native-runtime
    exec cargo run --quiet --locked -p skippy-package-builder --features skippy-runtime/dynamic-native-runtime -- parse-package-reference "$1"

# Resolve exact existing package cache custody without downloading.
[private]
[unix]
[positional-arguments]
skippy-layer-package-cache *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    just with-lld cargo build --quiet --locked -p skippy-package-builder --features skippy-runtime/dynamic-native-runtime
    exec cargo run --quiet --locked -p skippy-package-builder --features skippy-runtime/dynamic-native-runtime -- resolve-layer-package-cache "$@"

# Explicit native package acquisition, using the same source-built product owner.
[private]
[unix]
[positional-arguments]
skippy-layer-package-fetch *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    just with-lld cargo build --quiet --locked -p skippy-package-builder --features skippy-runtime/dynamic-native-runtime
    exec cargo run --quiet --locked -p skippy-package-builder --features skippy-runtime/dynamic-native-runtime -- fetch-layer-package "$@"

# Qualify Decisions HTTP responses from an already running System One model.
[unix]
[positional-arguments]
decisions-smoke *ARGS:
    @just automation-run automation decisions-smoke "$@"



# --- skippy_api recipes ------------------------------------------------------

skippy-llama-api-test:
    #!/usr/bin/env bash
    set -euo pipefail
    build_dir="${SKIPPY_LLAMA_API_BUILD_DIR:-target/skippy-llama-api-build}"
    cmake -S skippy/skippy_api -B "$build_dir" -G Ninja \
      -DCMAKE_BUILD_TYPE=Debug \
      -DSKIPPY_API_BUILD_TESTS=ON \
      -DSKIPPY_LINK_COMPILE_COMMANDS=ON \
      -DSKIPPY_LLAMA_SOURCE_DIR="{{absolute_path(llama_dir)}}"
    cmake --build "$build_dir" --parallel
    ctest --test-dir "$build_dir" --output-on-failure --no-tests=error

skippy-llama-api-bench:
    #!/usr/bin/env bash
    set -euo pipefail
    build_dir="${SKIPPY_LLAMA_API_BENCH_BUILD_DIR:-target/skippy-llama-api-bench}"
    cmake -S skippy/skippy_api -B "$build_dir" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DSKIPPY_API_BUILD_TESTS=OFF \
      -DSKIPPY_API_BUILD_BENCHMARKS=ON \
      -DSKIPPY_LINK_COMPILE_COMMANDS=ON \
      -DSKIPPY_LLAMA_SOURCE_DIR="{{absolute_path(llama_dir)}}"
    cmake --build "$build_dir" --target skippy_api_benchmarks --parallel
    "$build_dir/skippy_api_benchmarks" \
      --benchmark_min_time=0.5s \
      --benchmark_repetitions=5 \
      --benchmark_report_aggregates_only=true

# Same benchmark, but print the old-vs-new speedup table instead of raw rows.
[unix]
[positional-arguments]
skippy-llama-api-ffi-bench-compare model="" *ARGS="":
    #!/usr/bin/env bash
    set -euo pipefail
    model="$1"
    shift
    while [[ $# -gt 0 && -z "$1" ]]; do shift; done
    report="$(mktemp "${TMPDIR:-/tmp}/skippy-ffi-vs-api.XXXXXX")"
    trap 'rm -f "$report"' EXIT
    just skippy-llama-api-ffi-bench "$model" \
      --benchmark_format=json \
      --benchmark_out="$report" \
      "$@" >/dev/null
    just automation-run automation skippy-benchmark-compare "$report"

# Stage planner probe: proves a GGUF is a valid single-shard package and prints
# what planning costs. Optional layer count and [start, end) stage range.
[unix]
[positional-arguments]
skippy-llama-api-planner-probe model *ARGS="":
    #!/usr/bin/env bash
    set -euo pipefail
    build_dir="target/skippy-planner-probe"
    cmake -S skippy/skippy_api -B "$build_dir" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DSKIPPY_API_BUILD_PLANNER_PROBE=ON \
      -DSKIPPY_LLAMA_SOURCE_DIR="{{absolute_path(llama_dir)}}"
    cmake --build "$build_dir" --target skippy_api_planner_probe --parallel
    "$build_dir/skippy_api_planner_probe" "$@"

# Everything a skippy_api/skippy-api-rs checkpoint must pass.
# Models are optional; pass both to include the real-model Rust tests.
[unix]
[positional-arguments]
skippy-api-verify embed_model="" reranker_model="" causal_model="" *ARGS="":
    #!/usr/bin/env bash
    set -euo pipefail
    embed="${1:-}"
    if [[ $# -gt 0 ]]; then shift; fi
    reranker="${1:-}"
    if [[ $# -gt 0 ]]; then shift; fi
    causal="${1:-}"
    if [[ $# -gt 0 ]]; then shift; fi
    printf '%s\n' '== clang-format' >&2
    just skippy-llama-api-format-check
    printf '%s\n' '== clang-tidy' >&2
    just skippy-tidy-plugin-build
    just skippy-llama-api-tidy '*'
    printf '%s\n' '== every target compiles' >&2
    just skippy-api-all-targets-build
    printf '%s\n' '== C++ tests' >&2
    just skippy-llama-api-test
    printf '%s\n' '== Rust fmt + clippy' >&2
    just skippy-api-rs-lint
    printf '%s\n' '== Rust tests' >&2
    just skippy-api-rs-e2e "$embed" "$reranker" "$causal"
    printf '%s\n' '== downstream static link closure' >&2
    just skippy-api-downstream-smoke
    printf '%s\n' 'ok: skippy_api is ready to commit' >&2

# Format and lint the Rust facade with the gates CI applies elsewhere.
[unix]
skippy-api-rs-lint:
    #!/usr/bin/env bash
    set -euo pipefail
    cd skippy/skippy-api-rs
    cargo fmt --all --check
    cargo clippy --workspace --all-targets -- -D warnings

# Verify, stage the API sources, and commit. Message must be a Conventional Commit.
[unix]
[positional-arguments]
skippy-api-checkpoint message="" embed_model="" reranker_model="" causal_model="":
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -z "${1:-}" ]]; then
      printf '%s\n' 'usage: just skippy-api-checkpoint "<conventional commit subject>" [embed.gguf] [reranker.gguf] [causal.gguf]' >&2
      exit 1
    fi
    subject="$1"
    if [[ $# -gt 0 ]]; then shift; fi
    embed="${1:-}"
    if [[ $# -gt 0 ]]; then shift; fi
    reranker="${1:-}"
    if [[ $# -gt 0 ]]; then shift; fi
    causal="${1:-}"
    if ! git diff --cached --quiet -- . \
        ':(exclude)skippy/skippy_api/**' ':(exclude)skippy/skippy-api-rs/**' \
        ':(exclude)just/skippy.just' ':(exclude).gitignore'; then
      printf '%s\n' 'checkpoint refuses staged files outside the API; commit or unstage those files first' >&2
      exit 1
    fi
    just skippy-api-verify "$embed" "$reranker" "$causal"
    git add skippy/skippy_api skippy/skippy-api-rs just/skippy.just .gitignore
    if git diff --cached --quiet; then
      printf '%s\n' 'nothing staged; tree already matches the checkpoint' >&2
      exit 0
    fi
    git status --short
    git commit -m "$subject"
    git --no-pager log -1 --oneline

# End-to-end Rust tests: real GGUF through llama.cpp, no mocks.
# skippy-api-rs-e2e <embed-model.gguf> <reranker-model.gguf> [causal-model.gguf] [cargo test args]
[unix]
[positional-arguments]
skippy-api-rs-e2e embed_model="" reranker_model="" causal_model="" *ARGS="":
    #!/usr/bin/env bash
    set -euo pipefail
    embed="${1:-}"
    if [[ $# -gt 0 ]]; then shift; fi
    reranker="${1:-}"
    if [[ $# -gt 0 ]]; then shift; fi
    causal="${1:-}"
    if [[ $# -gt 0 ]]; then shift; fi
    while [[ $# -gt 0 && -z "$1" ]]; do shift; done
    if [[ -d /opt/homebrew/opt/lld/bin ]]; then
      export PATH="/opt/homebrew/opt/lld/bin:$PATH"
    fi
    cd skippy/skippy-api-rs
    # With all three models supplied, a skipped test is a failure: this is the
    # run CI should use (the causal lane is part of required coverage). Any
    # subset may be passed to cover a single workload locally.
    require=""
    if [[ -n "$embed" && -n "$reranker" && -n "$causal" ]]; then
      require=1
    fi
    # Preserve the repository compiler wrapper for the nested workspace too.
    # All four variables are assigned identically and unconditionally: env is
    # not an input to this recipe, positionals are, and the Rust side reads
    # set-but-empty as absent.
    SKIPPY_TEST_MODEL="$embed" \
    SKIPPY_TEST_RERANKER="$reranker" \
    SKIPPY_TEST_CAUSAL="$causal" \
    SKIPPY_TEST_REQUIRE_MODELS="$require" \
      cargo test "$@"


# Configure and build EVERY optional target. Build only, no runs: an optional
# target is optional to run, never to build (TYPE-RULES §6). Two trees, the
# same split the rest of the recipes use: the llama-backed targets build with
# the default compiler (llama's own sources do not build under Homebrew
# clang's libc++), and the fuzzers build under the clang prefix the fuzz
# recipes use, llama off.
[unix]
skippy-api-downstream-smoke:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo build --locked --manifest-path skippy/skippy-api-rs/tests/downstream/Cargo.toml \
        --target-dir target/skippy-api-downstream
    env -u LD_LIBRARY_PATH -u DYLD_LIBRARY_PATH -u DYLD_FALLBACK_LIBRARY_PATH \
        target/skippy-api-downstream/debug/skippy-api-downstream-smoke

[unix]
skippy-api-all-targets-build:
    #!/usr/bin/env bash
    set -euo pipefail
    cmake -S skippy/skippy_api -B target/skippy-api-all -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DSKIPPY_BUILD_LLAMA=ON \
        -DSKIPPY_LLAMA_SOURCE_DIR="{{absolute_path(llama_dir)}}" \
        -DSKIPPY_API_BUILD_TESTS=ON \
        -DSKIPPY_API_BUILD_BENCHMARKS=ON \
        -DSKIPPY_API_BUILD_PLANNER_PROBE=ON \
        -DSKIPPY_API_BUILD_FFI_BENCHMARKS=ON \
        -DBENCHMARK_ENABLE_WERROR=OFF \
        -DBENCHMARK_ENABLE_TESTING=OFF >/dev/null
    cmake --build target/skippy-api-all --parallel
    llvm="${SKIPPY_LLVM_PREFIX:-/opt/homebrew/opt/llvm}"
    cxx="$llvm/bin/clang++"
    [[ -x "$cxx" ]] || cxx="$(command -v clang++)"
    cmake -S skippy/skippy_api -B target/skippy-api-fuzz -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_CXX_COMPILER="$cxx" \
        -DSKIPPY_BUILD_LLAMA=OFF \
        -DSKIPPY_API_BUILD_FUZZERS=ON >/dev/null
    cmake --build target/skippy-api-fuzz --parallel

# Run one A/B case from the patch manifest (list cases with no argument).
[unix]
[positional-arguments]
skippy-ab case="" model="":
    #!/usr/bin/env bash
    set -euo pipefail
    manifest="skippy/skippy_api/benchmarks/cases.txt"
    if [ -z "{{ case }}" ]; then
        awk -F'\t' '!/^#/ && NF>=5 {printf "%-4s %-2s %-24s %-18s %s\n", $1, $2, $3, $5, $6}' "$manifest"
        exit 0
    fi
    row=$(awk -F'\t' -v want="{{ case }}" '!/^#/ && $3==want {print; exit}' "$manifest")
    if [ -z "$row" ]; then
        echo "no such case: {{ case }}" >&2
        exit 1
    fi
    filter=$(printf '%s' "$row" | cut -f4)
    expected=$(printf '%s' "$row" | cut -f5)
    note=$(printf '%s' "$row" | cut -f6)
    if [ "$expected" = "pending" ]; then
        echo "{{ case }}: pending${note:+ — $note}"
        exit 0
    fi
    if [ "$expected" = "blocked-upstream" ]; then
        echo "{{ case }}: blocked-upstream${note:+ — $note}"
        echo "  work belongs in skippy/llama_cpp/patches, not in skippy_api (see AGENTS.md)"
        exit 0
    fi
    case "$1" in
        batch_build_*|normalize_*|sampling_*) model_required=false ;;
        *) model_required=true ;;
    esac
    if [ "$model_required" = true ] && [ -z "${2-}" ]; then
        echo "{{ case }} needs a model: just skippy-ab {{ case }} <model.gguf>" >&2
        exit 1
    fi
    out=$(mktemp "${TMPDIR:-/tmp}/skippy-ab.XXXXXX")
    log="${out}.log"
    trap 'rm -f "$out"' EXIT
    # The recipe builds what it runs: a benchmark binary nobody else builds is
    # how the stage_plan typo hid for a week (TYPE-RULES §6).
    build_dir="${SKIPPY_LLAMA_API_FFI_BENCH_BUILD_DIR:-target/skippy-llama-api-ffi-bench}"
    cmake -S skippy/skippy_api -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DSKIPPY_BUILD_LLAMA=ON \
        -DSKIPPY_LLAMA_SOURCE_DIR="{{absolute_path(llama_dir)}}" \
        -DSKIPPY_API_BUILD_BENCHMARKS=ON -DSKIPPY_API_BUILD_FFI_BENCHMARKS=ON \
        -DBENCHMARK_ENABLE_WERROR=OFF -DBENCHMARK_ENABLE_TESTING=OFF >/dev/null
    cmake --build "$build_dir" --target skippy_api_ffi_benchmarks --parallel
    if ! SKIPPY_BENCH_MODEL="${2-}" \
        "$build_dir/skippy_api_ffi_benchmarks" \
        --benchmark_filter="$filter" \
        --benchmark_min_time=0.3s --benchmark_repetitions=15 \
        --benchmark_enable_random_interleaving=true \
        --benchmark_format=json --benchmark_out="$out" >"$log" 2>&1; then
        echo "{{ case }}: benchmark exited non-zero (log: $log); last output:" >&2
        tail -20 "$log" >&2
        exit 1
    fi
    # A filter that matches no registered benchmark leaves the report empty
    # with exit 0 (google benchmark writes nothing in that case), so check
    # before the comparison script sees it.
    if [ ! -s "$out" ]; then
        echo "{{ case }}: filter '$filter' matched no benchmark; report is empty" >&2
        echo "  registered names ending in /real_time come from ->UseRealTime();" >&2
        echo "  manifest filters must prefix-match, never \$-anchor" >&2
        exit 1
    fi
    just automation-run automation skippy-benchmark-compare "$out" | sed "s/^/  /"
    echo "{{ case }}: acceptance=$expected (benchmark output: $log)"

# Run the old C ABI against the skippy_api facade on one model (model=<gguf>).
[unix]
[positional-arguments]
skippy-llama-api-ffi-bench model="" *ARGS="":
    #!/usr/bin/env bash
    set -euo pipefail
    # Positional, not interpolated: a filter such as 'Embed|Tokenize' must
    # reach the binary as one argument instead of becoming a shell pipeline.
    model="$1"
    shift
    # just passes an empty placeholder for unset trailing parameters; an empty
    # argument would reach google benchmark as an unknown flag.
    while [[ $# -gt 0 && -z "$1" ]]; do shift; done
    build_dir="${SKIPPY_LLAMA_API_FFI_BENCH_BUILD_DIR:-target/skippy-llama-api-ffi-bench}"
    llama_source="{{absolute_path(llama_dir)}}"
    if [[ ! -f "$llama_source/include/skippy/workloads.h" ]]; then
      scripts/prepare-llama.sh pinned
    fi
    cmake -S skippy/skippy_api -B "$build_dir" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DSKIPPY_API_BUILD_BENCHMARKS=ON \
      -DSKIPPY_API_BUILD_FFI_BENCHMARKS=ON \
      -DSKIPPY_LLAMA_SOURCE_DIR="$llama_source"
    cmake --build "$build_dir" --target skippy_api_ffi_benchmarks --parallel
    if [[ -z "$model" ]]; then
      printf '%s\n' 'no model= given: only BatchBuild and Normalize will run' >&2
    fi
    # Interleaved repetitions: running every FFI case before every API case
    # lets GPU clocks drift and fakes a few percent of difference.
    SKIPPY_BENCH_MODEL="$model" "$build_dir/skippy_api_ffi_benchmarks" \
      --benchmark_min_time=0.5s \
      --benchmark_repetitions=5 \
      --benchmark_enable_random_interleaving=true \
      "$@"

skippy-llama-api-compdb:
    #!/usr/bin/env bash
    set -euo pipefail
    build_dir="${SKIPPY_LLAMA_API_BUILD_DIR:-target/skippy-llama-api-build}"
    cmake -S skippy/skippy_api -B "$build_dir" -G Ninja \
      -DCMAKE_BUILD_TYPE=Debug \
      -DSKIPPY_API_BUILD_TESTS=ON \
      -DSKIPPY_LLAMA_SOURCE_DIR="{{absolute_path(llama_dir)}}"

skippy-llama-api-format:
    #!/usr/bin/env bash
    set -euo pipefail
    build_dir="${SKIPPY_LLAMA_API_BUILD_DIR:-target/skippy-llama-api-build}"
    cmake -S skippy/skippy_api -B "$build_dir" -G Ninja \
      -DCMAKE_BUILD_TYPE=Debug \
      -DSKIPPY_API_BUILD_TESTS=ON \
      -DSKIPPY_LLAMA_SOURCE_DIR="{{absolute_path(llama_dir)}}"
    cmake --build "$build_dir" --target format

skippy-llama-api-format-check:
    #!/usr/bin/env bash
    set -euo pipefail
    build_dir="${SKIPPY_LLAMA_API_BUILD_DIR:-target/skippy-llama-api-build}"
    cmake -S skippy/skippy_api -B "$build_dir" -G Ninja \
      -DCMAKE_BUILD_TYPE=Debug \
      -DSKIPPY_API_BUILD_TESTS=ON \
      -DSKIPPY_LLAMA_SOURCE_DIR="{{absolute_path(llama_dir)}}"
    cmake --build "$build_dir" --target format-check

# Build the out-of-tree clang-tidy module in tidy/. The plugin must match the
# LLVM that owns the clang-tidy binary, so both come from the same prefix.
[unix]
skippy-tidy-plugin-build:
    #!/usr/bin/env bash
    set -euo pipefail
    llvm="${SKIPPY_LLVM_PREFIX:-/opt/homebrew/opt/llvm}"
    if [[ ! -x "$llvm/bin/clang-tidy" && -x "$(command -v clang-tidy)" ]]; then
      llvm="$(dirname "$(dirname "$(command -v clang-tidy)")")"
    fi
    cmake -S skippy/skippy_api/tidy -B target/skippy-tidy -G Ninja \
      -DClang_DIR="$llvm/lib/cmake/clang" \
      -DCMAKE_CXX_COMPILER="$llvm/bin/clang++" \
      -DCMAKE_BUILD_TYPE=Release
    cmake --build target/skippy-tidy --parallel
    ctest --test-dir target/skippy-tidy --output-on-failure

# Run the tidy module's fixture suite (one FileCheck test per check).
[unix]
skippy-tidy-fixtures:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ ! -f target/skippy-tidy/SkippyTidyChecks.so ]]; then
      just skippy-tidy-plugin-build
    fi
    ctest --test-dir target/skippy-tidy --output-on-failure

# Fails on any finding. Pass check names to narrow the gate, e.g. `bugprone-*`.
# Boundary scope comes from the per-directory .clang-tidy files
# (docs/ENFORCEMENT.md §5): src/llama and tools carry the llama boundary
# checks, the bridge carries the Rust boundary checks, the core carries the
# migration checks. The plugin is loaded whenever it is built.
[positional-arguments]
skippy-llama-api-tidy *CHECKS="":
    #!/usr/bin/env bash
    set -euo pipefail
    repo_root="$(pwd)"
    build_dir="${SKIPPY_LLAMA_API_TIDY_BUILD_DIR:-target/skippy-llama-api-tidy}"
    case "$build_dir" in
      /*) build_path="$build_dir" ;;
      *) build_path="$repo_root/$build_dir" ;;
    esac
    tidy="${SKIPPY_CLANG_TIDY:-$(command -v clang-tidy || true)}"
    if [[ -z "$tidy" && -x /opt/homebrew/opt/llvm/bin/clang-tidy ]]; then
      tidy=/opt/homebrew/opt/llvm/bin/clang-tidy
    fi
    if [[ -z "$tidy" ]]; then
      printf '%s\n' 'clang-tidy not found; set SKIPPY_CLANG_TIDY to its path' >&2
      exit 1
    fi
    plugin="$repo_root/target/skippy-tidy/SkippyTidyChecks.so"
    if [[ ! -f "$plugin" ]]; then
      printf '%s\n' "plugin missing: $plugin (run: just skippy-tidy-plugin-build)" >&2
      exit 1
    fi
    # dlopen failure is not fatal to clang-tidy: the run would go ahead with
    # none of the skippy checks loaded and the gate would pass on nothing.
    # Prove the plugin is live in THIS binary before trusting the run.
    if ! "$tidy" --load="$plugin" -checks='-*,skippy-no-mutex' -list-checks 2>/dev/null \
      | grep -q 'skippy-no-mutex'; then
      printf '%s\n' "plugin failed to load in $tidy: $plugin" >&2
      exit 1
    fi
    load_args=(--load="$plugin")
    checks="${1:-}"
    if [[ $# -gt 0 ]]; then shift; fi
    # '*' on the command line appends after the config's WarningsAsErrors and
    # would re-promote the demoted debt checks; the gate run ('*') lets the
    # .clang-tidy set govern, an explicit check name still upgrades just it.
    wae_args=(--warnings-as-errors="${checks:-*}")
    if [[ "${checks:-*}" == "*" ]]; then wae_args=(); fi
    # Configure only: the compile database is written by CMake, and the direct
    # run below is the gate. Building through CXX_CLANG_TIDY would also
    # escalate newer-clang diagnostics in the tests, which is not this gate.
    cmake -S skippy/skippy_api -B "$build_dir" -G Ninja \
      -DCMAKE_BUILD_TYPE=Debug \
      -DSKIPPY_API_BUILD_TESTS=OFF \
      -DSKIPPY_API_BUILD_BENCHMARKS=OFF \
      -DSKIPPY_API_BUILD_PLANNER_PROBE=ON \
      -DSKIPPY_LLAMA_SOURCE_DIR="{{absolute_path(llama_dir)}}"
    cd skippy/skippy_api
    "$tidy" -p "$build_path" ${wae_args[@]+"${wae_args[@]}"} \
      "${load_args[@]+"${load_args[@]}"}" \
      $(find src tools fuzz -name '*.cpp' | sort)
    cd "$repo_root"
    # The bridge has no compile database (cxx_build compiles it); the flags
    # below mirror build.rs. Generated headers stay outside the filter.
    cxx_include="$(find "${CARGO_HOME:-$HOME/.cargo}/registry/src" -maxdepth 4 -path '*/cxx-*/include' -type d | sort | tail -1)"
    "$tidy" ${wae_args[@]+"${wae_args[@]}"} \
      "${load_args[@]+"${load_args[@]}"}" \
      skippy/skippy-api-rs/skippy-sys/cpp/bridge.cpp -- \
      -std=c++2b \
      -I skippy/skippy-api-rs \
      -I skippy/skippy_api/include \
      -I .deps/llama.cpp/include \
      -I .deps/llama.cpp/ggml/include \
      -I skippy/skippy-api-rs/target/cxxbridge \
      -I "$cxx_include"
    # A NOLINT carries a reason (docs/ENFORCEMENT.md §1); a bare one is a
    # rule violation.
    bare=$(grep -rn 'NOLINT(' skippy/skippy_api/include skippy/skippy_api/src skippy/skippy_api/tools skippy/skippy-api-rs/skippy-sys/cpp --include='*.h' --include='*.cpp' | grep -v ': ' || true)
    if [[ -n "$bare" ]]; then
      printf '%s\n' 'bare NOLINT (a NOLINT carries a reason):' "$bare" >&2
      exit 1
    fi

# just skippy-llama-api-sanitize                 -> address;undefined
# just skippy-llama-api-sanitize thread
# just skippy-llama-api-sanitize "memory"        (Clang only)
skippy-llama-api-sanitize sanitizers="address;undefined":
    #!/usr/bin/env bash
    set -euo pipefail
    sanitizers='{{sanitizers}}'
    build_dir="target/skippy-llama-api-san-${sanitizers//;/-}"
    cmake -S skippy/skippy_api -B "$build_dir" -G Ninja \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DSKIPPY_API_BUILD_TESTS=ON \
      -DSKIPPY_SANITIZERS="$sanitizers" \
      -DSKIPPY_LINK_COMPILE_COMMANDS=ON \
      -DSKIPPY_LLAMA_SOURCE_DIR="{{absolute_path(llama_dir)}}"
    cmake --build "$build_dir" --parallel
    export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=1:abort_on_error=1}"
    export UBSAN_OPTIONS="${UBSAN_OPTIONS:-print_stacktrace=1:halt_on_error=1}"
    export TSAN_OPTIONS="${TSAN_OPTIONS:-halt_on_error=1}"
    ctest --test-dir "$build_dir" --output-on-failure --no-tests=error

# Build and run the skippy_api suite under AddressSanitizer + UndefinedBehaviorSanitizer.
# Fuzz every target for <seconds> each. A crash is minimised into the corpus
# (checked in with the fix that answers it) and the run fails.
[unix]
skippy-api-fuzz seconds="60":
    #!/usr/bin/env bash
    set -euo pipefail
    llvm="${SKIPPY_LLVM_PREFIX:-/opt/homebrew/opt/llvm}"
    cxx="$llvm/bin/clang++"
    [[ -x "$cxx" ]] || cxx="$(command -v clang++)"
    cmake -S skippy/skippy_api -B target/skippy-api-fuzz -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_CXX_COMPILER="$cxx" \
        -DSKIPPY_BUILD_LLAMA=OFF \
        -DSKIPPY_API_BUILD_FUZZERS=ON >/dev/null
    cmake --build target/skippy-api-fuzz --parallel
    for fuzz_target in fuzz_inventory fuzz_arena_slice fuzz_stage_plan_chain; do
        corpus="skippy/skippy_api/fuzz/corpus/$fuzz_target"
        artifacts="target/skippy-api-fuzz/artifacts/$fuzz_target"
        mkdir -p "$corpus" "$artifacts"
        if ! target/skippy-api-fuzz/$fuzz_target "$corpus" -max_total_time={{ seconds }} \
                -max_len=4096 -artifact_prefix="$artifacts/"; then
            crash="$(ls -t "$artifacts" | head -1)"
            target/skippy-api-fuzz/$fuzz_target -minimize_crash=1 \
                -exact_artifact_path="$corpus/minimised-from-$crash" "$artifacts/$crash" || true
            printf '%s\n' "$fuzz_target crashed; minimised seed added to $corpus" >&2
            exit 1
        fi
    done

# A separate build tree: sanitizer flags do not leak into other targets.
[unix]
skippy-api-asan:
    #!/bin/bash
    set -euo pipefail
    cmake -S skippy/skippy_api -B target/skippy-api-asan -G Ninja         -DCMAKE_BUILD_TYPE=Debug         -DSKIPPY_BUILD_LLAMA=OFF         -DSKIPPY_API_BUILD_TESTS=ON         -DSKIPPY_SANITIZERS="address;undefined" >/dev/null
    cmake --build target/skippy-api-asan --target skippy_api_tests --parallel
    ctest --test-dir target/skippy-api-asan --output-on-failure
    # The fuzzers are part of the regression gate: build them and replay every
    # corpus seed deterministically. A missing clang is a hard error, not a
    # silent skip. The property tests run in the ctest suite above.
    llvm="${SKIPPY_LLVM_PREFIX:-/opt/homebrew/opt/llvm}"
    cxx="$llvm/bin/clang++"
    if [[ ! -x "$cxx" ]]; then
        printf '%s\n' "skippy-api-asan: $cxx not found; cannot replay the fuzz corpus" >&2
        exit 1
    fi
    cmake -S skippy/skippy_api -B target/skippy-api-fuzz -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_CXX_COMPILER="$cxx" \
        -DSKIPPY_BUILD_LLAMA=OFF \
        -DSKIPPY_API_BUILD_FUZZERS=ON >/dev/null
    cmake --build target/skippy-api-fuzz --parallel
    for fuzz_target in fuzz_inventory fuzz_arena_slice fuzz_stage_plan_chain; do
        target/skippy-api-fuzz/$fuzz_target -runs=0 skippy/skippy_api/fuzz/corpus/$fuzz_target
    done
