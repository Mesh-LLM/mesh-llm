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

# Build the binaries copied into the Skippy WAN Docker lab image.
[linux]
skippy-wan-lab-build-bins:
    just release-runtime-build cpu
    just skippy-cli-release-build
    cargo xtool native verify-host-dependencies target/release/skippy --report dist/skippy-host-imports.json
    just with-lld cargo build --release --locked -p metrics-server
    just with-lld cargo build --release --locked -p xtask --bin xtask

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
    report="$(mktemp -t skippy-ffi-vs-api)"
    trap 'rm -f "$report"' EXIT
    just skippy-llama-api-ffi-bench "$model" \
      --benchmark_format=json \
      --benchmark_out="$report" \
      "$@" >/dev/null
    python3 skippy/skippy_api/benchmarks/compare_benchmarks.py "$report"

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
skippy-api-verify embed_model="" reranker_model="" *ARGS="":
    #!/usr/bin/env bash
    set -euo pipefail
    embed="$1"
    if [[ $# -gt 0 ]]; then shift; fi
    reranker="$1"
    if [[ $# -gt 0 ]]; then shift; fi
    printf '%s\n' '== clang-format' >&2
    just skippy-llama-api-format-check
    printf '%s\n' '== clang-tidy' >&2
    just skippy-llama-api-tidy '*'
    printf '%s\n' '== C++ tests' >&2
    just skippy-llama-api-test
    printf '%s\n' '== Rust tests' >&2
    just skippy-api-rs-e2e "$embed" "$reranker"
    printf '%s\n' 'ok: skippy_api is ready to commit' >&2

# Verify, stage the API sources, and commit. Message must be a Conventional Commit.
[unix]
[positional-arguments]
skippy-api-checkpoint message="" embed_model="" reranker_model="":
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -z "${1:-}" ]]; then
      printf '%s\n' 'usage: just skippy-api-checkpoint "<conventional commit subject>" [embed.gguf] [reranker.gguf]' >&2
      exit 1
    fi
    subject="$1"
    if [[ $# -gt 0 ]]; then shift; fi
    embed="${1:-}"
    if [[ $# -gt 0 ]]; then shift; fi
    reranker="${1:-}"
    just skippy-api-verify "$embed" "$reranker"
    git add skippy/skippy_api skippy/skippy-api-rs just/skippy.just .gitignore
    if git diff --cached --quiet; then
      printf '%s\n' 'nothing staged; tree already matches the checkpoint' >&2
      exit 0
    fi
    git status --short
    git commit -m "$subject"
    git --no-pager log -1 --oneline

# Every-frontier split verification: every stage boundary of a model, with
# the terminal activation frame compared against the whole-model reference.
# Cheap on small models (seconds); intended for pin bumps and model_support
# patches, where frontier-local regressions historically slip past the canary.
skippy-frontier-matrix model="" *ARGS="":
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -z "${1:-}" ]]; then
      printf '%s\n' 'usage: just skippy-frontier-matrix <model.gguf> [--prompt TEXT] [--tolerance F] [--three-stage-stride N] [--json]' >&2
      exit 1
    fi
    model="$1"; shift || true
    cargo run --locked -p skippy --example every_frontier -- "$model" "$@"

# End-to-end Rust tests: real GGUF through llama.cpp, no mocks.
# skippy-api-rs-e2e <embed-model.gguf> <reranker-model.gguf> [cargo test args]
[unix]
[positional-arguments]
skippy-api-rs-e2e embed_model="" reranker_model="" *ARGS="":
    #!/usr/bin/env bash
    set -euo pipefail
    embed="$1"
    if [[ $# -gt 0 ]]; then shift; fi
    reranker="$1"
    if [[ $# -gt 0 ]]; then shift; fi
    while [[ $# -gt 0 && -z "$1" ]]; do shift; done
    if [[ -d /opt/homebrew/opt/lld/bin ]]; then
      export PATH="/opt/homebrew/opt/lld/bin:$PATH"
    fi
    cd skippy/skippy-api-rs
    # With both models supplied, a skipped test is a failure: this is the run CI
    # should use. One model may be passed to cover a single workload locally.
    if [[ -n "$embed" && -n "$reranker" ]]; then
      export SKIPPY_TEST_REQUIRE_MODELS=1
    fi
    # The workspace config points rustc at sccache; these tests link the native
    # library through the crate's own build script, so run rustc directly.
    RUSTC_WRAPPER= \
    SKIPPY_TEST_MODEL="$embed" \
    SKIPPY_TEST_RERANKER="$reranker" \
      cargo test "$@"

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
    if [ -z "{{ model }}" ]; then
        echo "{{ case }} needs a model: just skippy-ab {{ case }} <model.gguf>" >&2
        exit 1
    fi
    out=$(mktemp -t skippy-ab).json
    SKIPPY_BENCH_MODEL={{ model }} \
        ./target/skippy-llama-api-ffi-bench/skippy_api_ffi_benchmarks \
        --benchmark_filter="$filter" \
        --benchmark_min_time=0.3s --benchmark_repetitions=5 \
        --benchmark_enable_random_interleaving=true \
        --benchmark_format=json --benchmark_out="$out" >/dev/null 2>&1
    python3 skippy/skippy_api/benchmarks/compare_benchmarks.py "$out" | sed "s/^/  /"
    echo "{{ case }}: acceptance=$expected"

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

# Fails on any finding. Pass check names to narrow the gate, e.g. `bugprone-*`.
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
    checks="${1:-}"
    if [[ $# -gt 0 ]]; then shift; fi
    # Configure only: the compile database is written by CMake, and the direct
    # run below is the gate. Building through CXX_CLANG_TIDY would also
    # escalate newer-clang diagnostics in the tests, which is not this gate.
    cmake -S skippy/skippy_api -B "$build_dir" -G Ninja \
      -DCMAKE_BUILD_TYPE=Debug \
      -DSKIPPY_API_BUILD_TESTS=OFF \
      -DSKIPPY_API_BUILD_BENCHMARKS=OFF \
      -DSKIPPY_LLAMA_SOURCE_DIR="{{absolute_path(llama_dir)}}"
    cd skippy/skippy_api
    "$tidy" -p "$build_path" --warnings-as-errors="${checks:-*}" \
      $(find src -name '*.cpp' | sort)

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
