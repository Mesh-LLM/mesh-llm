# skippy clang-tidy checks

Eighteen out-of-tree clang-tidy checks for the `skippy_api` rules that a type
cannot carry (see `../TYPE-RULES.md` §1 for when a rule is a type and when it is a
check). Built and tested against LLVM 18; every API used is unchanged through
LLVM 20.

```
tidy/
  CMakeLists.txt            builds SkippyTidyChecks.{so,dylib}; registers the fixtures with ctest
  SkippyTidyModule.cpp      registers the checks and this tree's default options
  checks/<Name>Check.{h,cpp}
  checks/Support.h          enclosing function / enclosing block / file match / line text
  test/<name>.cpp           one fixture per check, FileCheck lines inline
  test/run.sh               clang-tidy --load … | FileCheck; a warning without a CHECK line fails
  test/stubs/               llama.h, rust_stub.h, expected_stub.h — no real dependencies
```

## Build

The plugin must be built against the exact LLVM that owns the clang-tidy binary.

```bash
# Homebrew (macOS)
cmake -S skippy/skippy_api/tidy -B target/skippy-tidy \
      -DClang_DIR=/opt/homebrew/opt/llvm/lib/cmake/clang -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++
# Debian/Ubuntu
cmake -S skippy/skippy_api/tidy -B target/skippy-tidy -DClang_DIR=/usr/lib/llvm-18/lib/cmake/clang
cmake --build target/skippy-tidy --parallel
ctest --test-dir target/skippy-tidy           # 18 fixtures
```

Loading needs a clang-tidy built with plugin support (Homebrew and Debian are).
`SKIPPY_CLANG_TIDY` already selects the binary; the plugin path is the same
prefix's build output.

## Wire it into the verify recipe

Scope is set by per-directory `.clang-tidy` files (clang-tidy uses the nearest
one), so boundary checks cannot leak into the core and the core's migration
checks cannot be forgotten at the boundary — see `../TYPE-RULES.md` §5.

```yaml
# skippy_api/.clang-tidy (core): existing checks + the migration checks
Checks: >
  ..., skippy-no-expected-void, skippy-optional-error-polarity, skippy-no-void-silencing,
  skippy-no-process-wide-state, skippy-no-raw-loops,
  bugprone-use-after-move, bugprone-implicit-widening-of-multiplication-result,
  cppcoreguidelines-narrowing-conversions, readability-implicit-bool-conversion
WarningsAsErrors: '*'
CheckOptions:
  skippy-no-process-wide-state.AllowedInFunctions: 'skippy::detail::next_generation'

# skippy_api/src/llama/.clang-tidy, skippy_api/tools/.clang-tidy (llama boundary)
InheritParentConfig: true
Checks: >
  skippy-preceded-by, skippy-single-call-site, skippy-no-lock-across-native-call,
  skippy-const-native-mutation, skippy-explicit-narrowing, skippy-no-alloc-in-oom-handler
CheckOptions:
  skippy-single-call-site.Functions: 'llama_model_n_cls_out;llama_n_ctx;llama_n_batch;llama_n_ubatch;llama_n_seq_max'
  skippy-single-call-site.AllowedCallers: 'skippy::llama::ClassifierHead::detect;skippy::llama::ContextCapacity::ContextCapacity;skippy::llama::BatchLimit::BatchLimit'
  skippy-explicit-narrowing.AllowedFiles: 'narrow\.h$'

# skippy-api-rs/skippy-sys/cpp/.clang-tidy (Rust boundary)
InheritParentConfig: true
Checks: >
  skippy-bridge-barrier, skippy-no-alloc-in-oom-handler, skippy-rust-string-lossy, skippy-explicit-narrowing
CheckOptions:
  skippy-bridge-barrier.FileRegex: 'bridge\.cpp$'
```

```just
skippy-llama-api-tidy:
    #!/usr/bin/env bash
    set -euo pipefail
    plugin="target/skippy-tidy/SkippyTidyChecks$( [[ "$(uname)" == Darwin ]] && echo .dylib || echo .so )"
    [[ -f "$plugin" ]] || just skippy-tidy-plugin-build
    "${SKIPPY_CLANG_TIDY:-clang-tidy}" --load="$plugin" -p target/skippy-llama-api --warnings-as-errors='*' \
        $(git ls-files 'skippy/skippy_api/src/*.cpp' 'skippy/skippy_api/tools/*.cpp' 'skippy/skippy-api-rs/skippy-sys/cpp/*.cpp')
```

## The checks

| check | rule (AGENTS.md) | would have caught |
|---|---|---|
| `skippy-no-expected-void` | never `std::expected<void, E>` | the 15 declarations in P2-1 |
| `skippy-optional-error-polarity` | an `optional<Error>` converted to bool reads backwards | the polarity-flip trap |
| `skippy-no-raw-loops` | ranges over loops; index loops only at an index-based ABI, with a NOLINT reason | the P2-3 table |
| `skippy-preceded-by` | `llama_memory_clear` immediately after `llama_synchronize` (configurable pair) | P1-1 |
| `skippy-no-alloc-in-oom-handler` | nothing allocates inside a `bad_alloc`/catch-all handler of a noexcept function | P2-11, the `run_pass` trap |
| `skippy-bridge-barrier` | every noexcept bridge member is `return <barrier>(...)` | the eight methods of P0-2 |
| `skippy-single-call-site` | a native getter with a default/padded value is read in one named place | P0-1/P0-B (`n_cls_out`), P0-A (`n_ctx`) |
| `skippy-no-lock-across-native-call` | no mutex held across a call into llama/ggml/skippy | AGENTS "State" (no instance yet) |
| `skippy-no-void-silencing` | `(void)x` on a local with no other use is a dead computation | the `(void)imports…` blocks in materialize/clone |
| `skippy-no-process-wide-state` | no mutable namespace-scope/static-local state outside the allowlist | the memo index (P3-5) |
| `skippy-rust-string-lossy` | native text crosses the bridge through `rust::String::lossy` | every `rust::String(message)` before 136020b3 |
| `skippy-boundary-isolation` | tree-wide | native includes, calls and types live only in src/llama, include/llama, tools, benchmarks, skippy-sys/cpp (ENFORCEMENT.md §0) |
| `skippy-no-throwing-accessor` | tree-wide | `.value()` never appears; results compose (ENFORCEMENT.md §4) |
| `skippy-no-failure-optional` | tree-wide | `std::optional<E>`-as-failure is banned; the failure-only spelling is `std::expected<void, E>` (ENFORCEMENT.md §4) |
| `skippy-no-mutex` | tree-wide | no mutex anywhere — RCU snapshots and CAS (ENFORCEMENT.md §3.2) |
| `skippy-no-raw-pointer` | tree-wide | views over raw pointers; raw pointers only in owning cells and the ABI struct (ENFORCEMENT.md §4) |
| `skippy-explicit-narrowing` | an integer cast that can lose value is `narrow<>()`, a fitting constant, or carries `// bounded by …` | `arena_put` ×2, the three layer casts |
| `skippy-const-native-mutation` | a const method never hands a member to a native function through a non-const pointer | AGENTS "const means what it says" |

Every check's `.h` carries the rule, the option list and the finding it maps to.

## Adding a check

1. Decide it is a check and not a type: `../TYPE-RULES.md` §1. If a signature change can make the wrong program ill-formed, stop — write the type and a `static_assert(!requires …)` in `tests/type_contracts_test.cpp` instead.
2. `checks/<Name>Check.{h,cpp}`: header comment = rule + options + the finding it would have caught. Options with defaults; `storeOptions` for each.
3. Register in `SkippyTidyModule.cpp`; put this tree's values in `getModuleOptions()`.
4. `test/<name>.cpp`: `// CHECKS:` line, one positive case with a `// CHECK: [[@LINE-1]]:<col>: warning: …` line, and one negative case with no line. `run.sh` fails on any unexpected warning.
5. Add the row to the table above and to the verify recipe if it needs a new option.

## Known limits

- clang-tidy sees one translation unit. A rule that needs whole-program knowledge is not a check; it is a type, a link-time symbol, or a test.
- `skippy-preceded-by` is strict about adjacency on purpose: a `llama_synchronize` three statements earlier with a native call in between proves nothing.
- `skippy-explicit-narrowing` accepts `// bounded by …` on the line as the AGENTS rule asks; that is a review contract, not a proof. `narrow<>()` is.
- A `NOLINT` without a reason is not caught yet (`grep -n 'NOLINT(' | grep -v ': '` in the verify recipe until a check exists).
