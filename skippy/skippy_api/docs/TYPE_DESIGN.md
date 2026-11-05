# Types that make the PR #2481 bug classes unrepresentable

```yaml
for: skippy/skippy_api (C++23), skippy/skippy-api-rs bridge
derived_from: review-pr-2481.md (heads 42b1bd57, a2252f53, 5f19c740)
goal: the wrong spelling does not compile, and the compiler error names the rule
```

The enforcement ladder these types sit at is
[`TYPE-RULES.md`](../TYPE-RULES.md) §1. Every type below is rung 1; each
also takes rung 2 (`SKIPPY_DELETE` + a pinned `static_assert`) where it
retires a spelling.

**Phase (TYPE-RULES.md §0):** these designs are the long-term order, not the
current mechanism. No new strong types land for now — the boundary stays put
with `skippy-boundary-isolation`, and a type is introduced only where it
backs a code change that closes a P0 (`ContextCapacity` behind the pool's
capacity label, `ClassifierHead` behind head detection). When the designs are
picked up again, each type replaces the check that held its rule, and the
check is deleted in the same commit.

## The three moves

Every type below uses one or more of these. They are what turns a review
comment into something a compiler enforces against the next author, human or
model.

1. **Parse, don't validate.** A private constructor and a
   `static expected<T, E> make(...)`. Downstream code takes `T` and cannot be
   reached with unchecked input, because there is no other way to obtain a
   `T`.
2. **The only constructor takes the source of truth.** `ContextCapacity` is
   constructible from a `const llama_context&` and nothing else;
   `ClassifierHead` from the tensor table and nothing else. There is no
   parameter through which a caller can supply a wrong label.
3. **Forbid the old spelling and pin it.** Ref-qualified or deleted overloads
   make the previous call shape ill-formed, and a
   `static_assert(!requires { ... })` in `tests/type_contracts_test.cpp`
   fails the build if anyone brings it back. A negative compile test is the
   cheapest regression test there is: it runs in zero time, on every build,
   with no model file.

Why this helps a model specifically: a model can rationalise past a comment
("the pool labels…"), a trap entry, or a review; it cannot rationalise past
`error: call to deleted member function 'recycle'` with a reason string
attached. Put the rule in the error message.

```cpp
// poison.h — C++26 reasons on deleted functions, degraded gracefully
#if defined(__cpp_deleted_function) && __cpp_deleted_function >= 202403L
#define SKIPPY_DELETE(reason) = delete(reason)
#else
#define SKIPPY_DELETE(reason) = delete
#endif
```

(`tests/toolchain_test.cpp`:
`EXPECT_TRUE(__cpp_deleted_function >= 202403L) << "SKIPPY_DELETE fallback can be dropped";`
— fails the day every supported compiler has it, per the house rule.)

---

## Map: bug class → type

| bug (review id) | what let it happen | type that removes it |
|---|---|---|
| polarity flip between `expected<void,E>` and `optional<E>` (AGENTS trap, P2-1) | two spellings where `!result` means the opposite; results inspected instead of composed | **T1 — removed.** one spelling tree-wide: `std::expected<void, E>`, consumed with `and_then`/`transform` (`skippy-no-failure-optional`, `skippy-no-throwing-accessor`) |
| `bool` as an error channel (`assign` lambda, P1-10) | `bool` converts from anything | the failure-only result is `std::expected<void, E>` (T1 disposition) |
| pool relabel → `GGML_ASSERT` (P0-A); earlier mislabel (P1-5); moved-from argument trap | capacity, request and `n_ctx` were all `uint32_t`; `recycle` took labels from the caller; `batch_limit_` was a field nobody had to read | **T2 `SessionGeometry` / `ContextCapacity`**, **T3 `AdmittedBatch`** |
| `n_cls_out` default read as presence (P0-1 → P0-B) | a count with a non-zero default stood in for "is there a head"; classification could be fed a family | **T4 `ClassifierHead`** (optional, detected from the tensor table) |
| identities into `const char*` (P2-7, the two arena traps, six copies of the `const char*` loop) | the C ABI takes `const char*` where the planner holds `(offset, length)` | **T5 — removed.** the fix is upstream-shaped: `skippy_bytes_ref` / `skippy_runtime_config_v2` (below) |
| `size_t → uint32_t` arena offsets, `int64 → int32` layers, `uint64` byte products (P0-3, P2-2) | `static_cast` is unconditional; `*` wraps | **T6 `narrow<To>`, `checked_mul`, `ArenaSlice::make`** |
| endianness / unchecked enum casts in the GGUF reader (P2-2) | `memcpy` into `T` for any `T`; `static_cast<ggml_type>(file_u32)` | **T7 `WireScalar`, `SizedGgmlType`** |
| `LlamaBackend{}.plan_model_stages(...)` (P1-4) | member callable on a temporary | **T8 ref-qualified `&` members** |
| allocation in the OOM arm of a `noexcept` barrier (P2-11, `run_pass` trap) | failure needed text to exist | **T9 `Failure<Code>::out_of_memory()`** — a code with no text |
| `BackendId{"metal"}` in six places; cache key missing fields (P1-6) | strong type with a public string constructor | **T10 `StageRequest` + private `BackendId`** |
| `data + row * dims` arithmetic (P2-4) | raw pointer maths | **T11 `std::mdspan`** |

---

## T1 — removed: the failure-only spelling is `std::expected<void, E>`

`Fallible<E>` was the failure-only result with no `operator bool`, so a
polarity flip could not compile. The rules have since settled the other way
(TYPE-RULES.md §4): **failure-only results have one spelling, and it
chains** — `std::expected<void, E>` composes with `and_then`, which a
`failed()`/`error()` inspection API cannot; `optional<E>`-as-failure composes
on the *failure* (`optional::and_then` runs on the engaged value), which is
why it is banned. The header `include/skippy/fallible.h` and its pins are
deleted; nothing used it. What replaces each guarantee:

- *one spelling* — the tree-wide flip commit plus `skippy-no-failure-optional`
  (was `skippy-no-expected-void`, reversed), `skippy-optional-error-polarity`
  for the transition;
- *no throwing accessor* — `skippy-no-throwing-accessor` bans `.value()` on
  both `expected` and `optional` everywhere, for good;
- *no `bool` error channel* — unchanged rule; a failure-only call returns
  `std::expected<void, E>`.

---

## T2 — `SessionGeometry` and `ContextCapacity`: request vs what llama built

The product `max_tokens * sequences` is formed in one constructor; a capacity
can only come from a live context and already takes the min of `n_ctx`,
`n_batch`, `n_ubatch` (llama pads `n_ctx` to 256 and asserts on
`n_batch`/`n_ubatch`: `llama-context.cpp:377, 1896-1898`).

```cpp
// include/skippy/geometry.h
#pragma once
#include <algorithm>
#include <cstdint>
#include <expected>
#include <utility>

namespace skippy {

struct TokensPerSequence { std::uint32_t value; constexpr auto operator<=>(const TokensPerSequence&) const = default; };
struct SequenceCount     { std::uint32_t value; constexpr auto operator<=>(const SequenceCount&) const = default; };

class SessionGeometry {
public:
    static constexpr std::uint32_t kMaxSequences = 64;
    static constexpr std::uint32_t kMaxTotalTokens = 0x7FFF'FFFF; // llama batch counts are int32

    [[nodiscard]] static constexpr std::expected<SessionGeometry, RuntimeError>
    make(TokensPerSequence tokens, SequenceCount sequences) noexcept {
        if (tokens.value == 0 || sequences.value == 0 || sequences.value > kMaxSequences)
            return std::unexpected(RuntimeError::UnsupportedConfiguration);
        if (tokens.value > kMaxTotalTokens / sequences.value)
            return std::unexpected(RuntimeError::UnsupportedConfiguration);
        return SessionGeometry{tokens, sequences};
    }

    [[nodiscard]] constexpr TokensPerSequence tokens() const noexcept { return tokens_; }
    [[nodiscard]] constexpr SequenceCount sequences() const noexcept { return sequences_; }
    [[nodiscard]] constexpr std::uint32_t total_tokens() const noexcept { return tokens_.value * sequences_.value; } // proven by make()

private:
    constexpr SessionGeometry(TokensPerSequence t, SequenceCount s) noexcept : tokens_(t), sequences_(s) {}
    TokensPerSequence tokens_;
    SequenceCount sequences_;
};

} // namespace skippy
```

```cpp
// include/llama/context_capacity.h — the only header in llama/ that needs llama.h for this
namespace skippy::llama {

class ContextCapacity {
public:
    explicit ContextCapacity(const llama_context& context) noexcept
        : sequences_{std::max(1U, llama_n_seq_max(&context))},
          tokens_{std::min({llama_n_ctx(&context), llama_n_batch(&context), llama_n_ubatch(&context)}) /
                  sequences_.value} {}

    [[nodiscard]] constexpr bool serves(const SessionGeometry& geometry) const noexcept {
        return tokens_ >= geometry.tokens() && sequences_ >= geometry.sequences();
    }
    [[nodiscard]] constexpr TokensPerSequence tokens() const noexcept { return tokens_; }

private:
    SequenceCount sequences_;
    TokensPerSequence tokens_;
};

struct PooledContext {
    llama_context_ptr context;
    ContextCapacity capacity; // computed from `context` in the pool, never supplied
};

} // namespace skippy::llama
```

Pool surface:

```cpp
void recycle(llama_context_ptr context) noexcept;                                  // label computed inside
void recycle(llama_context_ptr, std::uint32_t, std::uint32_t)
    SKIPPY_DELETE("the pool labels a context from llama_n_ubatch/n_seq_max itself; see P0-A");
[[nodiscard]] std::optional<PooledContext> take_idle_context(const SessionGeometry&) noexcept;
```

Contracts:

```cpp
static_assert(!std::is_constructible_v<ContextCapacity, std::uint32_t, std::uint32_t>);
static_assert(!std::is_default_constructible_v<SessionGeometry>);
static_assert(!requires(LlamaModel& m, llama_context_ptr c) { m.recycle(std::move(c), 512U, 1U); });
```

This also removes the moved-from-argument trap by construction:
`recycle(std::move(context))` has no second argument that could read
`context`.

---

## T3 — `AdmittedBatch`: a decode cannot be issued without the limit check

`batch_limit_` was a field nobody had to read. Make the check produce the
thing `run_pass` consumes.

```cpp
// include/llama/batch_limit.h
namespace skippy::llama {

class AdmittedBatch {
public:
    [[nodiscard]] std::span<const Token> tokens() const noexcept { return tokens_; }
private:
    friend class BatchLimit;
    explicit AdmittedBatch(std::span<const Token> tokens) noexcept : tokens_(tokens) {}
    std::span<const Token> tokens_;
};

class BatchLimit {
public:
    explicit BatchLimit(const llama_context& context) noexcept : ubatch_{llama_n_ubatch(&context)} {}

    // Both bounds, in one place: the session's contract and the context's physical batch.
    [[nodiscard]] std::expected<AdmittedBatch, WorkloadError>
    admit(std::span<const Token> tokens, TokensPerSequence configured) const noexcept {
        if (tokens.empty()) return std::unexpected(WorkloadError::EmptyInput);
        if (tokens.size() > configured.value || tokens.size() > ubatch_)
            return std::unexpected(WorkloadError::TooManyTokens);
        return AdmittedBatch{tokens};
    }

private:
    std::uint32_t ubatch_;
};

} // namespace skippy::llama
```

`LlamaSession::run_pass(const AdmittedBatch&, Pass)`; `fill_embedding_batch(Batch&,
const AdmittedBatch&, SequenceId)`. The batch path admits each sequence and
then the total through a second `admit_total`. Contract:

```cpp
static_assert(!std::is_constructible_v<AdmittedBatch, std::span<const Token>>);
static_assert(!std::is_invocable_v<decltype(&LlamaSession::run_pass), LlamaSession&, std::span<const Token>, Pass>);
```

---

## T4 — `ClassifierHead`: presence first, count second

The head is the `cls.output` tensor (`llama-arch.cpp:536`, loaded by bert,
modern-bert, neo-bert, qwen3, qwen3vl, qwen35, lfm2). `n_cls_out` is its width
and defaults to 1 whether or not it exists.

```cpp
// include/llama/classifier_head.h
namespace skippy::llama {

class ClassifierHead {
public:
    [[nodiscard]] static std::optional<ClassifierHead>
    detect(std::span<const probe::TensorInfo> inventory_sorted_by_name, const llama_model& model) noexcept {
        static constexpr std::string_view kHead = "cls.output.weight";
        const auto found = std::ranges::lower_bound(inventory_sorted_by_name, kHead, {}, &probe::TensorInfo::name);
        if (found == inventory_sorted_by_name.end() || found->name != kHead) return std::nullopt;
        // Width from the tensor itself; n_cls_out only confirms it.
        const auto width = found->dimension_count >= 2 ? narrow<std::uint32_t>(found->dimensions[1]) : std::nullopt;
        return width.transform([](std::uint32_t outputs) { return ClassifierHead{outputs}; });
    }
    [[nodiscard]] constexpr std::uint32_t outputs() const noexcept { return outputs_; }

private:
    constexpr explicit ClassifierHead(std::uint32_t outputs) noexcept : outputs_(outputs) {}
    std::uint32_t outputs_;
};

} // namespace skippy::llama
```

```cpp
// workload_profile.h — the pure table takes structure only
struct ModelFacts {
    bool has_encoder = false;
    bool has_decoder = false;
    std::optional<std::uint32_t> classifier_outputs; // engaged iff the head tensor exists
    DeclaredPooling declared_pooling = DeclaredPooling::Unspecified;
};
```

Contracts:

```cpp
static_assert(!std::is_constructible_v<ClassifierHead, std::uint32_t>);
static_assert(!requires(ModelFacts f) { f.family; });   // classification never sees a ModelFamily
```

House rule to write next to it: *a wrapper over any llama getter with a
non-zero default returns `std::optional` and is named `*_if_present`; the raw
getter is called in exactly one place.*

---

## T5 — removed: the ABI gets views (`skippy_bytes_ref`), not a string type

`ZString` / `CStringTable` were the façade's answer to `const char*` in the
C ABI. The rules now route that problem upstream (TYPE-RULES.md §4, "The ABI
gets views too"): where the C ABI takes `const char*` for bytes the planner
hands out as `(offset, length)`, the fix is a llama.cpp-shaped patch, not a
string type above the boundary. The header
`include/skippy/cstrings.h` and its pins are deleted.

Upstream shape (patches blocked-upstream, tracked in `UPSTREAM.md`):

```c
/* skippy.h */
typedef struct {
    const uint8_t* data;   /* not NUL-terminated: length is authoritative */
    uint32_t        length;
} skippy_bytes_ref;

typedef struct skippy_runtime_config_v2 {
    uint32_t struct_size;            /* SKIPPY_RUNTIME_CONFIG_V2_SIZE */
    skippy_bytes_ref execution_contract;
    skippy_bytes_ref* resident_identities;  uint32_t resident_count;
    skippy_bytes_ref* activation_imports;   uint32_t import_count;
    skippy_bytes_ref* activation_exports;   uint32_t export_count;
    /* v1 fields by embedded struct, so old callers still compile */
    skippy_runtime_config v1;
} skippy_runtime_config_v2;
```

- callers that hold the plan (the arena) pass `skippy_bytes_ref{ptr, len}`
  pairs; nothing copies, nothing NUL-terminates, and the arena/NUL trap class
  is gone at the source instead of fenced in the façade;
- the planner probe, the pipeline probe, the staged probe, the harness and
  `staged.cpp` all speak `skippy_runtime_config_v2`, which deletes the six
  hand-rolled `string_view → const char*` loops in one move;
- until the patch lands, the arena-terminated fallback is **feature-tested on
  the ABI version macro** (`#if SKIPPY_ABI_VERSION < ...`) with a
  "fallback can go" test that fails the day the macro moves — the same
  pattern as `SKIPPY_DELETE` in `tests/toolchain_test.cpp`.

`UPSTREAM.md` carries the patch link and a `blocked-upstream` row; the entry
names what the fallback costs (one copy per config build) so the incentive to
rebase stays visible.

---

## T6 — widths: `narrow`, `checked_mul`, `ArenaSlice::make`

```cpp
// include/skippy/narrow.h
#pragma once
#include <concepts>
#include <limits>
#include <optional>
#include <utility>

namespace skippy {

template <std::integral To, std::integral From>
[[nodiscard]] constexpr std::optional<To> narrow(From value) noexcept {
    return std::in_range<To>(value) ? std::optional<To>{static_cast<To>(value)} : std::nullopt;
}

template <std::unsigned_integral T>
[[nodiscard]] constexpr std::optional<T> checked_mul(T a, T b) noexcept {
#if __has_builtin(__builtin_mul_overflow)
    T out{};
    return __builtin_mul_overflow(a, b, &out) ? std::nullopt : std::optional<T>{out};
#else
    return (b != 0 && a > std::numeric_limits<T>::max() / b) ? std::nullopt : std::optional<T>{a * b};
#endif
}

template <std::unsigned_integral T>
[[nodiscard]] constexpr std::optional<T> checked_add(T a, T b) noexcept {
    return a > std::numeric_limits<T>::max() - b ? std::nullopt : std::optional<T>{a + b};
}

} // namespace skippy
```

Apply at the three places that had unexplained casts:

```cpp
// stage_plan.h — one arena type instead of two arena_put copies
class IdentityArena {
public:
    [[nodiscard]] std::expected<ArenaSlice, PlanError> put(std::string_view bytes) {
        const auto offset = narrow<std::uint32_t>(bytes_.size());
        const auto length = narrow<std::uint32_t>(bytes.size());
        if (!offset || !length || !checked_add(*offset, *length)) return std::unexpected(PlanError::ArenaOverflow);
        bytes_.append(bytes);
        return ArenaSlice{*offset, *length};
    }
    [[nodiscard]] std::string_view view(ArenaSlice slice) const noexcept; // clamped, as today
private:
    std::string bytes_;
};
```

`row_size_bytes` already uses the divide-check form; swapping to
`checked_mul` makes it one word. `LayerCount`/`LayerIndex` as `int32_t`
strong types with `LayerRange::make(start, end)` rejecting `start >= end`
closes the planner input side. Then: `-Wconversion -Wsign-conversion -Wshadow
-Wold-style-cast` on every target, and a
`grep -n 'static_cast<std::u\?int[0-9]*_t>' include src` step in
`skippy-api-verify` whose allowlist is `narrow.h` plus the ABI-boundary files
— every other integer cast must be `narrow`.

Contract:
`static_assert(std::is_same_v<decltype(narrow<std::int32_t>(std::int64_t{})), std::optional<std::int32_t>>);`

---

## T7 — the GGUF reader: `WireScalar` and bounded native enums

```cpp
// include/skippy/wire.h
template <typename T>
concept WireScalar = (std::integral<T> || std::floating_point<T>) && std::has_unique_object_representations_v<T>;

template <WireScalar T>
[[nodiscard]] T from_little_endian(std::span<const std::byte, sizeof(T)> bytes) noexcept {
    T value;
    std::memcpy(&value, bytes.data(), sizeof value);
    if constexpr (std::endian::native == std::endian::big && std::integral<T>) value = std::byteswap(value);
    return value;
}
```

`Reader::read` is constrained on `WireScalar` so a struct can never be
`memcpy`'d off the file, and the big-endian path exists rather than being
assumed away. The enum side is already right (`GgufMetadataType` compared by
`to_underlying`); finish the pattern for ggml:

```cpp
// include/llama/sized_type.h — a ggml type that ggml can size; the only input row_size_bytes accepts
class SizedGgmlType {
public:
    [[nodiscard]] static std::optional<SizedGgmlType> parse(std::uint32_t wire) noexcept {
        if (wire >= static_cast<std::uint32_t>(GGML_TYPE_COUNT)) return std::nullopt;
        const auto type = static_cast<ggml_type>(wire);          // the one cast, after the bound
        const std::int64_t block = ggml_blck_size(type);
        const std::size_t element = ggml_type_size(type);
        if (block <= 0 || element == 0) return std::nullopt;     // removed slots
        return SizedGgmlType{type, block, element};
    }
    [[nodiscard]] ggml_type native() const noexcept { return type_; }
    [[nodiscard]] std::int64_t block() const noexcept { return block_; }
    [[nodiscard]] std::size_t element() const noexcept { return element_; }
private: /* ... */
};
static_assert(!std::is_constructible_v<SizedGgmlType, ggml_type>);
```

---

## T8 — `LlamaBackend`: planning needs an owner that outlives the plan

```cpp
class LlamaBackend {
public:
    [[nodiscard]] std::expected<PlanLease, PlanFailure>
    plan_model_stages(const std::string& path, std::span<const LayerRange> ranges) &;
    std::expected<PlanLease, PlanFailure>
    plan_model_stages(const std::string&, std::span<const LayerRange>) &&
        SKIPPY_DELETE("a temporary backend's plan store dies with the statement; pass the runtime's backend (P1-4)");
    // same pair for plan_model_plans
};

class StageChain {
public:
    [[nodiscard]] static std::expected<StageChain, StagedFailure>
    open(LlamaBackend& backend, const std::string& path, std::span<const LayerRange> ranges);
};
```

`PlanLease::configs` takes the same pair — it hands out views into the
store the lease pins (TYPE-RULES.md §4).

```cpp
static_assert(!requires(std::string p, std::span<const LayerRange> r) { LlamaBackend{}.plan_model_stages(p, r); });
static_assert(!requires(std::string p, std::span<const LayerRange> r) { StageChain::open(p, r); });
```

The `&`/`&&` pair is the smallest C++ that encodes "this object must outlive
what it returns" without a lifetime system. `PlanLease` already does the rest.

---

## T9 — `Failure<Code>`: out-of-memory is a code, never text

```cpp
// include/skippy/failure.h
template <typename Code>
    requires std::is_enum_v<Code> && requires { Code::OutOfMemory; Code::Internal; }
struct Failure {
    Code code;
    std::string message; // empty for OutOfMemory by construction

    [[nodiscard]] static Failure out_of_memory() noexcept { return {Code::OutOfMemory, {}}; }
    [[nodiscard]] static Failure internal(std::string_view what) noexcept {
        Failure f{Code::Internal, {}};
        try { f.message = what; } catch (...) {}     // text is optional; the code is not
        return f;
    }
};
```

The bridge barrier's arms become
`return in_band(Failure<Code>::out_of_memory())` — no `std::string`
concatenation, no `rust::String::lossy` call on the OOM path, so `noexcept`
is honest by construction rather than by care. Unifying `PlanFailure`,
`StagedFailure`, `WorkloadError` (P1-2) and the parser's `std::string` on
`Failure<Code>` is P3-3; the OOM arm is the part that cannot wait.

Contract: `static_assert(noexcept(Failure<PlanError>::out_of_memory()));`

---

## T10 — `StageRequest`: no literal backends, and the key comes from the type

```cpp
class BackendId {
public:
    [[nodiscard]] static BackendId detect();                       // first ggml GPU device name, else "cpu"
    [[nodiscard]] static std::optional<BackendId> named(std::string_view);  // validated against ggml's device list
    [[nodiscard]] std::string_view name() const noexcept;
private:
    explicit BackendId(std::string name);
    std::string name_;
};
static_assert(!std::is_constructible_v<BackendId, const char*>);
static_assert(!std::is_constructible_v<BackendId, std::string>);

struct StageRequest {
    ProfileSpec profile{.id = ProfileId{"decode"}, .tokens = 512, .sequences = 1, .outputs = 512};
    GraphConfigurationId graph_configuration{"stage"};
    BackendId backend = BackendId::detect();
    GpuLayers gpu_layers{999};

    // The only producer of a plan-store key. New fields are added here and nowhere else.
    [[nodiscard]] std::string key(std::string_view file_stamp, std::span<const LayerRange>) const;
};
```

`plan_model_stages(path, ranges, const StageRequest&)`; `StagedModel::open`
reads `ctx/n_batch/n_ubatch` from `request.profile`. `cache_key` as a free
function is deleted. Six `"metal"` literals stop compiling.

---

## T11 — `std::mdspan` for row-major outputs

```cpp
#if defined(__cpp_lib_mdspan) && __cpp_lib_mdspan >= 202207L
using Rows = std::mdspan<float, std::dextents<std::size_t, 2>>;   // [row][dim]
#else
struct Rows { /* minimal (data, rows, dims) with operator[](row) -> span<float> */ };
#endif
```

`embed_many` returns `Rows`; `embed_batch` writes `output[row]`. The
`data + row * dims` arithmetic disappears from `backend.cpp` and `embed.h`,
and a row index that exceeds `rows` is a bounds error under a hardened
standard library (`-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG` in
the ASan build) instead of a silent read.

---

## The contracts file

`tests/type_contracts_test.cpp` compiles with the mocked suite and contains
only `static_assert`s: the forbidden spellings above, the layout facts the
ABI depends on (`std::is_layout_compatible_v<Token, std::int32_t>`), and the
"fallback can be dropped" checks. It is the file an agent is told to extend
when it closes a finding: the rule, as a line of C++ that fails the build.
Pair it with the tidy checks that catch the remaining dynamic shapes —
`bugprone-use-after-move` (the moved-from argument),
`bugprone-implicit-widening-of-multiplication-result`,
`cppcoreguidelines-narrowing-conversions`,
`readability-implicit-bool-conversion` — with `WarningsAsErrors: '*'`.

## Order of introduction

Deferred per TYPE-RULES.md §0 — the current phase isolates the boundaries
with clang-tidy and introduces a type only where it backs a P0-closing code
change. The order for when the designs are picked up again:

1. `SessionGeometry`/`ContextCapacity`/`AdmittedBatch` (backs the pool
   capacity label; closes P0-A).
2. `ClassifierHead` (backs head detection; closes P0-B).
3. `&`-qualified planning + `StageChain::open(backend&)` (P1-4).
4. `Failure<Code>` OOM arm (P2-11), then the wider P3-3 unification.
5. `StageRequest` (P1-6), `mdspan`, `WireScalar`.
6. The `skippy_bytes_ref` / `skippy_runtime_config_v2` upstream patch rides
   the llama.cpp rebase queue, not this tree's commit order.
