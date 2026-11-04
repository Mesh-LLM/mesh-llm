# Ownership and memory in `skippy_api`, as types

```yaml
for: skippy/skippy_api (C++23), skippy/skippy-api-rs (cxx bridge + safe crate)
companion: docs/TYPE_DESIGN.md (value types), TYPE-RULES.md §3 (the rules these types enforce)
goal: C++ owns and frees; Rust borrows and checks. No borrow model in C++, no copies into Rust.
```

> Subject to TYPE-RULES §0 (defer the strong types — only P0-closing O4 is in scope now; O2/O8 deferred).
>
> **Triage.** Build the first three; the rest are aliases, one-liners or tabled.
>
> **Do now**
> 1. **Borrowing moves to Rust** (O2): expose `LlamaModel`/`LlamaSession`/`PlanLease` to cxx as owned opaque
>    types and return views (`&[i32]`, `&[f32]`, `&str`, `&[u8]`) from `&self` methods. cxx ties each view's
>    lifetime to the borrow of the object; rustc enforces "until the next mutating call" through the
>    `&mut self` on the mutating methods. Zero copies across the bridge; zero `unsafe` beyond the existing
>    `Token` transmute. Three C++ methods are affected today (`tokenize`, `embed`/`embed_many`,
>    `rerank_prompt_format`); the planner is the fourth consumer and loses all its hex `String`s.
> 2. **Pool hand-back as a stateful deleter over atomic slots** (O4): `std::unique_ptr<llama_context, ReturnToPool>`;
>    the deleter quiesces and CASes the context into a slot; capacity is read from the context, never stored. P0-A
>    closed structurally, no lock.
> 3. **Owner contracts** (O8): `static_assert`s on copyability and nothrow destruction, no new code.
>
> **Use what exists:** `Owned` = the `*_ptr` aliases (O1); `PlanLease` as it is, exposed whole (O3);
> `std::vector` for session scratch (O5); views into the arena instead of any string type, with a `skippy_bytes_ref` ABI patch so the loader takes views too (O6). **Tabled:** a
> generic `Pool<>` template, a `thread_model<T>` trait, `offsetof` order pins. **Not doing:** a C++ borrow type; a
> mutex anywhere (the pool is CAS); a hand-maintained ownership map — the libTooling pass in O9 generates one.

> **Status: deferred.** Per the current phase (TYPE-RULES §0) the boundaries are isolated with clang-tidy first and the strong types below are not introduced yet, except where a P0 fix happens to need one. Keep this file as the design the checks will be replaced by.

## The four moves

1. **Ownership has exactly one spelling per kind.** An owner is move-only and its destructor is the complete, nonthrowing release; a value holds nothing. Borrows are not a C++ kind here: a C++ method returns a view and says nothing more than "valid until the next mutating call on this object"; the bridge gives that sentence to rustc.
2. **Moving is the only way ownership changes hands, and a moved-from object is empty, destructible, and refuses to be used.** Handles clear their id (`std::exchange`); native pointers go `nullptr`; a method on a moved-from owner is a typed rejection.
3. **C++ holds the memory; everyone else gets a view.** Results live in object-owned storage (session scratch, the plan arena) and are handed out as `std::span`/`std::string_view` in C++ and `&[T]`/`&str` in Rust, with the object's lifetime. A raw pointer appears only inside an owning cell or in the C struct the ABI defines, and the ABI itself gets a view type (O6.1). Rust copies only when the caller asks (`to_vec()`), never because the bridge had to.
4. **Temporary ownership is a deleter.** A context a session holds *for now* and a pool holds *in between* is a `unique_ptr` whose deleter returns it; neither side can forget the hand-back.

---

## O1 — `Owned` is the alias you already have

`llama_model_ptr`, `llama_context_ptr`, `EmbeddingBatchPtr` are `std::unique_ptr` with named deleters. Add the two missing ones and the one-statement rule; no template.

```cpp
inline void skippy_model_release(skippy_model* model) noexcept { (void)skippy_model_free(model, nullptr); }
inline void skippy_session_release(skippy_session* session) noexcept { (void)skippy_session_free(session, nullptr); }
using skippy_model_ptr   = std::unique_ptr<skippy_model, decltype([](skippy_model* m) noexcept { skippy_model_release(m); })>;
using skippy_session_ptr = std::unique_ptr<skippy_session, decltype([](skippy_session* s) noexcept { skippy_session_release(s); })>;
```

```cpp
skippy_model* raw = nullptr;
ErrorSlot error;
const auto status = skippy_model_open(path.c_str(), &config.native(), &raw, &error.raw);
skippy_model_ptr model{raw};                                // owned from this statement, whatever status says
if (status != SKIPPY_STATUS_OK) return std::unexpected(error.take(status));
```

Rule: a raw native pointer never outlives the statement that produced it. Contract: `static_assert(!std::is_copy_constructible_v<skippy_model_ptr>)`.

---

## O2 — Borrowing lives in Rust; C++ returns views and holds the memory

Today the bridge is a registry (`Bridge` → `NativeRuntime` → slot → resource) and every result is copied out (`tokens.push_back`, normalise into the caller's buffer, hex-encode identities into `rust::String`s). The registry exists so that C++ callers cannot dereference a stale handle. Rust does not need it: `Model<'rt>`/`Session<'m>` already make a stale handle unrepresentable. So for Rust, expose the resources themselves and let the borrow checker do the rest.

### 2.1 The shape

cxx can return `&[T]`, `&str` and `&T` from a C++ method and elides the result lifetime to the `self` borrow. That is the whole borrow model: a `&self` method yields a shared view; the next `Pin<&mut self>` call cannot be made while the view is alive. No `Borrow<T, Owner, Until>`, no copy.

cxx shared structs cannot carry references, so a fallible view is two calls: a mutating call that returns a status, then a `&self` accessor that returns the view. The Rust wrapper makes it one method whose return borrows `&mut self`.

```rust
// skippy-sys/src/lib.rs — the bridge speaks resources, not a registry
unsafe extern "C++" {
    include!("skippy-sys/cpp/bridge.h");
    type LlamaModel;      // skippy::llama::LlamaModel
    type LlamaSession;    // skippy::llama::LlamaSession
    type PlanLease;       // skippy::llama::PlanLease

    fn load_model(path: &str) -> ModelResult;                                              // { status, model: UniquePtr<LlamaModel> }
    fn create_session(model: Pin<&mut LlamaModel>, config: SessionConfig) -> SessionResult; // { status, session: UniquePtr<LlamaSession> }

    // mutate → status; view → borrowed from &self
    fn tokenize(self: Pin<&mut LlamaSession>, text: &str) -> WorkloadStatus;
    fn tokens(self: &LlamaSession) -> &[i32];                        // session scratch, until the next tokenize
    fn embed(self: Pin<&mut LlamaSession>, tokens: &[i32]) -> WorkloadStatus;
    fn embed_many(self: Pin<&mut LlamaSession>, tokens: &[i32], lengths: &[u32]) -> WorkloadStatus;
    fn embedding(self: &LlamaSession) -> &[f32];                     // normalised rows, until the next pass
    fn embedding_width(self: &LlamaSession) -> usize;
    fn score_many(self: Pin<&mut LlamaSession>, tokens: &[i32], lengths: &[u32]) -> WorkloadStatus;
    fn scores(self: &LlamaSession) -> &[f32];
    fn rerank_prompt_template(self: &LlamaModel) -> &str;            // model traits, for the model's life
    fn workload(self: &LlamaSession) -> WorkloadInfo;                // a value: 40 bytes, copied
    fn last_error(self: &LlamaSession) -> &str;                      // the message run_pass kept (P1-2)

    fn plan_stages(model_path: &str, stage_count: u32, request: &StageRequest) -> PlanResult; // { status, lease: UniquePtr<PlanLease> }
    fn stage_count(self: &PlanLease) -> usize;
    fn layer_range(self: &PlanLease, stage: usize) -> LayerRange;    // value
    fn execution_contract(self: &PlanLease, stage: usize) -> &[u8];  // arena bytes, for the lease's life
    fn resident_count(self: &PlanLease, stage: usize) -> usize;
    fn resident_identity(self: &PlanLease, stage: usize, index: usize) -> &[u8];
    // imports / exports / requests the same way
}
```

```rust
// skippy/src/lib.rs — the borrow checker does what a C++ Borrow<T, Owner, Until> tried to document
pub struct Session<'m> { handle: UniquePtr<sys::LlamaSession>, _model: PhantomData<&'m Model<'m>> }

impl Session<'_> {
    pub fn tokenize(&mut self, text: &str) -> Result<&[Token], WorkloadError> {
        workload_result(self.handle.pin_mut().tokenize(text))?;
        Ok(tokens_from_native(self.handle.tokens()))      // &[i32] → &[Token]: repr(transparent), the one existing unsafe
    }
    pub fn embed(&mut self, tokens: &[Token]) -> Result<&[f32], WorkloadError> {
        workload_result(self.handle.pin_mut().embed(tokens_as_native(tokens)))?;
        Ok(self.handle.embedding())
    }
    pub fn embed_many(&mut self, groups: &[&[Token]]) -> Result<Rows<'_>, WorkloadError> {
        /* flatten + lengths; then embedding() chunked by embedding_width() */
    }
}
```

What rustc now enforces, with no C++ help:

| C++ comment today | Rust fact after O2 |
|---|---|
| "returned span borrows session scratch until the next `tokenize()`" | `tokenize(&mut self) -> &[Token]`: the slice holds the `&mut` borrow; a second `tokenize` while it lives is a compile error |
| "returned span borrows the context until the next decode or encode" | `embed(&mut self) -> &[f32]`: same |
| "`RerankPromptFormat` views borrow the model's traits" | `rerank_prompt_template(&self) -> &str` on `Model<'rt>`; sessions already borrow the model |
| "the lease pins the plans the configs borrow from" | `UniquePtr<PlanLease>` inside `StagePlans<'rt>`; every accessor returns `&[u8]` with the lease's lifetime; the hex `String`s and `request_*_hex` go away |

### 2.2 What C++ has to do

- **Keep result memory object-local.** `LlamaSession` owns `std::vector<Token> tokens_`, `std::vector<float> embedding_` (normalised in place — the caller-buffer shape goes), `std::vector<float> scores_`. Views come from these. `PlanLease` already owns its data.
- **Session methods are the bridge surface.** `tokenize`, `embed`, `embed_many`, `score_many` on `LlamaSession` directly; the free functions in `embed.h`/`rerank.h` remain the C++-only convenience over the registry. One implementation, two front doors.
- **`const` is the view contract.** Every accessor returning a view is `const` and touches nothing. `llama_get_embeddings_*` synchronises, so the *mutating* call copies the rows into `embedding_` and the `const` accessor returns the vector (`skippy-const-native-mutation` keeps that honest).
- **Raw pointers end at the statement that receives them.** `const float* data = llama_get_embeddings_seq(…)` becomes a `std::span<const float>` in the same statement (or `std::nullopt`); `LlamaSession::rerank_prompt_` stops being a `const RerankPromptSource*` and becomes the `RerankPromptFormat` view struct it hands out anyway (string_views into model traits; the model outlives the session by the runtime's rule). `skippy-no-raw-pointer` flags the rest.
- **Pool safety across sessions.** With sessions driven through `&mut self` and no bridge-wide mutex, two sessions of one model can drop on two threads at once; the pool is atomics and CAS (O4), so there is nothing to lock. `LlamaModel` is `Send + Sync` (immutable traits, lock-free pool); `LlamaSession` is `Send`, not `Sync`.

### 2.3 Why not a C++ borrow type

A C++ `Borrow<T, Owner, Until>` restates in a template what rustc checks for free, needs a friend per owner, and is bypassed by the first `.get()`. The only thing C++ must guarantee is that a view is valid until the next mutating call on the same object and that the object frees its storage in its destructor. Both are the RAII contract this tree already follows.

---

## O3 — `PlanLease` stays; it is exposed whole

No `Lease<Index, View>` template. `PlanLease{configs, pinned}` is correct as is. Two one-liners: `const std::vector<StageLoadConfig>& configs() const& noexcept;` and `configs() && = delete;`. For Rust it is a `UniquePtr<PlanLease>` with the `&self` accessors in O2 — plan data is read in place from the arena, which is the zero-copy path the AGENTS hex-encoding note was working around.

---

## O4 — The pool hand-back is a deleter, and the pool is atomics

No label is stored anywhere: capacity is read from the context when it is needed (`llama_n_ctx` is padded to 256; `n_batch`/`n_ubatch` are not), so the P1-5 → P0-A relabel class has nothing to relabel. No lock: two slots of `std::atomic<llama_context*>`, claim-then-inspect with `exchange`, place with a CAS from `nullptr`, evict round-robin.

```cpp
// llama/context_pool.h
class ContextPool;

struct ReturnToPool {
    std::reference_wrapper<ContextPool> pool;        // the pool outlives every holder: the runtime forbids unload while a session lives
    void operator()(llama_context* context) const noexcept;
};
using PooledContext = std::unique_ptr<llama_context, ReturnToPool>;

// An atomic cell that owns what it holds. std::atomic<std::unique_ptr> does not exist (C++23 or 26), so this
// is the one place a raw pointer is stored — behind an interface that only speaks unique_ptr.
template <typename T, auto Free>
class AtomicOwned {
public:
    using Ptr = std::unique_ptr<T, decltype([](T* p) noexcept { Free(p); })>;
    ~AtomicOwned() { Ptr{cell_.exchange(nullptr)}; }
    [[nodiscard]] Ptr take() noexcept { return Ptr{cell_.exchange(nullptr, std::memory_order_acq_rel)}; }
    [[nodiscard]] bool try_place(Ptr& owned) noexcept {                       // CAS from empty; on success the cell owns
        T* empty = nullptr;
        if (!cell_.compare_exchange_strong(empty, owned.get(), std::memory_order_acq_rel)) return false;
        std::ignore = owned.release();
        return true;
    }
    [[nodiscard]] Ptr swap(Ptr owned) noexcept { return Ptr{cell_.exchange(owned.release(), std::memory_order_acq_rel)}; }
    static_assert(std::atomic<T*>::is_always_lock_free);
private:
    std::atomic<T*> cell_{nullptr};
};

class ContextPool {
public:
    static constexpr std::size_t kSlots = 2;
    using Slot = AtomicOwned<llama_context, llama_free>;

    // Claims every idle context, keeps the best fit, parks the rest. A concurrent take may see empty
    // slots for that instant and build fresh: a spurious miss, never a double hand-out.
    [[nodiscard]] std::optional<PooledContext> take(const SessionGeometry& request) noexcept {
        std::array<Slot::Ptr, kSlots> claimed;
        std::ranges::transform(slots_, claimed.begin(), [](Slot& slot) { return slot.take(); });
        auto fitting = claimed | std::views::filter([&](const Slot::Ptr& c) { return c && ContextCapacity{*c}.serves(request); });
        const auto best = std::ranges::min_element(fitting, {}, [](const Slot::Ptr& c) { return ContextCapacity{*c}.tokens(); });
        Slot::Ptr chosen = best == fitting.end() ? Slot::Ptr{} : std::move(*best);
        std::ranges::for_each(claimed | std::views::filter([](const Slot::Ptr& c) { return static_cast<bool>(c); }),
                              [&](Slot::Ptr& c) { park(std::move(c)); });
        if (!chosen) return std::nullopt;
        return PooledContext{chosen.release(), ReturnToPool{*this}};
    }

    [[nodiscard]] PooledContext adopt(Slot::Ptr fresh) noexcept { return PooledContext{fresh.release(), ReturnToPool{*this}}; }

private:
    friend ReturnToPool;

    void put_back(Slot::Ptr context) noexcept {
        llama_synchronize(context.get());             // the pass may still be in flight; a native call, so before any atomics
        park(std::move(context));
    }

    void park(Slot::Ptr context) noexcept {
        if (std::ranges::any_of(slots_, [&](Slot& slot) { return slot.try_place(context); })) return;
        // Every slot is taken: evict round-robin. The evicted context is freed here, on the releasing thread.
        const std::size_t victim = next_.fetch_add(1, std::memory_order_relaxed) % kSlots;
        Slot::Ptr evicted = slots_[victim].swap(std::move(context));
    }

    std::array<Slot, kSlots> slots_{};
    std::atomic<std::size_t> next_{0};
};

inline void ReturnToPool::operator()(llama_context* context) const noexcept {
    pool.get().put_back(ContextPool::Slot::Ptr{context});   // the deleter's one raw pointer: handed straight back to an owner
}
```

`LlamaSession` holds `PooledContext context_;` and has no destructor body; `recycle` no longer exists. ABA is not a concern: placement only ever CASes from empty, and eviction is an unconditional swap. `ContextCapacity(const llama_context&)` reads four plain fields; it is the one allowed caller of those getters (`skippy-single-call-site`). The raw pointer exists in exactly two spellings — inside `AtomicOwned`'s cell and in the deleter's parameter, both immediately re-owned — which is what `skippy-no-raw-pointer` allows and nothing more. Contracts: `static_assert(!std::is_copy_constructible_v<PooledContext>)`; `static_assert(!requires(ContextPool& p, ContextPool::Slot::Ptr c) { p.put_back(std::move(c)); })`.

---

## O5 — Moves and scratch

- **Handles:** the C++ registry tokens `Model`/`Session` keep `id_(std::exchange(other.id_, std::nullopt))` and deleted move assignment; pin with `static_assert(!std::is_move_assignable_v<Model>)`. Rust never sees them after O2.
- **Owners:** `= default` moves over `unique_ptr` members; a user-written move constructor needs a one-line reason.
- **Pooled contexts:** `PooledContext` is movable (into a session); the deleter travels with the pointer.
- **Arguments:** nothing in a call reads an object another argument of the same call moves from (`bugprone-use-after-move`); O4 removed the only instance.
- **Factories** return by value, `[[nodiscard]]`, inside `expected`; `std::move` on a return is never written (`-Wpessimizing-move -Wredundant-move`).
- **Scratch is a `std::vector` sized once.** `native_tokens_.resize(max_tokens)` at construction (d9822260), `embedding_` sized to `rows × width` on first use, `prompt_`/`batch_tokens_` for `rerank_batch` (P1-11). The policy — *no allocation in a steady-state `embed`/`rerank`/`tokenize`* — is pinned by an allocation-count test in the Rust suite, not by a `Scratch<T>` type.

---

## O6 — Arena: owner, positions, and views all the way to the ABI

The staged C ABI is asymmetric: the planner hands identities *out* as `(offset, length)` refs (`skippy_stage_plan_string_ref_v1`, patch 0001:1590), but the loader config takes them *in* as `const char* const*` (patch 0001:1131). Every `const char*` in the façade exists only to satisfy that second half. Two halves to the fix, per the AGENTS porting procedure: the upstream-shaped half, and what the façade does until it lands.

### 6.1 The upstream-shaped half: give the loader a view type

A new patch in `skippy/llama_cpp/patches`, owned by the ABI-definition boundary, additive, with the PATCH bump mirrored in `skippy-ffi`:

```c
/** @brief A borrowed byte range. Not NUL-terminated; any byte is valid. */
struct skippy_bytes_ref { const char * data; size_t length; };

struct skippy_runtime_config_v2 {
    uint32_t abi_version;                       /* SKIPPY_RUNTIME_CONFIG_V2_ABI_VERSION */
    uint32_t struct_size;
    /* …the v1 scalar fields unchanged… */
    const struct skippy_bytes_ref * resident_tensor_names;  size_t resident_tensor_name_count;
    const struct skippy_bytes_ref * activation_import_identities; size_t activation_import_identity_count;
    const struct skippy_bytes_ref * activation_import_bindings;
    const struct skippy_bytes_ref * activation_export_identities; size_t activation_export_identity_count;
    const struct skippy_bytes_ref * activation_export_bindings;
    struct skippy_bytes_ref execution_contract;
};
skippy_status skippy_model_open_v2(const char * path, const struct skippy_runtime_config_v2 * config,
                                   skippy_model ** out_model, skippy_error ** out_error);
```

The native side already copies into `std::vector<std::string>` (patch 0003:2510), so each field is a one-line `std::string(ref.data, ref.length)`. `UPSTREAM.md` gets the entry; `benchmarks/cases.txt` gets a `blocked-upstream` row until it merges. This also retires the terminator-and-no-NUL rule below and the whole "are identities text or binary?" question (P2-7): a view carries any bytes.

### 6.2 The façade: views everywhere, `const char*` nowhere outside the ABI struct

```cpp
class IdentityArena {                                             // owns the bytes; the only writer
public:
    [[nodiscard]] std::expected<ArenaSlice, PlanError> put(std::string_view bytes);   // narrow<> inside; see fallback below
    [[nodiscard]] std::string_view view(ArenaSlice slice) const noexcept;            // clamped
private:
    std::string bytes_;
};
struct ArenaSlice { std::uint32_t offset; std::uint32_t length; };                   // a value: survives copies of the plan

struct StageLoadConfig {                                           // the public surface: views only
    LayerRange layers;
    std::int32_t stage_index;
    std::span<const std::string_view> resident_tensor_names;       // into the plan arena; valid for the lease
    std::span<const std::string_view> import_identities, import_bindings, export_identities, export_bindings;
    std::string_view execution_contract;
};

class PlanLease {
public:
    [[nodiscard]] std::span<const StageLoadConfig> configs() const& noexcept { return configs_; }
    std::span<const StageLoadConfig> configs() && = delete;
    // The only function in the tree that produces the ABI's raw pointers. v2: refs over the views, no copy.
    // v1 (fallback): the arena is terminated, so each view's data() is a C string.
    [[nodiscard]] skippy_runtime_config_v2 native(std::size_t stage, const StageRequest& request) const noexcept;
private:
    friend class PlanStore;
    std::shared_ptr<const PlanIndex> pinned_;                      // first: destroyed last
    std::vector<std::string_view> views_;                          // backing for every span above
    std::vector<skippy_bytes_ref> refs_;                           // v2 arrays, built once per lease from views_
    std::vector<StageLoadConfig> configs_;
};
```

`StagedModel::open(path, lease.native(i, request))`; the probes and the harness call the same line; `own()`, `StagedSpec`, `hex_of` and every per-identity `std::string` go. Rust reads identities through the lease's `&self` accessors (O2) as `&[u8]` — the same views.

Until 6.1 lands, the house rule applies — feature-test, fallback, and a test that flags when the fallback can go:

```cpp
#if defined(SKIPPY_RUNTIME_CONFIG_V2_ABI_VERSION)
    // refs_ built from views_; arena stores raw bytes; put() accepts any byte
#else
    // put() appends a '\0' the slice does not count and rejects an interior NUL; native() hands out
    // view.data() through a lease-owned std::vector<const char*> — the one place a raw pointer is spelled
#endif
// tests/toolchain_test.cpp: EXPECT_TRUE(defined(SKIPPY_RUNTIME_CONFIG_V2_ABI_VERSION)) << "drop the terminated-arena fallback";
```

Positions are values, so `StagePlan` is copyable (`clone()` goes) and both `arena_put`s collapse into `put`. Contract: `static_assert(!requires(skippy::StageLoadConfig c) { c.resident_identities.data(); } || true);` — more usefully, `skippy-no-raw-pointer` keeps every raw pointer out of `StageLoadConfig` and `PlanLease`'s public surface; `native` is the one allow-listed name.

---

## O7 — Across the bridge

| C++ | crosses as | Rust holds | who frees |
|---|---|---|---|
| `std::unique_ptr<T>` in a result struct | `UniquePtr<T>` | owned | Rust's `Drop` runs the C++ destructor |
| `&self` accessor returning `std::span<const T>` / `std::string_view` | `&[T]` / `&str` with the `&self` lifetime | a borrow of the C++ object | nobody: it is a view |
| `Pin<&mut T>` method | mutation; earlier views are dead by the borrow rules | — | — |
| `&[T]` parameter | `rust::Slice<const T>`, consumed within the call, never stored | — | — |
| value structs (`WorkloadInfo`, `LayerRange`, statuses) | by value | owned copy | trivially |
| `Failure<Code>` | `{code}` + `last_error(&self) -> &str` while the object lives | — | — |

Rules this table encodes: C++ frees everything it allocates, in a destructor; Rust owns through `UniquePtr` and borrows through `&`; a view is valid until the next `Pin<&mut>` call on the same object, which is exactly what the Rust signature says; nothing is copied to cross.

---

## O8 — Contracts

```cpp
static_assert(!std::is_copy_constructible_v<skippy::llama::LlamaModel>);
static_assert(!std::is_copy_constructible_v<skippy::llama::LlamaSession>);
static_assert(!std::is_copy_constructible_v<skippy::llama::StagedModel>);
static_assert(!std::is_copy_constructible_v<skippy::llama::StageChain>);
static_assert(!std::is_copy_constructible_v<skippy::llama::PlanLease>);
static_assert(!std::is_copy_constructible_v<skippy::llama::PooledContext>);
static_assert(!std::is_move_assignable_v<skippy::Model>);
static_assert(std::is_nothrow_destructible_v<skippy::llama::LlamaSession>);
static_assert(std::is_nothrow_move_constructible_v<skippy::llama::StageChain>);
static_assert(std::is_trivially_copyable_v<skippy::Token>);
static_assert(std::is_trivially_copyable_v<skippy::ArenaSlice>);
static_assert(std::is_copy_constructible_v<skippy::StagePlan>);                 // positions are values (O6)
static_assert(!requires(skippy::llama::PlanLease&& l) { std::move(l).configs(); });
```

Destruction order stays as declaration order with its one-line comments (`backend_` first in `NativeRuntime`; `model_` before the pool in `LlamaModel`; `pinned` before `configs` in `PlanLease`); the ASan suite is the witness.

---

## O9 — The tool that replaces an ownership map

Not written by hand. A libTooling pass in `tidy/` (same build, same `Clang_DIR`) walks the public headers and emits, per class: owning members (`unique_ptr`/`*_ptr`) and their deleters, reference/pointer members, `const` methods returning `span`/`string_view` (views), non-`const` methods (invalidators). A second step reads `skippy-sys/src/lib.rs` and checks that every view accessor is declared `self: &T`, every invalidator `self: Pin<&mut T>`, and every `type T` with a C++ destructor is `Send` iff its C++ declaration says so. Output is a Markdown table under `docs/`, regenerated by the verify recipe: it cannot go stale and nobody edits it.

## Order of introduction

1. **O4** `ReturnToPool` + the min-of-three capacity (closes P0-A; smallest diff).
2. **O2** session surface in cxx: `tokenize`/`tokens`, `embed`/`embedding`, `rerank_prompt_template`; Rust wrappers; delete the copy loops in `bridge.cpp`; keep the registry for C++ callers.
3. **O2/O3** `PlanLease` as `UniquePtr<PlanLease>` with `&self` accessors; delete every `*_hex` field.
4. **O6** terminated `IdentityArena` + lease-owned pointer arrays + `StageLoadConfig::native()` (retires `clone()`, both `arena_put`s, `own()`, `hex_of`, and the six copy loops).
5. **O8** contracts; **O9** the generator, before any new type is added.
