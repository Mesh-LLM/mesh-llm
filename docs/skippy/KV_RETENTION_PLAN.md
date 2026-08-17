# KV Prefix Retention — Feature Plan

**Goal:** keep more prefixes reusable for longer, so repeated prompts skip cold
prefill. This is **not** about extending effective context length, and it is not
OS-style demand paging. The target is retention time and hit rate.

**Primary workload:** agentic traffic — stable system prompt plus tool schemas,
divergent tails, and the same logical conversation returning after a gap.

**Mesh premise:** requests from one user are round-robined across peers, so a
given peer sees a familiar prefix again after a long gap. Retention value per
hit is high and hit frequency is low. That shape favours a large slow tier —
but it also means routing can remove much of the problem before any tier is
built. Both are in scope below.

## Current behaviour

| Family | Payload | On eviction |
|---|---|---|
| Dense attention (llama, qwen3, deepseek2/3, glm4, minimax) | `ResidentKv` | native seq drop; **state is lost**, next request recomputes |
| Interleaved sliding-window attention (ISWA), including ISWA in a hybrid wrapper (Gemma 3/4 and any family using the same llama.cpp memory types) | `ResidentKv` or `KvRecurrent` | composite base-cache + SWA-suffix page is persisted; recurrent state is also retained when present |
| Hybrid/recurrent (Qwen3Next, Falcon-H1, RWKV/Mamba) | `KvRecurrent` | `export_kv_page` + `export_recurrent_state` into an in-RAM BLAKE3 block store |

`ResidentKv` is a *performance* default, not a capability boundary: borrowing
resident state beats serialize/restore when it fits. The gate at
`crates/skippy-server/src/kv_integration/config.rs` is one-directional — it
stops recurrent models using KV-only reuse; nothing stops dense models using
the serialized path. Everything below is therefore family-agnostic in principle
and gated only by measurement and per-family certification.

MoE vs dense is **not** the relevant axis. DeepSeek3 is MoE and uses
`ResidentKv`. The axis is whether attention KV alone is the full continuation
state.

## Motivating scenario: partial reuse across a long gap

This is the case the plan exists for, and it needs **two** workstreams that are
often mistaken for alternatives.

A large prompt (say 8k tokens of system prompt plus tool schemas) was served a
while ago. A new request arrives with the same bulk and a **new tail**.

| Sub-problem | Solved by | Not solved by |
|---|---|---|
| "new tail" — reuse the bulk, prefill only the divergent part | the candidate grid + suffix prefill (**W2b**) | a disk tier |
| "not seen for a while" — the page was LRU-evicted from RAM and is gone | the mmap tier (**W4**) | the candidate grid |

So **W2b decides whether a reusable page exists at a shareable length; W4
decides whether it survived the gap.** Either alone yields nothing for this
scenario.

Why mmap suits the "massive bulk" shape specifically: restore cost is bounded by
the page's bytes and the kernel only faults in pages actually touched. A 4 GB
page still warm in page cache is nearly free; cold, it is one sequential read
(~1.4 s at 3 GB/s) against ~8+ s of quadratic-attention prefill. **The larger
the reusable bulk, the better the ratio** — the opposite of most caches.

Caveats: restore lands on a `shared_prefix_stride_tokens` floor, so up to
`stride - 1` tokens get re-prefilled — irrelevant against a multi-thousand-token
bulk. And a very long new tail after a restore must not exceed the runtime's
suffix-prefill limits, or it falls back to full recompute and the win is lost
(see `.agents/skills/kv-tool-loop-stability`).

### Skippy specifics: this applies to split serving *and* solo serving

KV retention is **per stage**, not per model. `prefix_hash_with_namespace`
(`skippy-cache/src/identity.rs:50-56`) hashes `stage_id`, `stage_index`,
`layer_start`, and `layer_end`, so:

- **Solo serving** — one stage covering all layers; one KV page per prefix.
- **Split serving** — each node owns a layer range and caches KV **only for its
  own layers**. A node holding layers 0–19 stores a page for 0–19; the node
  holding 20–39 stores a separate page. They are distinct `page_id`s and are not
  interchangeable.

Consequences specific to split topologies:

- A cold prefill on **any** stage in the chain costs the whole request. Retention
  has to hold across every stage for the pipeline to benefit, so per-stage hit
  rate matters more than aggregate hit rate. One stage missing negates upstream
  hits.
- Per-stage pages are **smaller** than a whole-model page — bytes scale with that
  stage's layer count — which improves W5's bandwidth ratio per node and makes a
  disk tier cheaper per node than the solo numbers suggest.
- Package-backed stages already cache only their own layer range, so W4 composes
  with materialized stage caches without loading a monolithic GGUF.
- Because the layer range is in the hash, **re-splitting
  invalidates every page.** A mesh that replans topology loses its entire
  retention benefit. Worth measuring how often replanning happens before
  investing in W4/W5.
- **Gap: activation frames have no serialize path.** `ResidentActivationCache`
  (`skippy-cache/src/resident/activation.rs`) is resident-only — there are no
  `activation` references in `payload/mod.rs` or `exact_state.rs`. So the
  activation-frame reuse that removes work at a *stage boundary* cannot survive
  eviction or a restart at all, and W4 as scoped does not cover it. Whether to
  extend the mmap tier to activation frames is an open question; it may be the
  larger split-serving win, since an activation frame is far smaller than a KV
  page.

## Verified starting facts

Each confirmed against the tree at `5bf7330d` (branch
`feat/kv-cache-disk-tier`).

| Fact | Evidence |
|---|---|
| KV quantization is **already executable** | `inference/skippy/resolver/support.rs:163-167` maps `saver` → `cache_type_k/v = "q8_0"`, `kv_offload = true`; reaches `StageConfig` via `resolver/translation.rs:83`; `skippy-protocol/src/lib.rs:280-282`; parsed at `skippy-runtime/src/config.rs:229` → `GGML_TYPE_Q8_0`. Default policy is `balanced` (`resolver/resolution.rs:227`). |
| `saver` also regresses throughput | `support.rs:225-232` halves batch/ubatch and forces `parallel=1`, `continuous_batching=false`. Set `cache_type_k/v` explicitly instead of shipping the macro. |
| `export_kv_page`/`import_kv_page` are **already on the serving path** | record at `kv_integration/exact_state.rs:125`, restore at `:65`, plumbed via `runtime_state/lane_lifecycle.rs:229,257`. |
| …but only for `KvRecurrent`, and only whole-prefix | `exact_state.rs:118` selects on payload; `export_kv_page(session_id, 0, token_count)` hardcodes `token_start = 0`. |
| `KvPageDesc` is genuinely page-granular | `skippy-ffi/src/lib.rs:556-570` carries `token_start`, `token_count`, `k_type`, `v_type`, `k_row_bytes`, `v_row_bytes`, `payload_bytes`. The cache layer flattens this to a single page. |
| Prefix identity omits KV dtype and backend | `skippy-cache/src/identity.rs:47-70` — zero `cache_type` references. `NATIVE_KV_DTYPE` is the fixed string `"ggml-native-kv"` and does not vary with q8_0 vs f16. |
| The candidate policy already does approximate sharing | `skippy-cache/src/config.rs:114-190` synthesizes a stride-aligned grid; test at `config.rs:245` shows 2214/2231-token prompts sharing a 2176 candidate. |
| `max_resident_tokens = n_ctx/2` exists to fix a real wedge | `skippy-cache/src/config.rs:55-70`. |
| `trim_session` is used in serving | `runtime_state/frame_operations.rs:418`, `frontend/linear_proposal/execution.rs:303`, `binary_messaging/control_messages.rs:212`. |
| Prefix-affinity routing already exists | `mesh-llm-host-runtime/src/network/affinity.rs` (37 KB). |
| ABI is at 0.1.35 and requires exact match | `skippy-ffi/src/lib.rs:1-3, 25-27`. |

## Workstreams

### W0 — Identity completeness (blocker)

`prefix_hash_with_namespace` (`skippy-cache/src/identity.rs:47-70`) does not
hash `cache_type_k`/`cache_type_v`, backend (CUDA/Metal/Vulkan), or GPU-layer
split. In-process this is benign — one config, one layout. It becomes **silent
numerical corruption** the moment state outlives a process (W3) or crosses a
node (W6): flipping `kv_cache_policy` from `quality` to `saver` makes stale
q8_0 payloads collide with f16 `page_id`s and be imported as f16.

- Add KV dtypes, backend id, and GPU-layer split to the hash.
- Confirm from the patch queue whether `skippy_import_kv_page` validates
  `KvPageDesc.k_type`/`v_type`/`k_row_bytes` against the live context. If it
  does not, that is a native fix plus a `SKIPPY_ABI_VERSION_PATCH` bump in both
  `skippy/common.h` and `skippy-ffi/src/lib.rs`.
- Make a desc/payload mismatch a **hard error**. Today `exact_state.rs:66-68`
  `continue`s past a `None` desc with non-empty kv bytes — a silent miss.

One-file change, silently invalidates existing in-RAM entries (harmless).
**Prerequisite for W3 and W6.**

Evidence: unit test proving two configs differing only in `cache_type_k`
produce different `page_id`s.

### W1 — KV quantization as retention policy

Already executable; this is config, docs, and defaults work, not new
machinery. q8_0 roughly halves KV footprint → ~2× resident entries.

- Expose `cache_type_k/v` as a retention knob distinct from the `saver` macro,
  so operators get quantization without the `parallel=1` throughput hit.
- Decide whether `balanced` should default to q8_0 for large-context serving.

Note: q8_0 KV is lossy. It shifts logits, so `skippy-correctness` parity
baselines need rebaselining, and it interacts with speculative/MTP verify
acceptance rates.

This is **orthogonal to, not a substitute for, a disk tier**: 2× capacity does
not help when the gap exceeds working-set turnover. They compound — q8_0 also
halves disk payload and disk read time, improving W4's ratio by 2×.

Evidence: `evals/skippy-openai-cache-matrix.py` f16 vs q8_0; resident-entry
count from `ResidentPrefixCacheStats`; a `skippy-correctness` parity run
quantifying logit drift; MTP acceptance-rate delta.

### W2 — Miss-reason instrumentation (gate for everything expensive)

Before building any tier, instrument the existing cache with a miss-reason
histogram: `evicted_recently` vs `never_seen` vs `identity_mismatch`, bucketed
by gap length since last use.

**If evicted-recently misses are rare, W3/W4 are worthless and this saves the
entire effort.** This is the cheapest possible way to validate the mesh
round-robin premise with real numbers rather than reasoning.

Extends `skippy.kv.*` attributes; must follow `.agents/skills/skippy-metrics`
and `.agents/skills/telemetry-privacy-review`.

### W2b — Deepen the shared-prefix record ladder (highest bandwidth win)

Cross-session prefix sharing **already works by design**, and this is where the
agentic system-prompt/tool-schema win lives. But the recording side is throttled
to the point where the shared prefix is almost never captured.

What already works:

- `prefix_hash_with_namespace` (`skippy-cache/src/identity.rs:47-70`) contains
  **zero `session_id` references**. Two unrelated sessions with the same leading
  tokens produce the same `prefix_hash` and the same `page_id`.
- The namespace is `base.chat_template_id`
  (`kv_integration/identity.rs:22`), fed from `ids.cache.namespace()`
  (`frontend/prefix_cache.rs:198`), which is `Some` **only** when the client
  sends `prompt_cache_key`
  (`frontend/generation/cache_hints.rs:111-115`). Ordinary requests get `None`
  → the shared default namespace → cross-session reuse.

The throttle — `family_policy.rs:107-109` sets
`shared_prefix_stride_tokens: 128` and **`shared_prefix_record_limit: 2`**.
`record_candidate_token_counts` (`skippy-cache/src/config.rs:150-184`) always
keeps the full length first, so with a limit of 2 only two lengths are ever
recorded. Simulating the real policy for an 8000-token request:

| | value |
|---|---|
| Lengths **probed** on lookup | 62 (8000 down to 256 in 128-token steps) |
| Lengths actually **recorded** | `[8000, 7936]` |
| Is a 2048-token shared system prompt recorded? | **No** |
| Would 2048 be found if it had been recorded? | **Yes** — it is probed |

So the lookup side is ready to exploit shared prefixes across sessions and the
record side never stores them. Both recorded entries sit at the *tail* of one
request, which is the least shareable part. A second session with the same
system prompt but a different tail probes 2048, finds nothing, and does a full
cold prefill.

This is a strong candidate for the largest win in the whole plan and it is
mostly a policy change:

- Record at least one **low, stable** candidate (near `min_tokens`, or aligned to
  a detected system-prompt/tool-schema boundary) rather than only the two
  longest.
- Consider a non-uniform ladder — a couple of tail candidates for
  same-session continuation plus a couple of low candidates for cross-session
  sharing. These two goals are currently in direct competition for 2 slots.
- Pairs naturally with W3: page-granular export makes recording several
  candidates cheap instead of O(prefix) bytes each.
- Note `derive_max_entries_from_kv_cells` (`family_policy.rs:101,132-148`) bounds
  entries by `n_ctx / (2 * min_tokens)`, so a deeper ladder competes for resident
  cells. Recording more candidates without more capacity just churns the LRU —
  which is precisely why this pairs with W1 (q8_0 doubles capacity) and W4.
- **Caveat:** a client sending `prompt_cache_key` *partitions* the namespace and
  thereby **disables** cross-session sharing. Worth documenting, and worth
  checking that agent harnesses are not setting it by default and silently
  losing the biggest win.

Evidence: with a fixed shared system prompt and N distinct tails across distinct
sessions, measure `skippy.kv.matched_prefix_tokens` and
`skippy.kv.cached_prompt_tokens` at `record_limit` 2 vs a deeper ladder. The
expected result is near-zero cross-session matched tokens today.

### W3 — Page-granular export

`KvPageDesc` already carries `token_start`/`token_count` but
`export_kv_page(session_id, 0, token_count)` always exports from zero. Combined
with `PrefixCandidatePolicy::record_candidate_token_counts` recording up to
`record_limit` overlapping prefixes, the same leading bytes are exported
repeatedly and the 1 MiB BLAKE3 dedupe claws them back after the fact.

Page-granular export **eliminates that work instead of deduping it**. Needs a
`token_start` plumb-through and a `Vec<(desc, bytes)>` payload variant. No ABI
change — the symbols exist.

This is a contained win independent of any disk tier, and it de-risks W4 and
W5. **Do this before W4.**

Evidence: before/after `CacheDedupeStats.hash_ms` and `hash_bytes`, and
`physical_bytes` at fixed workload — should drop sharply if the overlapping
-record waste is real.

### W4 — Disk tier via whole-payload mmap

`skippy_import_state` and `skippy_import_kv_page` both take contiguous
`(ptr, len)`, so `mmap` is a direct fit: zero-copy restore, kernel page cache
handles residency.

**Deduped blocks on disk is the wrong design.** `CacheBytes::as_cow()`
(`payload/bytes.rs:60-79`) allocates and concatenates for any `Blocks` repr, so
a block-based disk tier costs read syscalls plus a full-size heap allocation
plus concatenation immediately before the runtime copies again into device
memory — roughly 2 GB of pointless traffic for a 1 GB payload. Block dedupe's
value scales with *cross-entry* overlap, which in the agentic target is a
shared leading prefix that W3 captures structurally and more cheaply.

- Add `CacheBytesRepr::Mapped(Arc<Mmap>)` whose `as_cow()` borrows.
- Keep `CacheBlobStore` for the RAM tier. Do **not** put blocks on disk.
- `ExactStateCache` has no tiering concept — `record()` dedupes straight into
  RAM and `evict_until_within_limits()` drops. Needs a demote-on-evict hook,
  not a new cache type.
- Size cap, GC of orphaned files, and a persisted entry index for
  cross-restart reuse (safe only after W0).
- `model_fit.cache_ram_mib` is currently schema-reserved per
  `docs/skippy/CONFIGURATION.md` — natural home for the caps.

Evidence: hit-rate-over-gap-length curve from W2; restore latency vs cold
prefill at 2k/8k/32k.

### W5 — Export-on-eviction for dense families

Today dense eviction calls `drop_evicted(seq_id)` and the state is gone
(`resident/prefix.rs:230-260`). This is the change that makes retention
actually apply to the models people run.

Rough economics — disk wins iff
`kv_bytes_per_token / disk_BW < prefill_time_per_token`. For a 32-layer GQA-8
×128-dim shape at f16 (~128 KB/token) on NVMe at ~3 GB/s vs prefill at
~4k tok/s: ~4–6× favourable at f16, ~8–12× at q8_0. Quadratic attention makes
longer prefixes progressively better for the cache.

Where it loses: MLA/DeepSeek-style compressed KV (tiny bytes/token, fast
prefill); wide-KV MHA on slow or network-backed disk; and write amplification
starving read bandwidth during serving.

Two hard constraints:

- Dense eviction currently runs **on the decode hot path**
  (`binary_transport/kv_eviction.rs:104` → `evict_resident_prefix_for_tokens`).
  A synchronous multi-GB export there will spike TTFT badly. Export must be
  async/deferred.
- Deferred export means the seq cannot be dropped until export completes — a
  lifecycle change in `ResidentPrefixCache::evict_lru_entry`, which today drops
  synchronously then removes the entry. Getting this wrong gives either
  use-after-drop or a cell leak that re-triggers the 502 wedge
  `max_resident_tokens` was added to fix.

Gate behind a **runtime** admission check on measured bytes/token vs measured
disk bandwidth. Do not export unconditionally and do not hardcode the ratio.
Per-family restore certification required per
`.agents/skills/skippy-family-certification`.

Evidence: measured `bytes_per_token` per target model × measured local disk BW
vs measured prefill tok/s at 2k/8k/32k; TTFT distribution before/after to prove
the async path does not regress the hot path.

### W6 — Prefix-affinity routing (mesh)

The highest-leverage mesh item, and it was not in the original framing:
`network/affinity.rs` already exists. Hashing the request's leading-prefix
identity into peer selection makes the same prefix land on the same peer, which
**removes the round-robin premise motivating the disk tier at all**.

Routing change only. No wire-protocol impact, no correctness risk, no new
storage. Should be evaluated before committing to W4/W5 scope.

Evidence: per-peer prefix hit rate before/after on a 2-node private mesh, per
the confidence-testing shapes in `AGENTS.md`.

### W7 — Peer prefix fetch (speculative, likely not worth it)

Fetching a cached prefix from a peer over QUIC instead of recomputing. Three
problems:

1. **Identity is not peer-portable as written** — omits KV dtype and backend
   (W0). Two peers with different `kv_cache_policy` or different GPU vendors
   produce identical `page_id`s for incompatible bytes.
2. **`stage_id`/`layer_start`/`layer_end` are in the hash**, so a
   hit requires the same split. In a heterogeneous mesh that is exactly what
   does not hold. Unsplit peers serving the whole model would match; split
   meshes mostly would not.
3. **Economics** — 1 GB over LAN QUIC at ~1 GB/s is ~1 s, worse than local NVMe
   and comparable to recomputing. Over WAN it is strictly worse than recompute.
   Value exists only at 10 GbE+.

Also a new mesh wire surface: new stream type, additive gossip advertising held
prefixes, and a cache-poisoning trust problem — a malicious peer serving wrong
KV bytes is undetectable without re-verification. Mixed-version rules apply:
`mesh-llm/0` and older `mesh-llm/1` nodes must ignore it cleanly.

**Gate:** measure QUIC peer-to-peer throughput vs local NVMe vs local prefill
first. If peer BW < disk BW, W7 is dominated by W4 and should be dropped.

### Rejected

**Radix/tree prefix sharing (SGLang RadixAttention style).** The current design
is not a naive exact map — `PrefixCandidatePolicy` already does stride-quantized
approximate sharing (`config.rs:114-190`, test at `:245`). A radix tree upgrades
stride-floor-LCP to exact LCP, leaving ≤`stride_tokens` of prefill on the table
per hit; at a 128-token stride on a 2k+ shared system prompt that is <6%.

Cost: `resident/prefix.rs` is a flat `HashMap<String, ResidentPrefixEntry>` with
seq_id pooling, borrow tracking, and a hard `seq_id < 1024` ceiling
(`prefix.rs:315`). A radix tree needs node-level refcounting, partial-node
splits, and one llama.cpp seq per *node* rather than per entry — against that
1024 ceiling and the `max_resident_tokens` budget. Realistically a rewrite of
`prefix.rs` plus `resident_prefix.rs` plus the eviction path, and
`prefix.rs` is already 885 lines (the 1k rule in `AGENTS.md` applies).

Where a tree would genuinely win is *storage* — overlapping full-length copies
from `record_candidate_token_counts` — and W3 fixes that far more cheaply.

Revisit only if evidence says otherwise: distribution of
`LCP − stride_floor(LCP)` on real agentic traffic. If the median is <5% of
prefix length, do not build it.

**Partial/position-shift eviction via `trim_session`.** Wrong shape.
`skippy_trim_session` truncates a *session* to a length; cache entries are bare
`seq_id`s in the unified pool dropped via `skippy_session_drop_sequence`. You
would need `skippy_session_create_from_resident_prefix` + trim + drop a temp
session per eviction. And trimming keeps the prefix and discards the tail —
backwards for a shared-system-prompt workload, and already achievable by
recording a shorter grid candidate.

**KV defragmentation.** Zero `defrag` hits across `crates/` is expected, not a
gap. Cell allocation and compaction live inside llama.cpp's unified KV cache.
`NATIVE_KV_LAYER_CONTIGUOUS_LAYOUT` is a *serialization layout tag* hashed into
identity so exported bytes cannot be misread — not a residency invariant.
Fragmentation surfaces as "failed to find a memory slot" and the correct
mitigation is already in place (`max_resident_tokens = n_ctx/2`,
`config.rs:55-70`).

## What llama.cpp cannot provide

Block-table indirection. vLLM's non-contiguous KV blocks would need deep
surgery — contiguity is a load-bearing assumption
(`NATIVE_KV_LAYER_CONTIGUOUS_LAYOUT`, patch
`0062-Harden-Inkling-MTP-and-KV-contiguity-state`). Not proposed.

## Sequence

1. **W0** identity completeness — blocker, one file
2. **W2** miss-reason instrumentation — gate for the expensive work; also
   measures W2b's baseline
3. **W2b** deepen the shared-prefix record ladder — likely the largest win,
   mostly policy
4. **W1** KV quantization policy — config, already executable; supplies the
   capacity W2b needs
5. **W6** prefix-affinity routing — may remove the need for W4/W5
6. **W3** page-granular export — contained win, makes a deeper ladder cheap
7. **W4/W5** decide from W2 + W6 data
8. **W7** only if peer BW beats local disk BW

Dropped now: radix tree, partial eviction, defrag.

## Cross-cutting risks

- **Identity under-specification** (W0) — blocker for W4 and W7.
- **Silent-miss semantics** — `exact_state.rs:66-68` skips rather than errors on
  a `None` desc with non-empty kv bytes. Must become a hard error on a disk tier.
- **Async export lifecycle** (W5) — use-after-drop or cell leak re-triggering the
  502 wedge.
- **Hot-path latency** (W5) — eviction is on the decode path; export must not be
  synchronous there.
- **Lossy q8_0 KV** (W1) — rebaseline `skippy-correctness`; check MTP acceptance.
- **Mesh protocol** — W0–W6 are node-local. Only W7 touches the wire and must be
  additive per the mixed-version rules in `AGENTS.md`.
- **ABI sync** — W3/W5 need no ABI change. If page-granular export exposes a
  patch-queue bug, bump `SKIPPY_ABI_VERSION_PATCH` in `skippy/common.h` **and**
  `skippy-ffi/src/lib.rs` in the same change.
- **File size** — `resident/prefix.rs` at 885 lines is near the 1k threshold; any
  change touching it should extract rather than grow it.

---

## Implementation status (branch `feat/kv-prefix-retention`)

### Landed

| WS | State | Notes |
|---|---|---|
| **W0** identity completeness | done | Hashes `cache_type_k/v`, flash-attn, `n_gpu_layers`, backend device. Also **removes `topology_id`** and adds explicit **weight identity**. |
| **W2** miss-reason instrumentation | done | `PrefixMissTracker`: `evicted_recently` / `never_seen` / `identity_mismatch`, bucketed by gap. Bounded tombstone table, O(log n) trim. |
| **W2b** deeper record ladder | done | Keeps exact + near-tail, then geometric low candidates. Bounded by a **token** budget, not slot count. |
| **W4** mmap disk tier | done | `CacheBytesRepr::Mapped` borrows; `PrefixDiskTier` with checksums, atomic rename, LRU, directory lock. |
| **dense bridge** | done | Dense (`ResidentKv`) families archive at *record* time, restore on resident miss. This is what makes W4 useful for llama/qwen/gemma rather than recurrent-only. |

### Two findings that changed the design

**1. `topology_id` made persistence impossible.** Local serving derives it as
`topology-mesh-skippy-{unix_nanos}` (`runtime/local.rs`), so it is unique per
process. While it was hashed, every restart produced fresh `page_id`s: a
persistent tier could never read anything back. Removing it is required.

But removing it also removed an *accidental* safety property. `model_id` is a
display name, not a content digest — two runs can serve different tensors under
the same alias (different quant, repacked layer package, swapped GGUF). So
identity now hashes `manifest_sha256`, `source_model_sha256`, `package_ref` and
`load_mode` explicitly. Without that, removing `topology_id` would turn a
never-collides namespace into one keyed on a user-facing string, with silent
numerical corruption as the failure mode.

**2. Archiving the request's own tail is useless.** Capping to one archive per
request initially picked the *longest* recorded candidate — the request's own
tail, which nothing else ever asks for. Switching to the *lowest* candidate
restored cross-restart hits but archived too little to matter (see the split
section below). `ArchiveCandidate` now selects the longest prefix that still
excludes the request tail.

### Measured

Two shapes, single node, same binary, cold control = identical run against an
empty cache directory.

**Qwen2.5-0.5B dense, ~2k-token prompt** — the *worst* case for this work:

| Scenario | Cold | Warm |
|---|---|---|
| Cross-session, same prefix, different tail | 0.31–0.42s | **0.13–0.20s** |
| First request after restart | 0.31s | **0.25s** |

**Qwen3-8B Q4_K_M layer package, ~12.4k-token agent prompt** — the shape the
plan was actually written for:

| Scenario | Cold | Warm | Saved |
|---|---|---|---|
| Cross-session, same prefix, different tail | 21.50s | **8.85s** | 12.65s |
| First request after **process restart** | 21.60s | **8.85s** | 12.75s (2.44×) |

The archived page is **0.96 GB** for 12288 tokens. Restoring it beats
recomputing it by ~12.7s, and the restart case matches the cross-session case
exactly — the mmap restore costs the same whether the page was written by this
process or a previous one.

This is the ratio argument in the plan holding up: the win scales with the size
of the reusable bulk, because restore cost is bounded by bytes while prefill
cost is superlinear in tokens. The 0.5B number is small because there was
almost nothing to save, not because the mechanism is weak.

Historical note (superseded): at the time this section was written, multi-node
split topologies were unmeasured. They have since been measured on loopback and
on a physical LAN — see "Split serving" and the two-machine results below.

Still unmeasured: how often topology replanning invalidates pages in a live
mesh, WAN-latency splits, and very large MoE models.


## Split serving

Retention is per stage: `prefix_hash_with_namespace` hashes `stage_id`,
`stage_index`, `layer_start`, `layer_end`, so each stage caches only its own
layer range and the pages are not interchangeable.

### What was already there

The investigation found more existing machinery than the plan assumed:

- `PrefillChunkMessage` carries `token_ids` alongside activations, so
  downstream stages *can* compute a prefix hash.
- `KvStageIntegration` is constructed for every stage, and the per-stage
  resident cache in `binary_transport/binary_kv.rs` is not gated on
  `stage_index`.
- **A cross-stage agreement protocol already exists.**
  `try_restore_embedded_split_prefill` has stage 0 state its restore length on
  the wire, and `prefix_cache.rs` vetoes the entire attempt unless every stage
  reports a hit. Middle stages fold downstream stats into their own reply, so
  a miss at depth N propagates back to stage 0.

So per-stage disk restore did **not** need new negotiation. It is a third
tier under an existing gate.

### Three invariants that keep splits correct

These are load-bearing and were undocumented. Relaxing any one gives wrong
tokens rather than an error:

1. `config.layer_start == 0` — suffix-only execution after a partial hit is
   permitted only on the first stage.
2. `restored.token_count >= token_ids.len()` when `downstream.is_some()` —
   full-restore-only whenever a stage has a downstream.
3. The all-or-nothing veto in `prefix_cache.rs`.

Invariant 1 was previously enforced only as a side effect of a payload-size
check in the encoder. It is now an explicit named assertion in
`forwarding.rs`, with tests.

### Measured, 2-node split on loopback (0.5B, ~2.1k tokens)

| Scenario | Cold | Warm |
|---|---|---|
| Cross-session, same prefix, new tail | 1.34s | **0.57s** |
| First request after restarting both nodes | 1.33s | 1.32s |

Cross-session split reuse works. Restart reuse did **not** pay off at this
point in the branch, and the reason was visible in the archive index: stage 0
archived its full ladder including the 2048-token shared bulk, but the
downstream stage archived only a 512-token page. Because of the veto, one
stage's shallow archive negated the other's good one.

> **Superseded.** This was fixed by `e1c4af42` and `b32dec97`. See
> "It works on splits too (re-measured)" below for the current numbers:
> 31.02s cold -> 1.54s after restarting both nodes on an 8B split.

This also exposed a real bug in the first implementation: archiving the
*lowest* ladder candidate. For a 2129-token prompt that is a 256-token page —
12% of the prefill, indistinguishable from noise. `ArchiveCandidate` now
selects the longest prefix that still excludes the request's tail (2048 here,
96%), which is the actual shared bulk. That fix is in and unit-tested; the
remaining gap is why the downstream stage's ladder is shallower.

Not yet measured: a large MoE model across a real multi-machine split, which
is the shape where the solo 8B result (2.44x) suggests the payoff should be
largest.

### MoE

MoE is **not** a special case here. `family_policy.rs` maps MoE families
(`qwen3moe`, `glm4_moe`, `deepseek3`, `openai_moe`, `llama4`, `hunyuan_moe`,
`phimoe`, `ernie4_5_moe`, ...) to the same `resident_kv_policy` as dense
models. MoE changes FFN weights, not KV cache shape; attention KV is still the
full continuation state. Only hybrid/recurrent families (`qwen3next`,
`falcon_h1`, `mamba`, `rwkv`, `qwen35moe`, `nemotron_h_moe`) take the
`kv_recurrent` path. So the dense archive path covers the large MoE models
that motivate split serving.


## Expert review round 2

Two independent reviews (split correctness; disk tier + identity). Split
correctness came back clean on all five questions. The disk-tier review found
five real issues, all now fixed:

| Finding | Fix |
|---|---|
| Metadata (`extra`, the KV page descriptor) was not checksummed, only the payload. A corrupted-but-valid-JSON index could apply correct bytes under a wrong layout. | `extra_checksum` on every entry, verified before the payload. This was added before the format shipped, so the final on-disk contract remains `format_version: 1`; intermediate development builds are not compatible. Test: `tampered_metadata_is_rejected_even_though_the_payload_is_intact`. |
| Weights with no content digest could alias: two different GGUFs served under one `model_id` collide on disk. | The tier now refuses to open unless at least one of `manifest_sha256` / `source_model_sha256` is a valid 64-hex SHA-256 digest. A `package_ref` alone is not sufficient, so mutable local paths fail closed. |
| No directory locking on non-Unix; the tier silently ran unprotected. | Non-Unix now fails closed with an explicit error rather than running without a lock. |
| Crash-left `.tmp` files were never reclaimed, leaking a page's bytes per crash. | Reclaimed at open, which is safe because open holds the exclusive lock. Test: `crash_left_temp_files_are_reclaimed_on_open`. |
| Debug `eprintln!` instrumentation left in committed code. | Removed. |

Two review claims were checked and found **incorrect**: `page_id` is a full
BLAKE3 hex digest (not truncated to 64 bits), and the index *is* committed
atomically via write-temp-then-rename. Both were artifacts of the reviewer
seeing only an excerpt.

Re-measured after hardening, solo 0.5B, ~2.1k tokens: cross-session
0.42s -> 0.13s, restart 0.42s -> 0.21s. Two-node split smoke passes.


## Why retention matters more for splits than the solo numbers suggest

Three effects compound for large split models. The second is the largest and
was not in the original plan.

### 1. Prefix commonality is high in agent fleets

A fleet running one agent sends near-identical system prompt plus tool schemas
on every request. Hit *frequency* is high and the value per hit is a full cold
prefill, so the ideal is that a given prefix is warmed once per node ever.
That is a retention-time argument, and it is what the disk tier addresses:
survive eviction, survive restart, survive a redeploy.

### 2. A chain restore removes stage-boundary traffic almost entirely

This is the significant finding, and it is not a KV-cache effect --- it is a
*network* effect specific to split serving.

Prefill activations cross every stage boundary. For the restored span,
`embedded_prefix_cache_message` (`frontend/wire_messages.rs:264-288`) builds
the `TryRestorePrefill` message with **`activation: Vec::new()`** and carries
only `tokens`. So on a chain restore the restored prefix costs 4 bytes per
token on the wire instead of `hidden_width x dtype_bytes`:

| width | dtype | tokens | cold (activations) | warm (token ids) | ratio |
|---|---|---|---|---|---|
| 4096 | f32 | 128k | 2.10 GB | 0.5 MB | 4096x |
| 4096 | f16 | 128k | 1.05 GB | 0.5 MB | 2048x |
| 8192 | f16 | 128k | 2.10 GB | 0.5 MB | 4096x |

Per stage boundary, per request. On a thin or shared link this can dominate
total prefill time, which means **for splits the bandwidth saving may matter
more than the compute saving** --- and it is invisible in single-node
wall-clock benchmarks like the ones measured so far. Prefill compute
parallelizes across stages; the boundary transfer does not.

Corollary: the all-or-nothing veto is expensive here. A single stage missing
does not just lose that stage's compute, it forces full activation traffic
across *every* boundary for the whole prompt.

### 3. Pre-seeding is a natural consequence, not a separate feature

If prefixes are fleet-common and a hit removes both prefill compute and
boundary traffic, there is no reason to wait for organic traffic to warm a
node. The disk tier is already a content-addressed, checksummed,
identity-anchored page store; seeding it is a *write* into that store rather
than new machinery.

What makes this plausible now: the identity no longer contains `topology_id`
or any per-process value, so a page is valid for any process with the same
weights, layer range, KV dtype, and backend. What still blocks it: the layer
range and stage identity *are* hashed, so a seed is only valid for an
identical split. Seeding must therefore either happen after topology is
chosen, or be keyed by topology and matched at plan time.

Not designed or built. Recorded because the identity work needed for it is
already done.


## Seeding by replaying a prefix (measured)

The cheap version of pre-seeding needs no new machinery at all: **send the
canonical prefix through the node once as an ordinary request.** Recording
happens on the normal serving path, so a warmup run populates the disk tier
exactly as real traffic would.

Solo, 0.5B, ~2.1k-token agent prefix. The seed run sends the system prompt
with *no user turn*; the measured request then uses a tail the seed never saw,
on a freshly restarted process with cold RAM:

| | First real request |
|---|---|
| Unseeded (empty disk, cold RAM) | 0.41s |
| Seeded (disk warmed by one prior prefix-only run, cold RAM) | **0.21s** |

The seed run archived a single 2048-token page --- the shared bulk --- and a
later process with an unrelated tail hit it. So a node can be useful on its
*first ever* real request, which is the "never warm up again" property.

Why it works without new code: identity contains no `session_id` and no
per-process value, so a page recorded by a warmup run is valid for any later
session or process with the same weights, layer range, KV dtype and backend.
`ArchiveCandidate` also prefers the longest *partial* candidate, which is
correct here --- a seed prompt's full length is exactly the shared prefix.

### It works on splits too (re-measured)

The earlier ratchet is fixed. `e1c4af42` stopped `min_record_tokens` from
starving the archive selector on a warm restore (candidates below the restore
point are still offered to the archive; only resident re-recording is skipped)
and moved archival out of the `config.downstream.is_some()` branch, so the last
stage of a chain archives at all. `b32dec97` then removed the resident-admission
gate on stage-0 full prefill, which had been declining every rung of a prompt
larger than `max_resident_tokens`.

Re-measured on a 2-node loopback split, Qwen3-8B Q4_K_M layer package,
~16.9k-token agent prompt, `--max-vram 10` per stage, disk tier at 8 GiB per
stage:

| Scenario | Time |
|---|---|
| Cold, empty disk cache both stages | 31.02s |
| Cross-session, same prefix, new tail | **1.29s** / 0.91s |
| First request after restarting **both** nodes | **1.54s** |

That is a **20x** restart win where the previous measurement showed none
(1.33s vs 1.32s on 0.5B). The per-stage archive ladders now look like:

```text
stage 0 : [16768]                                   <- one page, the shared bulk
stage 1 : [16768, 16640, 16512, ... ] (88 entries)  <- full ladder
```

Both stages archive the 16768-token shared bulk, so the all-or-nothing veto is
satisfied and the chain restores. Seeding is therefore **not** solo-only.

### Confirmed on two physical machines

The measurement above is a loopback split -- two processes on one host, sharing
a disk and a GPU. That validates the agreement protocol but not the things only
a real network can break: page identity agreeing across two independently
configured hosts, restore negotiation over real QUIC latency, and each node
owning a separate cache directory on separate storage.

Repeated across an M4 Pro and a Mac mini on a LAN (13-14ms direct QUIC),
Qwen3-8B Q4_K_M layer package, `--max-vram 4 --ctx-size 16384` per node, 8 GiB
tier per node. Stage 0 held layers 0-22 on one machine, stage 1 layers 22-36 on
the other:

| Scenario | Time | cached_tokens |
|---|---|---|
| Cold, both caches empty | 12.45s | 0 |
| Cross-session, same prefix, new tail | **1.49s** / 1.53s | 4096 |
| First request after restarting **both** machines' nodes | **1.72s** | 4096 |

Both nodes independently persisted `format_version: 1` indexes of
`resident-kv-archive` entries and reloaded them into a freshly negotiated
topology, so the restored page ids agreed across hosts. Stage 1 also exercised
budget eviction (`disk_evictions` 0 -> 3 -> 5 against a 2 GiB stage share) while
still serving hits, which the loopback run never reached.

#### Forcing the split deterministically

Do **not** reproduce this by tuning `--max-vram` until a split happens. That is
what the first attempt did, and it is both unreliable and actively misleading:
at `--max-vram 5` one node quietly took all 36 layers and served solo while
still showing a healthy two-node mesh, and equal artificial caps on both nodes
can also flip coordinator election, since the coordinator is the participant
with the greatest advertised VRAM.

Use the documented mechanism instead (`docs/SKIPPY_SPLITS.md`): `--split` to
force staged serving even when the model would fit locally, plus
`--split-topology-lock` to pin the exact nodes and layer ranges. The lock is
fail-closed -- if the pinned ranges do not fit, startup fails rather than
silently falling back.

```json
{
  "version": 1,
  "model": "meshllm/Qwen3-8B-Q4_K_M-layers",
  "manifest_sha256": "<sha256 of the raw model-package.json bytes>",
  "stages": [
    { "node": "<stage-0 endpoint id>", "layer_start": 0,  "layer_end": 32 },
    { "node": "<stage-1 endpoint id>", "layer_start": 32, "layer_end": 36 }
  ]
}
```

Points worth knowing, each of which cost time to rediscover:

- Stage 0 must be the node that would be elected coordinator: highest advertised
  VRAM, endpoint id as tie-break.
- Use **full endpoint ids**, not hostnames. Both lab machines advertise
  `mac.lan`, and a hostname selector must resolve uniquely among participants.
- `manifest_sha256` is the SHA-256 of the `model-package.json` file's bytes, not
  a digest recorded inside it.
- Place the identical lock on every serving node and pass both flags on each.
- The join must be directed at the node whose advertised addresses are
  reachable; joining toward an unreachable address times out at 30s and falls
  back to standalone, which presents as a discovery failure but is not one.

Verify with `GET /api/runtime/stages` on the coordinator, not `/v1/models`.
`/v1/models` proves the model is routable; only the stages endpoint proves two
distinct endpoint ids own disjoint layer ranges. That distinction is exactly
what the accidental solo-serving run above would have hidden.

#### Re-measured under a locked topology

Same two machines, `--split --split-topology-lock`, `--ctx-size 8192`, stage 0
layers 0-32 on the M4 Pro and stage 1 layers 32-36 on the mini, confirmed
through `/api/runtime/stages`:

| Scenario | Time | cached_tokens |
|---|---|---|
| Cold, both caches empty | 5.48s | 0 |
| Cross-session, same prefix, new tail | **1.39s** / 1.29s | 4096 |
| First request after restarting **both** nodes | **1.48s** | 4096 |

Both nodes again persisted `format_version: 1` indexes independently (stage 0
one entry, stage 1 a 34-entry ladder) and restored into a freshly negotiated
topology.

### Not done

- **W1** KV quantization, **W3** page-granular export, **W5** export-on-eviction,
  **W6** prefix-affinity routing, **W7** peer fetch.
- Activation frames still have no serialize path, so stage-boundary reuse does
  not survive eviction. Likely the larger split-serving win.
- Miss-reason stats are collected but **not yet exported** as OTLP metrics, so
  W2 cannot yet answer the W4/W5 go/no-go question in production.

### Operational notes

The tier is **opt-in**: `SKIPPY_KV_DISK_TIER_MIB=<size>` (or
`SKIPPY_KV_DISK_TIER=1`), with `SKIPPY_KV_DISK_TIER_DIR` overriding the
location. Default-off means the serving path is unchanged apart from the
identity hash and ladder depth.

Cache directories are per stage-shape and hold an exclusive lock; a second
instance on the same directory declines the tier rather than sharing it, because
the index is last-writer-wins and orphan reclaim would delete the other
instance's live files.

## Model coverage

Validated by live measurement:

| Model / topology | Attention memory | Actual validation |
|---|---|---|
| Qwen3-8B Q4_K_M, single node, ~6.3k-token agent prompt | plain KV cache | disk reuse: 5.16s -> 0.47s, 6272 cached |
| Qwen3-8B Q4_K_M, two physical nodes, ~16.9k-token agent prompt | plain KV cache, split by layer range | 31.02s cold -> 1.27s cross-session -> 1.54s after restarting both nodes |
| Gemma 4 26B A4B | composite ISWA | the earlier run correctly declined before composite support existed; **no live post-support restart-reuse result has been recorded yet** |

**MoE is not the axis.** Experts live in feed-forward layers and carry no state
between tokens. The relevant question is whether Mesh can export the model's
complete continuation state.

Current composite ISWA support handles the two-cache shape used by
`llama_kv_cache_iswa`: a full-prefix base cache plus the visible suffix from the
window-bounded SWA cache. It also recognizes `llama_memory_hybrid_iswa`; in that
case the composite attention page is retained together with the recurrent state
already carried by `KvRecurrent`. Import is intentionally restricted to a fresh
session so both component caches advance from the same restored prefix.

Coverage added with the codec includes Rust tests for descriptor JSON
round-tripping, payload/range validation, overflow and corruption rejection,
and legacy/single-page compatibility. That is structural validation, not a
model-level correctness or performance result. A live Gemma 3/4 cold, warm,
and post-restart probe remains required before claiming the composite path is
certified.

Remaining unsupported cases are hybrid or future llama.cpp memory layouts that
cannot expose their *complete* continuation state as plain KV, KV plus recurrent
state, or the two-component ISWA codec. Those layouts still fail closed: the
unsupported result is latched per stage, telemetry reports
`skipped_unsupported_memory`, and serving continues without disk retention.
Do not infer support merely from a family label such as “hybrid” or “SWA”; the
runtime memory type and a live restore validation are authoritative.
