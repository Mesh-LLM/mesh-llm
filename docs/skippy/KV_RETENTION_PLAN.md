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
| Dense attention (llama, qwen3, deepseek2/3, glm4, gemma, minimax) | `ResidentKv` | native seq drop; **state is lost**, next request recomputes |
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
- Because `topology_id` and the layer range are in the hash, **re-splitting
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
2. **`topology_id`/`stage_id`/`layer_start`/`layer_end` are in the hash**, so a
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
