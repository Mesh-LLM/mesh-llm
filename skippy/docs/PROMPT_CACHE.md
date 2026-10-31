# Skippy OpenAI prompt cache

Skippy reuses exact token prefixes through a radix tree. A cache entry owns its
native continuation state independently of a request or an execution lane.
Recycling a lane releases the request's state; it does not discard cache-owned
prefixes.

## Shared attention and recurrent checkpoints

The loaded runtime admits complete cache representations from its graph and
memory capabilities. Serving does not choose a representation by model name.

For an admitted resident representation:

- Attention KV remains in the native memory pool. Cache sequence references
  share existing cells; retaining a prefix does not export those cells to RAM.
- A hybrid or recurrent prefix also owns an immutable recurrent checkpoint at
  a recorded token boundary. Restoring it copies the recurrent component into
  an execution lane and shares the corresponding attention cells.
- A request holds a lease while restoring an entry. Eviction cannot remove a
  leased source. Divergent continuations cannot mutate the saved checkpoint.
- Dense attention can reuse the common part of a longer stored prefix.
  Recurrent state can resume only at a saved checkpoint whose entire token
  path is a prefix of the request. A longer recurrent checkpoint cannot be
  truncated to an arbitrary common-prefix length.

This follows SGLang's radix ownership model: token paths identify shared KV
blocks, while recurrent state has separate checkpoint ownership. It is an
adaptation to llama.cpp's native memory interfaces, rather than a dependency
on SGLang's Torch-based runtime. See the pinned upstream
[radix cache](https://github.com/sgl-project/sglang/blob/1f152be6c5e1a36f27f1800762f5e0131f127a41/python/sglang/srt/mem_cache/radix_cache.py)
and [Mamba cache component](https://github.com/sgl-project/sglang/blob/1f152be6c5e1a36f27f1800762f5e0131f127a41/python/sglang/srt/mem_cache/unified_cache/components/mamba.py).

## Identity and lifetime

A hit must match the model weights, layer range, runtime state layout,
position rules, namespace, and token prefix. The radix key describes these
inputs; serving does not discover common prefixes by hashing exported KV
payloads. An optional client `prompt_cache_key` separates namespaces. Requests
without a key share the default namespace.

Run and topology instance IDs do not fragment the namespace. Equivalent
placements can reuse state when the numerical model and state layout match.

A saved native prefix remains usable after the originating session is freed.
Eviction drops the cache's native attention references and recurrent checkpoint.
The attention pool releases cells once no cache entry or request references
them. Unloading the model releases its retained state.

## Capacity

Attention admission uses physical occupancy of the shared native pool. Adding
logical prefix lengths would count shared cells repeatedly and evict reusable
state prematurely. Active restore leases protect their source from eviction.

Retained attention cells also affect the physical span read by llama.cpp's
attention kernels. Masked idle cells can increase attention work even when a
request gets a cache hit. This ownership change does not port SGLang's paged
attention kernels; benchmark both reuse and attention cost as the retained
working set grows.

Recurrent checkpoints consume host RAM independently of the attention pool.
Their budget is bounded and does not require a disk cache. An entry that cannot
fit the checkpoint budget is not retained; it cannot remain indefinitely over
budget merely because it is the last entry. Standalone
`--prefix-cache-ram SIZE` or Mesh's `model_fit.cache_ram_mib` supplies this
budget. A zero/unset value selects the bounded 1 GiB native-cache default;
disable prefix caching itself to retain no entries.

## Persistence boundary

The resident path does not eagerly export complete KV state into a blob store
or mirror every new prefix to disk. When persistent L3 is enabled, eviction is
the cold offload boundary: serving exports the retained attention cells and
their recurrent checkpoint, then queues a bounded durable write before
releasing native cache ownership. Disk payload integrity remains verified.

An L3 fill restores the complete state after a resident miss and can warm the
native radix cache again. Completed durable records survive a process restart.
Entries still held only in native memory, and writes interrupted before their
manifest commits, do not. Model unload and shutdown do not promise to flush
every hot prefix. A refused or failed best-effort offload becomes a cache miss
and cold prefill, rather than an inference failure.

Durable identity includes the native ABI and state layout. Older serialized
entries are not imported into the new runtime; they miss and are recomputed.
No migration of the superseded cache representation is required.

Portable state export/import also remains available for explicit transfer and
correctness checks. Runtimes without a complete resident representation use
their admitted serialized representation. See [the disk-tier
guide](KV_CACHE_DISK.md) for persistent store controls.

## Observing reuse

`usage.prompt_tokens_details.cached_tokens` reports cached input tokens in the
OpenAI response. Generation telemetry also reports cache status, matched prefix
length, suffix-prefill tokens, and hit kind. A match identifies reusable state;
it is not proof that generated tokens are correct.

Validate cold, repeated, changed-tail, and lane-pressure requests separately.
Use uninterrupted continuations as the correctness reference for restored and
forked state, and compare cache reuse and timings against vanilla llama on the
same model, prompt tokens, output limits, and concurrency. Report actual native
and Rust binary provenance with the results.
