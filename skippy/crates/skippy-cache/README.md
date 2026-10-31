# skippy-cache

`skippy-cache` owns prefix identity, the radix index, reference leases, eviction
policy, activation caching, and portable state storage. It does not execute a
model, call native code, serve HTTP, or choose topology. Its primary consumer
is [`skippy-serving`](../skippy-serving/README.md), which connects these data
structures to [`skippy-runtime`](../skippy-runtime/README.md).

## Resident ownership

Token paths in the radix tree identify retained native state. Entries own
native cache sequence references independently of execution lanes. Attention
cells can be shared by several prefixes and active requests. Recurrent models
also need immutable checkpoints at exact saved token boundaries; a common
prefix inside a later checkpoint is not a valid recurrent restore point.

The index distinguishes these matching rules and pins selected entries while
serving restores them. Eviction removes only unpinned entries. Serving releases
the corresponding native resources when it commits an eviction.

Identity binds tokens and namespace to numerical model weights, layer range,
runtime ABI, layout, dtype, and position configuration. Payload hashes are not
used to discover shared resident prefixes. Attention admission uses physical
native occupancy rather than summing overlapping logical prefix lengths.
Recurrent checkpoint RAM is bounded separately and works with disk disabled.

This adopts SGLang's separation of radix-owned attention blocks and recurrent
checkpoints through llama.cpp's native memory interfaces. It does not embed
SGLang's Torch runtime. See [the prompt-cache design](../../docs/PROMPT_CACHE.md)
for lifetime, matching, budget, and persistence behavior.

## Portable state

The crate also owns `FullState`, `RecurrentOnly`, and `KvRecurrent` payloads,
BLAKE3-addressed storage, and durable manifests for explicit transfer and
runtimes without complete resident support. These APIs remain separate from
warm resident retention. Resident-capable serving does not eagerly export,
reconstruct, or hash the attention payload on each cache record. Configured
persistent L3 receives cold eviction offloads and can fill resident misses
after a restart; payload integrity verification stays at that storage boundary.

Activation frames are cached separately so a split stage can reuse the boundary
activation associated with its exact prefix. Cache data structures remain
backend neutral; native memory ownership and import/export belong to runtime
and serving adapters.

Run the crate's tests from the workspace root:

```bash
cargo test -p skippy-cache
```

The [previous serialized-cache measurements](https://github.com/Mesh-LLM/mesh-llm/blob/18fef3a042847463f13844304b0e9100e49a2f6b/skippy/crates/skippy-cache/README.md#benchmark-evidence)
remain historical evidence; they do not qualify the new ownership path.
