# KV disk tier on-disk format

This is the normative description of the format the KV prefix disk tier writes,
implemented by `crates/skippy-cache/src/disk_tier.rs`. It documents what is on
disk, what guarantees the reader makes, and what a change to any of it obliges
you to do.

The tier is a **cache**. Nothing in it is authoritative and nothing in it is
migrated. Every recovery path in this document ends in "discard and recompute",
because a cache that starts empty is merely slow while a cache that serves the
wrong bytes is numerically wrong.

The tier also declines to open at all when the stage configuration carries no
valid content digest (`manifest_sha256` or `source_model_sha256`). A page that
outlives its process must be anchored to the weights' *content*, never to the
display name, or a re-quantized model published under the same alias is served
as a hit.

## Directory layout

```text
<base>/<stage-key>/
  owner.lock            # advisory exclusive lock, held for the tier's lifetime
  index.json            # entry metadata for the whole directory
  pages/
    <blake3(page_id)>.kvp             # one contiguous file per entry
    <...>.kvp.<pid>.<counter>.tmp     # in-flight write, reclaimed on open
```

`<base>` defaults to `$MESH_LLM_HOME/kv-cache` (or `~/.mesh-llm/kv-cache`).

`<stage-key>` is the first 16 hex characters of a BLAKE3 over `model_id`,
`stage_id`, `stage_index`, `layer_start`, and `layer_end`. It is hashed so no
model name or filesystem path text appears in a directory name and so the
component has a fixed length.

Page file names are `blake3(page_id)`, not the page id itself, for the same
reason. A name read back from the index is validated as a plain path component
before it is ever joined onto the root: no empty string, no `.` or `..`, no
separators, no NUL. An entry that fails that check is dropped.

## Index schema

`index.json` is a single JSON object:

```json
{
  "format_version": 1,
  "entries": [
    {
      "page_id": "…",
      "token_count": 16768,
      "file_name": "…​.kvp",
      "total_bytes": 2147483648,
      "components": [
        { "offset": 0, "len": 2147483648, "checksum": "<blake3 hex>" }
      ],
      "payload_kind": "resident-kv-archive",
      "extra": { "…": "opaque caller metadata" },
      "extra_checksum": "<blake3 hex>",
      "written_at_secs": 1786500000,
      "last_used_secs": 1786500100,
      "use_sequence": 42
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `format_version` | Whole-directory compatibility gate; a mismatch discards the directory |
| `page_id` | Prefix identity from `skippy_cache::prefix_identity` |
| `token_count` | Tokens the page covers, always starting at token 0 |
| `file_name` | Plain component inside `pages/`; never an absolute path, so the directory stays relocatable |
| `total_bytes` | Expected size of the page file; checked against the real file at load |
| `components` | Ordered byte ranges within the file, each independently checksummed |
| `payload_kind` | Payload discriminant; a load asking for a different kind is an error |
| `extra` | Opaque caller metadata, e.g. the `RuntimeKvPageDesc` a KV page needs to be importable |
| `extra_checksum` | BLAKE3 over the canonical encoding of `extra` |
| `written_at_secs` / `last_used_secs` | Wall-clock seconds, for observability and LRU |
| `use_sequence` | Monotonic counter that breaks LRU ties within one second |

`payload_kind` values are the stable strings from
`ExactStatePayloadKind::as_str`: `full-state`, `recurrent-only`, `kv-recurrent`,
`resident-kv-archive`. Component counts are fixed per kind — `kv-recurrent` has
two components, every other kind has one. **Changing any of these strings
requires a format-version bump.**

The index is JSON on purpose. It holds metadata only, it is read exactly once at
startup, and entry counts are in the hundreds, so parsing cost is not on any hot
path. A binary index would be a change made in response to a profile, not in
anticipation of one.

### Endianness and portability

The index is text, so it is endianness-neutral. Page files are **raw runtime
bytes** and are not portable: their layout depends on the backend, the KV
dtypes, the GPU layer split, and the model's tensors. Portability is not
attempted; it is *prevented*, by binding page identity to all of those inputs
(see Identity below). A cache directory copied to a different machine or a
differently configured stage produces misses, never wrong hits.

## Lifecycle

**Open.** Create the directory, acquire `owner.lock` exclusively (`flock`,
non-blocking), read the index, drop unusable entries, delete orphan page files,
then enforce the byte budget. Content is *not* verified at startup — only file
size — so opening a large cache does not read it.

**Store.** Write all components sequentially into a per-attempt temp file,
`fsync` it, `rename` it into place, insert the index entry, enforce the budget,
then rewrite the index. A page larger than the whole budget is skipped rather
than admitted: it would evict everything else and then itself.

**Load.** Check `payload_kind`, map the file (reusing a live mapping if one
exists), check the mapped length against `total_bytes`, verify `extra_checksum`,
then verify each component's payload checksum. Return `CacheBytes` that *borrow*
the mapping; no copy is made.

**Evict.** When the tier is over budget, drop the least recently used entry by
`(last_used_secs, use_sequence)`, remove its file and mapping, and rewrite the
index.

**Close.** The lock is released when the file descriptor closes, including on
crash. Entries survive; the next open reclaims anything half-written.

### Atomicity

Only two operations mutate the directory's visible state, and both are `rename`:
publishing a page file and publishing the index. A crash can therefore leave
only two artifacts, both reclaimed on the next open:

- A `.tmp` page file, from a crash mid-write. Reclaimed by prefix match.
- A page file with no index entry, from a crash between the file rename and the
  index commit. Reclaimed as an orphan.

The reverse — an index entry with no file, or with a file of the wrong size — is
dropped at load-index time.

Temp names include the pid and a per-process counter. A deterministic temp name
would let two writers interleave into one file and rename a torn result into
place.

### Concurrency

One process at a time, enforced by `owner.lock`. This is not a degraded mode
that is merely discouraged: `commit_index` is last-writer-wins, and orphan
reclaim deletes page files absent from *this* process's index — including files
another instance just wrote and still has mapped. Two instances sharing a
directory destroy each other's caches, so a second instance declines the tier
instead.

Platforms without advisory directory locking are refused rather than run
unprotected.

## Integrity

Two checksummed regions, both BLAKE3, both verified before any bytes reach the
runtime:

1. **Payload** — each component's bytes, verified against `components[].checksum`.
2. **Metadata** — the canonical encoding of `extra`, verified against
   `extra_checksum`.

Metadata is checksummed because the payload checksum says nothing about how to
*interpret* the bytes. For a KV page, `extra` carries the layer range, K/V ggml
types, and row strides; an index corrupted into still-valid JSON could otherwise
hand correctly-checksummed bytes to the runtime under the wrong layout, which is
silent numerical corruption on a path that looks verified.

Payload verification runs on the first load of a mapping and not again while
that mapping stays live. On a multi-gigabyte page hashing is roughly 95% of
restore cost, and re-hashing bytes this process already verified — while holding
an exclusive directory lock and a live mapping of a file published by atomic
rename and never modified in place — detects nothing. A remap re-verifies,
because that is where a different file could appear.

### Corruption behaviour

A verification failure is an **error, not a miss**. The entry is quarantined
(removed from the index, its file and mapping deleted) and the tier returns
`Err`. Callers treat that as a miss for the purposes of continuing to probe
shorter candidates, but the entry is gone and cannot be served again.

| Condition | Response |
|---|---|
| Index unreadable or not valid JSON | Reset the whole directory |
| `format_version` mismatch | Reset the whole directory |
| Unsafe `file_name` | Drop the entry |
| Page file missing | Drop the entry |
| Page file size ≠ `total_bytes` at open | Delete the file, drop the entry |
| Mapped length ≠ `total_bytes` at load | Quarantine, return `Err` |
| `payload_kind` mismatch | Quarantine, return `Err` |
| `extra_checksum` mismatch or absent | Quarantine, return `Err` |
| Component checksum mismatch | Quarantine, return `Err` |
| Page file with no index entry | Delete on open |
| Leftover `.tmp` file | Delete on open |

## Identity

The tier never decides whether two pages are interchangeable; `page_id` does,
via `skippy_cache::prefix_identity`. Identity covers the token sequence plus
every input that changes the exported bytes: weights (`model_id` plus content
digests, package ref, load mode), KV dtypes, flash-attention mode, GPU layer
split, backend device, stage, and layer range.

`topology_id` is deliberately excluded. It is derived per process from
`unix_nanos`, so hashing it would change every page id on restart — which is
precisely the reuse this tier exists to provide. Excluding it also removed an
accidental protection, which is why weight identity is now explicit and why the
tier refuses to open without a content digest.

## Versioning

`DISK_TIER_FORMAT_VERSION` (currently **1**) gates the entire directory. A
mismatch discards it. Bump it whenever any of the following changes:

- the index schema, including adding a field a reader must not ignore
- the page file layout or component ordering
- any `payload_kind` string
- the checksum algorithm or what either checksum covers
- the identity contract in `skippy-cache/src/identity.rs`

There is no migration path and none should be added. The contents are
regenerable, and the cost of being wrong is numerical corruption.

The bump rule applies from the first **released** version onward. Identity
changes made before version 1 ships do not need a bump, because no on-disk
directory claiming version 1 exists yet in any build a user could have run.

## Platform binding

Page files are raw runtime memory, so their interpretation depends on the CPU
architecture, native byte order, and pointer width of the process that wrote
them. `update_platform_identity` hashes all three into the page id.

This matters because the identity hash deliberately encodes its own integers
as little-endian, so two hosts of different native endianness would otherwise
compute the *same* page id for the same tokens. In the default machine-local
cache directory that is harmless, but `SKIPPY_KV_DISK_TIER_DIR` accepts any
path, including shared or copied storage; the stage directory key holds only
model and stage shape; and `backend_device` does not distinguish an x86_64
CUDA host from an aarch64 CUDA host, nor two CPU-only hosts that both record
`<no-selected-device>`. Without the platform tag, a copied directory could
produce a hit whose checksums pass -- the bytes did arrive intact -- and whose
import is a silent misread.

With it, a page from a different platform is a miss, never a wrong hit, and
several platforms can share one directory without quarantining each other.

## Models the disk tier declines

Sliding-window attention models -- Gemma 3/4, and anything else llama.cpp backs
with `llama_memory_hybrid_iswa` -- keep attention state in **two** caches: a
full-context base cache for the non-SWA layers and a window-bounded cache for
the SWA layers. For a prefix of N tokens the correct continuation state is
`0..N` for the base layers but only the still-visible suffix for the SWA
layers.

A native KV page carries one token range over one cache, so it cannot describe
that shape. The runtime declines the export with `runtime memory type is not
supported for native KV pages`, and it is right to: exporting the base cache
alone would produce a page that silently omits every SWA layer, and importing
it would advance `n_past` over state that was never restored. That is numerical
corruption, not a missing optimisation.

So the tier treats it as a permanent, expected property of the stage rather
than a per-request failure. The first attempt latches archiving off for that
stage and reports `archive_status=skipped_unsupported_memory`; subsequent
prefills report `skipped_tier_disabled` and cost nothing. Resident, in-process
reuse is unaffected, because a sequence copy duplicates both caches.

Supporting these models on disk needs a composite page -- a full-prefix base
page plus an SWA suffix page -- which is tracked in issue #1264. Adding the
ISWA types to the export path without that would be actively unsafe.

Inkling is the hardest case and has its own issue, #1265: it is both
hybrid-recurrent and sliding-window, so its recurrent state round-trips while
its attention KV half is declined. It needs the composite page *and* the
recurrent component restored together.

## Retention policy

Retention is LRU under a hard byte budget, with no TTL, and that is deliberate.

The workload this tier exists for is an agent prefix that is returned to after
a long gap; age alone does not make a content-bound page wrong. Entries become
*unreachable* rather than incorrect when the model or configuration changes,
because identity binds both. If no new writes arrive, retained entries consume
no more than the budget already allotted them, and deleting them only forfeits
future hits.

A TTL would therefore be a privacy or disk-hygiene control, not a correctness
one. The genuine gap is different: stage directories abandoned when `model_id`
or stage shape changes are outside any tier subsequently opened, so nothing
reclaims them. That wants a base-directory quota or GC, not expiry inside
active caches.

## Observability

Disk-tier counters are exported as telemetry attributes on every KV decision
event: `skippy.kv.disk_tier_enabled`, `disk_entries`, `disk_bytes`,
`disk_max_bytes`, `disk_demotions`, `disk_promotions`, `disk_evictions`,
`disk_corrupt_entries`, `disk_verifications`, `disk_verifications_skipped`.

Exact-state hits carry `skippy.exact_cache.hit_source` (`ram` or `disk`), and
archive attempts carry `skippy.kv.archive_status` with the reason
(`archived`, `skipped_too_short`, `skipped_already_archived`,
`skipped_tier_disabled`, `failed_export`, `failed_write`, `failed_error`) plus
`archive_bytes`, `archive_export_ms`, and `archive_write_ms` on success.

Those exist because the two states worth distinguishing — a tier that is never
probed and a tier that is probed and silently never serving — look identical
without them.
