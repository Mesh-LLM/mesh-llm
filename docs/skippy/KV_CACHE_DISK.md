# KV-Cache Disk Tier — Operator Guide

The node-local **disk prompt cache** is the durable tier under Skippy's radix
cache: exported continuation (KV) state is cut into content-addressed segments
and committed under manifests so a later run can restore an exact prefix from
disk instead of paying cold prefill. This guide covers operating the tier that
ships today (the exact L1/L3 stack): configuration, defaults, status, restart
behavior, prune/clear, corruption handling, and troubleshooting.

This is a different cache from the in-memory OpenAI prompt-prefix cache
documented in [PROMPT_CACHE.md](PROMPT_CACHE.md). That one lives in the serving
process; this one is the on-disk L3 store described here.

**Fail-closed configuration, fail-open runtime.** Invalid disk-cache
*configuration* fails **closed**: config validation and resolution reject bad
values — fixed mode without a budget, a non-IEC or zero size, a relative
directory, or a minimum-free reserve below 1 GiB — by returning an error, so a
node refuses to start on a broken cache setting rather than silently ignoring
it. A *valid* configuration is then **fail-open for inference**: if the store
cannot open, reaches low space, or cannot admit a write, the node logs a warning
and serves with cold prefill. Valid runtime unavailability never blocks
generation.

## Modes and safe defaults

The tier has three modes, selected by `mode` / `--kv-cache-disk`:

| Mode | Meaning |
|---|---|
| `off` | Disabled. **This is the default** — no disk cache unless you opt in. |
| `auto` | Enabled with an automatically computed budget from live filesystem free space (see [Auto budget](#auto-budget)). |
| `fixed` | Enabled with an explicit hard byte cap you set. A budget is **required** in this mode. |

Because the default is `off`, an out-of-the-box node writes nothing to disk for
the prompt cache. Turn it on deliberately.

## Configuration surface

The same four settings are expressible through a config file, environment
variables, and CLI flags. Sizes everywhere use **explicit IEC suffixes**
(`KiB`, `MiB`, `GiB`, `TiB`) on a positive whole number — for example `32GiB`.
Bare numbers and decimal/`GB`-style units are rejected.

| Setting | Config (`[runtime.kv_cache.disk]`) | Environment | CLI flag |
|---|---|---|---|
| Mode / fixed budget | `mode` (`off`/`auto`/`fixed`) + `budget_mib` | `MESH_LLM_KV_CACHE_DISK` (`off`/`auto`/`SIZE`) | `--kv-cache-disk off\|auto\|SIZE` |
| Directory | `directory` (absolute) | `MESH_LLM_KV_CACHE_DISK_DIR` (absolute) | `--kv-cache-disk-dir ABSOLUTE_PATH` |
| Minimum free reserve | `minimum_free_mib` | `MESH_LLM_KV_CACHE_MIN_FREE` (SIZE) | `--kv-cache-min-free SIZE` |

Notes:

- In the config file, `budget_mib` is a plain MiB integer and is **only valid
  when `mode = "fixed"`**; it is **required** there. Setting it under `off`/`auto`
  is a config error.
- On the CLI and in the environment, mode and budget share one value:
  `--kv-cache-disk 32GiB` selects fixed mode at 32 GiB; `--kv-cache-disk auto`
  and `--kv-cache-disk off` select those modes.
- Directories must be absolute. Relative paths (including bare-drive forms like
  `C:\cache` on non-Windows hosts) fail closed rather than resolving under the
  working directory.

### Precedence

Each of the four settings is resolved **independently, field by field**, with
later sources overriding earlier ones:

```text
CLI flag > MESH_LLM_KV_CACHE_* env > config file > built-in default
```

Legacy `SKIPPY_L3_*` variables only fill a field for which none of those public
sources supplied a value; they never override a public config field.

So you can pin the directory in the config file and still override just the
budget with `--kv-cache-disk`, without disturbing the other fields. `mesh-llm
kv-cache status` reports the winning source per field (`default`, `config`,
`environment`, `cli`, or `legacy_environment`).

### Legacy environment variables (deprecated)

`SKIPPY_L3_DIR` and `SKIPPY_L3_BUDGET_BYTES` are still honored but **only as a
field-level fallback** when no public setting is present, and they emit a
deprecation warning. Migrate to `[runtime.kv_cache.disk]` or the
`MESH_LLM_KV_CACHE_*` variables. Two behaviors to know:

- Presence of `SKIPPY_L3_DIR` (without a public mode) implies `fixed` mode, and
  the budget defaults to the legacy **32 GiB** when unset.
- `SKIPPY_L3_BUDGET_BYTES=0` **no longer means unbounded** — it is treated as
  the 32 GiB legacy default, with a warning. `SKIPPY_L3_BUDGET_BYTES` is ignored
  entirely if `SKIPPY_L3_DIR` is not set.

## Directory, budget, and minimum-free behavior

**Directory.** When unset, the root resolves to `$MESH_LLM_HOME/kv-cache`, or
`~/.mesh-llm/kv-cache` when `MESH_LLM_HOME` is unset. On startup the node
creates the root and its store subdirectories (`segments/`, `manifests/`,
`prefixes/`, and the packed-store dirs; `quarantine/` is used when an object
fails verification), restricts them to owner-only permissions (`0700`), and
takes an **exclusive lock** on `.owner.lock` in the root. One process owns a cache root at a time; a second node pointed at the same
root fails to acquire the lock rather than corrupting it. The root must not
contain symlinks — a symlinked entry under the root is refused.

**Minimum free reserve.** `minimum_free` is the free space the store preserves
for everything else on the filesystem. Default **16 GiB**
(`DEFAULT_KV_DISK_MINIMUM_FREE_MIB`); the floor is **1 GiB**
(`MIN_KV_DISK_MINIMUM_FREE_MIB`) and a smaller value is rejected. When the
filesystem sits at the reserve, the store goes **read-only**: existing entries
still serve restores, but new writes are refused (`read_only_low_space`) so the
cache never eats into the reserve.

**Fixed budget.** The hard whole-node cap. Eviction runs oldest-manifest-first
and evicts to ~85% of the budget (a low-water margin that amortizes the
O(manifests×segments) eviction scan); the newest manifest is never evicted.

<a id="auto-budget"></a>**Auto budget.** In `auto` mode the budget is resolved
from live filesystem facts immediately before the root opens, as the **minimum**
of:

- 20% of the capacity basis (current filesystem available + bytes already
  managed under this root),
- what is actually allocatable after honoring `minimum_free`, and
- a hard **64 GiB** ceiling.

If that resolves to `0` (for example, the filesystem is already at or below the
minimum-free reserve), the disk cache stays disabled and the node logs a warning
— again, cold prefill still serves.

## Status and observability

Inspect a node with:

```bash
mesh-llm kv-cache status            # human-readable
mesh-llm kv-cache status --json     # machine-readable
```

Without `--endpoint`, the command talks to the local node's loopback control
API (default port `3131`; the control endpoint is loopback-only). To inspect
nodes you own remotely, pass one or more `--endpoint <addr>` values (repeatable).

The status payload reports:

- `configured` — the resolved `mode`, `directory`, `budget_bytes`,
  `minimum_free_bytes`, and the winning **source** for each field.
- `effective.state` — one of:
  - `off` — mode is off.
  - `active` — store is open and admitting writes.
  - `read_only_low_space` — at the minimum-free reserve; reads serve, writes
    refused.
  - `degraded` — configured on but no manager: `reason` is
    `storage_unavailable` (couldn't open the root),
    `budget_below_entry_floor` (auto budget resolved to zero), or a runtime
    storage error.
- `usage` — `budget_bytes`, `used_bytes`, `reserved_inflight_bytes`,
  `filesystem_available_bytes`, `minimum_free_bytes`, `manifests`,
  `unique_segments`, `evicted_manifests`, and `quarantined_objects`.
- `activity`, `reconciliation` (see below), and `inventory` (per-model entries).

At **startup**, any resolution warnings (deprecated legacy vars, zero auto
budget, unavailable store) are emitted as `Warning` events in the node log.
Check the log first when the cache "isn't caching."

## Restart vs. live-apply semantics

On a config reload, disk-cache changes split into two classes:

| Change | Applied |
|---|---|
| `budget_mib`, `minimum_free_mib` | **Live.** Limits update in place; shrinking evicts inactive entries immediately, pinned entries stay valid, and writes stay refused until usage fits. |
| `mode`, `directory` | **Restart required.** These are preserved across a live reload and only take effect when the node restarts. |

If a reload changes only `mode`/`directory`, those fields are held at their
previous values (and the mode-coupled budget with them) until restart. Plan a
node restart when you move the cache directory or turn the tier on/off via
config reload.

## Prune, clear, and shutdown

Both operations act on **inactive** entries only — pinned/in-use state is never
removed — and both are gated behind an explicit confirmation (or `--yes`):

```bash
# Evict least-recently-used inactive entries, optionally down to a target size
# and/or scoped to one exact model identity.
mesh-llm kv-cache prune [--target 16GiB] [--model-identity <ID>] [--yes]

# Remove inactive entries entirely (inference falls back to cold prefill),
# optionally scoped to one model identity; omit the filter to clear the root.
mesh-llm kv-cache clear [--model-identity <ID>] [--yes]
```

- `--model-identity` takes an **exact numerical model identity**; display names
  are not accepted, and a blank/whitespace filter is rejected (it would match
  nothing and report a no-op success).
- `prune` without `--target` trims toward the low-water margin of the budget.
  On an uncapped (legacy budget `0`) store this default would remove everything,
  so that case is refused and steered to `clear`.
- Both accept `--endpoint`/`--port` for owner-controlled remote nodes and
  `--json`.

**Shutdown.** There is no flush step to run. Manifests only commit after every
referenced segment is present and the assembled payload digest matches, so a
process that stops mid-write leaves temp files, never a partial cache entry.
The root lock is released on exit; leftover temp files are reconciled on the
next start (below). A hard kill is safe — you lose only in-flight writes.

## Corruption, quarantine, and cold fallback

Integrity is content-addressed end to end. Every segment is addressed by the
BLAKE3 digest of its bytes and reads verify that digest, so corruption is
**detected, never silently imported**. A manifest is only loadable once all its
segments are present and the reassembled payload digest matches; partial state
is unreadable by construction.

On startup the node **reconciles** the root before serving and records what it
repaired in `reconciliation`:

- `removed_temporary_files` — abandoned in-flight temp files.
- `quarantined_manifests` — manifests that failed verification, moved aside into
  `quarantine/`.
- `removed_prefix_links` — dangling prefix index links.
- `removed_orphan_bytes` — unreferenced segment bytes garbage-collected.

Objects that fail verification while serving are moved to `quarantine/` and
counted in `usage.quarantined_objects`. A quarantined or missing entry is simply
a cache miss: the request falls back to cold prefill. Rising
`quarantined_objects` points at underlying storage trouble (bad disk, truncation,
external tampering) — investigate the filesystem; the cache itself stays safe.

## Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| No disk caching at all | Mode is `off` (the default) | Set `--kv-cache-disk auto` (or `fixed SIZE`), or `[runtime.kv_cache.disk] mode`. |
| Node logs "disk prompt cache is unavailable; inference will use cold prefill" | Root couldn't be opened (permissions, missing parent, symlink in root, lock held by another process) | Check the directory exists and is writable, contains no symlinks, and no other node owns `.owner.lock`. `status` shows `degraded`/`storage_unavailable`. |
| `auto` mode caches nothing | Auto budget resolved to `0` — filesystem at/below `minimum_free` | Free space or lower `--kv-cache-min-free` (floor 1 GiB). `status` shows `degraded`/`budget_below_entry_floor`. |
| Writes refused, reads still work | At the minimum-free reserve | `status` state `read_only_low_space`; free disk, `prune`, or lower `minimum_free`. |
| Startup fails to open the cache | Another process holds the root lock | Only one node per cache root; point the second node at a different `directory`. |
| Node refuses to start with a config/resolution error (e.g. "must use an explicit IEC suffix", "must be an absolute path", "fixed disk prompt-cache mode requires a positive budget", minimum-free below 1 GiB) | **Invalid configuration** — rejected up front, not a runtime fallback | Fix the value: use `32GiB` (not `32`/`32GB`), absolute directories, a positive `budget_mib` under `fixed`, and `minimum_free ≥ 1GiB`. These are config errors, distinct from the store being unavailable at runtime. |
| Deprecation warning about `SKIPPY_L3_*` | Legacy env vars in use | Migrate to `[runtime.kv_cache.disk]` or `MESH_LLM_KV_CACHE_*`. |
| Config reload didn't move the cache / toggle the mode | `mode`/`directory` are restart-only | Restart the node; only `budget`/`minimum_free` apply live. |
| `quarantined_objects` climbing | Segments failing digest verification | Inspect the underlying storage; entries fall back to cold prefill, cache stays safe. |
| `prune` reports it freed nothing | Wrong/blank `--model-identity`, or only pinned entries present | Use the exact numerical model identity; pinned/in-use entries are never pruned. |

## Reference

- Config resolution and precedence: `crates/mesh-llm-host-runtime/src/runtime/kv_disk_config.rs`
- Config schema and constants: `crates/mesh-llm-config/src/model.rs`, `crates/mesh-llm-config/src/validate.rs`
- L3 store, integrity, reconciliation, status contract: `crates/skippy-cache/src/l3.rs`
- CLI and control API: `crates/mesh-llm-cli/src/parser/commands.rs`, `crates/mesh-llm-commands/src/kv_cache.rs`, `crates/mesh-llm-host-runtime/src/api/routes/kv_cache.rs`
