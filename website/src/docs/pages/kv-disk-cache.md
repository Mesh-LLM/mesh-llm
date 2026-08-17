---
title: "KV Disk Cache"
---

# KV Disk Cache

Mesh can keep the KV cache for a long shared prompt prefix on disk, so the same
prefix does not have to be prefilled again after it falls out of memory — or
after the node restarts.

This matters most for agent workloads. An agent sends the same system prompt and
tool schemas on every turn with a different tail. Prefill cost grows roughly
quadratically with prefix length, while restoring a saved prefix is linear in
bytes, so **the bigger the shared prefix, the better this pays off**.

Measured on a 2-node split serving Qwen3-8B with a ~16.9k-token agent prompt:

| Scenario | Time to first token |
|---|---|
| Cold, nothing cached | 31.0s |
| Same prefix, new tail, later session | 1.3s |
| First request after restarting both nodes | 1.5s |

## Turning it on

The disk cache is **off by default**. It uses real disk space and writes model
state to it, so it is opt-in.

```sh
# Give the cache 8 GB
mesh-llm serve --model <model-ref> --kv-cache-disk 8

# Or let Mesh size it from free space
mesh-llm serve --model <model-ref> --kv-cache-disk auto

# Store it somewhere other than ~/.mesh-llm/kv-cache
mesh-llm serve --model <model-ref> --kv-cache-disk 8 --kv-cache-disk-dir /data/kv
```

| Flag | Effect |
|---|---|
| `--kv-cache-disk <GB>` | Enable with an explicit node-wide budget |
| `--kv-cache-disk auto` | Enable and size from free space |
| `--kv-cache-disk-dir <path>` | Store the cache somewhere other than the default |

The equivalent environment variables still work and take precedence, which is
useful for systemd units and containers:

| Variable | Effect |
|---|---|
| `SKIPPY_KV_DISK_TIER=1` | Enable with the default budget |
| `SKIPPY_KV_DISK_TIER_MIB=<mib>` | Enable with an explicit budget |
| `SKIPPY_KV_DISK_TIER_DIR=<path>` | Store the cache somewhere other than the default |

The budget is a **whole-node total**, shared across every model the node is
serving. It does not multiply by the number of loaded models.

By default the cache lives under `~/.mesh-llm/kv-cache/`. Put it on your fastest
local disk. Do not put it on a network filesystem: the cache relies on
memory-mapping files and on exclusive local file locking.

## When it will not turn on

Mesh declines to enable the disk cache, and says why on stderr, when:

- **There is no content digest for the model.** A cached prefix must be tied to
  the exact weights that produced it. A display name is not enough — two
  different GGUFs can be served under one name — so without a
  `manifest_sha256` or `source_model_sha256` the cache stays off rather than
  risk serving a prefix computed from different weights.
- **There is not enough free disk space.**
- **Another Mesh instance already owns the cache directory.** Only one process
  may use a cache directory at a time.

None of these stop the node serving. You simply get today's behaviour: every
prefix is recomputed.

## What is safe about it

A cached prefix is only reused when it was produced by an identical setup. The
cache key covers the model weights, the KV cache dtypes, flash attention mode,
the CPU/GPU layer split, the backend device, and which layers this stage owns.
Change any of those and old entries are ignored, not reinterpreted.

Every entry also carries checksums over both its bytes and the metadata that
describes how to interpret them. An entry that fails verification is deleted and
the request falls back to a normal prefill. A cache miss is cheap; a wrong
restore would be silently wrong output, so the cache always chooses the miss.

Interrupted writes and stale files are cleaned up the next time the node starts.

## Clearing it

Stop the node and delete the directory:

```sh
rm -rf ~/.mesh-llm/kv-cache
```

Everything in it is regenerable. Deleting it costs you a slow first request and
nothing else.

## Details

For the on-disk format, integrity guarantees, and versioning rules, see
[KV disk tier on-disk format](https://github.com/Mesh-LLM/mesh-llm/blob/main/docs/skippy/KV_DISK_TIER_FORMAT.md).

## Which models it works for

Most dense and mixture-of-experts models are supported. Whether the cache
applies depends on the shape of the model's attention state, not on whether it
is MoE: experts sit in the feed-forward layers and carry no state between
tokens.

Interleaved sliding-window attention (ISWA) is supported. For models such as
Gemma 3 and Gemma 4, llama.cpp keeps a full-context base cache and a
window-bounded SWA cache. Mesh saves those as one composite page: the complete
base range plus the visible SWA suffix, each with its own descriptor. The same
codec also covers ISWA inside a hybrid attention/recurrent memory wrapper; the
recurrent state is retained alongside the composite attention page.

Other hybrid memory layouts are not assumed to be safe. If the native runtime
cannot export the model's complete continuation state in one of the supported
plain-KV, KV-plus-recurrent, or composite-ISWA forms, Mesh declines disk
retention for that stage and continues serving normally. In-memory prefix reuse
is unaffected.

The composite descriptor and corruption checks have Rust unit coverage. Live
restart validation with Muse-Glimmer-30B Q4_K_XL restored 1,920 cached tokens
from a 2,038-token prompt: wall time fell from 3.76 seconds cold to 0.43 seconds
after restart. The 54.3 MB composite page contained 13 full-context layers and
39 SWA layers. Inkling remains unsupported because it combines recurrent state
with ISWA and requires both forms to be restored together.

To check what a running node decided, look at `skippy.kv.archive_status`. It is
emitted through telemetry, not written to the ordinary log: with no exporter
endpoint configured it is not recorded anywhere. Set `SKIPPY_TELEMETRY_STDERR=1`
to print it to stderr, or point the node at an OTLP endpoint. A status of
`skipped_unsupported_memory` means the model's attention state cannot be saved
to disk.
