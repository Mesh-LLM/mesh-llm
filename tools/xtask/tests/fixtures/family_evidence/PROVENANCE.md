# Legacy family sharding oracle

This directory freezes output from the existing local `scripts/plan-family-battery.py` before the task-24 Rust port. It is fixture data, not an alternate planner. Capture ran from the detached worktree root at commit `a4e04070db2c6b8e2644a40f5c4f207789df6c2d`; the planner file SHA-256 was `24f383830e7231f1506647b7a9f3537f0fe45dbb96620b0b44916f91a66ef470`. The checked-in real manifest `ci/llama-canary/family-certified.json` had raw SHA-256 `90834e6f52cf494cff216b3bce40953919109596512eb68efd07b4e33f938cdf`. The synthetic manifest in this directory has raw SHA-256 `3643e767cfe6d547c009cb1b6954a66fc83e66b07ee2f2a2cb7a57ecf6530916`. These SHA values identify input bytes, not a reserialized manifest or plan.

From the worktree root, run each command below with `python3`. Redirect fd 1 to `<case>.stdout` and fd 2 to `<case>.stderr`; record the exit status separately. Empty files are intentional. A successful plan is complete sorted-key, indented schema-v1 JSON with a trailing newline. Its SHA-256 below hashes **exact stdout bytes**. `--verify-plan` emits no stdout on success, and rejects the changed matrix weight in `tampered-plan.json` here. Neither option checks a model cache unless `--check-cache` is supplied.

| Case | Arguments after `python3 scripts/plan-family-battery.py` | Exit | stdout SHA-256 | stderr SHA-256 |
| --- | --- | ---: | --- | --- |
| `real-1` | `--shard-count 1` | 0 | `45ee6ee3461963e34936c12efe5870e612b4c3dacd94ebbb8e656899ffcb043d` | empty |
| `real-4` | `--shard-count 4` | 0 | `470cbde3fe82cda0e63134004063b440117754467953d5e0b6a1116343b4c9dc` | empty |
| `real-256` | `--shard-count 256` | 0 | `fa78863c0a0c09b81c012f41d2a03da9fb685ce4bcc7573bf41ac852127342c0` | empty |
| `real-reversed` | `--families llama,qwen3-dense --shard-count 1` | 0 | `72c7b7d03959dbe2f7528316d141da9674fcdb14c65d9ab425c42e5527f7bc93` | empty |
| `synthetic-equal` | `--manifest tools/xtask/tests/fixtures/family_evidence/synthetic-manifest.json --shard-count 2` | 0 | `f404d1ff18cc164e412f4326e95aae1439de1c2acde8170d2ee831631803e8c0` | empty |
| `synthetic-uneven` | `--manifest tools/xtask/tests/fixtures/family_evidence/synthetic-manifest.json --shard-count 3` | 0 | `fe2bc7ab9b34354e03b4e2390d6643cf9d2d69fef080848add119f98b1f01908` | empty |
| `real-zero` | `--shard-count 0` | 2 | empty | `2acbdc3a0505f0f89cd6728a8defce7cf5e29302f86a52436020c7b3363d2cc5` |
| `tampered-plan` | `--verify-plan tools/xtask/tests/fixtures/family_evidence/tampered-plan.json` | 2 | empty | `a3e58ddb10d286dee214822849da0c6bc17773cad2398bac1b6e8708f9758783` |
| `malformed-manifest` | `--manifest tools/xtask/tests/fixtures/family_evidence/malformed-manifest.json` | 2 | empty | `774a33f544e5ac7177f0d382827bfc7ce6433a30017d00c9ca2e5901a7c45ea0` |

All empty streams have SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. `tampered-plan.json` is derived from the **real-reversed** output by incrementing `.github_matrix.include[0].estimated_work_bytes` using `jq '.github_matrix.include[0].estimated_work_bytes += 1'`; the original selected filter is retained. The synthetic manifest uses five valid causal rows with weights 9, 9, 5, 4, 4 in manifest order `zeta,beta,alpha,gamma,delta`. The artifact coordinates in it are structural placeholders; no model is fetched or loaded.

The next Rust worker should compare `shards` and `github_matrix` structurally against these independent stdout files. `real-256` selects 95 families and yields 95 nonempty shards, but matrix order differs from shard-index order. `real-reversed` requests `llama` before `qwen3-dense` yet emits shard families in original manifest order. `synthetic-equal` proves equal-weight tie and original-index output ordering; `synthetic-uneven` proves stable shard-index allocation and ascending `(estimated_work_bytes, families-array)` scheduling order. Full JSON byte parity, plan digest, verification/error parity and stale-source protection belong to the later enclosing CLI/aggregate work, not just the pure sharder. A successful planner exit alone says nothing about executable handoffs, family receipts, CPU oracle evidence or publication eligibility.
