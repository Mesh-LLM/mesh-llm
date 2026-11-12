# Test model artifacts

`registry.json` is the canonical source for shared model fixtures used by the
family battery, ordinary CI, and the parity/radix/competitive suites migrated
to this contract. Every artifact records an immutable
Hugging Face revision, its exact file set, byte sizes, SHA-256 digests, family
capabilities, and the suites and cadences allowed to use it.

A suite built from models pinned elsewhere is declared, not typed in:
`derived_suites` names a pinned evals model list
(`skippy/evals/kv-validation/models.json`), the selectors to register it at
and an id prefix, and the generator refuses the registry until every derived
row exists, so a member cannot be forgotten. `cargo xtool models pin` with no
arguments registers exactly the missing rows: each file name is the list's
with its selector suffix swapped (`Qwen3.5-4B-Q4_K_M.gguf` at `BF16` is
`Qwen3.5-4B-BF16.gguf`), each id is `<prefix>-<key>` with the selector slug
swapped (`perf-qwen3.5-4b-bf16`), and the size and LFS SHA-256 come from the
Hub over the pinned HTTPS transport, with the list's own digest enforced at
its own selector. A single file enters with
`cargo xtool models pin <repo>@<revision> <file> --id <id> --family <family>
--selector <text> --suite <suite> --tag <tag>...`, which resolves the revision
to its commit the same way; `--expect-sha256` checks the Hub against a digest
pinned elsewhere and `--dry-run` prints the rows instead of writing them.

Run the generator after changing the registry:

```bash
cargo xtool models generate
```

`cargo test -p xtask --test migration_models` runs the generator in check mode.
It rejects stale suite manifests, invalid revisions, unsafe paths,
missing integrity metadata, undeclared suites/cadences, and accidental changes
to the generated llama family battery.

Consumers should read the generated manifest for their suite with
`cargo xtool models resolve <manifest> --cadence <cadence>`. The resolver
rejects artifacts outside their declared cadence, emits repository, revision,
selector, files, URLs, and integrity metadata, and can stream-verify downloaded
files with `--verify-root`. Single-file consumers additionally pass
`--require-single-file` so multipart artifacts fail closed. CI caches use the
artifact digest in their exact key; a cache hit is still verified before use.

A whole suite is materialized on one machine with
`cargo xtool models fetch <manifest> --cadence <cadence> --root <dir>`. Each
artifact lands in `<root>/<artifact id>/<file>` through the pinned HTTPS
transfer and is held to the manifest's size and SHA-256; a file that already
verifies is kept, one that does not is reported and left for the operator.
`--dry-run` lists what would be fetched and how much, `--artifact-id` limits
the fetch to one artifact, and `models resolve --artifact-id <id>
--verify-root <root>/<id>` checks a fetched artifact afterwards. The
`performance-*` suites are the manual performance tiers: `performance-coverage`
pins one representative of each architecture class (dense, MoE, MLA, MXFP4,
sliding window, Mamba2, RWKV7, Granite hybrid, embedding, reranker) at the
same revision the family battery certifies.

Multi-artifact manifests may declare `default_artifact_id` for callers that
intentionally consume the suite's canonical fixture without selecting an
artifact explicitly. Explicit `--artifact-id` selection always takes
precedence; a multi-artifact manifest without a declared default remains an
error.

Different repositories, revisions, files, or quantizations remain separate
named artifacts even when they cover the same model family. This preserves
intentional coverage differences while preventing each workflow or script from
inventing another floating or partially pinned download identity.

Entries in `unverified_consumers` document in-scope variants that cannot enter
the registry yet because their checked-in source lacks an immutable revision
or complete integrity metadata. Unrelated release QA, WAN lab, and latency
benchmark downloads keep their existing owner-specific contracts.
