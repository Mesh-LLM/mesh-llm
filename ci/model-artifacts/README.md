# Test model artifacts

`registry.json` is the canonical source for shared model fixtures used by the
family battery, ordinary CI, and the parity/radix/competitive suites migrated
to this contract. Every artifact records an immutable
Hugging Face revision, its exact file set, byte sizes, SHA-256 digests, family
capabilities, and the suites and cadences allowed to use it.

Run the generator after changing the registry:

```bash
python3 scripts/generate-test-model-manifests.py
```

`just ci-validate` runs the generator in check mode through the Python contract
tests. It rejects stale suite manifests, invalid revisions, unsafe paths,
missing integrity metadata, undeclared suites/cadences, and accidental changes
to the generated llama family battery.

Consumers should read the generated manifest for their suite with
`scripts/resolve-test-model-manifest.py --cadence <cadence>`. The resolver
rejects artifacts outside their declared cadence, emits repository, revision,
selector, files, URLs, and integrity metadata, and can stream-verify downloaded
files with `--verify-root`. Single-file consumers additionally pass
`--require-single-file` so multipart artifacts fail closed. CI caches use the
artifact digest in their exact key; a cache hit is still verified before use.

Different repositories, revisions, files, or quantizations remain separate
named artifacts even when they cover the same model family. This preserves
intentional coverage differences while preventing each workflow or script from
inventing another floating or partially pinned download identity.

Entries in `unverified_consumers` document in-scope variants that cannot enter
the registry yet because their checked-in source lacks an immutable revision
or complete integrity metadata. Unrelated release QA, WAN lab, and latency
benchmark downloads keep their existing owner-specific contracts.
