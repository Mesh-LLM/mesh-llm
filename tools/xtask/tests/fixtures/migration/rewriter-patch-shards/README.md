# Patch shard candidate fixtures

These are data-only inputs and historical output copies. They are not fresh
legacy executions and do not certify native rewriting or the generated queue.

## Independent output sources

The following files are exact copies of checked-in legacy mail patches at
`a4e04070db2c6b8e2644a40f5c4f207789df6c2d`. Their expected digests were copied
from the checked-in `third_party/llama.cpp/patches/generated/series.json`, not
computed by the candidate sharder.

| Fixture | Source under `third_party/llama.cpp/patches/generated/` | SHA-256 |
| --- | --- | --- |
| `afmoe.patch` | `0001-family-afmoe.patch` | `f24da42643df886f3d854c46722ddafe0713d046a712547b976c77ece30f9b95` |
| `deepseek2--mistral4.patch` | `0009-family-deepseek2--mistral4.patch` | `cfe4e22719a7b3597ac5067d86eb7d21528af6d7be1b3ec9ca0f8c52c69ccd26` |
| `rwkv6.patch` | `0070-family-rwkv6.patch` | `513f9d910dc2e57461f42a2eebe8a903fca411a3ad2b30ca5a11cb897e3640e0` |

The input section extractor in `migration_patch_shards/contracts.rs` removes
only the mail envelope from these copies. The reduced case reuses their entire
diff sections. It does not ask the implementation under test to generate the
expected mail bytes. Reduced filenames differ from the full queue numbering;
mail subjects and full-mail digests do not include the filename.

`family-map.json` projects the four relevant mappings from
`ci/llama-canary/generated-family-map.json`, SHA-256
`63db182a4a539ef45e9910565d79d0f3b92ab57b590ceef6175b2769c1e7ad83`.
Its insertion order and RWKV6 source order are deliberately reversed.
`certified.json` projects the four causal family/class/profile triples from
`ci/llama-canary/family-certified.json`, SHA-256
`90834e6f52cf494cff216b3bce40953919109596512eb68efd07b4e33f938cdf`.
It adds a synthetic embedding row to exercise the workload-only exemption.
Neither reduced JSON file is represented as historical generator output.

The mixed case prepends a synthetic unowned section containing Git binary-patch
text to RWKV6, shared-source, and AFMoE sections, in that order. The reordered
case uses AFMoE, shared-source, RWKV6, then the same unowned section. This checks
opaque preservation of binary-patch text, not whether Git can apply that text.
Additional inline inputs are source-derived contract cases, not captured oracle
receipts. `test_select_skippy_family_shards.py:120` is the independent source
for the complete two-section control.

## Pending legacy capture

No independent mixed/reordered `series.json`, complete reduced shard directory,
combined patch, or raw-diff digest has been captured yet. The existing full-queue
`series.json` must not be used as the expected result for these reduced inputs.

The ignored test `migration_patch_shards_compare_independent_legacy_capture`
requires an explicit `PATCH_SHARD_ORACLE_ROOT`. Missing captures fail when the
test is explicitly selected with `--ignored`; they are not silently accepted.
Each of its `mixed/` and `reordered/` directories must contain:

- `input.diff`, exactly the corresponding data-only input from `contracts.rs`.
- `family-map.json` and `certified.json`, exact bytes from this directory.
- `combined.patch`, emitted by the unchanged legacy generator.
- `shards/`, containing each legacy shard plus `series` and `series.json`.
- `capture.json`, with `generator_sha256`, `exit_code`, `diff_sha256`,
  `family_map_sha256`, and `manifest_sha256`.

The generator identity must be
`99e269c4025ff8c9eb091c1ad4fd74f6c5e6f607e57928b84b1e20a4baf36cad`.
An independent capture owner must additionally retain the exact invocation,
interpreter identity, cwd/environment, raw stdout/stderr, exit status, file
modes/hashes and side-effect inventory. The test checks complete bytes and
capture identities; a self-authored `capture.json` is not provenance proof.

Do not generate expected files with the Rust candidate. Do not add Python
helpers or modify checked-in generated shards. Capture, differential execution,
red/green mutation checks, formatting, compilation and linting await separate
serial queue authorization.

## API boundary

The candidate accepts `&[u8]` diff data and two decoded `serde_json::Value`
documents, then returns in-memory output. Map and manifest semantic validation
belongs to this owner. Raw JSON decoding is not implemented by this slice.
Nonfinite JSON extensions, arbitrary-precision integer decoding, duplicate-key
decoding, unpaired surrogates and JSON parser resource limits remain questions
for the eventual input adapter, not new rejection policies asserted here.
Existing repository map/manifest data is finite ordinary JSON.
