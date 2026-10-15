# Test vectors: where they come from

These files are copied verbatim from their sources and are not edited here
(`parity/README.md`, which describes the parity set, is written here, and the
two Rust bundles are generated here: see their row).
`aac-capsule/` is BSD-3-Clause: its license, which must travel with the
copies, is `aac-capsule/LICENSE`, copied verbatim from the source repository.
Everything else here is Apache-2.0, like the rest of this repository.
`SHA256SUMS` lists every file; `shasum -a 256 -c SHA256SUMS` from this
directory checks them.

| Directory / file | Source | Pin | Used by |
| --- | --- | --- | --- |
| `aac-capsule/` (`vectors.json` and the `canonical-*` cases) | [agent-action-capsule](https://github.com/action-state-group/agent-action-capsule), `vectors/capsule/` | tag `v0.6.0` (commit `13a1f4534c31369e927db73af6298e8b9ffdc36a`); each file also matches that tag's own `vectors/capsule/SHA256SUMS` | `mesh/crates/mesh-llm-capsules-plugin/tests/aac_canonical_vectors.rs` |
| `split-stage/` | the split-stage vector generator in [capsule-emit-mesh](https://github.com/action-state-group/capsule-emit-mesh) (`tests/fixtures/split-stage/generate.py`) | commit `7867311132418f5302cdf7142cb529387fe67663` | the plugin's split-stage tests (`hop-cases.json`); `block-cases`, `fold-vectors` and `receipt-cases` are the full split-stage set for the stage-record code |
| `split-stage/rust-split-bundle.json`, `record_push_bundle_rust.json` | written in this repository by the plugin's own ignored generator tests (`writes_the_split_bundle_fixture` in `capsule_emit.rs`, `writes_the_cross_language_bundle_fixture` in `record_push_bridge.rs`); records sealed with this plugin's `developer` header (`capsule-emit-mesh/<version>`) | this repository: regenerate with those tests after a change to what the plugin seals | cross-implementation checks |
| `parity/` (the `.json` files) | the parity harness in [capsule-emit-mesh](https://github.com/action-state-group/capsule-emit-mesh) (`tests/parity/`). Record push: the corpus is built by `build_record_push_corpus.py` and the golden answers are the Python reference's (`record_push_python.py --golden`), both generated at the pinned commit; CI here does not rerun the Python reference. Evidence request: written by the plugin's own generator, reviewed against draft-mih-agent-evidence-request-00 | commit `b63d6e235097825064f1573bd07807c21913612b` | `record_push_parity.rs`, `evidence_request_parity.rs` in the plugin crate |
| `parity/referee/` | the referee parity corpus in [capsule-emit-mesh](https://github.com/action-state-group/capsule-emit-mesh) (`tests/parity/referee/`): `corpus/` and `rule_answers/` built by `build_referee_corpus.py`, `golden/`, `intended_differences.json` and `mutants.json` written by `referee_python.py --golden`, `port_tests.json` by hand; copied unchanged (the source's scripts and README stay there), except that `golden/service.json` and `intended_differences.json` were regenerated there with `referee_python.py --golden` after a verdict's `chain.relation` became the registered `confirms` (only those relation lines differ; that change is not yet in the source branch) | commit `a34abf4cd3efc953c2b90f660a33460ff6a6f10e` | `referee::parity` in the plugin crate; `scripts/referee-mutants.sh` |
| `tdx/tdx_quote.bin` | a TDX DCAP v4 quote taken by this project on its own cloud TDX test VM on 2026-10-12, with the go-tdx-guest `attest` tool; its REPORT_DATA is the SHA-512 of a test string, and it holds no keys or secrets | taken here (sha256 in `SHA256SUMS`) | `producer::tee_attest`'s parser tests; the binding tests splice their own REPORT_DATA into a copy |

To update a set: regenerate it at the source, copy it here unchanged, bump the
pin in this table, and regenerate `SHA256SUMS`.
