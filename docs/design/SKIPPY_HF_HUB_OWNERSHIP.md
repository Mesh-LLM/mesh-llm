# Skippy Hugging Face client ownership

Decision for [#1897](https://github.com/Mesh-LLM/mesh-llm/issues/1897): resolve the
published HF Hub fork identity during M2 model acquisition, before declaring the
complete no-Mesh-dependency gate satisfied. This is a sequencing decision within
the single extraction PR, not an exception to its final acceptance criteria.

## Evidence at ff7fb85c83aad3e755419035ee4ef74491a5ed29

`model-hf` and `model-package` depend on the registry package
`mesh-llm-hf-hub` 1.0.2 under the Rust import alias `hf_hub`, with the `blocking`
feature enabled. See their Cargo manifests and the root Cargo.lock. Renaming
only the import alias does not remove that package from the dependency graph.

The installed package's `.cargo_vcs_info.json` identifies source commit
[c47b78835eeb98b7339718f8701adf80b8685446](https://github.com/Mesh-LLM/hf-hub/commit/c47b78835eeb98b7339718f8701adf80b8685446),
path `hf-hub`, in the Apache-2.0 fork of `huggingface/hf-hub`. That commit fixes
blocking-download lifetimes by owning download parameters and captured values.
A replacement must preserve the required client surface and those semantics;
this audit does not establish equivalence with any upstream release.

The locked `cargo metadata --locked --format-version 1` graph contains no other
`mesh-*` package in this fork's transitive closure. The recorded count of 334
uses unique package IDs reachable through `resolve.nodes[].dependencies`,
starting with the fork's dependencies and excluding the fork itself, from the
unfiltered default-feature metadata invocation above. Different target/feature
or edge-kind filtering can produce different counts; the no-other-Mesh-package
conclusion is the relevant result. A source scan of
the installed 1.0.2 package for Mesh, Skippy and Iroh names finds only its own
package installation example. This is a generic registry client fork, not a
reverse call into Mesh discovery, plugins or serving. Its package identity still
fails the literal no-`mesh-*` dependency criterion.

The lockfile includes paths from `skippy-bench`, `skippy-correctness` and
`skippy-model-package` through `model-hf` to that registry package. Standalone
acquisition will also use it when the lifecycle API consumes model resolution.
The current check must therefore cover aliases and transitive registry packages,
not only workspace paths or Rust import names.

## M2 work and acceptance

1. Evaluate an upstream client release against the exact API and fork fixes used
   by model acquisition. Prefer upstream if equivalent; otherwise rehome the
   required generic client under a Skippy-owned package with its license and
   source provenance intact. Do not introduce a name-only alias as a solution.
2. Switch model acquisition and any shared Mesh consumers together as needed to
   avoid incompatible client types. Keep credentials, endpoints, retry behavior,
   Hugging Face/Xet cache layout, integrity checks and TLS provider setup under
   explicit validation. `model-hf/src/tls.rs` documents the current reqwest 0.13
   and CPU-safe provider contract.
3. Run complete affected-package tests and exercise real model download, cached
   reuse and removal from a clean Skippy installation. Include blocking and async
   callers and failure behavior; a successful Cargo check is insufficient.
4. Re-audit Cargo metadata across normal, build, optional and target-specific
   dependencies, resolving dependency aliases to actual package identities.
   The final Skippy closure must contain no Mesh-owned product or fork package.

Until these steps are complete, retain the current locked client and keep the
full dependency gate open. This decision changes no dependency or download
behavior and claims no M2 or standalone-serving acceptance result.

## Rehome implementation

The client is now imported as workspace package `skippy-hf-hub`; the four
consumer manifests resolve their `hf_hub` alias to that path. This supersedes
the instruction above to retain the registry package while evaluating options.
The final acceptance gate remains open.

The upstream v0.5.0 API does not provide the fork's typed HFClient/repository
surface. Rehoming the pinned fork avoids rewriting acquisition consumers while
preserving its Apache-2.0 license. `crates/skippy-hf-hub/SOURCE_PROVENANCE.json`
records hashes of all 24 original Rust files, verified against the pinned Git
blobs, and the imported hashes and explicit local changes.

Besides formatting and subprocess isolation of eight token-precedence scenarios,
the import boxes four inner download futures inside their existing async public
methods. The retained future-size tests failed against both the unmodified
pinned source and the initial import on this toolchain; boxing restores their
original limits. This changes allocation and future representation, not the
public async signatures. An uncalled private HEAD helper was removed and its
independent relative-location predicate retained under test configuration.

The imported suite and consumer checks are migration evidence, not a substitute
for clean-install acquisition, cache reuse/removal, and standalone serving.
