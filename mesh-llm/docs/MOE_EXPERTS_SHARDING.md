# MoE Expert Components Sharding

Proposed design for publishing MoE split artifacts to Hugging Face without
creating a separate full split set for every supported node count.

## Goal

Today mesh-llm publishes and reuses MoE rankings, but each runtime MoE split is
still built locally as a topology-specific shard.

That works, but it scales poorly if we try to publish prebuilt split files:

- 2 nodes means 2 shard files
- 3 nodes means 3 different shard files
- 4 nodes means 4 different shard files
- every new node count creates a new split set

This leads to combinatorial expansion in stored artifacts as we support more
topologies.

The better approach is to publish a topology-independent decomposition:

- one shared trunk artifact
- one artifact per expert
- one manifest describing how to reassemble a runnable shard

Then `mesh-llm serve` downloads only the expert components needed for the
current node's assignment and materializes a local shard from those pieces.

## Proposed Hugging Face Layout

These artifacts should live in the existing `meshllm/moe-rankings` dataset
under the same canonical prefix as the ranking that produced them.

Example:

```text
data/
  unsloth/
    Qwen3.6-35B-A3B-GGUF/
      9280dd353ab587157920d5bd391ada414d84e552/
        gguf/
          Qwen3.6-35B-A3B-UD-Q4_K_XL/
            full-v1/
              ranking.csv
              metadata.json
              analysis.json
              experts/
                manifest.json
                trunk.gguf
                experts/
                  000-031/
                    expert-000.gguf
                    expert-001.gguf
                    ...
                  032-063/
                    ...
                  064-095/
                    ...
                  096-127/
                    ...
```

The exact bucketing of expert files is flexible. The important property is that
each expert remains independently addressable so a node can download only the
experts it needs.

## Why This Is Better

- Storage grows with expert count, not with every supported node count.
- Published artifacts are reusable across 2-node, 3-node, 4-node, and future
  topologies.
- Node join and leave events become more natural: a node only needs to fetch
  new experts when its assignment changes.
- The artifact model is cleaner: one canonical MoE decomposition per exact
  model revision.

## Runtime Model

`mesh-llm serve` should continue to resolve MoE rankings as it does today.

After ranking resolution, the runtime flow becomes:

1. Compute the node's expert assignment from the ranking and current mesh plan.
2. Check whether matching expert component artifacts exist in the dataset.
3. Download `experts/manifest.json`.
4. Download `trunk.gguf` if it is not already cached locally.
5. Download only the expert files needed by this node's assignment.
6. Materialize a local runnable shard GGUF from `trunk + selected experts`.
7. Start `llama-server` using the materialized local shard.
8. If any component is missing or invalid, fall back to today's local
   `llama-moe-split` path.

This keeps the current local split path as the safety net while letting us try
published expert reuse incrementally.

## Important Compatibility Rule

Published expert components must only be used when they match the ranking that
the runtime selected.

The components are not identified by model name alone. They are valid only for
an exact combination of:

- source repo
- source revision
- distribution id
- analyzer id (`full-v1` / `micro-v1`)
- ranking hash
- expert count and expert-used count
- any component schema version we introduce

If runtime resolves a newer or stronger local ranking, the published expert
components must be ignored and the node should assemble from components that
match that ranking or fall back to local splitting.

## Manifest

The `experts/manifest.json` file should describe the published decomposition and
let runtime validate it before downloading large files.

Example shape:

```json
{
  "schema_version": 1,
  "source_repo": "unsloth/Qwen3.6-35B-A3B-GGUF",
  "source_revision": "9280dd353ab587157920d5bd391ada414d84e552",
  "distribution_id": "Qwen3.6-35B-A3B-UD-Q4_K_XL",
  "analyzer_id": "full-v1",
  "ranking_sha256": "sha256:...",
  "format": "gguf-moe-components",
  "expert_count": 128,
  "expert_used_count": 8,
  "trunk": {
    "path": "trunk.gguf",
    "sha256": "sha256:..."
  },
  "experts": [
    {
      "expert_id": 0,
      "path": "experts/000-031/expert-000.gguf",
      "sha256": "sha256:..."
    },
    {
      "expert_id": 1,
      "path": "experts/000-031/expert-001.gguf",
      "sha256": "sha256:..."
    }
  ]
}
```

The runtime should use this manifest to validate:

- model identity
- ranking identity
- expert count
- presence of each required expert artifact
- hashes when a local cached copy already exists

## CLI Workflow

The preferred workflow is to extend the existing share command instead of
creating a separate publishing command.

Recommended behavior:

```bash
mesh-llm moe share MODEL
mesh-llm moe share MODEL --with-experts
```

Meaning:

- `mesh-llm moe share MODEL`
  publishes the existing ranking artifacts only:
  - `ranking.csv`
  - `metadata.json`
  - `analysis.json`
- `mesh-llm moe share MODEL --with-experts`
  additionally publishes:
  - `experts/manifest.json`
  - `experts/trunk.gguf`
  - `experts/expert-XYZ.gguf`

This keeps the current lightweight share path intact and makes the heavier
expert publishing flow explicit while we validate the design.

## Why `--with-experts`

Publishing rankings is lightweight.

Publishing expert components is much heavier:

- many large GGUF files
- slower upload time
- more expensive reruns if the component format changes

So expert publishing should start as opt-in.

## Upload Path

All MoE share uploads should move to the Rust `huggingface_hub_rust` client.

Do not keep two separate upload implementations where:

- ranking-only share uses the current hand-rolled NDJSON path
- `--with-experts` uses a different large-file upload backend

Instead, the entire MoE share workflow should use one upload path based on the
Hub repo commit APIs exposed by `huggingface_hub_rust`.

That gives us:

- one implementation for ranking-only and expert-component uploads
- one auth path
- one progress and retry path
- one PR workflow
- one codepath to maintain

The Rust client already supports:

- normal repo commit creation
- preupload classification for regular files vs. large files
- xet/LFS-backed upload for large files
- `create_pr = true` on commit creation

So the intended implementation is:

- `mesh-llm moe share MODEL`
  uploads `ranking.csv`, `metadata.json`, `analysis.json`, and optional
  `run.log` through `huggingface_hub_rust`
- `mesh-llm moe share MODEL --with-experts`
  uploads those same files plus `experts/manifest.json`, `experts/trunk.gguf`,
  and `experts/expert-XYZ.gguf` through the same Rust path

User-facing behavior should still preserve the contribution PR workflow.

In other words:

- small artifacts should not continue using the current custom NDJSON request
- large artifacts should not require shelling out to `hf`
- all MoE share uploads should go through `huggingface_hub_rust` with PR
  creation enabled

## Required llama.cpp Support

This design only works if we have a primitive to move between:

- full original GGUF
- trunk-only artifact
- per-expert artifact
- runnable local shard

## llama-moe-split Decision

The decision is:

- keep `llama-moe-split` as the stable runnable-shard tool
- do not turn it into a kitchen-sink CLI for the new artifact workflow
- extract shared internals underneath it as needed
- add a new sibling tool in the llama.cpp fork for expert component work

Why:

- `llama-moe-split` already has a clear meaning: take a full GGUF and emit a
  runnable shard GGUF
- mesh-llm already depends on that behavior today
- the new expert-components path is a different abstraction: extract trunk,
  extract per-expert components, and assemble a shard later
- keeping the current split path stable gives us a low-risk fallback while the
  new path is being proven out

Recommended implementation structure:

- move shared tensor-selection and GGUF-writing logic into reusable internal
  code in the llama.cpp fork
- keep `llama-moe-split` as the compatibility and fallback path
- add a sibling tool for the expert-components workflow

That new sibling tool should support:

- extract trunk
- extract a single expert
- assemble `trunk + selected experts -> runnable shard.gguf`

The exact tool name is still open, but it should be a separate tool or cleanly
separated subcommand surface, not a semantic rewrite of `llama-moe-split`.

Without that assembly step, `serve` cannot consume the published expert
components because `llama-server` still expects a runnable shard GGUF.

## Fastest Path To Try It

The smallest path that gives us a real end-to-end experiment is:

1. Add `--with-experts` to `mesh-llm moe share`.
2. Define the dataset layout and manifest schema.
3. Implement trunk/expert extraction and local assembly in the llama.cpp fork.
4. Add a hidden or developer-focused round-trip command that materializes a
   shard from published components.
5. Validate that the materialized shard loads in `llama-server`.
6. Run a smoke test on at least one small MoE model.
7. Only then teach `mesh-llm serve` to prefer published expert components.

## Suggested Phases

### Phase 1: Local artifact proof

Implement `mesh-llm moe share MODEL --with-experts`, but initially focus on
generating the component layout locally.

Goal:

- prove the format
- prove the extractor
- prove the assembler

Do not change `serve` yet.

### Phase 2: Round-trip command

Add a hidden or dev command such as:

```bash
mesh-llm moe materialize MODEL --from-experts --nodes 2 --shard-index 0
```

That command should:

- read the published ranking
- compute the node assignment
- fetch or reuse `trunk + needed experts`
- assemble a local runnable shard
- validate that it loads successfully

Goal:

- test the full artifact loop without adding runtime risk to `serve`

### Phase 3: Runtime integration

After the round-trip path is stable, update `mesh-llm serve`:

1. resolve ranking as today
2. look for matching `experts/manifest.json`
3. fetch trunk and only the required experts
4. materialize the shard in local cache
5. launch from that shard
6. fall back to the existing local split flow on any miss or validation failure

Goal:

- reuse published expert components when available
- preserve today's known-good behavior as fallback

## Non-Goals For The First Iteration

- publishing a separate full split set for every node count
- making `--with-experts` the default immediately
- removing the local `llama-moe-split` fallback
- introducing a per-tensor artifact format that is too granular to manage

## Open Questions

- Should `trunk.gguf` include all router and shared-expert tensors, or should
  some of those be split into separate component classes?
- Should each expert be one file, or should we bundle experts into coarse blocks
  while still allowing selective download?
- What is the cleanest local cache layout for materialized shards and downloaded
  components?
- Should runtime eagerly prefetch likely future experts after a mesh expansion,
  or stay strictly demand-driven at first?

## Current Recommendation

Build this as an extension to `mesh-llm moe share` behind `--with-experts`,
store the artifacts in the existing dataset under `.../full-v1/experts/`,
migrate the entire MoE share upload path to `huggingface_hub_rust` with PR
creation, prove the round-trip locally first, and only then integrate it into
`serve`.
