# Reviewed catalog generation defaults

These JSON files are publisher inputs for the active layer packages listed in
[`docs/model-package-generation-defaults-migration.md`](../../../../docs/model-package-generation-defaults-migration.md).
Each file is a standalone `GenerationRequestDefaults` object accepted by
`mesh-llm models package --generation-defaults`.

The values come from an official model card or `generation_config.json` at the
immutable revision embedded in every profile. Fields without an official
recommendation remain absent. The package schema test parses and validates all
19 files, including their revision-pinned provenance.

These files cover 20 of the 22 active package mappings because the two Nemotron
Super packages share one base-model profile. Kimi K3 and Inkling have no
general deployment recommendation in their official repositories at the
audited revisions; they intentionally use the bounded Mesh fallback recorded
in the migration inventory.

Publish these defaults only after a compatible runtime is released. Mesh
v0.76.1 rejects the new canonical `generation.request_defaults` field.
