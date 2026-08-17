---
name: config-settings-management
description: Use this skill when adding, renaming, removing, validating, or exposing mesh-llm config settings, including built-in settings, plugin config schemas, owner-control apply behavior, CLI validation, and UI configuration surfaces.
metadata:
  short-description: Keep mesh-llm config settings complete
---

# config-settings-management

Use this skill before changing any setting that appears in
`~/.mesh-llm/config.toml`, the owner-control configuration API, the runtime
configuration UI, or an installed plugin's `config_schema`.

## Mental Model

Config settings are not just struct fields. A complete setting has:

- A persisted TOML shape in `crates/mesh-llm-config/src/model.rs`.
- Authoring/editor support in `crates/mesh-llm-config/src/authoring.rs` when
  code needs to create or mutate it.
- Built-in schema metadata in
  `crates/mesh-llm-config/src/model/built_in_schema.rs` when it is a core
  mesh-llm setting.
- Validation diagnostics in `crates/mesh-llm-config/src/validate.rs`, with
  stable `ConfigPath` and canonical path metadata.
- Runtime schema aggregation/export in
  `crates/mesh-llm-host-runtime/src/config_schema.rs`.
- Owner-control apply behavior in
  `crates/mesh-llm-host-runtime/src/runtime/config_state.rs` when it can be
  changed dynamically.
- API/protocol conversion coverage in `crates/mesh-llm-host-runtime/src/api/`,
  `crates/mesh-llm-host-runtime/src/protocol/`, and
  `crates/mesh-llm-protocol/proto/node.proto` when it crosses process or node
  boundaries.
- UI adapter and fixture coverage under
  `crates/mesh-llm-ui/src/features/configuration/` and
  `crates/mesh-llm-host-runtime/tests/fixtures/`.

## Built-In Settings Checklist

When adding or removing a built-in setting:

- Update `MeshConfig` or the owning nested config struct in
  `crates/mesh-llm-config/src/model.rs`.
- Update defaults and editor helpers in `authoring.rs` if generated configs,
  tests, or command flows need to write the setting.
- Add, rename, or remove the corresponding descriptor in
  `model/built_in_schema.rs`. Include owner, value schema, support state,
  control surfaces, apply mode, restart scope, visibility, constraints, aliases,
  and description.
- Update validation in `validate.rs`. Prefer structured `ConfigDiagnostic`
  helpers over plain string errors.
- Preserve compatibility with existing TOML when possible. Use aliases and
  warnings for renamed keys; reserve `version = 1` bumps for actual incompatible
  persisted config format changes.
- Update schema fixtures and UI adapter expectations when exported schema JSON
  changes.
- Run `mesh-llm config validate --config-path <fixture> --json` for at least one
  valid and one invalid representative file.

## Plugin Settings Checklist

Plugin settings are install-time schemas, not hard-coded built-in settings.

- The plugin manifest owns its schema through `config_schema` in
  `crates/mesh-llm-plugin/src/manifest.rs` and
  `crates/mesh-llm-plugin/proto/plugin.proto`.
- Keep `schema_version` at
  `mesh_llm_config::SUPPORTED_PLUGIN_CONFIG_SCHEMA_VERSION` unless the schema
  format itself becomes incompatible. Tightening validation of existing v1
  fields such as `required`, type, enum, object, array, or constraints does not
  by itself require a schema version bump.
- Host-side installed plugin schema loading and strict validation live in
  `crates/mesh-llm-host-runtime/src/plugin/config.rs` and
  `crates/mesh-llm-config/src/plugin_validation.rs`.
- Required plugin settings must be rejected even when `[plugin.settings]` is
  absent.
- Missing or unavailable schemas should reject custom settings, but plugin
  entries without custom settings should remain loadable when possible.
- `allow_unvalidated_config` should produce warnings, not silently drop
  diagnostics from success responses.

## Owner-Control And UI

- Dynamic apply behavior belongs in
  `crates/mesh-llm-host-runtime/src/runtime/config_state.rs`.
- The management API should return diagnostics for both rejected applies and
  successful applies with warnings.
- Protobuf changes must be additive unless explicitly approved as breaking.
  Older nodes and clients should ignore unknown fields.
- The UI should consume exported schema metadata instead of duplicating setting
  ownership, labels, constraints, or apply behavior.
- Snapshot fixtures in `crates/mesh-llm-host-runtime/tests/fixtures/` are the
  cross-check between Rust schema export and the TypeScript adapter.

## Validation

Run cargo commands serially. For config-surface changes, start with:

```bash
cargo test -p mesh-llm-config --lib
cargo test -p mesh-llm-host-runtime --lib schema_export
cargo test -p mesh-llm-host-runtime --lib runtime_config
cargo test -p mesh-llm-host-runtime --lib plugin_config
cargo test -p mesh-llm-plugin --lib
cargo test -p mesh-llm-plugin-manager --lib
cargo test -p mesh-llm-cli config_validate --lib
cargo test -p mesh-llm config_validate --lib
cargo check -p mesh-llm
cargo clippy -p mesh-llm-config -p mesh-llm-plugin -p mesh-llm-plugin-manager -p mesh-llm-host-runtime -p mesh-llm-cli -p mesh-llm --all-targets -- -D warnings
```

Also run the UI checks when the schema export or adapter changes:

```bash
cd crates/mesh-llm-ui
npm test -- --run src/features/configuration/api/config-adapter.test.ts
npm run typecheck
```

Use the repo build gate before publishing broad changes:

```bash
just build
```
