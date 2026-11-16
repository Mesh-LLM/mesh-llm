# mesh-llm-plugin-manager

Plugin package management primitives for Mesh LLM.

This crate owns install-reference parsing, platform target naming, native
release asset selection, validation of packaged plugin manifest metadata
(including local web UI bundle roots and entry scripts), the plugin catalog
(including named plugin sets), host-version compatibility of plugin releases,
and local installed-plugin metadata. Catalog/GitHub installs use
`install_plugin`; authoring and release validation can pass a `.tar.gz` or
`.zip` directly to `install_plugin_archive`, which uses the same extraction
and package-validation boundary. Host upgrades judge every installed plugin
through `plan_plugin_upgrades` and repair verified catalog pins through
`repair_incompatible_plugins`. The retired bundled-default system survives
only as the surface grants `legacy_defaults` records for migrated installs.
It deliberately does not render CLI output; callers should render progress
and status events in the host application. Primary consumers are
`mesh-llm-commands` (the `plugins`, `setup`, and `update` command surfaces)
and `mesh-llm-host-runtime` (installed-plugin resolution at startup).

