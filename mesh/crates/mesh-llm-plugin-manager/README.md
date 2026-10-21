# mesh-llm-plugin-manager

Plugin package management primitives for Mesh LLM.

This crate owns install-reference parsing, platform target naming, native
release asset selection, validation of packaged plugin manifest metadata
(including local web UI bundle roots and entry scripts), and local
installed-plugin metadata. Catalog/GitHub installs use `install_plugin`;
authoring and release validation can pass a `.tar.gz` or `.zip` directly to
`install_plugin_archive`, which uses the same extraction and package-validation
boundary. It deliberately
does not render CLI output; callers should render progress and status events in
the host application.

It also owns the default plugin list (`defaults`) and installing a default
from a release's bundled copy (`bundled`). The `capsules` pin is compiled in
by `build.rs` from `MESH_LLM_BUNDLED_CAPSULES_SHA256`, which the release job
sets to the SHA-256 of the capsules archive it built from the same commit; a
build without it pins no capsules archive.
