# mesh-llm-hardware-profile

`mesh-llm-hardware-profile` detects the local operating system, architecture,
GPU labels, and compatible native runtime flavors used by Mesh LLM native
runtime selection.

The crate is intentionally small and publishable. It avoids depending on the
host application runtime so the SDK, installer, updater, and CLI can share the
same flavor ranking input without pulling in the full Mesh LLM app graph.

