# mesh-llm-provider-runtime

This crate defines MeshLLM's package boundary for executable model providers.
It is deliberately separate from `mesh-llm-native-runtime`: native runtimes
contain backend libraries selected by the Skippy ABI, while provider runtimes
are independently signed processes with their own protocol and model surface.

The crate owns:

- the versioned `provider-runtime.json` bundle manifest;
- the `provider-runtimes.json` release index;
- platform, provider, protocol, and model selection;
- file and archive checksum validation;
- safe ZIP extraction and immutable cache installation;
- bundled, cached, and downloadable runtime resolution.

It does not launch provider processes or expose SDK-specific lifecycle APIs.
Those belong to the host supervisor and language SDK layers built on this
contract.

Provider bundles are installed beneath:

```text
<cache>/<artifact-id>/<version>/<os>-<arch>/
```

Artifact coordinates are immutable. Installing different bytes at an existing
coordinate is rejected rather than overwriting a previously verified runtime.
