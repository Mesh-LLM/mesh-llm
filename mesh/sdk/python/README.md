# MeshLLM Python SDK

The package exposes one real embedded `Node` with client, serve-only, and
combined roles. The legacy thin `Client` is removed. See the
[Python usage guide](../../docs/sdk/python.md) for current constructors,
streaming, role selection, and native runtime requirements.

The `serve` role serves local models; `combined` also permits mesh inference.
Client mode starts no native serving runtime. Supply a Mesh LLM owner keystore
path through `ownerKeyPath`/`owner_key_path` if owner identity is required.

## Component tests

From the repository root, use an existing Python 3.10 or newer interpreter:

```bash
python3 -I -B mesh/sdk/python/tests/test_client.py
```

This existing unittest runner adds only this component's source directory to
its import path. It tests the actual Python node lifecycle, cancellation and
stream behavior with local native-handle fixtures; no external Python test
packages, installed wheel, native build, model or endpoint are required for
these cases. Packaging separately uses the Hatchling build dependency declared
in `mesh/sdk/python/pyproject.toml`; building and publishing a wheel still requires
the matching native bridge.

This is an optional language-component command, separate from generic
`just ci-validate`. It does not replace or qualify the four locked ecosystem
clients, their required smoke/release or explicit embedding-workload cadence,
or real platform/model compatibility.

## Native bridge bindings

Generate bindings from the exact prebuilt Mesh FFI library using UniFFI 0.32.0.
The main repository supplies the matching UDL and crate metadata. The library
supplies the exported Rust metadata and API checksums; UDL-only generation
can produce incompatible checksums for Mesh's combined UDL/proc-macro surface.

```bash
mesh/sdk/python/scripts/generate-python-bindings.sh /absolute/uniffi-bindgen /absolute/libmeshllm_ffi.dylib
mesh/sdk/python/scripts/build-native.sh /absolute/libmeshllm_ffi.dylib
python3 -I -B mesh/sdk/python/tests/test_binding.py
```

Use the matching `.so` or `.dll` on other platforms. The SDK-owned lazy loader
checks every generated API checksum before caching the native module. Generated
UniFFI output is preserved apart from trailing whitespace normalization.
After regenerating bindings, update the matching file hash in `sdk-inputs.json`;
for `mesh_ffi.py`, use the SHA-256 of
`src/meshllm/_generated/mesh_ffi.py` for the
`files["src/meshllm/_generated/mesh_ffi.py"]` entry. Verify the updated source
manifest from the repository root with
`cargo xtool automation smoke-observation sdk-source --kind root`.
The component tests above use mock handles; the binding tests require the real
sibling bridge and test checksum refusal and native version readback without
constructing a node or starting services. Neither test suite qualifies model
serving or a published wheel.

The SDK's source distribution includes the native bridge prepared by
`scripts/build-native.sh`, so the default `python -m build` sdist-to-wheel flow
can build the platform wheel without compiling or downloading a bridge.
