# MeshLLM Python SDK

The package exposes one real embedded `Node` with client, serve-only, and
combined roles. The legacy thin `Client` is removed. See the
[python usage guide](../../docs/sdk/python.md) for current constructors,
streaming, role selection, and native runtime requirements.

The `serve` role serves local models; `combined` also permits mesh inference.
Client mode starts no native serving runtime. Supply a Mesh LLM owner keystore
path through `ownerKeyPath`/`owner_key_path` if owner identity is required.

## Component tests

From the repository root, use an existing Python 3.10 or newer interpreter:

```bash
python3 -I mesh/sdk/python/tests/test_client.py
```

This existing unittest runner adds only this component's source directory to
its import path. It tests the actual Python node lifecycle, cancellation and
stream behavior with local native-handle fixtures; no external Python test
packages, installed wheel, native build, model or endpoint are required for
these cases. Packaging separately uses the Hatchling build dependency declared
in `mesh/sdk/python/pyproject.toml`; building and publishing a wheel still requires
the matching native bridge.

This is an optional language-component command, separate from generic
`just ci-validate`. It does not replace or qualify the four retained external
SDK clients, their required smoke/release or explicit embedding-workload
cadence, or real platform/model compatibility.
