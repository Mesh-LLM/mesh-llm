# MeshLLM Python SDK

The package exposes one real embedded `Node` with client, serve-only, and
combined roles. The legacy thin `Client` is removed. See the
[python usage guide](../../docs/sdk/python.md) for current constructors,
streaming, role selection, and native runtime requirements.

The `serve` role serves local models; `combined` also permits mesh inference.
Client mode starts no native serving runtime. Supply a Mesh LLM owner keystore
path through `ownerKeyPath`/`owner_key_path` if owner identity is required.
