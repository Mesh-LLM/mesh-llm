# skippy-engine

Engine-neutral staged execution contract for Skippy.

The crate contains shared stage descriptors and the `StageEngine` trait. It has
no model runtime of its own: concrete engines keep native arrays, model handles,
and KV caches private while exchanging token IDs and Skippy-owned activation
buffers with the server transport.

The first concrete second-engine implementation is `skippy-engine-mlx`. The
existing llama.cpp implementation remains in `skippy-runtime` while its broader
cache, multimodal, MTP, and session surface is migrated behind this contract.
