# skippy-ane

Apple Neural Engine support for the paired Qwen3.6 DFlash drafter. This crate
owns GGUF to Core ML conversion, compiled-model placement checks, and stateful
Core ML draft execution. `skippy-runtime` owns speculative decoding and target
verification; it calls this crate's draft API on macOS. The crate does not
depend on MeshLLM or a Skippy runtime crate.

Convert from the workspace root with `just dflash-ane-convert`. The executable
is `skippy-ane-convert`; it accepts `--draft`, `--target`, `--out`,
`--embeddings-out`, `--context`, and `--rows`. Rust callers can use the same
conversion without a child process:

```rust,no_run
use skippy_ane::conversion::{ConversionOptions, convert};

let options = ConversionOptions {
    draft: "draft.gguf".into(),
    target: "target.gguf".into(),
    out: "draft.mlpackage".into(),
    embeddings_out: "embeddings.f16".into(),
    context: 512,
    rows: 7,
};
convert(&options)?;
# Ok::<(), anyhow::Error>(())
```

The `conversion` feature is enabled by default. `skippy-runtime` disables it
so serving does not compile the GGUF reader or MIL package writer.

See [the ANE guide](../../evals/dflash_ane/README.md) for package requirements
and runtime selection.
