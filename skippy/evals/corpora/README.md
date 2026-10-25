# Shared validation corpora

These inputs are shared by family certification, speculative decoding checks,
and the optional `trajectory-reader` corpus generator:

- `kv_mixed_prompts.jsonl`: mixed prompts for correctness and draft checks.
- `speculative_coding_prompts.jsonl`: family certification's default draft corpus.
- `bench_corpus_sources.json`: dataset source manifest for `just bench-corpus`.

Generated datasets remain under `target/`. These inputs do not require the
retired Skippy benchmark launcher.
