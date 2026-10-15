# Qwen3.5-4B native differential findings

The focused DGX Spark test found identical Skippy and upstream llama.cpp logits
and token choices when prompt tokens, continuation history, sampling, and prefill
batch sizes matched. Restoring a 256-token prefix with its token history also
matched cold execution exactly. Changing prefill batching changed logits in both
engines; at temperature 1 it changed some sampled choices despite an identical seed.

## Identity and method

- Model: `unsloth/Qwen3.5-4B-GGUF`, `Qwen3.5-4B-Q4_K_M.gguf`.
- GGUF SHA-256: `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`.
- Upstream source: `8212c7802455255460ab8e18fc34754560031b34`.
- Mesh source supplying the retained native libraries:
  `0d29c797794890af4aa2e72dfb8286dce739509d`.
- Patched llama.cpp source: `7c1b680d5c413d60177f3a4789d839232aa8ff88`.
- The retained cold warmup request rendered to 593 identical prompt token IDs.
  Thinking was disabled, with no API tool definitions.
- Context 131072, F16 KV, flash attention on, one sequence, all layers on GPU,
  MTP disabled, batch and microbatch capacity 512.
- Seed 42, top-k 20, top-p 0.95, min-p 0, presence penalty 1.5,
  repetition penalty 1, penalty history 64; temperature as shown below.

A temporary C++ probe linked against each retained native library. It compared
ordinary upstream decoding, ordinary decoding through the patched library, and
the Skippy native session API. It prefills all but the final prompt token in
chunks of 512 or 64, then decodes that final token in a one-token call. This is a
controlled native comparison, not a reproduction of every llama-server scheduling
decision.

Each temperature's upstream baseline supplied the continuation token IDs. Other
cases consumed that exact history even if their sampled choice differed, isolating
per-step differences from divergence caused by different preceding answers.
Every vocabulary logit was compared before sampling. Skippy's sampler was also
compared with the common sampler on the same Skippy logits and accepted history;
there were no mismatches.

The restore case exported full native state after 256 prompt tokens, recreated
the session, imported state, set its position, restored token history, and
continued. This exercises the full-state path; it does not certify every cache
payload or replay path.

## Results

| Comparison | Temperature | Steps | Largest absolute logit difference | Different token choices |
|---|---:|---:|---:|---:|
| Upstream 512 vs patched ordinary 512 | 0 | 64 | 0 | 0 |
| Upstream 512 vs Skippy 512 | 0 | 64 | 0 | 0 |
| Upstream 512 vs Skippy 64 | 0 | 64 | 2.591862 | 0 |
| Upstream 64 vs Skippy 64 | 0 | 64 | 0 | 0 |
| Cold Skippy 64 vs restored Skippy 64 | 0 | 64 | 0 | 0 |
| Upstream 512 vs Skippy 64 | 1 | 63 | 2.591862 | 3 |
| Upstream 64 vs Skippy 64 | 1 | 63 | 0 | 0 |

Temperature 0 is valid for the recorded comparison. It selected identical tokens
throughout this trace, including across these batch sizes. That observation is
not a guarantee of identical greedy output for all prompts or model families.

Batch-dependent numerical differences provide a demonstrated mechanism for
cross-engine sampled divergence. This test does not reconstruct the entire earlier
8192-versus-70-token response pair or establish why either answer was semantically
unusable. The greedy trace itself repeated empty code-context delimiters. Matching
engines does not establish answer quality.

## Output limits and scope

The separate Rust change removes the fixed 8192-token default: an omitted output
limit now resolves to the remaining context. Explicit client and operator limits
remain effective. Unit tests cover default resolution, explicit/configured limits,
context clamping, reasoning-budget preservation, and Mesh/Skippy default parity.
This prevents the old fixed-cap truncation; it cannot make a looping answer useful,
and reaching the context boundary can still produce `length`.

These native probes used unchanged retained libraries, not a rebuilt Mesh server.
The 64-step ceiling was an intentional diagnostic bound, not a successful complete
generation or a sustained replay result. Strict replay continues to reject `length`
and structurally incomplete responses during warmup and measurement. No broad
replay was run for this diagnostic, and this is not all-model certification.

Retained evidence includes the probe source and binaries, request, rendered
prompts, per-step token choices, full vocabulary logits, comparison summaries,
and SHA-256 manifests. Probe source SHA-256:
`aadca65f5bb595b6ff77be94631d771b44b0216076d544727db103ac47f295cf`.
