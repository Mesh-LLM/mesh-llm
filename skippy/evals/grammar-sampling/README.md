# Grammar sampler A/B/C experiment

This experiment compares the existing Skippy sampler (A), the real llama.cpp
`common_sampler_sample` entrypoint (B), and Skippy with sampling-first grammar
validation (C). The owning native patch remains `0002`; its logprobs-dependent
patch is refreshed for the extracted `chat_sampling` module.

The user selected **JY-only measurements on Apple M6**, using Qwen3.5-9B Q4_K_M
because the available M6 nodes have 24 GB memory. The exact model, controls and
qualification gates are in [plan.json](plan.json). Results do not qualify 35B
decode speed or Max agent latency.

`runner.cpp` calls the production entrypoints without reproducing their algorithms.
Both arms use identically configured native llama contexts. Five synthetic cases
cover the actual model-generated lazy tool grammar, an HTML file argument,
Unicode, forced proposal rejection, closing delimiters and EOS. One baseline
process serves A/B; a second process serves C. Only one computes at a time.

The prerequisite phase captures teacher-forced model logits once. The restricted
grammar case deliberately makes an invalid proposal most likely; other cases
retain the model's logits. Fixture bytes, generated grammar, token IDs and logits
are hashed before any timing and reused for screening and qualification. This
isolates sampler cost from different model outputs. Matched decode uses the same
teacher-forced sequence and includes native decoding and sampling; a separate
free-running check requires seeded B/C token decisions to match.

`controller.cpp` preserves raw per-token timings in balanced A/B/C block order.
`audit.cpp` checks completeness, decisions and thermal observations, and computes
paired confidence bounds. Speed qualification is separate from external identity,
MTP correctness, co-tenancy and MLflow readback gates. No PR is authorized until
all gates pass. The earlier M1 Ultra prerequisite job measured no inference.

Build on JY from the candidate repository, supplying its sibling baseline checkout:

```sh
just grammar-sampling-build /absolute/path/to/baseline
just grammar-sampling-tests
```

Source delivery uses two immutable JY workspace revisions. The launch configuration
and phase attempt must be recorded and read back from shared MLflow before JY
submission. Keep credentials outside specs and artifacts. Retain failed attempts;
source repairs require new immutable execution identities and fresh measurements.

The input corpus is entirely synthetic, authored for this campaign under the
repository's license. No user documents, credentials or original Instruments
trace are included. Do not run timing or inference on the developer's machine.
