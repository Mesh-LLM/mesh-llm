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

Revision 3 completed all 30 qualification blocks and failed the sampler's 1% gate.
It reduced sampling latency by 83.7% versus existing Skippy but remained 14.5%
slower than llama.cpp. A JY/M6 Instruments profile and disassembly localized the
remaining cost to a reference-capturing candidate-fill lambda: it reloaded the
vocabulary size per token and compiled as a scalar loop. The same-binary reference
control and sampled CPU core classes did not explain the gap.

Revision 4 replaces that lambda with a helper receiving loop inputs by value. It
requires fresh correctness checks, the existing five-block development screen,
and 30 new qualification blocks with unchanged gates. Earlier timings cannot
qualify the revised source. Revision 2's planned screen was stopped incomplete
before outcome inspection; it contributed no accepted performance result.

Revision 4 restored vectorization but its five-block screen still failed: sampler
C/B was 1.00893 overall (upper bound 1.01176), and 1.01853 before lazy trigger
activation. Revision 5 leaves the native patch queue unchanged, moves evaluator
validity assertions outside timing, and adds a diagnostic reference control in
the candidate process. It runs all five cases in six balanced B/R/C blocks
after three discarded warmup blocks, with 20 repetitions per cell. `R` is the
real llama.cpp sampler inside the candidate binary, sharing its process with C.
No decode or qualification claim is made from this diagnostic.

Select `diagnostic` as the controller phase after the normal build/correctness
prerequisites. Preserve the raw observations and independently audit C/B, R/B,
and C/R comparisons. A later production change still requires its own fresh
correctness, passing development screen, and 30-block qualification.

Revision 6 adds a literal-search path for nonempty printable ASCII trigger
patterns with no regex syntax and default ECMAScript flags. Regex creation and
all other matching stay unchanged, including escaped and non-ASCII patterns.
This preserves trigger words assembled across token pieces. New core patch
`0012` owns this upstream grammar capability, which no earlier patch modifies;
it is independent of Skippy's sampling dispatch in `0002`. No ABI changes.

Before screening, the revision-5 control must be independently verified and the
[preregistered decision map](diagnostic-decisions.json) followed. The frozen
baseline remains primary. A same-binary control cannot replace a failed baseline
comparison. Native audit now enforces the aggregate gate explicitly as well as
the per-case gates; the external verifier already required both.
