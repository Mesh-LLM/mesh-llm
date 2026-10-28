# Portable SIMD top-k investigation

This is a separate experiment on top of the qualified upstream-backed sampler.
It changes top-k selection only, using Highway as the portable SIMD candidate.
Grammar caching and other arithmetic changes are separate hypotheses.

[plan.json](plan.json) freezes the arms, controls, workload and gates. B is the
original frozen llama reference; C is the current qualified implementation;
D changes top-k; R disables that change in the candidate build and is diagnostic
only. All timing and model execution run through JY on the admitted Apple M6.

First profile the exact qualified C binary across all five existing grammar
cases, verify the stack exports and inspect the selection instructions. This
phase produces no qualifying timing result and needs no candidate rebuild:

```sh
just sampler-simd-profile /absolute/qualified-workspace /absolute/model.gguf /absolute/artifacts
```

The launch must be recorded and read back in shared MLflow before JY dispatch.
The job account must be able to read the retained immutable workspace and model.
Only exported sampled stacks and disassembly leave the profiling job; raw trace
bundles remain private on the node. The profiler/target environment is scrubbed.

The candidate source, Highway source pin, build flags and correctness fixtures
must be frozen before build admission. Every required correctness check must pass
before a five-block screen. Only a passing screen admits 30 fresh qualification
blocks; R never replaces either primary comparator or normalizes their results.

The predecessor's source, captured inputs and historical results stay immutable.
Its [qualified figures](../grammar-sampling/figures/README.md) describe the earlier
comparison, not evidence that this proposed SIMD change is faster.

The separately authorized [Spark plan](spark-plan.json) replicates B/C/D/R on one
Linux/CUDA DGX Spark. It has its own built reference binaries and captured logits,
frozen before timing. The same semantic cases and numerical gates apply. Spark
and M6 results are never pooled; each stratum must qualify independently.

```sh
just sampler-simd-spark-probe /absolute/model.gguf /absolute/artifacts
```

The four-arm controller creates a fresh worker for each arm and block, with one
resident model at a time. Every worker completes three whole warmup blocks before
its measured block. Screen and qualification ranges and arm orders are fixed by
the plan; it refuses to overwrite existing observations. Build it independently
with `just sampler-simd-controller /absolute/native-source` so later orchestration
changes cannot silently rebuild admitted worker binaries.

The B build uses the original native sampler interface. ABI recovery verification
is built for C, D and R, which expose the newer public implementation being tested.
This harness selection does not alter the frozen B native source.

M6 reuses its admitted B/C binaries and input hashes. Build only the new D/R
workers with `just sampler-simd-candidate-build metal`; their paths are separate
from the frozen references. The controller-only build also compiles `simd-audit`,
which checks every arm, case, scope, warmup and process identity before evaluating
the frozen paired bootstrap gates. Its statistical result still requires the
separate source, correctness, placement and MLflow audits.

The controller-only build runs model-free synthetic contracts. They exercise the
whole five-block controller, exact paired constant-ratio confidence bounds,
failure gates, and rejection of missing/duplicate cells, unknown thermal state,
changed ranges/orders, malformed timings and divergent token decisions. Their
synthetic times are test data and never enter MLflow performance metrics.
Variable ratios also have an independent closed-form check: five case factors
and two doubled blocks give known case-balanced geometric means and binomial
bootstrap quantiles. This checks aggregation and interval direction without
copying the auditor's random-number or reduction implementation.
