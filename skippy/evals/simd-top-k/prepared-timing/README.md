# Prepared heap-filter timing jobs

These launch configurations implement the unchanged [M6](../plan.json) and
[Spark](../spark-plan.json) gates for the [restricted candidate](../heap-filter-plan.json).
Prepared files are not submitted jobs.

Each hardware lane has a five-block screen, three qualification jobs covering
**0–9, 10–19 and 20–29**, and a final combined audit. Qualification uses all
30 fresh blocks; screen observations never enter its statistics. The qualification
chunks have no numerical early-stopping decision. Preserve every chunk, including
failures, and audit all 4,800 rows and 120 fresh workers together. R remains a
diagnostic control, and each hardware lane qualifies independently.

[timing-stages.json](timing-stages.json) records the exact dependencies, source,
node, binary/input bindings, output directories and spec hashes. M6 bindings are
resolved. Spark bindings now match its verified build and once-only frozen-B CUDA
capture on `spark-000907`; [spark-bindings.json](spark-bindings.json) records the
verified evidence and preserves the original template lineage. Both lanes still
require correctness/ISA and screen gates before qualification. Never submit a
`.template.json` file or substitute an unverified executable, input manifest or node.

Before submission, require the shared correctness/ISA audit and its downloaded
artifact to match the admitted source and binaries. Qualification additionally
requires the screen's numerical gates and metric/artifact readback. Later chunks
wait for the previous chunk's execution/evidence audit; the aggregate waits for
all three. Record and read back the resolved spec before one submission, persist
its intent and JY ID, and recover that existing job after an uncertain response.
Do not blindly resubmit.

Each job verifies source, native tree, model, inputs and executables; refuses to
overwrite its deterministic output directory; and retains observations and host
records even on failure. The aggregate concatenates the original rows without
renumbering, normalization or exclusions, then invokes the native 30-block auditor.
Independent statistics, per-job placement/thermal evidence and shared MLflow
readback remain required before any qualification claim.

The model-free `simd-orchestration-contracts` test runs all three chunk ranges,
checks their combined qualification result, and rejects a missing middle chunk
or an unregistered start block. Its synthetic values are test data, never measured
performance. Run it with `just sampler-simd-controller NATIVE_SOURCE`.
