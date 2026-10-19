# AgentX smoke v8 comparison charts

Qwen3.5-4B Q4_K_M, three same-host repetitions per engine at offered root-session
concurrency 1/2/4, in llama/Skippy/Skippy/llama/llama/Skippy order.
These charts document the campaign in [PR #2463](https://github.com/Mesh-LLM/mesh-llm/pull/2463).
They are local comparative smokes, not an official AgentX leaderboard submission.

| Platform | Chart | Plot specification | Evidence summary |
|---|---|---|---|
| DGX Spark 2 / CUDA | [PNG](cuda.png) | [Vega-Lite](cuda.vl.json) | [JSON](cuda-summary.json) |
| M3 Ultra / Metal | [PNG](metal.png) | [Vega-Lite](metal.vl.json) | [JSON](metal-summary.json) |

The self-contained Vega-Lite v6 specifications embed their plotted data. They can
be rendered using Vega-Lite's `vl2svg` command or opened in the Vega editor.
The checked-in PNGs were rendered through Vega-Lite/Vega and rasterized with Sharp.
No runtime or benchmark configuration is changed by these assets.

Throughput and TTFT bars show the median across completed arms. Error bars show
the observed minimum and maximum, not confidence intervals. TTFT is each arm's
p50; derived decode is 1000 divided by median inter-token latency in milliseconds.
Memory bars show the maximum observed process RSS across completed arms. This is
not isolated GPU residency. Metal physical footprint remains available in the
JSON and PR table. CUDA footprint and GPU residency counters are unavailable.
Compare engines within a platform; actual completed workloads differ between
platforms. Concurrency is offered root sessions, not GPU decode batch size.
At c4, child branching produces five overlapping HTTP requests. Metal c1 output
throughput is dispatch-window limited, not a saturated-capacity measurement.

CUDA c2 is deliberately absent from every chart: llama completes four requests
and 1,821 output tokens, while Skippy completes three requests and 691 tokens.
A follow-up crosses the 60-second sending cutoff only on Skippy. One Skippy c2
arm also fails before inference on model-file identity verification. The original
c2 values remain in the summary JSON and the PR's explicitly qualified table;
missing bars are not zero-valued measurements. CUDA c1/c4 and all Metal cells
have matching measured trace/turn/output tuples across the six arms.

Performance qualification remains open. Native functional qualification is
Metal 12/12 and CUDA 10/12; the two strict CUDA cross-batch continuation failures
are not waived by these performance observations.

## Proposed correction for a clean c2 rerun

Use a versioned fixed-work smoke: select the same root traces, issue each once,
and let all admitted parent turns, child branches and joins finish. Preserve
recorded delays and requested outputs. Use an external safety deadline that
fails an incomplete run, rather than a dispatch cutoff that silently changes
its measured work. The harness has session-count admission support; adapting
its agentic scenario must also remove the scenario's default duration and the
duration-only grace setting, and verify sampler exhaustion and DAG completion.
Merely dropping the explicit 60-second flag would restore the scenario default.

Before accepting paired results, validate completed trace/turn multiplicities,
requested and actual output totals, no omitted descendants, zero request errors
or cancellations, and matching source cohort identity. Test the case where a
parent response arrives after the old cutoff, including delayed child and join
paths. Keep the duration-based full benchmark separately versioned and labelled.

This proposal has not been implemented or rerun. A longer sending window alone
can move the same boundary problem to a later turn. The model-file identity
startup failure also needs diagnosis and successful reruns with its integrity
guard preserved. A new comparison must repeat both engines in ABBAAB order;
it must not replace only a failed or unfavorable arm.
