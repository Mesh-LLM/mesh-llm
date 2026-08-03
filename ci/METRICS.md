# CI timing metrics

Use `scripts/collect-ci-metrics.py` to establish repeatable timing baselines for
PR Builds, main CI, or exact workflow runs. The script is dependency-free beyond
Python and the GitHub CLI, and every GitHub operation it performs is read-only.

Collect successful PR Builds and save the raw observations as well as JSON and
Markdown summaries:

```bash
python3 scripts/collect-ci-metrics.py \
  --repo Mesh-LLM/mesh-llm \
  --workflow pr_builds.yml \
  --event pull_request \
  --limit 30 \
  --label provider=github \
  --raw-out /tmp/pr-builds-runs.json \
  --json-out /tmp/pr-builds-metrics.json \
  --markdown-out /tmp/pr-builds-metrics.md
```

Collect main CI over a bounded date range:

```bash
python3 scripts/collect-ci-metrics.py \
  --repo Mesh-LLM/mesh-llm \
  --workflow ci.yml \
  --branch main \
  --created '>=2026-07-01' \
  --limit 50 \
  --json-out /tmp/main-ci-metrics.json \
  --markdown-out /tmp/main-ci-metrics.md
```

Analyze exact runs or reprocess saved observations without another API request:

```bash
python3 scripts/collect-ci-metrics.py \
  --run-id 30435682397 \
  --run-id 30460057494

python3 scripts/collect-ci-metrics.py \
  --input /tmp/pr-builds-runs.json \
  --json-out /tmp/pr-builds-recomputed.json \
  --markdown-out /tmp/pr-builds-recomputed.md
```

When no output path is supplied, the Markdown report is written to stdout. Use
`--json-out -` for machine-readable stdout.

## Timing definitions

- Workflow wall time is GitHub's run `created_at` to `updated_at`.
- Workflow queue time is run `created_at` to `started_at`.
- Workflow wall time, workflow queue time, and job start delay exclude rerun
  attempts. GitHub retains the original run-level timestamps while its jobs API
  returns the latest attempt, so combining them would create false queue and
  wall measurements. Job duration and job queue remain valid for the latest
  attempt.
- Job duration is job `started_at` to `completed_at`.
- Job queue time is job `created_at` to `started_at`. Live collection uses the
  read-only jobs API because `gh run view --json jobs` omits job creation times.
- Job start delay is workflow `created_at` to job `started_at`; it includes
  dependency wait and must not be presented as runner queue time. It is only
  reported for first attempts.
- A terminal job is the last non-skipped job to finish. It is a critical-path
  candidate, not a reconstructed Actions dependency graph.

The JSON output includes p50, p90, and p95 summaries; exact slow observations;
job-family summaries; terminal-job counts; runner labels; and individual run
metadata. Raw output intentionally excludes logs and step output.

For before/after comparisons, use the same workflow, event, change class, run
conclusion, and sample size. Keep documentation-only and full native-build PRs
in separate cohorts, and record the runner provider/image revision with
`--label`. Compare both wall time and queue time: a faster compiler does not
explain provider-capacity delays, and a shorter routed workflow is not evidence
that an unchanged build became faster.

## Migration baseline and targets

The pre-migration snapshot was collected on 2026-07-29 from the 20 successful
`pull_request` runs of `pr_builds.yml` recorded in the
[normalized baseline report](metrics/2026-07-29-pr-builds-baseline.json). It
includes the legacy workflow graph and its historical runner mix, so it is
historical workload-mix context, not a provider comparison or a controlled
before/after cohort.

| Cohort | Samples | p50 | p90 | p95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| PR Builds historical workload mix before product-v2 graph cleanup | 20 | 33m 12s | 45m 21s | 55m 33s | 1h 9m 1s |

The snapshot mixes different routed workloads: 13 runs executed 31 jobs, four
executed four jobs, and one each executed 12, 19, and 21 jobs. Three head SHAs
appear twice. Depot acceptance comparisons must group exact run IDs by change
class and executed-job graph, use the same sample size, and state whether
repeated SHAs are retained or deduplicated.

The slowest job families in that historical snapshot were Windows CUDA
(42m 9s p95), Windows ROCm (39m 23s), Windows CPU (32m 6s), Swift SDK smoke
(27m 12s), and Linux ROCm (25m 55s). These values identify legacy hotspots;
they are not a controlled before/after cohort. A single warm run is not
sufficient to replace them.

Migration success targets:

| Cohort | Target |
| --- | --- |
| Typical affected-Rust PR | p50 under 10m and p95 under 20m |
| PR native-backend build | p95 under 30m |
| Trusted main CI | p95 under 45m |
| Rust compilation cache | at least 80% hit rate on comparable warm runs |
| Artifact consumers | zero host, runtime, or ABI rebuilds |

`collect-ci-metrics.py` measures workflow, job, and (when returned by the jobs
API) step timing, runner queue, and runner dimensions derived from labels. It
does not measure sccache. Compile jobs retain machine-readable
`sccache --show-stats --stats-format json` evidence separately. Zero the
counters immediately before the measured compilation and define aggregate hit
rate as:

```text
sum(cache_hits.counts) /
  (sum(cache_hits.counts) + sum(cache_misses.counts))
```

Compare the warm member of same-SHA, same-provider, same-runner-size, and
same-image cold/warm pairs. The 80% row is an unmeasured rollout gate until
those JSON artifacts have been retained and aggregated; human-readable log
output alone is not acceptance evidence.

After downloading the selected jobs' `sccache-*` artifacts into one directory,
evaluate the gate offline:

```bash
python3 scripts/summarize-sccache-stats.py \
  --minimum-hit-rate 0.80 \
  /tmp/sccache-evidence
```

For this migration, a run is capacity-contaminated when executed-job runner
queue p95 is at least five minutes or its terminal job waits at least five
minutes for a runner. Such a run can validate correctness and artifact reuse,
but it is excluded from provider performance acceptance.

The first composable-graph quality observation is
[run 30486038630](https://github.com/Mesh-LLM/mesh-llm/actions/runs/30486038630):
27m 20s wall time, while the three clippy rows executed for 7m 4s–8m 43s.
Individual job queues reached 14m 47s, so this run is recorded as
capacity-contaminated and is not evidence of a compile-time regression.

The first green composable-graph build observation is
[run 30486038843](https://github.com/Mesh-LLM/mesh-llm/actions/runs/30486038843).
Its 36 executed jobs took 1h 7m 34s wall time, but the median job executed for
only 3m 53s while waiting 10m 33s for a runner. Job execution p95 was 17m 9s;
runner-queue p95 was 21m 52s. The terminal Kotlin SDK smoke waited 18m 4s and
then executed for 11m 55s. Queue delay, rather than product composition, was
the dominant wall-time constraint: the nine Linux, macOS, and Windows
composition action steps each took 10s–70s and rebuilt neither the host nor the
runtime. This single capacity-contaminated observation validates artifact reuse
but does not replace the multi-run baseline. It also predates the final split
of the Linux GPU matrix into independent runtime producers and thin composers.

## Runner-image publication observation

The latest read-only runner-image observation is
[run 30248081255](https://github.com/Mesh-LLM/mesh-llm-runner-images/actions/runs/30248081255)
at source `890cdc6a1472028a67f7013baa29e29be57e6529`. GitHub's run and job
timestamps provide the following measured evidence:

| Observation | Measured value |
| --- | ---: |
| Workflow wall time | 39m 15s |
| Completed jobs | 55 |
| Slowest initial `Build and verify test image` step | 14m 25s |
| Later public ROCm 7.2 AMD64 `Build and push architecture image by digest` step | 18m 03s |

The initial test matrix and later publish matrix both build architecture
images. The second timing is therefore evidence of duplicate image
construction on the publication path. It is not evidence of a cold pull,
compressed image size, or the amount of reusable cache.

The runner-image migration must measure one explicit lifecycle:

```text
build once -> stage immutable digest -> verify exact digest -> promote digest
```

Record the role, platform, backend, source SHA, staged digest, build duration,
verification duration, promotion duration, compressed bytes, provider, and
runner class. Measure cold pull only in a controlled fresh-worker cohort and
retain the raw observations. Until those records exist, the proposed role-size,
cold-pull, and publication-time thresholds in
[`DEPOT_MIGRATION.md`](DEPOT_MIGRATION.md) are design budgets, not baselines.

## Depot Registry pull-through measurements

Use the manual `depot-registry-canary.yml` workflow for registry comparisons.
The upstream input must be digest-pinned, and the Depot repository must mirror
that exact upstream repository. Each source receives five fresh ephemeral
Depot-managed runners so local layer reuse cannot turn a warm local pull into a
false registry result. Depot pre-authenticates each ephemeral runner with a
short-lived organization Registry job credential. Downloaded observations can
be reevaluated with:

```bash
python3 scripts/summarize-depot-registry-pulls.py \
  --enforce /tmp/depot-registry-pull-observations
```

Adoption requires identical resolved digests, at least five unique samples per
source, at least 20% median improvement, and at least 10 seconds saved at the
median. This result applies only to image transfer. Do not attribute it to
`apt`, Cargo, pnpm/npm, native backend compilation, or Docker layer export.

The first valid cohorts ran on 2026-08-02. Each row used five fresh runners per
source, and every sample resolved to the listed digest. None met the gate:

| Image | Digest | Run | Upstream median | Depot median | Result |
| --- | --- | --- | ---: | ---: | --- |
| GHCR Actions runner | `sha256:0cfdcc701ce933c6d243c6b0b2da767366dc9f2e99961d4c3754b0b78084cdda` | [30776030734](https://github.com/Mesh-LLM/mesh-llm/actions/runs/30776030734) | 12.055s | 12.047s | Fail; 8ms (0.1%) faster |
| Ubuntu 24.04 | `sha256:4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90` | [30776128516](https://github.com/Mesh-LLM/mesh-llm/actions/runs/30776128516) | 1.452s | 1.363s | Fail; 89ms (6.1%) faster |
| CUDA 12.9.2 | `sha256:6d2a0dabc50c3bf14d27fc66822b6b1f94a325807ace17bd1997762307790587` | [30776197769](https://github.com/Mesh-LLM/mesh-llm/actions/runs/30776197769) | 38.661s | 79.495s | Fail; 40.834s slower |
| CUDA 13.1.2 | `sha256:bff001d3257971cc4752e15ac2d354befa70995ded8e141741ade50569fc192e` | [30776298367](https://github.com/Mesh-LLM/mesh-llm/actions/runs/30776298367) | 21.537s | 43.256s | Fail; 21.719s slower |
| ROCm 7.0 | `sha256:41faf6a0e3d2302db28d5112f8896ae6b8e2d4637c4280115e1b213271c9d3f8` | [30776371194](https://github.com/Mesh-LLM/mesh-llm/actions/runs/30776371194) | 27.349s | 45.253s | Fail; 17.904s slower |
| Arch `base-devel` | `sha256:40d14ac9db5af04f695eacd82a53181ad685fecc2534a66e05a51182a077cbd5` | [30776449087](https://github.com/Mesh-LLM/mesh-llm/actions/runs/30776449087) | 7.691s | 13.071s | Fail; 5.380s slower |
| Arch `base` | `sha256:3406a568f45d68f0bef35dc80b3eacec8bda59b0292b2e50d5932ba1667f20cf` | [30776499761](https://github.com/Mesh-LLM/mesh-llm/actions/runs/30776499761) | 4.637s | 6.440s | Fail; 1.803s slower |

The runner-images and packaging repository cache gates therefore remain
`false`. Alpine was not measured because its packaging release and matrix rows
are disabled. Earlier authentication-failure runs are excluded from these
cohorts because they produced no pull observations.

| Phase | Change class | Provider / runner | Samples | p50 | p90 | p95 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Product-v2 PR graph | full CI refactor | hosted mix | 1 | 1h 7m 34s | 1h 7m 34s | 1h 7m 34s | Green; queue-contaminated; composition 10s–70s |
| Depot canary cold | trusted main canary | Depot | 1 | 37s | 37s | 37s | Six resource/cache probes; not a full build |
| Depot canary warm | trusted main canary | Depot | 1 | 37s | 37s | 37s | Six resource/cache probes; not a full build |
| Main after rollout | full main | mixed | pending | — | — | — | Five comparable green runs minimum |

## Post-Depot observations and tuning decision

The following read-only observations are from the MeshLLM repository. The two
canary runs compile one tiny C file and validate runner/cache plumbing; their
wall times are not native-build or packaging measurements.

| Run | Workload | Measured wall time | Measured Depot job evidence |
| --- | --- | ---: | --- |
| [30525111329](https://github.com/Mesh-LLM/mesh-llm/actions/runs/30525111329) | Cold canary | 37s; jobs 30–34s | Six labels: default, 4, 8, 16, ARM default, ARM-8 |
| [30525247727](https://github.com/Mesh-LLM/mesh-llm/actions/runs/30525247727) | Warm canary | 37s; jobs 29–33s | Same six labels; probe-only cache hit signal |
| [30590595090](https://github.com/Mesh-LLM/mesh-llm/actions/runs/30590595090) | Same-SHA warm non-GPU release | 53m 11s; 9 Depot jobs | CPU/native SDK/static ABI producers on x86/ARM-8 took 1m 37s–4m 20s; composers on x86/ARM-4 took 1m 17s–1m 31s |
| [30586470043](https://github.com/Mesh-LLM/mesh-llm/actions/runs/30586470043) | Same-SHA prerelease backend slice | 1h 41m 51s; 15 Depot jobs | x86 CPU `-8`: 2m 23s; x86 Vulkan `-16`: 3m 46s; x86 ROCm `-16`: 13m 25s; native SDK x86/ARM-8: 11m 16s/12m 38s |

The two full-build runs share SHA `851888d0`, but they are not a controlled
cold/warm pair: the release run restored exact static-ABI cache entries and
reported native-SDK sccache hit rates of 95.5% (x86) and 95.0% (ARM), while the
prerelease run has different workflow inputs and cache state. The warm release
also shows architecture variance within the same role: native runtime x86/ARM
was 1m 37s/2m 02s, native SDK was 4m 09s/4m 20s, static ABI was 1m 57s/1m 47s,
and composition was 1m 31s/1m 17s. These are observations, not runner-size
experiments.

No same-backend `-4`/`-8`/`-16` A/B exists in the retained evidence. The `-16`
ROCm and Vulkan rows therefore do not justify moving CPU or SDK rows to a
larger runner; their elapsed-time benefit and additional runner cost were not
measured. Keep the checked-in role split: `-8` for CPU/native-SDK/static-ABI
producers, `-4` for composition, and `-16` only for the existing high-parallel
ROCm/Vulkan runtime rows. No matrix-concurrency change is justified without a
queue/capacity series or a controlled completion-time comparison.

This repository has no Bake-group, Depot-project/billing, CPU-utilization, or
BuildKit context-upload/cache-import/export telemetry. The runner-image
repository owns the image BuildKit lifecycle; MeshLLM owns the workflow graph,
runner selectors, and cache-key contracts. Consequently this change does not
alter runner size, matrix concurrency, bake groups, cache project boundaries,
or cache identities. Step timestamps and label-derived runner dimensions are
now retained by `collect-ci-metrics.py` so a future comparable cohort can
measure build phases without inferring unavailable cache or upload timings.
