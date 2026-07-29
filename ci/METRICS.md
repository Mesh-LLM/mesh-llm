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

`collect-ci-metrics.py` measures timing and runner queue only; it does not
measure sccache. Compile jobs retain machine-readable
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

| Phase | Change class | Provider / runner | Samples | p50 | p90 | p95 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Product-v2 PR graph | full CI refactor | hosted mix | 1 | 1h 7m 34s | 1h 7m 34s | 1h 7m 34s | Green; queue-contaminated; composition 10s–70s |
| Depot canary cold | trusted main canary | Depot | pending | — | — | — | Restricted workflow allowlist |
| Depot canary warm | trusted main canary | Depot | pending | — | — | — | Same SHA, runner size, and image |
| Main after rollout | full main | mixed | pending | — | — | — | Five comparable green runs minimum |
