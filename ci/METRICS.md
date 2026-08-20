# CI measurement contract

Use `scripts/collect-ci-metrics.py` for repeatable, read-only GitHub Actions
measurements. The collector emits schema version 3 reports and deterministic
rollout signals; this document defines method, not a historical baseline.
Store raw observations as workflow artifacts or in the owning tracking issue;
do not add dated run conclusions to authoritative CI documentation.

## Collection

Collect a bounded comparable cohort:

```bash
python3 scripts/collect-ci-metrics.py \
  --repo Mesh-LLM/mesh-llm \
  --workflow pr_linux.yml \
  --event pull_request \
  --limit 20 \
  --label profile=pr-ready \
  --label provider=github \
  --raw-out /tmp/pr-ci-runs.json \
  --json-out /tmp/pr-ci-metrics.json \
  --markdown-out /tmp/pr-ci-metrics.md
```

Collect exact runs when validating a migration canary:

```bash
python3 scripts/collect-ci-metrics.py \
  --run-id RUN_ID \
  --run-id RUN_ID \
  --raw-out /tmp/ci-canary-runs.json \
  --json-out /tmp/ci-canary-metrics.json
```

Reprocess saved observations without another API request:

```bash
python3 scripts/collect-ci-metrics.py \
  --input /tmp/ci-canary-runs.json \
  --json-out /tmp/ci-canary-recomputed.json \
  --markdown-out /tmp/ci-canary-recomputed.md
```

Build a historical PR cohort and compare a candidate cohort without mixing
provider dimensions. The input labelled `provider=github` below is a cohort
label for the report; the runner labels in the report remain the source of
truth for the actual provider.

```bash
python3 scripts/collect-ci-metrics.py \
  --repo Mesh-LLM/mesh-llm \
  --workflow pr_linux.yml \
  --event pull_request \
  --limit 20 \
  --label cohort=historical-pr \
  --label provider=github \
  --raw-out /tmp/pr-github-history.json \
  --json-out /tmp/pr-github-history.metrics.json

python3 scripts/collect-ci-metrics.py \
  --input /tmp/pr-depot-candidate.json \
  --label cohort=depot-candidate \
  --label provider=depot \
  --compare-input /tmp/pr-github-history.json \
  --json-out /tmp/pr-depot-comparison.metrics.json \
  --markdown-out /tmp/pr-depot-comparison.metrics.md
```

Use the same workflow, event, plan/profile, selected slice IDs and source
conditions for both cohorts. A single `--limit` query is not a provider filter;
inspect `jobs.by_runner` and retain the included/excluded run IDs in the
tracking artifact. Run the command separately for each focused PR entrypoint
when monitoring the complete five-lane graph.

## Definitions

- **Workflow wall time:** run `created_at` to terminal `updated_at` for a first
  attempt.
- **Workflow queue time:** run `created_at` to `started_at` for a first attempt.
- **Job queue time:** job `created_at` to `started_at`.
- **Dependency wait:** the interval from job `created_at` to an instrumented
  dependency-ready time (or explicit `dependency_wait_seconds`), distinct from
  runner queue. The standard GitHub jobs API does not expose dependency-ready
  timestamps; the collector reports `n/a`, never a fabricated zero, when they
  are absent.
- **Runner queue time:** `started_at` minus dependency-ready time when that
  timestamp exists. Without instrumentation, the collector falls back to the
  raw creation-to-start interval for compatibility and the report's
  definitions identify that dependency/runner separation is unavailable.
- **Job execution:** job `started_at` to `completed_at` (`execution_seconds`,
  with `duration_seconds` retained as a compatibility alias).
- **Job wall time:** job `created_at` to `completed_at`; this includes runner
  queue and any measured dependency wait.
- **Step execution:** step start to completion as returned by the jobs API.
- **Runner-minutes:** sum of allocated job execution seconds divided by 60.
- **Peak workers:** maximum overlap of started, incomplete allocated jobs,
  grouped by operating system, provider and semantic runner role when those
  dimensions are present.
- **Critical path candidate:** the dependency chain ending at the last required
  successful job, reconstructed from checked workflow `needs` where possible.
- **Cancelled work:** execution time consumed by a run later cancelled or
  superseded.

GitHub retains original run-level timestamps across reruns while its job API may
return the latest attempt. Do not combine original run wall/queue fields with
latest-attempt jobs. Mark rerun workflow timing excluded and analyze valid job
durations separately.

## Cohort rules

Before comparing results, match:

- workflow and event;
- CI plan/profile and required slice IDs;
- change class;
- source SHA when evaluating cold/warm behavior;
- provider and runner role/size;
- runner image/toolchain epoch;
- cache mode and expected warm/cold state;
- run conclusion and attempt policy;
- sample size.

Separate docs-only, UI, ordinary Rust, runtime product, backend, SDK and
CI-control changes. A faster routed workflow does not prove faster compilation,
and a larger runner does not explain reduced queue time.

Treat a run as capacity-contaminated for provider execution comparisons when
job runner-queue p95 or the terminal required job's runner queue is at least five
minutes. The report records this per cohort in `heuristics.capacity_contaminated`
and in the affected run reports. It may still prove correctness and artifact
reuse, but it must not be used as clean provider-latency evidence.

## Required observations

Every PR/main graph comparison records:

- plan schema/profile/digest;
- required slices and selection reasons;
- allocated and peak workers by provider/platform/role;
- wall, runner queue, dependency wait and execution time;
- runner-minutes and cancelled runner-minutes;
- setup/container time;
- producer artifact identities;
- fallback rebuild count in consumers;
- cache mode, hit/miss/error counts and cache publication authority;
- expected and unexpected skips.

The report's schema-v3 timing fields are deliberately separate:

| Report path | Meaning |
| --- | --- |
| `workflow.wall_seconds` | first-attempt workflow wall p50/p90/p95 and sample count |
| `workflow.queue_seconds` | first-attempt workflow queue p50/p90/p95 |
| `workflow.dependency_wait_seconds` | measured dependency wait, or zero samples when unavailable |
| `jobs.by_runner` | provider, OS, architecture, role, runner size and timing percentiles |
| `jobs.execution_seconds` | allocated build execution timing |
| `jobs.runner_queue_seconds` | runner assignment queue, separated when dependency-ready instrumentation exists |
| `capacity` | runner-minutes, cancelled runner-minutes and peak workers |
| `heuristics` | queue thresholds, contamination and deterministic state |
| `comparison` | provider separation and candidate-vs-baseline p95 deltas when `--compare-input` is used |

Every percentile is accompanied by `count`; a missing count or a cohort with
fewer than three job samples is not rollout evidence.

Do not infer compiler-cache behavior from workflow time. Instrument sccache
jobs with machine-readable statistics, reset counters immediately before the
measured build, and aggregate:

```text
sum(cache_hits) / (sum(cache_hits) + sum(cache_misses))
```

Evaluate retained sccache artifacts offline:

```bash
python3 scripts/summarize-sccache-stats.py \
  --minimum-hit-rate 0.80 \
  /tmp/sccache-evidence
```

The threshold is an acceptance gate only for comparable warm cohorts. CI's
capture action labels each observation `cold`, `opportunistic`, `warm-pass`,
or `warm-failure`; only an exact seed restore is held to its configured warm
threshold. A cache hit is not correctness evidence; native artifacts must
still pass their build stamp, manifest and checksum verification.

## Composition migration targets

Target values are budgets, not baselines:

| Signal | Budget |
| --- | --- |
| Draft PR required signal | 12 minutes when runner queue is below one minute |
| Ready ordinary runtime PR | 18 minutes under the same queue condition |
| Ordinary PR peak allocation | At most ten workers |
| Artifact consumer rebuilds | Zero |
| Duplicate producer identity per run | Zero |
| Main workspace test coverage | Every member exactly once |
| Selected PR/main row parity | Same workflow, commands, profile and artifact contract |

Do not loosen correctness or expand fan-out merely to meet a timing target.

## Depot comparisons

Depot comparisons must use the same checked plan and slice inputs as GitHub.
Record provider-specific queue and startup separately from build execution and
keep the provider cohorts disjoint. Provider comparison is scoped to build and
test executors; hosted orchestration, summaries, runner selectors, and
credentialed smoke jobs remain visible in the full report but are excluded from
the provider cohort. `compare_reports` marks a comparison `hold` when those
executor provider sets overlap, no job family or OS/architecture family is
common, either baseline or candidate has fewer than three runner-queue samples,
or the candidate queue heuristic is not eligible. This prevents hosted control
jobs or weak historical evidence from masking a provider or build-graph change.

The built-in, date-independent heuristic constants are:

| Signal | Rule |
| --- | --- |
| Minimum evidence | fewer than 3 `jobs.runner_queue_seconds` samples → `insufficient_sample` (checked before queue warning and capacity contamination) |
| Queue warning | with minimum evidence met, `jobs.runner_queue_seconds` p95 above 60 seconds, or unavailable → `hold` |
| Capacity contamination | `jobs.runner_queue_seconds` or terminal-job runner queue p95 at least 300 seconds → `rollback` |
| Provider cohort separation | baseline and candidate runner-provider sets must be disjoint |

These are rollout/rollback heuristics, not historical conclusions. A maintainer
may override a state only with an issue/artifact containing comparable runs,
correctness results and an explicit reason.

PR-on-Depot cache isolation is a security gate, not a performance experiment.
Do not run untrusted PR content on Depot until `DEPOT_MIGRATION.md` requirements
pass. Never use cache keys as evidence of isolation.

## Reporting

A migration report states:

- exact plan/change class;
- included and excluded run IDs in the tracking issue or artifact;
- provider, runner role, image/toolchain epoch and cache mode;
- p50/p90/p95 with sample count;
- queue-contaminated samples;
- correctness, artifact and cache conclusions separately;
- rollback decision.

For a PR-Depot rollout, include the five focused lane reports, a comparable
historical PR cohort, and the exact plan/profile and cache mode. Keep queue,
dependency wait, execution and wall-time conclusions separate. Do not infer
cache or compiler throughput from workflow wall time, and do not treat a Depot
run as proof of cache isolation.

Do not commit raw timing JSON, dated tables, canary anecdotes or transient run
URLs under `ci/`. Promote only durable methodology or policy into these docs.
