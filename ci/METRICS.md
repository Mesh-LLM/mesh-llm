# CI measurement contract

Use `scripts/collect-ci-metrics.py` for repeatable, read-only GitHub Actions
measurements. This document defines method, not a historical baseline. Store raw
observations as workflow artifacts or in the owning tracking issue; do not add
dated run conclusions to authoritative CI documentation.

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

## Definitions

- **Workflow wall time:** run `created_at` to terminal `updated_at` for a first
  attempt.
- **Workflow queue time:** run `created_at` to `started_at` for a first attempt.
- **Job queue time:** job `created_at` to `started_at`.
- **Dependency wait:** workflow/job readiness delay caused by `needs`, distinct
  from runner queue.
- **Job execution:** job `started_at` to `completed_at`.
- **Step execution:** step start to completion as returned by the jobs API.
- **Runner-minutes:** sum of allocated job execution seconds divided by 60.
- **Peak workers:** maximum overlap of started, incomplete allocated jobs,
  grouped by platform, provider and runner role.
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
minutes. It may still prove correctness and artifact reuse.

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

The threshold is an acceptance gate only for comparable warm cohorts. A cache
hit is not correctness evidence; native artifacts must still pass their build
stamp, manifest and checksum verification.

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
Record provider-specific queue and startup separately from build execution.

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

Do not commit raw timing JSON, dated tables, canary anecdotes or transient run
URLs under `ci/`. Promote only durable methodology or policy into these docs.
