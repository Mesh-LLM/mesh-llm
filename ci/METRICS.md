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
