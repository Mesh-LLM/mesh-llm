//! `ci validate-lane` parity with `scripts/validate-ci-lane-results.py`,
//! plus the parsed lane/entrypoint workflow graph contracts it enforces when
//! given `--workflow`.
//!
//! Plans come from the frozen planner goldens under `fixtures/ci_plan`, and
//! the expected planned jobs per case and lane come from
//! `fixtures/ci_graph/lane_jobs.json`, which jq derived from the legacy
//! script's rules. Workflow mutations run on temporary copies of the checked-in
//! workflows. Default runs start no Python; set
//! `MIGRATION_CI_GRAPH_LEGACY_PYTHON` to also run the legacy script side by
//! side on the legacy argv.

#[path = "migration_ci_graph/support.rs"]
mod support;

#[path = "migration_ci_graph/graph_failures.rs"]
mod graph_failures;
#[path = "migration_ci_graph/happy.rs"]
mod happy;
#[path = "migration_ci_graph/result_failures.rs"]
mod result_failures;
