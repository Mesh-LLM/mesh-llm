//! `automation parity --suite ci`: shadow comparison of the legacy
//! `scripts/plan-ci.py` planner (and its protected caller contract in
//! `.github/actions/plan-ci/action.yml`) with `ci plan`.
//!
//! Default runs never start Python: the happy test runs `--rust-only`,
//! comparing Rust with the frozen goldens and the recorded action outputs,
//! and every failure contract fails before a legacy process could start.
//! Set `MIGRATION_CI_PLAN_LEGACY_PYTHON` (and `MIGRATION_CI_PLAN_LEGACY_BASH`
//! for a bash 5) to add the legacy side-by-side comparison.

#[path = "migration_ci_shadow/support.rs"]
mod support;

#[path = "migration_ci_shadow/failures.rs"]
mod failures;
#[path = "migration_ci_shadow/happy.rs"]
mod happy;
