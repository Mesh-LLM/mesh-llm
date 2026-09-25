//! `ci plan` parity with `scripts/plan-ci.py`. Every frozen case under
//! `fixtures/ci_plan/cases` has a golden produced by the legacy planner
//! (Python 3.13, bash 5, the fixture `cargo metadata`) and normalized with
//! `jq -c .` exactly as `.github/actions/plan-ci/action.yml` hashes it. Plans
//! must match those bytes in full, and failures must match the legacy
//! diagnostic and status. Set `MIGRATION_CI_PLAN_LEGACY_PYTHON` (and
//! `MIGRATION_CI_PLAN_LEGACY_BASH` for a bash 5) to also run the legacy
//! planner side by side on identical inputs.

#[path = "migration_ci_plan/support.rs"]
mod support;

#[path = "migration_ci_plan/command_surface.rs"]
mod command_surface;
#[path = "migration_ci_plan/frozen_cases.rs"]
mod frozen_cases;
