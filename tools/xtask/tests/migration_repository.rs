//! Behavioral parity for the repository checks ported from `scripts/`:
//! affected-crate closure, Conventional Commits, the environment-mutation
//! census and the llama.cpp upstream-pin guard. Expected outputs are the
//! legacy scripts' observed bytes (see `.omo/evidence/task-8-*`).

#[path = "migration_repository/support.rs"]
mod support;

#[path = "migration_repository/affected_crates.rs"]
mod affected_crates;
#[path = "migration_repository/conventional_commit.rs"]
mod conventional_commit;
#[path = "migration_repository/env_mutation_census.rs"]
mod env_mutation_census;
#[path = "migration_repository/upstream_pin.rs"]
mod upstream_pin;
