#![cfg(unix)]

#[path = "../src/automation/command_interrupt/mod.rs"]
pub(crate) mod command_interrupt;

#[path = "../src/automation"]
mod automation {
    pub(crate) mod client_readiness;
    pub(crate) use crate::command_interrupt;
    pub(crate) mod daemon_readiness;
    mod private_state;
    pub(crate) mod retained_session;
    #[cfg(test)]
    #[path = "../../tests/migration_lifecycle/shared_owners.rs"]
    pub(crate) mod shared_owner_tests;
}
#[path = "migration_lifecycle/blocked_signals.rs"]
mod blocked_signals;
#[cfg(unix)]
#[path = "migration_lifecycle/cli.rs"]
mod cli;
#[path = "migration_lifecycle/daemon/mod.rs"]
mod daemon;
#[path = "migration_lifecycle/daemon/header_bound.rs"]
mod daemon_header_bound;
#[path = "migration_lifecycle/interruption.rs"]
mod interruption;
#[path = "migration_lifecycle/observation.rs"]
mod observation;
#[path = "../src/process/mod.rs"]
pub mod process;
#[path = "migration_lifecycle/protocol.rs"]
mod protocol;
#[path = "migration_lifecycle/pty.rs"]
mod pty;
#[path = "migration_lifecycle/runner_cleanup.rs"]
mod runner_cleanup;
#[path = "migration_lifecycle/support.rs"]
mod support;

#[path = "migration_lifecycle/canary_timeout_cli.rs"]
mod canary_timeout_cli;
