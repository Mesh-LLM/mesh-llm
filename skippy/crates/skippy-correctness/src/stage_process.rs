//! Child server startup policy shared by correctness workloads.
use std::{path::Path, process::Command};

/// Keep the child's readiness budget consistent with the harness waiting for it.
pub(crate) fn command(binary: &Path, startup_timeout_secs: u64) -> Command {
    let mut command = Command::new(binary);
    command.args([
        "serve",
        "--startup-timeout-secs",
        &startup_timeout_secs.to_string(),
    ]);
    command
}

#[cfg(test)]
mod tests {
    use super::command;
    use std::path::Path;

    #[test]
    fn stage_commands_forward_the_full_readiness_budget() {
        for timeout in [1, 60, 790, 1800] {
            let command = command(Path::new("skippy"), timeout);
            let args = command.get_args().collect::<Vec<_>>();
            assert_eq!(
                args,
                ["serve", "--startup-timeout-secs", &timeout.to_string()]
            );
        }
    }
}
