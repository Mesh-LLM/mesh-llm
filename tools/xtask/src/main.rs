mod artifact;
mod attestation;
mod automation_bootstrap;
mod automation_parity;
mod ci_operations;
mod ci_plan;
mod ci_validation;
mod cli;
mod command;
mod installer_fixtures;
mod migration_inventory;
mod model_registry;
mod no_console_print;
mod prepared_input;
mod publish_consistency;
mod release_targets;
mod repo_consistency;
mod repository;

use command::DynResult;

#[cfg(test)]
mod tests;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> DynResult<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let parsed = cli::Cli::parse(&args)?;
    let explicit_root = parsed
        .root
        .as_deref()
        .map(|path| repository::RepositoryRoot::resolve(Some(path)))
        .transpose()?;
    match parsed.command {
        cli::CliCommand::GenerateKeypair(rest) => {
            attestation::generate_release_attestation_keypair(rest)
        }
        cli::CliCommand::Inspect(rest) => attestation::inspect_release_attestation(rest),
        cli::CliCommand::Check(check, rest) => repository::run_check(check, rest, explicit_root),
        cli::CliCommand::CiPlan(rest) => {
            let root = match explicit_root {
                Some(root) => root,
                None => repository::RepositoryRoot::resolve(None)?,
            };
            ci_plan::run(root.as_path(), rest)
        }
        cli::CliCommand::CiValidate(verb, rest) => ci_validation::lane_results::run(verb, rest),
        cli::CliCommand::AutomationParity(rest) => {
            let root = match explicit_root {
                Some(root) => root,
                None => repository::RepositoryRoot::resolve(None)?,
            };
            automation_parity::run(root.as_path(), rest)
        }
        cli::CliCommand::PreparedInput(rest) => prepared_input::run(rest),
        cli::CliCommand::Artifact(command, rest) => artifact::run(command, rest),
        cli::CliCommand::Models(command, rest) => model_registry::run(command, rest, || {
            let root = match explicit_root {
                Some(root) => root,
                None => repository::RepositoryRoot::resolve(None)?,
            };
            Ok(root.as_path().to_path_buf())
        }),
        cli::CliCommand::CiOperations(command, rest) => ci_operations::run(command, rest, || {
            let root = match explicit_root {
                Some(root) => root,
                None => repository::RepositoryRoot::resolve(None)?,
            };
            Ok(root.as_path().to_path_buf())
        }),
        cli::CliCommand::Stamp(rest) => attestation::stamp_release_attestation(
            rest,
            explicit_root
                .as_ref()
                .map(repository::RepositoryRoot::as_path),
        ),
        cli::CliCommand::Repository(command) => {
            let root = match explicit_root {
                Some(root) => root,
                None => repository::RepositoryRoot::resolve(None)?,
            };
            let root = root.as_path();
            match command {
                cli::RepositoryCommand::Automation(rest) => {
                    std::env::set_current_dir(root)?;
                    migration_inventory::run(rest)
                }
                cli::RepositoryCommand::AutomationBootstrap(rest) => {
                    automation_bootstrap::run(root, rest)
                }
                cli::RepositoryCommand::ReleaseTargets => {
                    repo_consistency::check_release_targets_command(root)
                }
                cli::RepositoryCommand::CiCrateLists => {
                    repo_consistency::check_ci_crate_lists_command(root)
                }
                cli::RepositoryCommand::PublishCrates => {
                    repo_consistency::check_publish_crates_command(root)
                }
                cli::RepositoryCommand::TestAllCoverage => {
                    repo_consistency::check_test_all_coverage_command(root)
                }
                cli::RepositoryCommand::NoConsolePrint(rest) => {
                    no_console_print::check_no_console_print_command(root, rest)
                }
            }
        }
    }
}
