mod attestation;
mod command;
mod installer_fixtures;
mod publish_consistency;
mod release_targets;
mod repo_consistency;
mod workflow_checks;

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
    match args.as_slice() {
        [command, scope] if command == "repo-consistency" && scope == "release-targets" => {
            repo_consistency::check_release_targets_command()
        }
        [command, scope] if command == "repo-consistency" && scope == "ci-crate-lists" => {
            repo_consistency::check_ci_crate_lists_command()
        }
        [command, scope] if command == "repo-consistency" && scope == "publish-crates" => {
            repo_consistency::check_publish_crates_command()
        }
        [command, scope, rest @ ..]
            if command == "release-attestation" && scope == "generate-keypair" =>
        {
            attestation::generate_release_attestation_keypair(rest)
        }
        [command, scope, rest @ ..] if command == "release-attestation" && scope == "stamp" => {
            attestation::stamp_release_attestation(rest)
        }
        [command, scope, rest @ ..]
            if command == "release-attestation" && scope == "inspect" =>
        {
            attestation::inspect_release_attestation(rest)
        }
        _ => Err(
            "usage:\n  cargo run -p xtask -- repo-consistency release-targets\n  cargo run -p xtask -- repo-consistency ci-crate-lists\n  cargo run -p xtask -- repo-consistency publish-crates\n  cargo run -p xtask -- release-attestation generate-keypair --private-key-out <path> --public-key-out <path>\n  cargo run -p xtask -- release-attestation stamp --binary <path> --signing-key-file <path> [--node-version <semver>] [--build-id <id>] [--commit <sha>] [--target-triple <triple>] [--protocol-min <n>] [--protocol-max <n>]\n  cargo run -p xtask -- release-attestation inspect --binary <path> [--public-key-file <path>] [--json]"
                .to_string()
                .into(),
        ),
    }
}
