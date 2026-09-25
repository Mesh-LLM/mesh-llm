use crate::command::DynResult;
use std::path::{Path, PathBuf};

const USAGE: &str = "usage:\n  cargo xtool artifact verify-checksum <artifact>\n  cargo xtool artifact extract-tar <archive> <destination>\n  cargo xtool artifact extract-zip <archive> <destination>\n  cargo xtool automation {inventory|policy} --check\n  cargo xtool automation bootstrap\n  cargo xtool automation parity --suite ci [--evidence <dir>] [--interpreter <path> | --rust-only]\n  cargo xtool ci plan [--manifest-root <path>] < plan-input.json\n  cargo xtool ci validate-lane --lane-plan <json> --needs <json> [--workflow <lane.yml>] [--plan-digest <sha256> --canonical-plan <json>]\n  cargo xtool ci validate-graph --workflows <dir>\n  cargo xtool ci-ops runner-identity [--root <path>] [--catalog <path>] {validate|check|diagnose|lookup|seed-key|bind} ...\n  cargo xtool ci-ops build-cache {status|prune|build} [--workspace <path>] [--target-dir <path>] [--max-size <size>] [--max-age <days>] [--json] [--execute] [-- <build command>...]\n  cargo xtool ci-ops sccache-stats --artifact-name <name> --output <path> [--github-output <path>] [--cache-expectation {cold|warm|opportunistic}] [--minimum-hit-rate <float>]\n  cargo xtool ci-ops collect-metrics --input <path|-> [--json-out <path|->] [--status <status>] [--top <n>] [--label KEY=VALUE]...\n  cargo xtool models generate [--registry <path>] [--check]\n  cargo xtool models resolve <manifest> --cadence <cadence> [--artifact-id <id>] [--require-single-file] [--github-output <path> [--github-output-prefix <name_>]] [--verify-root <dir>]\n  cargo xtool models restore-inputs --github-output <path> [--model-url <url> --model-file <name> | --model-manifest <path> --model-cadence <cadence> [--model-artifact-id <id>]]\n  cargo xtool prepared-input <consumer> ... (UI, static ABI, native SDK inputs)\n  cargo xtool repo-consistency release-targets\n  cargo xtool repo-consistency ci-crate-lists\n  cargo xtool repo-consistency publish-crates\n  cargo xtool repo-consistency test-all-rust-crate-coverage\n  cargo xtool repo-consistency no-console-print\n  cargo xtool repository affected-crates [--stdin | <path>...]\n  cargo xtool repository conventional-commits (--message <subject> | --range <range> | <file>) [--trailers-only]\n  cargo xtool repository env-mutation-census [--root <path>] [--file <path>]...\n  cargo xtool repository llama-upstream-pin [--repository <path>] [--upstream-url <url>] <base-sha> <head-sha>\n  cargo xtool release-attestation generate-keypair --private-key-out <path> --public-key-out <path>\n  cargo xtool release-attestation stamp --binary <path> --signing-key-file <path> [--node-version <semver>] [--build-id <id>] [--commit <sha>] [--target-triple <triple>] [--protocol-min <n>] [--protocol-max <n>]\n  cargo xtool release-attestation inspect --binary <path> [--public-key-file <path>] [--json]\n  (cargo run -p xtask -- <domain> <command> ... remains supported)";

pub(crate) struct Cli<'a> {
    pub(crate) root: Option<PathBuf>,
    pub(crate) command: CliCommand<'a>,
}

pub(crate) enum CliCommand<'a> {
    Repository(RepositoryCommand<'a>),
    GenerateKeypair(&'a [String]),
    Stamp(&'a [String]),
    Inspect(&'a [String]),
    Check(RepositoryCheck, &'a [String]),
    CiPlan(&'a [String]),
    CiValidate(&'a str, &'a [String]),
    AutomationParity(&'a [String]),
    Models(crate::model_registry::ModelsCommand, &'a [String]),
    Artifact(crate::artifact::ArtifactCommand, &'a [String]),
    CiOperations(crate::ci_operations::CiOperationsCommand, &'a [String]),
    PreparedInput(&'a [String]),
}

/// Ported repository checks. They take paths from their own arguments or the
/// working directory, like the scripts they replace, so they work in fixture
/// checkouts that lack xtask's workspace markers.
#[derive(Clone, Copy)]
pub(crate) enum RepositoryCheck {
    AffectedCrates,
    ConventionalCommits,
    EnvMutationCensus,
    LlamaUpstreamPin,
}

pub(crate) enum RepositoryCommand<'a> {
    Automation(&'a [String]),
    AutomationBootstrap(&'a [String]),
    ReleaseTargets,
    CiCrateLists,
    PublishCrates,
    TestAllCoverage,
    NoConsolePrint(&'a [String]),
}

impl<'a> Cli<'a> {
    pub(crate) fn parse(args: &'a [String]) -> DynResult<Self> {
        let (root, command_args) = match args {
            [flag, value, rest @ ..] if flag == "--repo-root" => {
                if value.starts_with("--") {
                    return Err("missing value for --repo-root".into());
                }
                (Some(Path::new(value).to_path_buf()), rest)
            }
            [flag, ..] if flag == "--repo-root" => {
                return Err("missing value for --repo-root".into());
            }
            _ => (None, args),
        };
        let command = match command_args {
            [domain, scope, rest @ ..] if domain == "automation" && scope == "parity" => {
                CliCommand::AutomationParity(rest)
            }
            [domain, scope, rest @ ..] if domain == "automation" && scope == "bootstrap" => {
                CliCommand::Repository(RepositoryCommand::AutomationBootstrap(rest))
            }
            [domain, rest @ ..] if domain == "automation" => {
                CliCommand::Repository(RepositoryCommand::Automation(rest))
            }
            [domain, scope] if domain == "repo-consistency" && scope == "release-targets" => {
                CliCommand::Repository(RepositoryCommand::ReleaseTargets)
            }
            [domain, scope] if domain == "repo-consistency" && scope == "ci-crate-lists" => {
                CliCommand::Repository(RepositoryCommand::CiCrateLists)
            }
            [domain, scope] if domain == "repo-consistency" && scope == "publish-crates" => {
                CliCommand::Repository(RepositoryCommand::PublishCrates)
            }
            [domain, scope]
                if domain == "repo-consistency" && scope == "test-all-rust-crate-coverage" =>
            {
                CliCommand::Repository(RepositoryCommand::TestAllCoverage)
            }
            [domain, scope, rest @ ..]
                if domain == "repo-consistency" && scope == "no-console-print" =>
            {
                CliCommand::Repository(RepositoryCommand::NoConsolePrint(rest))
            }
            [domain, scope, rest @ ..] if domain == "ci" && scope == "plan" => {
                CliCommand::CiPlan(rest)
            }
            [domain, scope, rest @ ..]
                if domain == "ci" && (scope == "validate-lane" || scope == "validate-graph") =>
            {
                CliCommand::CiValidate(scope, rest)
            }
            [domain, rest @ ..] if domain == "prepared-input" => CliCommand::PreparedInput(rest),
            [domain, scope, rest @ ..] if domain == "artifact" => {
                match crate::artifact::ArtifactCommand::parse(scope) {
                    Some(command) => CliCommand::Artifact(command, rest),
                    None => return Err(USAGE.into()),
                }
            }
            [domain, scope, rest @ ..] if domain == "models" => {
                match crate::model_registry::ModelsCommand::parse(scope) {
                    Some(command) => CliCommand::Models(command, rest),
                    None => return Err(USAGE.into()),
                }
            }
            [domain, scope, rest @ ..] if domain == "ci-ops" => {
                match crate::ci_operations::CiOperationsCommand::parse(scope) {
                    Some(command) => CliCommand::CiOperations(command, rest),
                    None => return Err(USAGE.into()),
                }
            }
            [domain, scope, rest @ ..] if domain == "repository" => {
                let check = match scope.as_str() {
                    "affected-crates" => RepositoryCheck::AffectedCrates,
                    "conventional-commits" => RepositoryCheck::ConventionalCommits,
                    "env-mutation-census" => RepositoryCheck::EnvMutationCensus,
                    "llama-upstream-pin" => RepositoryCheck::LlamaUpstreamPin,
                    _ => return Err(USAGE.into()),
                };
                CliCommand::Check(check, rest)
            }
            [domain, scope, rest @ ..]
                if domain == "release-attestation" && scope == "generate-keypair" =>
            {
                CliCommand::GenerateKeypair(rest)
            }
            [domain, scope, rest @ ..] if domain == "release-attestation" && scope == "stamp" => {
                CliCommand::Stamp(rest)
            }
            [domain, scope, rest @ ..] if domain == "release-attestation" && scope == "inspect" => {
                CliCommand::Inspect(rest)
            }
            _ => return Err(USAGE.into()),
        };
        Ok(Self { root, command })
    }
}
