use crate::command::DynResult;
use std::path::{Path, PathBuf};

const HF_CONVERTED_ARTIFACT_USAGE: &str =
    "usage: cargo xtool hf-converted-artifact preflight --artifact-dir <directory>";

pub(crate) fn print_usage() {
    println!("  cargo xtool automation local-ports COUNT");
    println!("  cargo xtool automation system-one-cases --help");
    println!("  cargo xtool automation system-one-smoke --help");
    println!("  cargo xtool automation binary-stage-readiness --help");
    println!("  cargo xtool automation workload-monolithic-oracle --help");
    println!("  cargo xtool automation workload-media-oracle --help");
    println!("  cargo xtool automation workload-tts-oracle --help");
    println!("  cargo xtool automation openai-smoke-config cache --help");
    println!(
        "  cargo xtool automation agent-fixture-inputs {{sha256 FILE | soak MODEL TARGET_CHARS OUTPUT | surface MODEL OUTPUT}}"
    );
    println!(
        "  cargo xtool automation agent-client-config {{pi BASE MODEL JSON | goose BASE MODEL PROVIDER_JSON CONFIG_YAML | opencode [BASE MODEL]}}"
    );
    println!(
        "  cargo xtool automation agent-fixture-evidence {{soak RESPONSE LABEL | result JSONL LABEL REQUIRE_TOOLS | probe RESPONSE LABEL | opencode-session JSONL | opencode-result JSONL}}"
    );
    println!("  cargo xtool automation family-model-identity MODEL_ID MODEL_PATH");
    println!("  cargo xtool automation family-model-identity --snapshot-revision PATH");
    println!(
        "  cargo xtool automation family-battery-policy ROOT MANIFEST PLAN SHARD_INDEX_OR_EMPTY"
    );
    println!(
        "  cargo xtool automation family-battery-policy --environment ARTIFACT_ROOT MODEL_ROOT MINIMUM_GIB OUTPUT"
    );
    println!(
        "  cargo xtool automation family-battery-policy --cache ROOT MANIFEST PLAN CACHE_ROOT"
    );
    println!("  cargo xtool automation family-battery-policy --inspect-gguf PATH");
    println!(
        "  cargo xtool automation openai-smoke-config --output PATH --model-id ID --model-path PATH --layer-end N --ctx-size N"
    );
    println!(
        "  cargo xtool automation workload-smoke-config --output PATH --model-id ID --model-path PATH --model-sha256 SHA --layer-end N --n-gpu-layers N [--projector-path PATH]"
    );
    println!("  cargo xtool prepared-input swift-api-checksum LIBRARY GENERATED_SWIFT");
    println!(
        "  cargo xtool automation smoke-inputs <product-root> <binary-name> <expected-backend>"
    );
    println!("  cargo xtool automation smoke-observation <verb> [model] < observation.json");
    println!(
        "  cargo xtool automation workload-smoke --base-url <url> --model <id> --class <class> [--media-path <path>]"
    );
    println!("  cargo xtool automation hf-xet-smoke <download-output> <isolated-cache-root>");
    println!(
        "  cargo xtool automation cache-family-report --input PATH... [--output PATH] [--use-case-corpus PATH]"
    );
    println!("  cargo xtool automation split-probe <verb> ...");
    println!(
        "{USAGE}\n  {HF_CONVERTED_ARTIFACT_USAGE}\n  {}\n  {}\n  {}\n  {}\n  {}\n  {}\n  {}\n  {}\n  {}\n  {}\n  {}\n  {}\n  cargo xtool ci family-plan ...",
        crate::automation::client_readiness::USAGE,
        crate::automation::stability::USAGE,
        crate::automation::daemon_readiness::USAGE,
        crate::automation::REPLAY_EXPORT_USAGE,
        crate::automation::REPLAY_RUN_FAMILY_USAGE,
        crate::automation::sdk_advisory::USAGE,
        crate::automation::rollout::USAGE,
        crate::automation::WORKLOAD_ORACLE_EVIDENCE_USAGE,
        crate::automation::canary_aggregate_command::USAGE,
        crate::automation::qualification::USAGE,
        crate::automation::required_smoke::USAGE,
        crate::automation::native_generator::USAGE
    );
    println!(
        "  cargo xtool repository cargo-packages --generation {{legacy|current}} --crates <JSON> [--batches <JSON>] {{--cargo <absolute-executable> [--timeout <seconds>] | --metadata <fixture-path>}}"
    );
    println!("  {}", crate::automation::split_evidence::USAGE);
    println!("  cargo xtool native package-source-version {{workspace|abi}} SOURCE (experimental)");
    println!("  cargo xtool repository cargo-target-directory < cargo-metadata.json");
    println!(
        "  cargo xtool repository publish-order [--dependency-pairs | --selected-script PATH] < cargo-metadata.json"
    );
    println!("  cargo xtool product attestation-status < inspection.json");
    println!("  cargo xtool product rc-ok {{model|request MODEL|verify}}");
    println!(
        "  cargo xtool native runtime-manifest-write MANIFEST ID VERSION ABI OS ARCH TARGET PLATFORM BACKEND CUDA_MAJOR PRIMARY UPSTREAM PATCHED PATCH_DIGEST LIBRARY... -- TOOL... -- LICENSE... -- RELOCATABLE..."
    );
    println!(
        "  cargo xtool prepared-input native-sdk-manifest-write MANIFEST ID VERSION TARGET PLATFORM OS ARCH BACKEND FLAVOR PROFILE LIBRARY UNIFFI UPSTREAM PATCHED PATCH_DIGEST"
    );
}

const NATIVE_USAGE: &str = "usage:\n  cargo xtool native select-runtime --root <dir> --os <os> --arch <arch> --backend <backend> [--cuda-major <major>]\n  cargo xtool native verify-host-dependencies <binary> [--format {elf|macho|pe}] [--report <path>] [--no-import-policy] [--max-glibc <version|declared>]\n  cargo xtool native linux-runtime-deps {collect|verify|order} --lib-dir <dir> [--scan-dir <dir>]... [--arch {x86_64|aarch64|arm}] [--search-dir <dir>]... [--cuda-major {12|13}] [--primary <name>]\n  cargo xtool native windows-runtime-deps {collect|verify} --lib-dir <dir> [--scan-dir <dir>]... [--search-dir <dir>]...\n  cargo xtool native release-matrix --manifest <json> [--required-target <os>/<arch>/<backend>]... [<artifact>...]\n  cargo xtool native verify-runtime-package [--portable] <artifact>...\n  cargo xtool native package-source-version {workspace|abi|runtime} SOURCE (experimental)";

const USAGE: &str = "usage:\n  cargo xtool artifact verify-checksum <artifact>\n  cargo xtool artifact extract-tar <archive> <destination>\n  cargo xtool artifact extract-zip <archive> <destination>\n  cargo xtool automation {inventory|policy} --check\n  cargo xtool automation bootstrap\n  cargo xtool automation ui-build --ui-dir PATH [--logs-dir PATH] [--timeout-secs 1..3600] [--pnpm-command PATH [--pnpm-script PATH]]\n  cargo xtool automation parity --suite ci [--evidence <dir>] [--interpreter <path> | --rust-only]\n  cargo xtool automation replay-matrix validate --matrix <path>\n  cargo xtool ci plan [--manifest-root <path>] < plan-input.json\n  cargo xtool ci validate-lane --lane-plan <json> --needs <json> [--workflow <lane.yml>] [--plan-digest <sha256> --canonical-plan <json>]\n  cargo xtool ci validate-graph --workflows <dir>\n  cargo xtool ci-ops runner-identity [--root <path>] [--catalog <path>] {validate|check|diagnose|lookup|seed-key|bind} ...\n  cargo xtool ci-ops build-cache {status|prune|build} [--workspace <path>] [--target-dir <path>] [--max-size <size>] [--max-age <days>] [--json] [--execute] [-- <build command>...]\n  cargo xtool ci-ops sccache-stats --artifact-name <name> --output <path> [--github-output <path>] [--cache-expectation {cold|warm|opportunistic}] [--minimum-hit-rate <float>]\n  cargo xtool ci-ops sccache-summary [--format text|json] [--minimum-hit-rate <0..1>] <evidence-path>...\n  cargo xtool ci-ops performance-history --artifact <dir> --output <jsonl> --report <md> [--baseline <path>] [--gate]\n  cargo xtool ci-ops collect-metrics --input <path|-> [--json-out <path|->] [--status <status>] [--top <n>] [--label KEY=VALUE]...\n  cargo xtool models generate [--registry <path>] [--check]\n  cargo xtool models resolve <manifest> --cadence <cadence> [--artifact-id <id>] [--require-single-file] [--github-output <path> [--github-output-prefix <name_>]] [--verify-root <dir>]\n  cargo xtool models restore-inputs --github-output <path> [--model-url <url> --model-file <name> | --model-manifest <path> --model-cadence <cadence> [--model-artifact-id <id>]]\n  cargo xtool models parity-download --manifest <path> --model-manifest <path> --cadence manual --hf-command <absolute-path> [--dry-run] [--status <csv>] [--priority <csv>] [--timeout-secs <1..86400>]\n  cargo xtool prepared-input <consumer> ... (UI, static ABI, native SDK inputs)\n  cargo xtool release inventory [--repo <owner/name>] [--head <ref>] [--release-tag <tag>] [--output <json>]\n  cargo xtool release notes-base <target-tag> < tags.txt\n  cargo xtool release notes-link --body <md> --range <a..b> --repo <owner/name> --out-body <md> --out-links <json> [--repo-root <dir>] [--api-budget <n>]\n  cargo xtool release notes-classify --body <md> (--has-entries | --range <a..b> --version <v> --date <d> --out <json>) [--repo-root <dir>] [--links <json>]\n  cargo xtool repo-consistency release-targets\n  cargo xtool repo-consistency ci-crate-lists\n  cargo xtool repo-consistency publish-crates\n  cargo xtool repo-consistency test-all-rust-crate-coverage\n  cargo xtool repo-consistency no-console-print\n  cargo xtool repository affected-crates [--stdin | <path>...]\n  cargo xtool repository conventional-commits (--message <subject> | --range <range> | <file>) [--trailers-only]\n  cargo xtool repository env-mutation-census [--root <path>] [--file <path>]...\n  cargo xtool repository llama-upstream-pin [--repository <path>] [--upstream-url <url>] <base-sha> <head-sha>\n  cargo xtool repository selected-ref --ref <branch-or-sha> --expected-origin <url> --event workflow_dispatch [--repository <path>] [--upstream <sha>] [--github-output <path> --summary <path>] [--timeout-secs <seconds>]\n  cargo xtool release-attestation generate-keypair --private-key-out <path> --public-key-out <path>\n  cargo xtool release-attestation stamp --binary <path> --signing-key-file <path> [--node-version <semver>] [--build-id <id>] [--commit <sha>] [--target-triple <triple>] [--protocol-min <n>] [--protocol-max <n>]\n  cargo xtool release-attestation inspect --binary <path> [--public-key-file <path>] [--json]\n  (cargo run -p xtask -- <domain> <command> ... remains supported)";

pub(crate) struct Cli<'a> {
    pub(crate) root: Option<PathBuf>,
    pub(crate) command: CliCommand<'a>,
}

pub(crate) enum CliCommand<'a> {
    AgentClientConfig(&'a [String]),
    CacheFamilyReport(&'a [String]),
    AgentFixtureEvidence(&'a [String]),
    AgentFixtureInputs(&'a [String]),
    FamilyBatteryPolicy(&'a [String]),
    FamilyModelIdentity(&'a [String]),
    LocalPorts(&'a [String]),
    OpenaiSmokeConfig(&'a [String]),
    WorkloadSmokeConfig(&'a [String]),
    SmokeInputs(&'a [String]),
    WorkloadSmoke(&'a [String]),
    Stability(&'a [String]),
    SystemOneCases(&'a [String]),
    SystemOneSmoke(&'a [String]),
    BinaryStageReadiness(&'a [String]),
    WorkloadMonolithicOracle(&'a [String]),
    WorkloadMediaOracle(&'a [String]),
    WorkloadTtsOracle(&'a [String]),
    HfXetSmoke(&'a [String]),
    SmokeObservation(&'a [String]),
    SplitProbe(&'a [String]),
    RuntimeCacheInstall(&'a [String]),
    SdkFixture(&'a [String]),
    LoggingConsole(&'a [String]),
    UiBuild(&'a [String]),
    StartupRecovery(&'a [String]),
    DaemonLifecycle(&'a [String]),
    LoggingRecovery(&'a [String]),
    ControlPlaneQa(&'a [String]),
    HfConvertedArtifact(&'a [String]),
    Rollout(&'a [String]),
    Repository(RepositoryCommand<'a>),
    GenerateKeypair(&'a [String]),
    Stamp(&'a [String]),
    Inspect(&'a [String]),
    Check(RepositoryCheck, &'a [String]),
    CiPlan(&'a [String]),
    CiFamilyPlan(&'a [String]),
    NativeGenerator(&'a [String]),
    SplitEvidence(&'a [String]),
    CiValidate(&'a str, &'a [String]),
    Qualification(&'a str, &'a [String]),
    ReplayMatrix(&'a [String]),
    Laya(&'a [String]),
    AgentPickModel(&'a [String]),
    WorkloadOracleEvidence(&'a [String]),
    CanaryTimeout(&'a [String]),
    CanaryReceipts(&'a [String]),
    RewriterReport(&'a [String]),
    Models(crate::model_registry::ModelsCommand, &'a [String]),
    Artifact(crate::artifact::ArtifactCommand, &'a [String]),
    CiOperations(crate::ci_operations::CiOperationsCommand, &'a [String]),
    Native(crate::native_policy::NativeCommand, &'a [String]),
    Product(crate::product::ProductCommand, &'a [String]),
    PreparedInput(&'a [String]),
    Release(crate::release::ReleaseCommand, &'a [String]),
}

/// Ported repository checks. They take paths from their own arguments or the
/// working directory, like the scripts they replace, so they work in fixture
/// checkouts that lack xtask's workspace markers.
#[derive(Clone, Copy)]
pub(crate) enum RepositoryCheck {
    AffectedCrates,
    CargoPackages,
    CargoTargetDirectory,
    PublishOrder,
    ConventionalCommits,
    EnvMutationCensus,
    LlamaUpstreamPin,
    SelectedRef,
}

pub(crate) enum RepositoryCommand<'a> {
    SdkAdvisory(&'a [String]),
    ClientReadiness(&'a [String]),
    DaemonReadiness(&'a [String]),
    RequiredSmoke(&'a [String]),
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
            [domain, scope, rest @ ..]
                if domain == "automation" && scope == "agent-fixture-inputs" =>
            {
                CliCommand::AgentFixtureInputs(rest)
            }
            [domain, scope, rest @ ..]
                if domain == "automation" && scope == "agent-fixture-evidence" =>
            {
                CliCommand::AgentFixtureEvidence(rest)
            }
            [domain, scope, rest @ ..]
                if domain == "automation" && scope == "agent-client-config" =>
            {
                CliCommand::AgentClientConfig(rest)
            }
            [domain, scope, rest @ ..]
                if domain == "automation" && scope == "cache-family-report" =>
            {
                CliCommand::CacheFamilyReport(rest)
            }
            [domain, scope, rest @ ..]
                if domain == "automation" && scope == "family-battery-policy" =>
            {
                CliCommand::FamilyBatteryPolicy(rest)
            }
            [domain, scope, rest @ ..]
                if domain == "automation" && scope == "family-model-identity" =>
            {
                CliCommand::FamilyModelIdentity(rest)
            }
            [domain, scope, rest @ ..] if domain == "automation" && scope == "local-ports" => {
                CliCommand::LocalPorts(rest)
            }
            [domain, scope, rest @ ..]
                if domain == "automation" && scope == "openai-smoke-config" =>
            {
                CliCommand::OpenaiSmokeConfig(rest)
            }
            [domain, scope, rest @ ..]
                if domain == "automation" && scope == "workload-smoke-config" =>
            {
                CliCommand::WorkloadSmokeConfig(rest)
            }
            [domain, scope, rest @ ..] if domain == "automation" && scope == "laya" => {
                CliCommand::Laya(rest)
            }
            [domain, scope, rest @ ..] if domain == "automation" && scope == "stability" => {
                CliCommand::Stability(rest)
            }
            [domain, scope, rest @ ..] if domain == "automation" && scope == "split-probe" => {
                CliCommand::SplitProbe(rest)
            }
            [domain, scope, rest @ ..]
                if domain == "automation" && scope == "smoke-observation" =>
            {
                CliCommand::SmokeObservation(rest)
            }
            [domain, scope, rest @ ..] if domain == "automation" && scope == "hf-xet-smoke" => {
                CliCommand::HfXetSmoke(rest)
            }
            [domain, scope, rest @ ..] if domain == "automation" && scope == "system-one-smoke" => {
                CliCommand::SystemOneSmoke(rest)
            }
            [domain, scope, rest @ ..] if domain == "automation" && scope == "system-one-cases" => {
                CliCommand::SystemOneCases(rest)
            }
            [domain, scope, rest @ ..]
                if domain == "automation" && scope == "binary-stage-readiness" =>
            {
                CliCommand::BinaryStageReadiness(rest)
            }
            [domain, scope, rest @ ..]
                if domain == "automation" && scope == "workload-monolithic-oracle" =>
            {
                CliCommand::WorkloadMonolithicOracle(rest)
            }
            [domain, scope, rest @ ..]
                if domain == "automation" && scope == "workload-media-oracle" =>
            {
                CliCommand::WorkloadMediaOracle(rest)
            }
            [domain, scope, rest @ ..]
                if domain == "automation" && scope == "workload-tts-oracle" =>
            {
                CliCommand::WorkloadTtsOracle(rest)
            }
            [domain, scope, rest @ ..] if domain == "automation" && scope == "workload-smoke" => {
                CliCommand::WorkloadSmoke(rest)
            }
            [domain, scope, rest @ ..] if domain == "automation" && scope == "smoke-inputs" => {
                CliCommand::SmokeInputs(rest)
            }
            [domain, scope, rest @ ..] if domain == "automation" && scope == "control-plane-qa" => {
                CliCommand::ControlPlaneQa(rest)
            }
            [domain, scope, rest @ ..] if domain == "automation" && scope == "logging-recovery" => {
                CliCommand::LoggingRecovery(rest)
            }
            [domain, scope, rest @ ..] if domain == "automation" && scope == "daemon-lifecycle" => {
                CliCommand::DaemonLifecycle(rest)
            }
            [domain, scope, rest @ ..] if domain == "automation" && scope == "startup-recovery" => {
                CliCommand::StartupRecovery(rest)
            }
            [domain, scope, rest @ ..] if domain == "automation" && scope == "ui-build" => {
                CliCommand::UiBuild(rest)
            }
            [domain, scope, rest @ ..] if domain == "automation" && scope == "logging-console" => {
                CliCommand::LoggingConsole(rest)
            }
            [domain, scope, rest @ ..] if domain == "automation" && scope == "sdk-fixture" => {
                CliCommand::SdkFixture(rest)
            }
            [domain, scope, rest @ ..]
                if domain == "automation" && scope == "runtime-cache-install" =>
            {
                CliCommand::RuntimeCacheInstall(rest)
            }
            [domain, scope, rest @ ..] if domain == "automation" && scope == "agent-pick-model" => {
                CliCommand::AgentPickModel(rest)
            }
            [domain, scope, verb, rest @ ..]
                if domain == "automation" && scope == "required-smoke" && verb == "run" =>
            {
                CliCommand::Repository(RepositoryCommand::RequiredSmoke(rest))
            }
            [domain, scope, rest @ ..] if domain == "automation" && scope == "native-generator" => {
                CliCommand::NativeGenerator(rest)
            }
            [domain, scope, rest @ ..] if domain == "ci" && scope == "family-plan" => {
                CliCommand::CiFamilyPlan(rest)
            }
            [domain, scope, rest @ ..]
                if domain == "hf-converted-artifact" && scope == "preflight" =>
            {
                CliCommand::HfConvertedArtifact(rest)
            }
            [domain, ..] if domain == "hf-converted-artifact" => {
                return Err(HF_CONVERTED_ARTIFACT_USAGE.into());
            }
            [domain, scope, rest @ ..] if domain == "automation" && scope == "client-readiness" => {
                CliCommand::Repository(RepositoryCommand::ClientReadiness(rest))
            }
            [domain, scope, rest @ ..] if domain == "automation" && scope == "daemon-readiness" => {
                CliCommand::Repository(RepositoryCommand::DaemonReadiness(rest))
            }
            [domain, scope, rest @ ..] if domain == "automation" && scope == "rewriter-report" => {
                CliCommand::RewriterReport(rest)
            }
            [domain, scope, rest @ ..] if domain == "automation" && scope == "replay-matrix" => {
                CliCommand::ReplayMatrix(rest)
            }
            [domain, scope, rest @ ..]
                if domain == "automation" && scope == "workload-oracle-evidence" =>
            {
                CliCommand::WorkloadOracleEvidence(rest)
            }
            [domain, scope, rest @ ..] if domain == "automation" && scope == "canary-timeout" => {
                CliCommand::CanaryTimeout(rest)
            }
            [domain, scope, rest @ ..] if domain == "automation" && scope == "canary-receipts" => {
                CliCommand::CanaryReceipts(rest)
            }
            [domain, scope, rest @ ..] if domain == "automation" && scope == "bootstrap" => {
                CliCommand::Repository(RepositoryCommand::AutomationBootstrap(rest))
            }
            [domain, scope, rest @ ..] if domain == "automation" && scope == "sdk-advisory" => {
                CliCommand::Repository(RepositoryCommand::SdkAdvisory(rest))
            }
            [domain, scope, rest @ ..] if domain == "automation" && scope == "rollout" => {
                CliCommand::Rollout(rest)
            }
            [domain, verb, rest @ ..]
                if domain == "automation"
                    && matches!(verb.as_str(), "qualify" | "qualify-replay") =>
            {
                CliCommand::Qualification(verb, rest)
            }
            [domain, scope, rest @ ..] if domain == "automation" && scope == "split-evidence" => {
                CliCommand::SplitEvidence(rest)
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
            [domain, scope, rest @ ..] if domain == "native" => {
                match crate::native_policy::NativeCommand::parse(scope) {
                    Some(command) => CliCommand::Native(command, rest),
                    None => return Err(NATIVE_USAGE.into()),
                }
            }
            [domain, scope, rest @ ..] if domain == "product" => {
                match crate::product::ProductCommand::parse(scope) {
                    Some(command) => CliCommand::Product(command, rest),
                    None => return Err(USAGE.into()),
                }
            }
            [domain, scope, rest @ ..] if domain == "release" => {
                match crate::release::ReleaseCommand::parse(scope) {
                    Some(command) => CliCommand::Release(command, rest),
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
                    "cargo-packages" => RepositoryCheck::CargoPackages,
                    "cargo-target-directory" => RepositoryCheck::CargoTargetDirectory,
                    "publish-order" => RepositoryCheck::PublishOrder,
                    "conventional-commits" => RepositoryCheck::ConventionalCommits,
                    "env-mutation-census" => RepositoryCheck::EnvMutationCensus,
                    "llama-upstream-pin" => RepositoryCheck::LlamaUpstreamPin,
                    "selected-ref" => RepositoryCheck::SelectedRef,
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
            _ => {
                return Err(format!(
                    "{USAGE}\n  {HF_CONVERTED_ARTIFACT_USAGE}\n  {}\n  {}\n  {}\n  {}",
                    crate::automation::REPLAY_EXPORT_USAGE,
                    crate::automation::sdk_advisory::USAGE,
                    crate::automation::rollout::USAGE,
                    crate::automation::WORKLOAD_ORACLE_EVIDENCE_USAGE
                )
                .into());
            }
        };
        Ok(Self { root, command })
    }
}
