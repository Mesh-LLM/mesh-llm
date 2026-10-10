mod artifact;
mod attestation;
mod automation;
mod automation_bootstrap;
mod ci_operations;
mod ci_plan;
mod ci_validation;
mod cli;
mod cli_output;
mod command;
#[path = "automation/command_interrupt/mod.rs"]
pub(crate) mod command_interrupt;
mod installer_fixtures;
mod model_registry;
mod native_policy;
mod no_console_print;
mod prepared_input;
pub mod process;
mod product;
mod publish_consistency;
mod release;
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
    if args == ["--help"] {
        cli::print_usage();
        return Ok(());
    }
    let parsed = cli::Cli::parse(&args)?;
    let explicit_root = parsed
        .root
        .as_deref()
        .map(|path| repository::RepositoryRoot::resolve(Some(path)))
        .transpose()?;
    match parsed.command {
        cli::CliCommand::EndpointModelDiscovery(rest) => {
            automation::endpoint_model_discovery::run(rest)
        }
        cli::CliCommand::GuardrailCorpus(rest) => automation::guardrail_corpus::run(rest),
        cli::CliCommand::SuffixProposer(rest) => automation::suffix_proposer::run(rest),
        cli::CliCommand::EventBenchmarkRun(rest) => automation::event_benchmark_runner::run(rest),
        cli::CliCommand::AgentClientConfig(rest) => automation::agent_client_config::run(rest),
        cli::CliCommand::CacheFamilyMoe(rest) => automation::cache_family_moe::run(rest),
        cli::CliCommand::CacheFamilyRun(rest) => automation::cache_family_run::run(rest),
        cli::CliCommand::CacheFamilyReport(rest) => automation::cache_family_report::run(rest),
        cli::CliCommand::CacheFamilyCorrectness(rest) => {
            automation::cache_family_correctness::run(rest)
        }
        cli::CliCommand::CacheFamilyCell(rest) => automation::cache_family_cell::run(rest),
        cli::CliCommand::CacheFamilyMeasure(rest) => automation::cache_family_measure::run(rest),
        cli::CliCommand::CacheFamilyPlan(rest) => automation::cache_family_plan::run(rest),
        cli::CliCommand::AgentFixtureEvidence(rest) => {
            automation::agent_fixture_evidence::run(rest)
        }
        cli::CliCommand::AgentFixtureInputs(rest) => automation::agent_fixture_inputs::run(rest),
        cli::CliCommand::AgentRecordingProxy(rest) => automation::agent_recording_proxy::run(rest),
        cli::CliCommand::EventBenchmarkComparison(rest) => {
            automation::event_benchmark_comparison::run(rest)
        }
        cli::CliCommand::AgenticPromptManifest(rest) => {
            automation::agentic_prompt_manifest::run(rest)
        }
        cli::CliCommand::NativeRuntimeEvidence(rest) => {
            automation::native_runtime_evidence::run(rest)
        }
        cli::CliCommand::FamilyBatteryPolicy(rest) => automation::family_battery_policy::run(rest),
        cli::CliCommand::FamilyModelIdentity(rest) => automation::family_model_identity::run(rest),
        cli::CliCommand::LocalPorts(rest) => automation::local_ports::run(rest),
        cli::CliCommand::ManualSmoke(rest) => automation::manual_smoke::run(rest),
        cli::CliCommand::OpenaiCacheMatrix(rest) => automation::cache_matrix::run(rest),
        cli::CliCommand::OpenaiSmokeConfig(rest) => automation::openai_smoke_config::run(rest),
        cli::CliCommand::WorkloadSmokeConfig(rest) => automation::workload_smoke_config::run(rest),
        cli::CliCommand::SplitProbe(rest) => automation::split_probe::run(rest),
        cli::CliCommand::SmokeObservation(rest) => automation::smoke_observation::run(rest),
        cli::CliCommand::HfXetSmoke(rest) => automation::hf_xet_smoke::run(rest),
        cli::CliCommand::WorkloadSmoke(rest) => automation::workload_smoke::run(rest),
        cli::CliCommand::Stability(rest) => automation::stability::run(
            explicit_root
                .as_ref()
                .map(repository::RepositoryRoot::as_path),
            rest,
        ),
        cli::CliCommand::WanObservation(rest) => automation::wan_observation::run(rest),
        cli::CliCommand::WanStageDeployment(rest) => automation::wan_stage_deployment::run(rest),
        cli::CliCommand::SystemOneCases(rest) => automation::system_one_cases::run(rest),
        cli::CliCommand::SystemOneSmoke(rest) => automation::system_one_smoke::run(rest),
        cli::CliCommand::DecisionsSmoke(rest) => automation::decisions_smoke::run(rest),
        cli::CliCommand::BinaryStageReadiness(rest) => {
            automation::binary_stage_readiness::run(rest)
        }
        cli::CliCommand::WorkloadMonolithicOracle(rest) => {
            automation::workload_smoke::comparison::run(rest)
        }
        cli::CliCommand::WorkloadMediaOracle(rest) => {
            automation::workload_smoke::media_comparison::run(rest)
        }
        cli::CliCommand::WorkloadTtsOracle(rest) => {
            automation::workload_smoke::tts_oracle::run(rest)
        }

        cli::CliCommand::SmokeInputs(rest) => automation::smoke_inputs::run(rest),
        cli::CliCommand::RemoteHandoffSummary(rest) => {
            automation::remote_handoff_summary::run(rest)
        }
        cli::CliCommand::LightningCompatibility(rest) => {
            automation::lightning_compatibility::run(rest)
        }
        cli::CliCommand::ControlPlaneQa(rest) => {
            let root = repository::RepositoryRoot::resolve(None)?;
            automation::control_plane_qa::run(root.as_path(), rest)
        }
        cli::CliCommand::LoggingRecovery(rest) => {
            let root = repository::RepositoryRoot::resolve(None)?;
            automation::logging_recovery::run(root.as_path(), rest)
        }
        cli::CliCommand::DaemonLifecycle(rest) => {
            let root = repository::RepositoryRoot::resolve(None)?;
            automation::daemon_lifecycle::run(root.as_path(), rest)
        }
        cli::CliCommand::StartupRecovery(rest) => {
            let root = repository::RepositoryRoot::resolve(None)?;
            automation::startup_recovery::run(root.as_path(), rest)
        }
        cli::CliCommand::UiBuild(rest) => automation::ui_build::run(rest),
        cli::CliCommand::LoggingConsole(rest) => {
            let root = repository::RepositoryRoot::resolve(None)?;
            automation::logging_console::run(root.as_path(), rest)
        }
        #[cfg(unix)]
        cli::CliCommand::SdkCompat(rest) => {
            let root = repository::RepositoryRoot::resolve(None)?;
            automation::sdk_compat::run(root.as_path(), rest)
        }
        cli::CliCommand::SdkFixture(rest) => {
            let root = repository::RepositoryRoot::resolve(None)?;
            automation::sdk_fixture::run(root.as_path(), rest)
        }
        cli::CliCommand::RuntimeCacheInstall(rest) => automation::runtime_install::run(rest),
        cli::CliCommand::AgentPickModel(rest) => automation::agent_model::run(rest),
        cli::CliCommand::HfCertification(rest) => automation::hf_certify::run(rest),
        cli::CliCommand::HfMtpCompose(rest) => automation::hf_mtp_compose::run(rest),
        cli::CliCommand::MtpScheduler(rest) => automation::mtp_scheduler::run(rest),
        cli::CliCommand::MtpSchedulerWorker(rest) => automation::mtp_scheduler::run_worker(rest),
        cli::CliCommand::HfConvertedArtifact(rest) => automation::hf_converted_artifact::run(rest),
        cli::CliCommand::Rollout(rest) => automation::rollout::run(rest),
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
        cli::CliCommand::CiFamilyPlan(rest) => {
            let root = match explicit_root {
                Some(root) => root,
                None => repository::RepositoryRoot::resolve(None)?,
            };
            ci_plan::family::run(root.as_path(), rest)
        }
        cli::CliCommand::NativeGenerator(rest) => automation::native_generator::run(rest),
        cli::CliCommand::SplitEvidence(rest) => automation::split_evidence::run(rest),
        cli::CliCommand::ReplayMatrix(rest) => automation::run_replay_matrix(
            rest,
            explicit_root
                .as_ref()
                .map(repository::RepositoryRoot::as_path),
        ),
        cli::CliCommand::Laya(rest) => {
            let root = repository::RepositoryRoot::resolve(parsed.root.as_deref())?;
            automation::laya::run(root.as_path(), rest)
        }
        cli::CliCommand::CanaryTimeout(rest) => automation::canary_timeout::run(rest),
        cli::CliCommand::CanaryReceipts(rest) => automation::canary_aggregate_command::run(rest),
        cli::CliCommand::WorkloadOracleEvidence(rest) => {
            automation::run_workload_oracle_evidence(rest)
        }
        cli::CliCommand::RewriterReport(rest) => automation::rewriter_report::run(rest),
        cli::CliCommand::PreparedInput(rest) => prepared_input::run(rest),
        cli::CliCommand::Product(command, rest) => product::run(command, rest),
        cli::CliCommand::Release(command, rest) => release::run(command, rest),
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
        cli::CliCommand::Native(command, rest) => native_policy::run(command, rest, || {
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
                cli::RepositoryCommand::RequiredSmoke(rest) => {
                    automation::required_smoke::run(root, rest)
                }
                cli::RepositoryCommand::SdkAdvisory(rest) => {
                    automation::sdk_advisory::run(root, rest)
                }
                cli::RepositoryCommand::ClientReadiness(rest) => {
                    Ok(automation::client_readiness::run(root, rest)?)
                }
                cli::RepositoryCommand::DaemonReadiness(rest) => {
                    Ok(automation::daemon_readiness::run(root, rest)?)
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
