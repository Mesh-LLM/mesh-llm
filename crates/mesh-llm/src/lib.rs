#![recursion_limit = "256"]

use std::time::Duration;

use clap::{CommandFactory, Parser};

mod commands;

pub use mesh_llm_host_runtime::*;

pub async fn run_main() -> i32 {
    match run_cli_entrypoint().await {
        Ok(()) => 0,
        Err(err) => {
            let _ = mesh_llm_tui::emit_fatal_error(&err);
            tokio::time::sleep(Duration::from_millis(50)).await;
            1
        }
    }
}

async fn run_cli_entrypoint() -> anyhow::Result<()> {
    maybe_print_binary_help_and_exit();

    let normalized_args = mesh_llm_cli::normalize_runtime_surface_args(std::env::args_os());
    let cli = mesh_llm_cli::Cli::parse_from(normalized_args.normalized.clone());
    let warning = mesh_llm_cli::legacy_runtime_surface_warning(
        &cli,
        &normalized_args.original,
        normalized_args.explicit_surface,
    );
    let explicit_surface = normalized_args.explicit_surface.map(map_runtime_surface);

    if should_initialize_host_runtime_pre_dispatch(cli.command.as_ref()) {
        mesh_llm_host_runtime::initialize_host_runtime_with_config(cli.config.as_deref()).await?;
    }

    if commands::dispatch(&cli).await? {
        return Ok(());
    }

    mesh_llm_host_runtime::initialize_host_runtime_with_config(cli.config.as_deref()).await?;
    mesh_llm_tui::output::OutputManager::init_global(
        cli.log_format,
        mesh_llm_host_runtime::console_session_mode_for_runtime_surface(explicit_surface),
    );
    mesh_llm_tui::install_terminal_panic_hook();

    mesh_llm_host_runtime::run_runtime_initialized(
        runtime_options_from_cli(cli),
        explicit_surface,
        warning,
    )
    .await
}

fn should_initialize_host_runtime_pre_dispatch(command: Option<&mesh_llm_cli::Command>) -> bool {
    matches!(
        command,
        Some(mesh_llm_cli::Command::Gpus { .. }) | Some(mesh_llm_cli::Command::Benchmark { .. })
    )
}

fn maybe_print_binary_help_and_exit() {
    let args: Vec<_> = std::env::args_os().collect();
    if binary_help_request(args.iter().cloned()) {
        mesh_llm_cli::Cli::command().print_help().ok();
        std::process::exit(0);
    }
    if let Some(surface) = runtime_surface_help_request(args.iter().cloned()) {
        print!("{}", mesh_llm_cli::parser::runtime_surface_help(surface));
        std::process::exit(0);
    }
    if args.iter().any(|arg| arg == "--help-advanced") {
        print_advanced_help();
        std::process::exit(0);
    }
}

fn binary_help_request<I>(args: I) -> bool
where
    I: IntoIterator<Item = std::ffi::OsString>,
{
    let args: Vec<_> = args.into_iter().collect();
    match args.as_slice() {
        [_program] => true,
        [_program, arg] => arg == "--help" || arg == "-h",
        [_program, help, arg] => help == "help" && (arg == "--help" || arg == "-h"),
        _ => false,
    }
}

fn runtime_surface_help_request<I>(args: I) -> Option<mesh_llm_cli::RuntimeSurface>
where
    I: IntoIterator<Item = std::ffi::OsString>,
{
    let args: Vec<_> = args.into_iter().collect();
    let help = args.last()?;
    if help != "--help" && help != "-h" {
        return None;
    }
    mesh_llm_cli::normalize_runtime_surface_args(args).explicit_surface
}

fn print_advanced_help() {
    let mut command = mesh_llm_cli::Cli::command();
    let args: Vec<clap::Id> = command
        .get_arguments()
        .map(|arg| arg.get_id().clone())
        .collect();
    for id in args {
        command = command.mut_arg(id, |arg| arg.hide(false));
    }
    let subcommands: Vec<String> = command
        .get_subcommands()
        .map(|subcommand| subcommand.get_name().to_string())
        .collect();
    for name in subcommands {
        command = command.mut_subcommand(name, |subcommand| subcommand.hide(false));
    }
    command.print_help().ok();
    eprintln!();
}

fn runtime_help_text() -> Option<String> {
    let mut bytes = Vec::new();
    mesh_llm_cli::Cli::command()
        .write_help(&mut bytes)
        .ok()
        .and_then(|()| String::from_utf8(bytes).ok())
}

fn runtime_options_from_cli(cli: mesh_llm_cli::Cli) -> mesh_llm_host_runtime::RuntimeOptions {
    let speculative_overrides = speculative_overrides_from_cli(&cli);
    mesh_llm_host_runtime::RuntimeOptions {
        log_format: cli.log_format,
        debug: cli.debug,
        skippy_metrics_otlp_grpc: cli.skippy_metrics_otlp_grpc,
        mesh_guardrails: map_mesh_guardrail_mode(cli.mesh_guardrails),
        help_text: runtime_help_text(),
        join: cli.join,
        discover: cli.discover,
        auto: cli.auto,
        mesh_discovery_mode: map_mesh_discovery_mode(cli.mesh_discovery_mode),
        model: cli.model,
        gguf: cli.gguf,
        mmproj: cli.mmproj,
        port: cli.port,
        client: cli.client,
        console: cli.console,
        headless: cli.headless,
        swarm_capture: cli.swarm_capture,
        publish: cli.publish,
        mesh_name: cli.mesh_name,
        region: cli.region,
        min_node_version: cli.min_node_version,
        max_node_version: cli.max_node_version,
        min_protocol_version: cli.min_protocol_version,
        max_protocol_version: cli.max_protocol_version,
        require_release_attestation: cli.require_release_attestation,
        release_signer_key: cli.release_signer_key,
        name: cli.name,
        plugin: cli.plugin,
        auto_update: cli.auto_update,
        command_is_update: matches!(cli.command, Some(mesh_llm_cli::Command::Update { .. })),
        command_uses_machine_output: command_uses_machine_output(cli.command.as_ref()),
        draft: cli.draft,
        draft_max: cli.draft_max,
        no_draft: cli.no_draft,
        speculative_overrides,
        split: cli.split,
        split_topology_lock: cli.split_topology_lock,
        ctx_size: cli.ctx_size,
        max_vram: cli.max_vram,
        no_enumerate_host: cli.no_enumerate_host,
        bin_dir: cli.bin_dir,
        llama_flavor: cli.llama_flavor.map(map_binary_flavor),
        device: cli.device,
        tensor_split: cli.tensor_split,
        relay: cli.relay,
        relay_auth: cli.relay_auth,
        disable_iroh_relays: cli.disable_iroh_relays,
        bind_port: cli.bind_port,
        bind_ip: cli.bind_ip,
        listen_all: cli.listen_all,
        max_clients: cli.max_clients,
        nostr_relay: cli.nostr_relay,
        no_console: cli.no_console,
        config: cli.config,
        owner_key: cli.owner_key,
        control_bind: cli.control_bind,
        control_advertise_addr: cli.control_advertise_addr,
        owner_required: cli.owner_required,
        node_label: cli.node_label,
        trust_policy: cli.trust_policy.map(map_trust_policy),
        trust_owner: cli.trust_owner,
        nostr_discovery: cli.nostr_discovery,
    }
}

fn speculative_overrides_from_cli(
    cli: &mesh_llm_cli::Cli,
) -> Option<mesh_llm_host_runtime::plugin::SpeculativeConfig> {
    let suppress_cooldown_drafts = if cli.speculative_native_mtp_allow_cooldown_drafts {
        Some(false)
    } else {
        cli.speculative_native_mtp_suppress_cooldown_drafts
            .then_some(true)
    };
    let mut overrides = mesh_llm_host_runtime::plugin::SpeculativeConfig::default();
    overrides.strategy = cli.speculative_strategy.clone();
    overrides.ngram_min = cli.speculative_ngram_min;
    overrides.ngram_max = cli.speculative_ngram_max;
    overrides.ngram_max_proposal_tokens = cli.speculative_ngram_max_proposal_tokens;
    overrides.ngram_proposer = cli
        .speculative_ngram_proposer
        .map(|proposer| proposer.as_str().to_string());
    overrides.extension_max_tokens = cli.speculative_extension_max_tokens;
    overrides.native_mtp_reject_cooldown_tokens = cli.speculative_native_mtp_reject_cooldown_tokens;
    overrides.native_mtp_suppress_cooldown_drafts = suppress_cooldown_drafts;
    overrides.native_mtp_suppress_cooldown_draft_limit =
        cli.speculative_native_mtp_suppress_cooldown_draft_limit;
    overrides.verify_window_min_tokens = cli.speculative_verify_window_min_tokens;
    overrides.verify_window_max_tokens = cli.speculative_verify_window_max_tokens;
    overrides.verify_window_pipeline_depth = cli.speculative_verify_window_pipeline_depth;
    (overrides != Default::default()).then_some(overrides)
}

fn command_uses_machine_output(command: Option<&mesh_llm_cli::Command>) -> bool {
    matches!(
        command,
        Some(mesh_llm_cli::Command::Doctor {
            json: true,
            command: None,
        }) | Some(mesh_llm_cli::Command::Runtime {
            command: Some(
                mesh_llm_cli::runtime::RuntimeCommand::List { json: true, .. }
                    | mesh_llm_cli::runtime::RuntimeCommand::Install { json: true, .. }
                    | mesh_llm_cli::runtime::RuntimeCommand::Remove { json: true, .. }
                    | mesh_llm_cli::runtime::RuntimeCommand::Prune { json: true, .. },
            ),
        })
    )
}

fn map_runtime_surface(
    surface: mesh_llm_cli::RuntimeSurface,
) -> mesh_llm_host_runtime::RuntimeSurface {
    match surface {
        mesh_llm_cli::RuntimeSurface::Serve => mesh_llm_host_runtime::RuntimeSurface::Serve,
        mesh_llm_cli::RuntimeSurface::Client => mesh_llm_host_runtime::RuntimeSurface::Client,
    }
}

fn map_mesh_discovery_mode(
    mode: mesh_llm_cli::MeshDiscoveryMode,
) -> mesh_llm_host_runtime::discovery::MeshDiscoveryMode {
    match mode {
        mesh_llm_cli::MeshDiscoveryMode::Nostr => {
            mesh_llm_host_runtime::discovery::MeshDiscoveryMode::Nostr
        }
        mesh_llm_cli::MeshDiscoveryMode::Mdns => {
            mesh_llm_host_runtime::discovery::MeshDiscoveryMode::Mdns
        }
    }
}

fn map_mesh_guardrail_mode(
    mode: mesh_llm_cli::MeshGuardrailCliMode,
) -> mesh_llm_host_runtime::MeshGuardrailMode {
    match mode {
        mesh_llm_cli::MeshGuardrailCliMode::Disabled => {
            mesh_llm_host_runtime::MeshGuardrailMode::Disabled
        }
        mesh_llm_cli::MeshGuardrailCliMode::Metrics => {
            mesh_llm_host_runtime::MeshGuardrailMode::Metrics
        }
        mesh_llm_cli::MeshGuardrailCliMode::Enforce => {
            mesh_llm_host_runtime::MeshGuardrailMode::Enforce
        }
    }
}

fn map_binary_flavor(flavor: mesh_llm_cli::BinaryFlavor) -> mesh_llm_system::backend::BinaryFlavor {
    match flavor {
        mesh_llm_cli::BinaryFlavor::Cpu => mesh_llm_system::backend::BinaryFlavor::Cpu,
        mesh_llm_cli::BinaryFlavor::Cuda => mesh_llm_system::backend::BinaryFlavor::Cuda,
        mesh_llm_cli::BinaryFlavor::Rocm => mesh_llm_system::backend::BinaryFlavor::Rocm,
        mesh_llm_cli::BinaryFlavor::Vulkan => mesh_llm_system::backend::BinaryFlavor::Vulkan,
        mesh_llm_cli::BinaryFlavor::Metal => mesh_llm_system::backend::BinaryFlavor::Metal,
    }
}

fn map_trust_policy(
    policy: mesh_llm_cli::TrustPolicy,
) -> mesh_llm_host_runtime::crypto::TrustPolicy {
    match policy {
        mesh_llm_cli::TrustPolicy::Off => mesh_llm_host_runtime::crypto::TrustPolicy::Off,
        mesh_llm_cli::TrustPolicy::PreferOwned => {
            mesh_llm_host_runtime::crypto::TrustPolicy::PreferOwned
        }
        mesh_llm_cli::TrustPolicy::RequireOwned => {
            mesh_llm_host_runtime::crypto::TrustPolicy::RequireOwned
        }
        mesh_llm_cli::TrustPolicy::Allowlist => {
            mesh_llm_host_runtime::crypto::TrustPolicy::Allowlist
        }
    }
}

#[cfg(test)]
mod cli_entrypoint_tests {
    use clap::Parser;
    use std::ffi::OsString;

    #[test]
    fn runtime_surface_help_request_handles_serve_and_client_help() {
        assert_eq!(
            super::runtime_surface_help_request([
                OsString::from("mesh-llm"),
                OsString::from("serve"),
                OsString::from("--help"),
            ]),
            Some(mesh_llm_cli::RuntimeSurface::Serve)
        );
        assert_eq!(
            super::runtime_surface_help_request([
                OsString::from("mesh-llm"),
                OsString::from("client"),
                OsString::from("-h"),
            ]),
            Some(mesh_llm_cli::RuntimeSurface::Client)
        );
    }

    #[test]
    fn runtime_surface_help_request_skips_leading_global_flags() {
        assert_eq!(
            super::runtime_surface_help_request([
                OsString::from("mesh-llm"),
                OsString::from("--log-format"),
                OsString::from("json"),
                OsString::from("serve"),
                OsString::from("--help"),
            ]),
            Some(mesh_llm_cli::RuntimeSurface::Serve)
        );
        assert_eq!(
            super::runtime_surface_help_request([
                OsString::from("mesh-llm"),
                OsString::from("--relay-auth=https://relay.example=token"),
                OsString::from("client"),
                OsString::from("-h"),
            ]),
            Some(mesh_llm_cli::RuntimeSurface::Client)
        );
    }

    #[test]
    fn binary_help_request_handles_help_help() {
        assert!(super::binary_help_request([
            OsString::from("mesh-llm"),
            OsString::from("help"),
            OsString::from("--help"),
        ]));
    }

    #[test]
    fn cli_ngram_override_reaches_runtime_config() {
        let normalized = mesh_llm_cli::normalize_runtime_surface_args([
            "mesh-llm",
            "serve",
            "--speculative-strategy",
            "mtp",
            "--speculative-ngram-min",
            "2",
            "--speculative-ngram-max",
            "6",
            "--speculative-ngram-max-proposal-tokens",
            "5",
        ]);
        let cli = mesh_llm_cli::Cli::try_parse_from(normalized.normalized).expect("CLI parses");

        let config = super::speculative_overrides_from_cli(&cli)
            .expect("speculative flags produce an override");

        assert_eq!(config.strategy.as_deref(), Some("mtp"));
        assert_eq!(config.ngram_min, Some(2));
        assert_eq!(config.ngram_max, Some(6));
        assert_eq!(config.ngram_max_proposal_tokens, Some(5));
    }
}
