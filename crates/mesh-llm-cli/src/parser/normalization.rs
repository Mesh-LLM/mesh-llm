use std::ffi::OsString;

use super::commands::Cli;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeSurface {
    Serve,
    Client,
}

#[derive(Clone, Debug)]
pub struct NormalizedRuntimeArgs {
    pub original: Vec<OsString>,
    pub normalized: Vec<OsString>,
    pub explicit_surface: Option<RuntimeSurface>,
}

/// Flags that consume the following token as their value.
///
/// Hand-maintained: a new value-taking flag has to be added here, or
/// [`first_positional_index`] will mistake its value for a subcommand.
/// Boolean flags (`--help-advanced`, `--auto`, `--client`, `--local-model-only`,
/// `--headless`, `--publish`, `--auto-update`, `--no-draft`, `--split`,
/// `--no-enumerate-host`, `--listen-all`, `--no-console`, `--owner-required`)
/// are deliberately absent.
const VALUE_TAKING_FLAGS: &[&str] = &[
    "--log-format",
    "--mesh-discovery-mode",
    "--max-vram",
    "--llama-flavor",
    "--device",
    "--tensor-split",
    "--bind-port",
    "--bind-ip",
    "--max-clients",
    "--port",
    "--console",
    "--swarm-capture",
    "--draft-max",
    "--ctx-size",
    "--parallel",
    "--model",
    "--gguf",
    "--mmproj",
    "--checkpoint-quantization",
    "--quant",
    "--checkpoint-imatrix",
    "--join",
    "--join-file",
    "--discover",
    "--mesh-name",
    "--region",
    "--name",
    "--plugin",
    "--draft",
    "--bin-dir",
    "--relay",
    "--relay-auth",
    "--nostr-relay",
    "--config",
    "--owner-key",
    "--control-bind",
    "--control-advertise-addr",
    "--node-label",
    "--trust-policy",
    "--trust-owner",
];

/// Index of the first positional argument, skipping leading global flags.
///
/// A value-taking flag consumes the token after it, so `mesh-llm --config
/// /tmp/config analytics` locates `analytics` rather than the config path.
/// `--flag=value` is self-contained and skipped as one token. An unknown flag
/// is skipped a token at a time so clap still gets to report it. Returns
/// `args.len()` when there is no positional argument.
fn first_positional_index(args: &[OsString]) -> usize {
    let mut pos = 1;
    while pos < args.len() {
        let arg = args.get(pos).and_then(|arg| arg.to_str()).unwrap_or("");

        if let Some((flag, _value)) = arg.split_once('=')
            && VALUE_TAKING_FLAGS.contains(&flag)
        {
            pos += 1;
            continue;
        }

        if VALUE_TAKING_FLAGS.contains(&arg) {
            let next_is_value = args
                .get(pos + 1)
                .and_then(|arg| arg.to_str())
                .is_some_and(|next| !next.starts_with('-'));
            // Advance by two when the value is present, by one otherwise and
            // let clap report the missing value.
            pos += if next_is_value { 2 } else { 1 };
            continue;
        }

        if arg.starts_with('-') {
            pos += 1;
            continue;
        }

        break;
    }
    pos
}

pub fn normalize_runtime_surface_args<I, S>(args: I) -> NormalizedRuntimeArgs
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    let original: Vec<OsString> = args.into_iter().map(Into::into).collect();
    let mut normalized = original.clone();
    let mut explicit_surface = None;

    // Skip leading global flags to find the pseudo-subcommand position.
    let pos = first_positional_index(&original);

    // Now apply the serve/client normalization logic at the discovered position
    match original.get(pos).and_then(|arg| arg.to_str()) {
        Some("serve") => match original.get(pos + 1).and_then(|arg| arg.to_str()) {
            Some(arg) if arg.starts_with('-') => {
                normalized.remove(pos);
                explicit_surface = Some(RuntimeSurface::Serve);
            }
            None => {
                normalized.remove(pos);
                explicit_surface = Some(RuntimeSurface::Serve);
            }
            _ => {}
        },
        Some("client") => {
            normalized.remove(pos);
            normalized.insert(pos, OsString::from("--client"));
            explicit_surface = Some(RuntimeSurface::Client);
        }
        _ => {}
    }

    NormalizedRuntimeArgs {
        original,
        normalized,
        explicit_surface,
    }
}

pub fn legacy_runtime_surface_warning(
    cli: &Cli,
    original_args: &[OsString],
    explicit_surface: Option<RuntimeSurface>,
) -> Option<String> {
    if explicit_surface.is_some() || cli.command.is_some() {
        return None;
    }

    if cli.client {
        return Some(format!(
            "⚠️ top-level `--client` now maps to `mesh-llm client`.\n  Please use: {}",
            suggested_client_command(original_args)
        ));
    }

    if !cli.model.is_empty()
        || !cli.gguf.is_empty()
        || cli.mmproj.is_some()
        || cli.checkpoint_quantization.is_some()
        || cli.checkpoint_imatrix.is_some()
    {
        return Some(format!(
            "⚠️ top-level serving flags now map to `mesh-llm serve`.\n  Please use: {}",
            suggested_serve_command(original_args)
        ));
    }

    None
}

fn suggested_serve_command(original_args: &[OsString]) -> String {
    let mut args = Vec::with_capacity(original_args.len() + 1);
    if let Some(program) = original_args.first() {
        args.push(program.clone());
    } else {
        args.push(OsString::from("mesh-llm"));
    }
    args.push(OsString::from("serve"));
    args.extend(original_args.iter().skip(1).cloned());
    shell_join(&args)
}

fn suggested_client_command(original_args: &[OsString]) -> String {
    let mut args = Vec::with_capacity(original_args.len());
    if let Some(program) = original_args.first() {
        args.push(program.clone());
    } else {
        args.push(OsString::from("mesh-llm"));
    }
    args.push(OsString::from("client"));
    let mut skipped_client = false;
    for arg in original_args.iter().skip(1) {
        if !skipped_client && arg.to_string_lossy() == "--client" {
            skipped_client = true;
            continue;
        }
        args.push(arg.clone());
    }
    shell_join(&args)
}

fn shell_join(args: &[OsString]) -> String {
    args.iter().map(shell_display).collect::<Vec<_>>().join(" ")
}

fn shell_display(arg: &OsString) -> String {
    let text = arg.to_string_lossy();
    if text.is_empty() {
        "\"\"".into()
    } else if text
        .chars()
        .any(|ch| ch.is_whitespace() || matches!(ch, '"' | '\'' | '\\'))
    {
        format!("{text:?}")
    } else {
        text.into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{Cli, Command, MeshDiscoveryMode};
    use clap::Parser;
    use mesh_llm_events::LogFormat;
    use std::ffi::OsString;
    use std::path::PathBuf;

    #[test]
    fn normalize_runtime_surface_args_rewrites_serve_invocation() {
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "serve",
            "--auto",
            "--model",
            "Qwen3-8B-Q4_K_M",
        ]);

        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Serve));
        assert_eq!(
            normalized.normalized,
            vec!["mesh-llm", "--auto", "--model", "Qwen3-8B-Q4_K_M"]
                .into_iter()
                .map(OsString::from)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn normalize_runtime_surface_args_skips_parallel_value_before_serve() {
        let args = normalize_runtime_surface_args([
            "mesh-llm",
            "--parallel",
            "32",
            "serve",
            "--model",
            "x.gguf",
        ]);
        assert_eq!(args.explicit_surface, Some(RuntimeSurface::Serve));
        assert_eq!(
            args.normalized,
            vec![
                OsString::from("mesh-llm"),
                OsString::from("--parallel"),
                OsString::from("32"),
                OsString::from("--model"),
                OsString::from("x.gguf"),
            ]
        );
    }

    #[test]
    fn normalize_runtime_surface_args_skips_quant_value_before_serve() {
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "--quant",
            "Q4_K_M",
            "serve",
            "--model",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
        ]);

        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Serve));
        assert_eq!(
            normalized.normalized,
            vec![
                "mesh-llm",
                "--quant",
                "Q4_K_M",
                "--model",
                "Qwen/Qwen2.5-Coder-7B-Instruct",
            ]
            .into_iter()
            .map(OsString::from)
            .collect::<Vec<_>>()
        );
    }

    #[test]
    fn normalize_runtime_surface_args_bare_serve_loads_default_config() {
        let normalized = normalize_runtime_surface_args(["mesh-llm", "serve"]);

        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Serve));
        assert_eq!(
            normalized.normalized,
            vec!["mesh-llm"]
                .into_iter()
                .map(OsString::from)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn normalize_runtime_surface_args_rewrites_client_invocation() {
        let normalized =
            normalize_runtime_surface_args(["mesh-llm", "client", "--auto", "--port", "9337"]);

        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Client));
        assert_eq!(
            normalized.normalized,
            vec!["mesh-llm", "--client", "--auto", "--port", "9337"]
                .into_iter()
                .map(OsString::from)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn normalize_runtime_surface_args_treats_relay_auth_as_value_taking_before_serve() {
        // Regression: --relay-auth carries a `URL=TOKEN` value, so the
        // pseudo-subcommand scanner must skip the value and still discover
        // `serve` (or `client`) as the runtime surface. If --relay-auth is not
        // in the value-taking list the scanner stops at the token and Clap
        // sees a malformed command.
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "--relay-auth",
            "https://gated.example/=token",
            "serve",
            "--relay",
            "https://gated.example/",
            "--auto",
        ]);

        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Serve));
        assert_eq!(
            normalized.normalized,
            vec![
                "mesh-llm",
                "--relay-auth",
                "https://gated.example/=token",
                "--relay",
                "https://gated.example/",
                "--auto",
            ]
            .into_iter()
            .map(OsString::from)
            .collect::<Vec<_>>()
        );

        // And the resulting argv must actually parse cleanly through Clap so
        // the relay-auth value reaches `Cli::relay_auth`.
        let cli = Cli::try_parse_from(&normalized.normalized).expect("clap parse");
        assert_eq!(
            cli.relay_auth,
            vec![("https://gated.example/".to_string(), "token".to_string())],
        );
    }

    #[test]
    fn normalize_runtime_surface_args_relay_auth_before_client_invocation() {
        // Same regression but for the `client` surface, including a token
        // containing `=` (NIP-98-style base64 padding).
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "--relay-auth",
            "https://gated.example/=eyJhbGciOiJFZERTQSJ9.payload==",
            "client",
            "--auto",
        ]);

        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Client));
        let cli = Cli::try_parse_from(&normalized.normalized).expect("clap parse");
        assert!(cli.client, "client surface flag should be set");
        assert_eq!(
            cli.relay_auth,
            vec![(
                "https://gated.example/".to_string(),
                "eyJhbGciOiJFZERTQSJ9.payload==".to_string()
            )],
        );
    }

    #[test]
    fn normalize_runtime_surface_args_keeps_non_runtime_subcommands() {
        let normalized = normalize_runtime_surface_args(["mesh-llm", "download", "foo"]);

        assert_eq!(normalized.explicit_surface, None);
        assert_eq!(
            normalized.normalized,
            vec!["mesh-llm", "download", "foo"]
                .into_iter()
                .map(OsString::from)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn legacy_runtime_surface_warning_for_top_level_serve_flags() {
        let normalized =
            normalize_runtime_surface_args(["mesh-llm", "--auto", "--model", "Qwen3-8B-Q4_K_M"]);
        let cli = Cli::parse_from(normalized.normalized.clone());

        let warning =
            legacy_runtime_surface_warning(&cli, &normalized.original, normalized.explicit_surface)
                .expect("warning should be present");

        assert!(warning.contains("mesh-llm serve --auto --model Qwen3-8B-Q4_K_M"));
    }

    #[test]
    fn legacy_runtime_surface_warning_for_top_level_client_flag() {
        let normalized = normalize_runtime_surface_args(["mesh-llm", "--auto", "--client"]);
        let cli = Cli::parse_from(normalized.normalized.clone());

        let warning =
            legacy_runtime_surface_warning(&cli, &normalized.original, normalized.explicit_surface)
                .expect("warning should be present");

        assert!(warning.contains("mesh-llm client --auto"));
    }

    #[test]
    fn explicit_runtime_surface_suppresses_legacy_warning() {
        let normalized = normalize_runtime_surface_args(["mesh-llm", "client", "--auto"]);
        let cli = Cli::parse_from(normalized.normalized.clone());

        assert!(
            legacy_runtime_surface_warning(&cli, &normalized.original, normalized.explicit_surface)
                .is_none()
        );
    }

    #[test]
    fn cli_accepts_headless_flag_for_serve_surface() {
        let args = vec!["mesh-llm", "serve", "--headless", "--auto"];
        let normalized = normalize_runtime_surface_args(args);
        let cli = Cli::try_parse_from(&normalized.normalized).unwrap();
        assert!(cli.headless);
    }

    #[test]
    fn cli_accepts_headless_flag_for_client_surface() {
        let args = vec!["mesh-llm", "client", "--headless", "--auto"];
        let normalized = normalize_runtime_surface_args(args);
        let cli = Cli::try_parse_from(&normalized.normalized).unwrap();
        assert!(cli.headless);
    }

    #[test]
    fn cli_accepts_swarm_capture_flag_for_client_surface() {
        let args = vec![
            "mesh-llm",
            "client",
            "--swarm-capture",
            "/tmp/mesh-capture",
            "--auto",
        ];
        let normalized = normalize_runtime_surface_args(args);
        let cli = Cli::try_parse_from(&normalized.normalized).unwrap();

        assert!(cli.client);
        assert_eq!(cli.swarm_capture, Some(PathBuf::from("/tmp/mesh-capture")));
    }

    #[test]
    fn cli_accepts_global_swarm_capture_before_client() {
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "--swarm-capture",
            "/tmp/mesh-capture",
            "client",
            "--auto",
        ]);
        let cli = Cli::parse_from(normalized.normalized);

        assert!(cli.client);
        assert_eq!(cli.swarm_capture, Some(PathBuf::from("/tmp/mesh-capture")));
        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Client));
    }

    #[test]
    fn legacy_no_console_remains_ignored_in_headless_tests() {
        let args = vec!["mesh-llm", "serve", "--no-console"];
        let normalized = normalize_runtime_surface_args(args);
        let cli = Cli::try_parse_from(&normalized.normalized).unwrap();
        assert!(
            !cli.headless,
            "--no-console must not activate headless mode"
        );
    }

    #[test]
    fn local_model_only_is_an_explicit_serve_topology() {
        let args = vec![
            "mesh-llm",
            "serve",
            "--local-model-only",
            "--model",
            "/models/model.gguf",
        ];
        let normalized = normalize_runtime_surface_args(args);
        let cli = Cli::try_parse_from(&normalized.normalized).unwrap();

        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Serve));
        assert!(cli.local_model_only);
        assert!(!cli.client);
    }

    #[test]
    fn unknown_top_level_command_is_captured_for_plugin_dispatch() {
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "goose-next",
            "--model",
            "auto",
            "--",
            "prompt.txt",
        ]);
        let cli = Cli::parse_from(normalized.normalized);

        match cli.command.expect("external plugin command expected") {
            Command::ExternalPlugin(args) => {
                assert_eq!(
                    args,
                    vec![
                        OsString::from("goose-next"),
                        OsString::from("--model"),
                        OsString::from("auto"),
                        OsString::from("--"),
                        OsString::from("prompt.txt"),
                    ]
                );
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn cli_defaults_log_format_to_pretty() {
        let normalized = normalize_runtime_surface_args(["mesh-llm", "serve", "--auto"]);
        let cli = Cli::parse_from(normalized.normalized);

        assert_eq!(cli.log_format, LogFormat::Pretty);
    }

    #[test]
    fn cli_accepts_json_log_format() {
        let normalized =
            normalize_runtime_surface_args(["mesh-llm", "serve", "--log-format", "json", "--auto"]);
        let cli = Cli::parse_from(normalized.normalized);

        assert_eq!(cli.log_format, LogFormat::Json);
    }

    #[test]
    fn cli_accepts_global_log_format_before_serve() {
        let normalized =
            normalize_runtime_surface_args(["mesh-llm", "--log-format", "json", "serve", "--auto"]);
        let cli = Cli::parse_from(normalized.normalized);

        assert_eq!(cli.log_format, LogFormat::Json);
        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Serve));
    }

    #[test]
    fn cli_accepts_global_log_format_before_serve_with_model() {
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "--log-format",
            "json",
            "serve",
            "--model",
            "Qwen3-8B-Q4_K_M",
        ]);
        let cli = Cli::parse_from(normalized.normalized);

        assert_eq!(cli.log_format, LogFormat::Json);
        assert_eq!(cli.model, vec![std::path::PathBuf::from("Qwen3-8B-Q4_K_M")]);
        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Serve));
    }

    #[test]
    fn cli_accepts_global_log_format_equals_before_serve() {
        let normalized =
            normalize_runtime_surface_args(["mesh-llm", "--log-format=json", "serve", "--auto"]);
        let cli = Cli::parse_from(normalized.normalized);

        assert_eq!(cli.log_format, LogFormat::Json);
        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Serve));
    }

    #[test]
    fn cli_accepts_global_log_format_before_client() {
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "--log-format",
            "json",
            "client",
            "--auto",
        ]);
        let cli = Cli::parse_from(normalized.normalized);

        assert_eq!(cli.log_format, LogFormat::Json);
        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Client));
    }

    #[test]
    fn cli_accepts_global_bind_ip_before_serve() {
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "--bind-ip",
            "10.1.2.3",
            "serve",
            "--bind-port",
            "47916",
        ]);
        let cli = Cli::parse_from(normalized.normalized);

        assert_eq!(cli.bind_ip, Some("10.1.2.3".parse().unwrap()));
        assert_eq!(cli.bind_port, Some(47916));
        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Serve));
    }

    #[test]
    fn cli_accepts_global_mesh_discovery_mode_before_serve() {
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "--mesh-discovery-mode",
            "mdns",
            "serve",
            "--auto",
        ]);
        let cli = Cli::parse_from(normalized.normalized);

        assert_eq!(cli.mesh_discovery_mode, MeshDiscoveryMode::Mdns);
        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Serve));
    }

    #[test]
    fn cli_defaults_mesh_discovery_mode_to_nostr() {
        let normalized = normalize_runtime_surface_args(["mesh-llm", "serve", "--auto"]);
        let cli = Cli::parse_from(normalized.normalized);

        assert_eq!(cli.mesh_discovery_mode, MeshDiscoveryMode::Nostr);
    }

    #[test]
    fn cli_accepts_mdns_discovery_mode_for_runtime_surfaces() {
        let normalized =
            normalize_runtime_surface_args(["mesh-llm", "client", "--mesh-discovery-mode", "mdns"]);
        let cli = Cli::parse_from(normalized.normalized);

        assert!(cli.client);
        assert_eq!(cli.mesh_discovery_mode, MeshDiscoveryMode::Mdns);
    }
}

/// Whether raw argv invokes the `analytics` subcommand.
///
/// Needed on the parse-failure path, where there is no parsed `Cli` to match
/// on. `mesh-llm analytics --typo` fails clap parsing, and without this it
/// would report a `cli_command` event for the very command family that
/// promises not to report.
///
/// Uses the same value-aware scan as [`first_positional_index`], so a
/// value-taking global flag before the subcommand
/// (`mesh-llm --config /tmp/config analytics --typo`) still matches instead of
/// handing the flag's value to the classifier as a subcommand name.
#[must_use]
pub fn raw_args_invoke_analytics(args: &[std::ffi::OsString]) -> bool {
    args.get(first_positional_index(args))
        .and_then(|arg| arg.to_str())
        == Some("analytics")
}

#[cfg(test)]
mod analytics_classification_tests {
    use super::{first_positional_index, raw_args_invoke_analytics};
    use std::ffi::OsString;

    fn args(raw: &[&str]) -> Vec<OsString> {
        raw.iter().map(OsString::from).collect()
    }

    #[test]
    fn matches_analytics_invocations_including_malformed_ones() {
        for raw in [
            &["mesh-llm", "analytics"][..],
            &["mesh-llm", "analytics", "status"][..],
            &["mesh-llm", "analytics", "--typo"][..],
            &["mesh-llm", "--debug", "analytics", "disable"][..],
            &["mesh-llm", "analytics", "--help"][..],
        ] {
            assert!(raw_args_invoke_analytics(&args(raw)), "missed {raw:?}");
        }
    }

    #[test]
    fn does_not_match_other_commands() {
        for raw in [
            &["mesh-llm", "gpus"][..],
            &["mesh-llm"][..],
            &["mesh-llm", "--version"][..],
            &["mesh-llm", "download", "analytics"][..],
        ] {
            assert!(!raw_args_invoke_analytics(&args(raw)), "matched {raw:?}");
        }
    }

    #[test]
    fn matches_analytics_after_a_value_taking_global_option() {
        // The value of a global option is not a subcommand name. Treating it
        // as one made `mesh-llm --config /tmp/config analytics --typo` report
        // a `cli_command` event for the family that promises not to report.
        for raw in [
            &["mesh-llm", "--config", "/tmp/config", "analytics", "--typo"][..],
            &["mesh-llm", "--config=/tmp/config", "analytics", "status"][..],
            &[
                "mesh-llm",
                "--log-format",
                "json",
                "--config",
                "/tmp/config",
                "analytics",
            ][..],
        ] {
            assert!(raw_args_invoke_analytics(&args(raw)), "missed {raw:?}");
        }
    }

    #[test]
    fn a_value_taking_option_value_is_not_a_subcommand() {
        // `analytics` here is a config path and a node label, not a command.
        for raw in [
            &["mesh-llm", "--config", "analytics"][..],
            &["mesh-llm", "--name", "analytics", "gpus"][..],
        ] {
            assert!(!raw_args_invoke_analytics(&args(raw)), "matched {raw:?}");
        }
    }

    #[test]
    fn first_positional_index_skips_flag_values() {
        assert_eq!(first_positional_index(&args(&["mesh-llm", "serve"])), 1);
        assert_eq!(
            first_positional_index(&args(&["mesh-llm", "--port", "9337", "serve"])),
            3
        );
        assert_eq!(
            first_positional_index(&args(&["mesh-llm", "--port=9337", "serve"])),
            2
        );
        // A boolean flag takes no value, so nothing after it is skipped.
        assert_eq!(first_positional_index(&args(&["mesh-llm", "--auto"])), 2);
        assert_eq!(first_positional_index(&args(&["mesh-llm"])), 1);
    }
}
