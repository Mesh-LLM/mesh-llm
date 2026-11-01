//! Validate the explicit local memberships launch surface before host startup.

use clap::{CommandFactory, error::ErrorKind, parser::ValueSource};

use crate::{Cli, NormalizedRuntimeArgs, RuntimeSurface};

/// Memberships own their topology and host limits. Reject options the shared
/// host cannot honor instead of silently accepting single-mesh settings.
pub fn validate_launch(cli: &Cli, args: &NormalizedRuntimeArgs) -> Result<(), clap::Error> {
    if cli.serving.memberships_config.is_none() {
        return Ok(());
    }
    if args.explicit_surface != Some(RuntimeSurface::Serve) || cli.command.is_some() {
        return Err(clap::Error::raw(
            ErrorKind::ArgumentConflict,
            "--memberships-config requires the explicit `serve` command",
        ));
    }
    if cli.serving.gguf.len() != 1 {
        return Err(clap::Error::raw(
            ErrorKind::ArgumentConflict,
            "--memberships-config requires exactly one --gguf local model",
        ));
    }
    let command = Cli::command();
    let matches = command.clone().try_get_matches_from(&args.normalized)?;
    for argument in command.get_arguments() {
        let id = argument.get_id().as_str();
        if !matches!(
            matches.value_source(id),
            Some(ValueSource::CommandLine | ValueSource::EnvVariable)
        ) {
            continue;
        }
        if matches!(
            id,
            "gguf"
                | "memberships_config"
                | "ctx_size"
                | "parallel"
                | "config"
                | "log_format"
                | "debug"
        ) {
            continue;
        }
        // Print only the declared flag name, never its value (e.g. an invite).
        let flag = argument.get_long().unwrap_or(id);
        return Err(clap::Error::raw(
            ErrorKind::ArgumentConflict,
            format!(
                "--{flag} cannot be combined with --memberships-config; configure memberships and host settings in the local TOML"
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn parse(args: &[&str]) -> Result<Cli, clap::Error> {
        let args = crate::normalize_runtime_surface_args(args.iter().copied());
        let cli = Cli::try_parse_from(&args.normalized)?;
        validate_launch(&cli, &args)?;
        Ok(cli)
    }

    #[test]
    fn configured_serve_uses_normal_model_and_capacity_options() {
        let cli = parse(&[
            "mesh-llm",
            "serve",
            "--gguf",
            "model.gguf",
            "--memberships-config",
            "memberships.toml",
            "--parallel",
            "2",
            "--ctx-size",
            "1024",
        ])
        .unwrap();
        assert_eq!(
            cli.serving.memberships_config.unwrap().to_str(),
            Some("memberships.toml")
        );
        assert_eq!(cli.placement.parallel.unwrap().get(), 2);
        assert_eq!(cli.placement.ctx_size, Some(1024));
    }

    #[test]
    fn rejects_ambiguous_topology_and_unused_flags_without_echoing_credentials() {
        for extra in [
            vec!["--join", "SECRET"],
            vec!["-j", "SECRET"],
            vec!["--join=SECRET"],
            vec!["--join-file", "work.token"],
            vec!["--auto"],
            vec!["--client"],
            vec!["--local-model-only"],
            vec!["--model", "other"],
            vec!["--gguf", "second.gguf"],
            vec!["--port", "9337"],
            vec!["--console", "3131"],
            vec!["--publish"],
            vec!["--no-update-check"],
        ] {
            let mut args = vec![
                "mesh-llm",
                "serve",
                "--gguf",
                "model.gguf",
                "--memberships-config",
                "memberships.toml",
            ];
            args.extend(extra);
            let error = parse(&args).unwrap_err().to_string();
            assert!(!error.contains("SECRET"));
        }
        for surface in ["client", "status"] {
            assert!(
                parse(&[
                    "mesh-llm",
                    "--gguf",
                    "model.gguf",
                    "--memberships-config",
                    "m.toml",
                    surface
                ])
                .is_err()
            );
        }
        assert!(parse(&["mesh-llm", "serve", "--memberships-config", "m.toml"]).is_err());
        assert!(
            parse(&[
                "mesh-llm",
                "--gguf",
                "model.gguf",
                "--memberships-config",
                "m.toml"
            ])
            .is_err()
        );
        assert!(
            Cli::command()
                .get_subcommands()
                .all(|command| command.get_name() != "serve-shared")
        );
    }

    #[test]
    fn ordinary_serve_and_repeated_join_keep_existing_semantics() {
        let cli = parse(&[
            "mesh-llm",
            "serve",
            "--gguf",
            "model.gguf",
            "--join",
            "one",
            "--join",
            "two",
        ])
        .unwrap();
        assert_eq!(cli.joining.join, ["one", "two"]);
        assert!(cli.serving.memberships_config.is_none());
    }

    #[test]
    fn config_validation_selects_exactly_one_schema() {
        assert!(
            parse(&[
                "mesh-llm",
                "config",
                "validate",
                "--memberships-config",
                "m.toml",
                "--json"
            ])
            .is_ok()
        );
        assert!(
            parse(&[
                "mesh-llm",
                "config",
                "validate",
                "--memberships-config",
                "m.toml",
                "--config-path",
                "mesh.toml"
            ])
            .is_err()
        );
    }
}
