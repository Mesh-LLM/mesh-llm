use super::{Cli, Command, SetupArgs};
use clap::{Parser, error::ErrorKind};

#[test]
fn setup_command_parses_without_plugin_fallback() {
    let cli = Cli::parse_from([
        "mesh-llm",
        "setup",
        "--yes",
        "--no-interactive",
        "--skip-runtime",
        "--skip-plugin-install",
        "--verbose",
    ]);

    match cli.command.expect("setup command expected") {
        Command::Setup(SetupArgs {
            yes,
            no_interactive,
            service,
            no_service,
            skip_runtime,
            skip_plugin_install,
            verbose,
        }) => {
            assert!(yes);
            assert!(no_interactive);
            assert!(!service);
            assert!(!no_service);
            assert!(skip_runtime);
            assert!(skip_plugin_install);
            assert!(verbose);
        }
        other => panic!("unexpected command: {other:?}"),
    }
}

#[test]
fn setup_command_rejects_conflicting_service_flags() {
    let err = Cli::try_parse_from(["mesh-llm", "setup", "--service", "--no-service"])
        .expect_err("setup should reject conflicting service flags");

    assert_eq!(err.kind(), ErrorKind::ArgumentConflict);
    assert!(err.to_string().contains("--service"));
    assert!(err.to_string().contains("--no-service"));
}

#[test]
fn setup_command_rejects_skip_doctor_flag() {
    let err = Cli::try_parse_from(["mesh-llm", "setup", "--skip-doctor"])
        .expect_err("setup should reject unknown skip-doctor flag");

    assert_eq!(err.kind(), ErrorKind::UnknownArgument);
    assert!(err.to_string().contains("--skip-doctor"));
}

#[test]
fn setup_installs_plugins_by_default_and_skip_plugin_install_opts_out() {
    for (flag, expected) in [(None, false), (Some("--skip-plugin-install"), true)] {
        let mut args = vec!["mesh-llm", "setup"];
        args.extend(flag);
        let cli = Cli::parse_from(args);
        let Some(Command::Setup(setup)) = cli.command else {
            panic!("setup command expected");
        };
        assert_eq!(setup.skip_plugin_install, expected);
    }
}
