use super::*;

#[test]
fn plugins_install_accepts_local_archive_options() {
    let cli = Cli::parse_from([
        "mesh-llm",
        "plugins",
        "install",
        "--archive",
        "/tmp/demo.tar.gz",
        "--name",
        "demo",
        "--version",
        "0.1.0",
    ]);

    match cli.command.expect("plugins command") {
        Command::Plugin {
            command:
                PluginCommand::Install {
                    reference,
                    set: None,
                    archive: Some(archive),
                    name: Some(name),
                    version: Some(version),
                },
        } => {
            assert!(reference.is_empty());
            assert_eq!(archive, PathBuf::from("/tmp/demo.tar.gz"));
            assert_eq!(name, "demo");
            assert_eq!(version, "0.1.0");
        }
        other => panic!("unexpected command: {other:?}"),
    }
}

#[test]
fn plugins_install_rejects_reference_with_local_archive() {
    let error = Cli::try_parse_from([
        "mesh-llm",
        "plugins",
        "install",
        "demo",
        "--archive",
        "/tmp/demo.tar.gz",
        "--name",
        "demo",
    ])
    .expect_err("reference and local archive must conflict");

    assert!(error.to_string().contains("cannot be used with"));
}

#[test]
fn accepts_one_or_multiple_references_in_order() {
    for references in [
        vec!["blackboard"],
        vec![
            "blackboard",
            "Mesh-LLM/metrics-plugin",
            "https://github.com/Mesh-LLM/agents-plugin",
        ],
    ] {
        let args = [vec!["mesh-llm", "plugins", "install"], references.clone()].concat();
        let cli = Cli::try_parse_from(args).expect("plugin references");
        let Some(Command::Plugin {
            command:
                PluginCommand::Install {
                    reference,
                    set: None,
                    archive: None,
                    name: None,
                    version: None,
                },
        }) = cli.command
        else {
            panic!("expected reference install");
        };
        assert_eq!(reference, references);
    }
}

#[test]
fn rejects_missing_references_and_invalid_archive_options() {
    for options in [
        vec![],
        vec!["--archive", "demo.tar.gz"],
        vec!["demo", "--name", "demo"],
        vec!["demo", "--version", "1.0.0"],
        vec![
            "first",
            "second",
            "--archive",
            "demo.tar.gz",
            "--name",
            "demo",
        ],
    ] {
        let args = [vec!["mesh-llm", "plugins", "install"], options].concat();
        assert!(Cli::try_parse_from(&args).is_err(), "accepted {args:?}");
    }
}

#[test]
fn plugins_install_accepts_a_catalog_set_alone() {
    let cli = Cli::try_parse_from(["mesh-llm", "plugins", "install", "--set", "recommended"])
        .expect("--set parses by itself");
    match cli.command.expect("plugins command") {
        Command::Plugin {
            command:
                PluginCommand::Install {
                    reference,
                    set: Some(set),
                    archive: None,
                    ..
                },
        } => {
            assert!(reference.is_empty());
            assert_eq!(set, "recommended");
        }
        other => panic!("unexpected command: {other:?}"),
    }
}

#[test]
fn plugins_install_rejects_a_set_with_references_or_an_archive() {
    for extra in [
        vec!["notes"],
        vec!["--archive", "/tmp/demo.tar.gz", "--name", "demo"],
    ] {
        let mut args = vec!["mesh-llm", "plugins", "install", "--set", "recommended"];
        args.extend(extra);
        let error = Cli::try_parse_from(args).expect_err("--set conflicts");
        assert_eq!(error.kind(), clap::error::ErrorKind::ArgumentConflict);
    }
}
