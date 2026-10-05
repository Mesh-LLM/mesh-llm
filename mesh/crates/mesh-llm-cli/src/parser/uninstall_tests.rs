use super::{Cli, Command};
use clap::Parser;

#[test]
fn uninstall_defaults_to_confirming_real_changes() {
    let cli = Cli::parse_from(["mesh-llm", "uninstall"]);

    let Some(Command::Uninstall {
        dry_run,
        yes,
        keep_cache,
        keep_service_files,
        purge_config,
        keep_config,
        binary_path,
        json,
        verbose,
    }) = cli.command
    else {
        panic!("expected uninstall command");
    };

    assert!(!dry_run);
    assert!(!yes);
    assert!(!keep_cache);
    assert!(!keep_service_files);
    assert!(!purge_config);
    assert!(!keep_config);
    assert!(binary_path.is_none());
    assert!(!json);
    assert!(!verbose);
}

#[test]
fn uninstall_accepts_automation_flags() {
    let cli = Cli::parse_from([
        "mesh-llm",
        "uninstall",
        "--dry-run",
        "--yes",
        "--keep-cache",
        "--keep-service-files",
        "--keep-config",
        "--binary-path",
        "/tmp/mesh-llm",
        "--json",
        "--verbose",
    ]);

    let Some(Command::Uninstall {
        dry_run,
        yes,
        keep_cache,
        keep_service_files,
        purge_config,
        keep_config,
        binary_path,
        json,
        verbose,
    }) = cli.command
    else {
        panic!("expected uninstall command");
    };

    assert!(dry_run);
    assert!(yes);
    assert!(keep_cache);
    assert!(keep_service_files);
    assert!(!purge_config);
    assert!(keep_config);
    assert_eq!(
        binary_path.expect("binary path"),
        std::path::Path::new("/tmp/mesh-llm")
    );
    assert!(json);
    assert!(verbose);
}

#[test]
fn uninstall_accepts_purge_config() {
    let cli = Cli::parse_from(["mesh-llm", "uninstall", "--purge-config"]);

    let Some(Command::Uninstall {
        purge_config,
        keep_config,
        ..
    }) = cli.command
    else {
        panic!("expected uninstall command");
    };

    assert!(purge_config);
    assert!(!keep_config);
}
