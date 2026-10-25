use crate::protocol::{Behavior, Destination};
use crate::support::{Case, Sentinel, assert_absent, ready, record, repository};

#[path = "cli_readiness.rs"]
mod readiness;

#[path = "shell_adapter.rs"]
mod shell_adapter;

#[test]
fn migration_lifecycle_cli_private_ready_and_graceful_shutdown() {
    let case = Case::new(Behavior::Clean, vec![ready()]);
    let mut sentinel = Sentinel::new(&case);

    let output = case.run();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let audit = case.audit();
    let port = audit.arguments[3].parse::<u16>().unwrap();
    assert_ne!(port, 0);
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        format!("client readiness observed on port {port}\n")
    );
    assert!(case.native.join("handler").is_file());
    assert_eq!(audit.cwd, repository());
    assert!(!audit.environment.contains_key("HF_TOKEN"));
    assert!(!audit.environment.contains_key("TASK20_AMBIENT"));
    for key in [
        "HOME",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "MESH_LLM_CONFIG",
        "MESH_LLM_RUNTIME_ROOT",
        "MESH_LLM_NATIVE_RUNTIME_CACHE_DIR",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_RUNTIME_DIR",
        "TMPDIR",
        "TEMP",
        "TMP",
    ] {
        assert!(std::path::Path::new(&audit.environment[key]).starts_with(&case.state));
    }
    assert_eq!(
        std::path::Path::new(&audit.environment["MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR"]),
        case.native
    );
    assert!(sentinel.0.try_wait().unwrap().is_none());
    case.assert_removed();
}

macro_rules! success_case {
    ($name:ident, $records:expr) => {
        #[test]
        fn $name() {
            let case = Case::new(Behavior::Clean, $records);

            let output = case.run();

            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(case.native.join("handler").is_file());
            case.assert_removed();
        }
    };
}

success_case!(
    migration_lifecycle_cli_legacy_stderr,
    vec![record(
        Destination::Stderr,
        b"{\"message\":\"cLiEnT ReAdY\"}\n"
    )]
);
success_case!(
    migration_lifecycle_cli_suppressed_ready_is_raw,
    vec![record(
        Destination::Stdout,
        b"{\"role\":\"client\",\"event\":\"passive_mode\",\"status\":\"ready\",\"tokens\":0}\n"
    )]
);
success_case!(
    migration_lifecycle_cli_duplicate_last_wins,
    vec![record(
        Destination::Stdout,
        b"{\"message\":\"no\",\"message\":\"Client ready\"}\n"
    )]
);
success_case!(
    migration_lifecycle_cli_fragmented_crlf,
    vec![
        record(Destination::Stdout, b"{\"message\":\"Client "),
        record(Destination::Stdout, b"ready\"}\r\n")
    ]
);

macro_rules! failure_case {
    ($name:ident, $behavior:expr, $records:expr) => {
        #[test]
        fn $name() {
            let case = Case::new($behavior, $records);

            let output = case.run();

            assert!(!output.status.success());
            assert!(output.stdout.is_empty());
            case.assert_removed();
        }
    };
}

failure_case!(
    migration_lifecycle_cli_plain_text_fails,
    Behavior::Clean,
    vec![record(Destination::Stdout, b"Client ready\n")]
);
failure_case!(
    migration_lifecycle_cli_scalar_json_fails,
    Behavior::Clean,
    vec![record(Destination::Stdout, b"\"Client ready\"\n")]
);
failure_case!(
    migration_lifecycle_cli_duplicate_last_rejects,
    Behavior::Clean,
    vec![record(
        Destination::Stdout,
        b"{\"message\":\"Client ready\",\"message\":null}\n"
    )]
);
failure_case!(
    migration_lifecycle_cli_cross_stream_fragments_fail,
    Behavior::Clean,
    vec![
        record(Destination::Stdout, b"{\"message\":"),
        record(Destination::Stderr, b"\"Client ready\"}\n")
    ]
);
failure_case!(
    migration_lifecycle_cli_unterminated_open_fails,
    Behavior::Clean,
    vec![record(
        Destination::Stdout,
        b"{\"message\":\"Client ready\"}"
    )]
);
failure_case!(
    migration_lifecycle_cli_unterminated_eof_with_live_leader_fails,
    Behavior::UnterminatedEof,
    vec![record(
        Destination::Stdout,
        b"{\"message\":\"Client ready\"}"
    )]
);
failure_case!(
    migration_lifecycle_cli_immediate_zero_without_readiness_fails,
    Behavior::EarlyZero,
    vec![]
);
failure_case!(
    migration_lifecycle_cli_deadline_cannot_be_repaired,
    Behavior::LateReady,
    vec![]
);
failure_case!(
    migration_lifecycle_cli_cleanup_lf_cannot_complete_startup_record,
    Behavior::LateNewline,
    vec![record(
        Destination::Stdout,
        b"{\"message\":\"Client ready\"}"
    )]
);
failure_case!(
    migration_lifecycle_cli_stubborn_ready_fails,
    Behavior::Stubborn,
    vec![ready()]
);
failure_case!(
    migration_lifecycle_cli_nonzero_handler_fails,
    Behavior::Nonzero,
    vec![ready()]
);
failure_case!(
    migration_lifecycle_cli_occupied_selected_port_fails,
    Behavior::OccupiedPort,
    vec![ready()]
);
failure_case!(
    migration_lifecycle_cli_flood_is_bounded,
    Behavior::Flood,
    vec![]
);

#[test]
fn migration_lifecycle_cli_early_failure_diagnostics_are_sanitized() {
    let case = Case::new(Behavior::EarlyNonzero, vec![]);

    let output = case.run();

    assert!(!output.status.success());
    let diagnostics = String::from_utf8_lossy(&output.stderr);
    assert!(diagnostics.contains("fixture startup diagnostic"));
    assert!(!diagnostics.contains("synthetic-never-print"));
    assert!(output.stdout.is_empty());
    case.assert_removed();
}

#[test]
fn migration_lifecycle_cli_cleanup_descendant_cannot_supply_leader_readiness() {
    let case = Case::new(Behavior::DescendantAfterExit, vec![]);
    let mut sentinel = Sentinel::new(&case);

    let output = case.run();

    assert!(!output.status.success());
    assert!(case.native.join("leaf.handler").is_file());
    let leaf = std::fs::read_to_string(case.native.join("leaf.pid"))
        .unwrap()
        .parse()
        .unwrap();
    assert_absent(leaf);
    assert!(sentinel.0.try_wait().unwrap().is_none());
    case.assert_removed();
}

#[test]
fn migration_lifecycle_cli_deletion_failure_is_nonzero_after_clean_stop() {
    let case = Case::new(Behavior::StateDeletionFailure, vec![ready()]);

    let output = case.run();

    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(case.native.join("handler").is_file());
    assert_absent(case.audit().pid);
    assert_ne!(std::fs::read_dir(&case.state).unwrap().count(), 0);
}

#[test]
fn migration_lifecycle_cli_help_does_not_launch() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(repository())
        .args(["automation", "client-readiness", "--help"])
        .output()
        .unwrap();

    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("--native-runtime-root"));
}

#[test]
fn migration_lifecycle_cli_invalid_budget_does_not_launch() {
    let case = Case::new(Behavior::Clean, vec![ready()]);
    let mut arguments = case.arguments();
    arguments[5] = "0".into();
    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_xtask"));
    command
        .current_dir(repository())
        .args(["automation", "client-readiness"])
        .args(arguments);

    let output = command.output().unwrap();

    assert!(!output.status.success());
    assert!(!case.native.join("audit.json").exists());
    assert_eq!(std::fs::read_dir(&case.state).unwrap().count(), 0);
}
