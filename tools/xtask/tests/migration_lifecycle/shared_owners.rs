use super::command_interrupt::{Interrupt, Reason};
use super::private_state::{self, FinishError, PrivateState};
use crate::process::Value;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub(super) enum SecondError {
    Interrupt(Reason),
    State(private_state::Error),
}

pub(super) fn second_consumer(parent: &Path) -> Result<(Interrupt, PrivateState), SecondError> {
    let interrupt = Interrupt::install().map_err(SecondError::Interrupt)?;
    interrupt.check().map_err(SecondError::Interrupt)?;
    let state = PrivateState::create(parent, "mld-state").map_err(SecondError::State)?;
    state.prepare().map_err(SecondError::State)?;
    Ok((interrupt, state))
}

#[cfg(unix)]
pub(crate) fn assert_second_consumer_refuses_blocked_signal(signal: libc::c_int) {
    let parent = tempfile::tempdir().unwrap();
    let second = second_consumer(parent.path());
    assert!(
        matches!(second, Err(SecondError::Interrupt(reason)) if matches!((signal, &reason), (libc::SIGINT, Reason::BlockedSigint) | (libc::SIGTERM, Reason::BlockedSigterm)))
    );
    assert_eq!(std::fs::read_dir(parent.path()).unwrap().count(), 0);
}

const HOST_SEARCH_PATHS: &[&str] = &[
    "PATH",
    "SYSTEMROOT",
    "WINDIR",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "DYLD_FALLBACK_LIBRARY_PATH",
    "SYSTEMDRIVE",
    "PROGRAMDATA",
    "PROGRAMFILES",
    "COMSPEC",
    "PATHEXT",
    "COMPUTERNAME",
    "USERNAME",
    "USERDOMAIN",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
    "OS",
];

#[test]
fn second_consumer_refuses_overlap_before_state_creation() {
    if std::env::var_os("TASK20_SECOND_CONSUMER").is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "automation::shared_owner_tests::second_consumer_refuses_overlap_before_state_creation",
                "--nocapture",
            ])
            .env("TASK20_SECOND_CONSUMER", "1")
            .output()
            .unwrap();
        assert!(output.status.success(), "{output:?}");
        assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed; 0 failed"));
        return;
    }
    let parent = tempfile::tempdir().unwrap();
    let first = Interrupt::install().unwrap();
    first.cancellation().cancel();

    let result = second_consumer(parent.path());

    assert!(matches!(
        result,
        Err(SecondError::Interrupt(Reason::ScopeBusy))
    ));
    assert!(first.cancellation().is_cancelled());
    assert_eq!(std::fs::read_dir(parent.path()).unwrap().count(), 0);
    let arguments = vec![
        "--binary".into(),
        std::env::current_exe().unwrap().to_str().unwrap().into(),
        "--native-runtime-root".into(),
        parent.path().to_str().unwrap().into(),
        "--state-parent".into(),
        parent.path().to_str().unwrap().into(),
    ];
    let client = super::client_readiness::run(parent.path(), &arguments);
    assert!(matches!(
        client,
        Err(super::client_readiness::Error::Invalid(
            "client readiness signal scope already active"
        ))
    ));
    assert!(first.cancellation().is_cancelled());
    assert_eq!(std::fs::read_dir(parent.path()).unwrap().count(), 0);
    assert!(matches!(first.finish(), Err(Reason::Interrupted)));
    let (interrupt, state) = second_consumer(parent.path()).unwrap();
    state.finish(Ok::<_, SecondError>(())).unwrap();
    interrupt.finish().unwrap();
}

#[test]
fn second_consumer_prefix_and_environment_are_distinct() {
    let parent = tempfile::tempdir().unwrap();
    let stale = parent.path().join("mlc-state.stale");
    std::fs::create_dir(&stale).unwrap();
    let client = PrivateState::create(parent.path(), "mlc-state").unwrap();
    let daemon = PrivateState::create(parent.path(), "mld-state").unwrap();
    client.prepare().unwrap();
    daemon.prepare().unwrap();
    let native = parent.path().join("native");

    let client_environment = client.environment(&native);
    let daemon_environment = daemon.environment(&native);

    let client_home = public_path(&client_environment[std::ffi::OsStr::new("HOME")]);
    let daemon_home = public_path(&daemon_environment[std::ffi::OsStr::new("HOME")]);
    for (home, prefix) in [(&client_home, "mlc-state."), (&daemon_home, "mld-state.")] {
        let root = home.parent().unwrap();
        assert_eq!(root.parent(), Some(parent.path()));
        let suffix = root
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .strip_prefix(prefix)
            .unwrap();
        assert_eq!(suffix.len(), 32);
        assert!(suffix.bytes().all(|byte| byte.is_ascii_hexdigit()));
    }
    assert_ne!(client_home, daemon_home);
    assert_eq!(
        client_environment.keys().collect::<Vec<_>>(),
        daemon_environment.keys().collect::<Vec<_>>()
    );
    for (key, value) in &daemon_environment {
        match value {
            Value::Public(value) => assert!(
                Path::new(value).starts_with(daemon_home.parent().unwrap())
                    || (key == "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR" && Path::new(value) == native)
                    || (HOST_SEARCH_PATHS.contains(&key.to_str().unwrap())
                        && matches!(client_environment.get(key), Some(Value::Public(client_value)) if value == client_value))
            ),
            Value::Secret(value) => assert!(
                matches!(client_environment.get(key), Some(Value::Secret(client_value)) if value == client_value)
            ),
        }
    }
    daemon.finish(Ok::<_, SecondError>(())).unwrap();
    assert!(client_home.is_dir());
    assert!(stale.is_dir());
    client.finish(Ok::<_, SecondError>(())).unwrap();
    assert!(stale.is_dir());
}

fn public_path(value: &Value) -> PathBuf {
    match value {
        Value::Public(path) => PathBuf::from(path),
        Value::Secret(_) => panic!("private state paths must be explicit"),
    }
}

#[derive(Debug, PartialEq, Eq)]
struct DaemonFailure(u16);

#[test]
fn deletion_retains_typed_second_consumer_failure() {
    let parent = tempfile::tempdir().unwrap();
    let state = PrivateState::create(parent.path(), "mld-state").unwrap();
    let root = state
        .output_files()
        .stdout
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    std::fs::remove_dir(&root).unwrap();
    std::fs::write(&root, b"obstruction").unwrap();

    let result: Result<(), _> = state.finish(Err(DaemonFailure(503)));

    match result {
        Err(FinishError::Deletion {
            kind,
            code,
            preceding,
        }) => {
            assert_eq!(*preceding.unwrap(), DaemonFailure(503));
            let expected = std::fs::remove_dir_all(&root).unwrap_err();
            assert_eq!(kind, expected.kind());
            assert_eq!(code, expected.raw_os_error());
        }
        result => panic!("expected deletion failure, got {result:?}"),
    }
    assert_eq!(std::fs::read(root).unwrap(), b"obstruction");
}

#[test]
fn successful_deletion_passes_through_typed_second_consumer_failure() {
    let parent = tempfile::tempdir().unwrap();
    let state = PrivateState::create(parent.path(), "mld-state").unwrap();

    let result: Result<(), _> = state.finish(Err(DaemonFailure(409)));

    assert!(matches!(
        result,
        Err(FinishError::Prior(DaemonFailure(409)))
    ));
    assert_eq!(std::fs::read_dir(parent.path()).unwrap().count(), 0);
}

#[test]
fn second_consumer_retains_private_state_io_detail() {
    let parent = tempfile::tempdir().unwrap();
    let missing = parent.path().join("missing");

    let result = PrivateState::create(&missing, "mld-state").map_err(SecondError::State);

    assert!(matches!(
        result,
        Err(SecondError::State(private_state::Error::Io {
            operation: "create private directory",
            kind: std::io::ErrorKind::NotFound,
            code: Some(_)
        }))
    ));
}
