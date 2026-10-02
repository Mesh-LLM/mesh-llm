use super::PrivateState;
use crate::automation::client_readiness::Error;
use crate::process::Value;

#[test]
fn native_logs_are_retained_without_private_runtime_metadata() {
    let parent = tempfile::tempdir().unwrap();
    let state = PrivateState::create(parent.path(), "mlc-state").unwrap();
    state.prepare().unwrap();
    let logs = state.root.join("runtime/123/logs");
    std::fs::create_dir_all(&logs).unwrap();
    std::fs::write(logs.join("skippy-native.log"), b"native evidence").unwrap();
    std::fs::write(state.root.join("runtime/123/identity.json"), b"private").unwrap();
    let destination = parent.path().join("evidence");

    state.retain_runtime_logs(&destination).unwrap();
    state.finish(Ok::<_, Error>(())).unwrap();

    assert_eq!(
        std::fs::read(destination.join("123/logs/skippy-native.log")).unwrap(),
        b"native evidence"
    );
    assert!(!destination.join("123/identity.json").exists());
}

#[test]
fn model_fit_writes_recurrent_sizing_when_requested() {
    let parent = tempfile::tempdir().unwrap();
    let state = PrivateState::create(parent.path(), "mlc-state").unwrap();
    state.prepare().unwrap();

    state.model_fit(Some((128, 64))).unwrap();

    let config = std::fs::read_to_string(state.root.join("config.toml")).unwrap();
    assert!(config.contains("batch = 128\nubatch = 64"));
    state.finish(Ok::<_, Error>(())).unwrap();
}

#[test]
fn migration_lifecycle_deletion_failure_overrides_success() {
    let parent = tempfile::tempdir().unwrap();
    let state = PrivateState::create(parent.path(), "mlc-state").unwrap();
    std::fs::remove_dir(&state.root).unwrap();
    std::fs::write(&state.root, b"not a directory").unwrap();

    let result = state.finish(Ok::<_, Error>(())).map_err(Error::from);

    assert!(matches!(
        result,
        Err(Error::StateDeletion {
            preceding: None,
            ..
        })
    ));
}

#[test]
fn migration_lifecycle_deletion_failure_retains_original_failure() {
    let parent = tempfile::tempdir().unwrap();
    let state = PrivateState::create(parent.path(), "mlc-state").unwrap();
    std::fs::remove_dir(&state.root).unwrap();
    std::fs::write(&state.root, b"not a directory").unwrap();

    let result: Result<(), _> = state
        .finish(Err(Error::Invalid("fixture")))
        .map_err(Error::from);

    assert!(matches!(
        result,
        Err(Error::StateDeletion {
            preceding: Some(_),
            ..
        })
    ));
}

#[test]
fn migration_lifecycle_private_roots_are_removed_without_removing_sentinel() {
    let parent = tempfile::tempdir().unwrap();
    let sentinel = parent.path().join("sentinel");
    std::fs::write(&sentinel, b"unrelated").unwrap();
    let state = PrivateState::create(parent.path(), "mlc-state").unwrap();
    state.prepare().unwrap();

    let result = state.finish(Ok::<_, Error>(42));

    assert_eq!(result.unwrap(), 42);
    assert_eq!(std::fs::read_dir(parent.path()).unwrap().count(), 1);
    assert_eq!(std::fs::read(sentinel).unwrap(), b"unrelated");
}

#[test]
fn migration_lifecycle_child_environment_has_only_execution_inputs_and_private_roots() {
    let parent = tempfile::tempdir().unwrap();
    let state = PrivateState::create(parent.path(), "mlc-state").unwrap();
    state.prepare().unwrap();
    let native = parent.path().join("native");

    let environment = state.environment(&native);

    for (key, value) in environment {
        match value {
            Value::Secret(_) => assert!(
                [
                    "PATH",
                    "SYSTEMROOT",
                    "WINDIR",
                    "LD_LIBRARY_PATH",
                    "DYLD_LIBRARY_PATH",
                    "DYLD_FALLBACK_LIBRARY_PATH",
                ]
                .iter()
                .any(|allowed| key == *allowed)
            ),
            Value::Public(value) => {
                let path = std::path::Path::new(&value);
                assert!(
                    path.starts_with(&state.root)
                        || (key == "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR" && path == native)
                );
            }
        }
    }
    state.finish(Ok::<_, Error>(())).unwrap();
}

#[cfg(unix)]
#[test]
fn migration_lifecycle_private_directory_modes_restrict_other_users() {
    use std::os::unix::fs::PermissionsExt;
    let parent = tempfile::tempdir().unwrap();
    let state = PrivateState::create(parent.path(), "mlc-state").unwrap();

    state.prepare().unwrap();

    for name in [
        "",
        "home",
        "cache",
        "config",
        "xdg-runtime",
        "runtime-cache",
        "runtime",
        "tmp",
    ] {
        assert_eq!(
            std::fs::metadata(state.root.join(name))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o700
        );
    }
    state.finish(Ok::<_, Error>(())).unwrap();
}
