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
            Value::Secret(_) => assert_eq!(key, "CUDA_VISIBLE_DEVICES"),
            Value::Public(_)
                if [
                    "PATH",
                    "SYSTEMROOT",
                    "WINDIR",
                    "LD_LIBRARY_PATH",
                    "DYLD_LIBRARY_PATH",
                    "DYLD_FALLBACK_LIBRARY_PATH",
                ]
                .iter()
                .any(|host| key == *host) => {}
            Value::Public(value) => {
                let path = std::path::Path::new(&value);
                assert!(
                    path.starts_with(&state.root)
                        || (key == "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR" && path == native)
                        || (key == "CUDA_VISIBLE_DEVICES" && value.is_empty())
                );
            }
        }
    }
    state.finish(Ok::<_, Error>(())).unwrap();
}

#[cfg(unix)]
#[test]
fn runtime_visibility_reaches_closed_child_without_ambient_inheritance() {
    use crate::process::{
        self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness,
    };
    use std::collections::BTreeMap;
    use std::time::Duration;
    let owner = module_path!().split_once("::").unwrap().1;
    let peer = format!("{owner}::runtime_visibility_subprocess_peer");
    for selection in [
        Some("GPU-6b7fe24c-5f15-4ac5-88d6-c8934135a4ea"),
        Some(""),
        None,
    ] {
        let expected = selection.map_or_else(|| "unset:".to_owned(), |v| format!("set:{v}"));
        let directory = tempfile::tempdir().unwrap();
        let mut environment = super::host_environment().collect::<BTreeMap<_, _>>();
        environment.insert(
            "MESH_FIXTURE_VISIBILITY_EXPECTED".into(),
            Value::Public(expected.clone().into()),
        );
        environment.insert(
            "MESH_FIXTURE_UNRELATED_AMBIENT".into(),
            Value::Public("must-not-reach-runtime".into()),
        );
        if let Some(value) = selection {
            environment.insert("CUDA_VISIBLE_DEVICES".into(), Value::Public(value.into()));
        } else {
            environment.remove(std::ffi::OsStr::new("CUDA_VISIBLE_DEVICES"));
        }
        let spec = ProcessSpec {
            executable: std::env::current_exe().unwrap(),
            arguments: ["--exact", &peer, "--ignored", "--nocapture"]
                .into_iter()
                .map(|arg| Value::Public(arg.into()))
                .collect(),
            cwd: directory.path().into(),
            environment,
        };
        let output = process::supervise(
            &spec,
            &Limits {
                execution: Duration::from_secs(15),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(2),
                retained_bytes_per_stream: 8192,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            OutputFiles::default(),
        )
        .unwrap();
        assert!(
            output.success() && output.cleanup.complete && output.cleanup.failure.is_none(),
            "{output:?}"
        );
        assert!(!output.stdout.truncated && !output.stderr.truncated);
        assert!(String::from_utf8_lossy(&output.stdout.bytes_retained).contains("1 passed"));
    }
}

#[cfg(unix)]
#[test]
#[ignore = "native subprocess peer invoked by the normal visibility test"]
fn runtime_visibility_subprocess_peer() {
    use crate::process::{
        self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness,
    };
    use std::time::Duration;
    let expected = std::env::var("MESH_FIXTURE_VISIBILITY_EXPECTED").unwrap();
    let parent = tempfile::tempdir().unwrap();
    let state = PrivateState::create(parent.path(), "cuda-visibility").unwrap();
    state.prepare().unwrap();
    let native = parent.path().join("native");
    let environment = state.environment(&native);
    assert!(!environment.contains_key(std::ffi::OsStr::new("MESH_FIXTURE_UNRELATED_AMBIENT")));
    let spec = ProcessSpec {
        executable: "/bin/sh".into(),
        arguments: vec![
            Value::Public("-c".into()),
            Value::Public(
                "test -z \"${MESH_FIXTURE_UNRELATED_AMBIENT+x}\" || exit 1; if test \"${CUDA_VISIBLE_DEVICES+x}\" = x; then actual=\"set:$CUDA_VISIBLE_DEVICES\"; else actual='unset:'; fi; test \"$actual\" = \"$1\" || exit 2; printf matched".into(),
            ),
            Value::Public("cuda-visibility-peer".into()),
            Value::Secret(expected.into()),
        ],
        cwd: parent.path().into(),
        environment,
    };
    let report = process::supervise(
        &spec,
        &Limits {
            execution: Duration::from_secs(5),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 4096,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert!(report.success(), "{report:?}");
    assert!(report.cleanup.complete && report.cleanup.failure.is_none());
    assert_eq!(report.stdout.bytes_retained, b"matched");
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
