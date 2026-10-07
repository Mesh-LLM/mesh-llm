use super::*;
use std::{fs, os::unix::fs::PermissionsExt, path::PathBuf};

struct Fixture {
    owned: tempfile::TempDir,
    input: admission::Input,
}
impl Fixture {
    fn new() -> Self {
        Self::with_python_symlink(false)
    }
    fn with_python_symlink(symlink_python: bool) -> Self {
        let owned = tempfile::tempdir().unwrap();
        let root = owned.path().canonicalize().unwrap();
        fs::create_dir(root.join("scripts")).unwrap();
        fs::create_dir_all(root.join("ci/required-sdk-python")).unwrap();
        fs::write(
            root.join("scripts/ci-compat-smoke.sh"),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../scripts/ci-compat-smoke.sh"
            )),
        )
        .unwrap();
        for leaf in [
            "ci-openai-python-smoke.py",
            "ci-litellm-smoke.py",
            "ci-langchain-openai-smoke.py",
            "ci-openai-node-smoke.cjs",
        ] {
            fs::copy(
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../scripts")
                    .join(leaf),
                root.join("scripts").join(leaf),
            )
            .unwrap();
        }
        fs::copy(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../ci/required-sdk-python/requirements.lock"),
            root.join("ci/required-sdk-python/requirements.lock"),
        )
        .unwrap();
        for leaf in ["binary", "python", "node"] {
            if leaf == "python" && symlink_python {
                std::os::unix::fs::symlink("/bin/sh", root.join(leaf)).unwrap();
            } else {
                fs::copy("/bin/sh", root.join(leaf)).unwrap();
                fs::set_permissions(root.join(leaf), fs::Permissions::from_mode(0o700)).unwrap();
            }
        }
        fs::write(root.join("model"), b"inert fixture model").unwrap();
        fs::create_dir(root.join("native")).unwrap();
        fs::create_dir_all(root.join("modules/openai")).unwrap();
        fs::write(
            root.join("modules/openai/package.json"),
            br#"{"name":"openai"}"#,
        )
        .unwrap();
        let mut args = vec!["run".to_owned()];
        for (flag, leaf) in [
            ("--binary", "binary"),
            ("--python", "python"),
            ("--node", "node"),
            ("--model", "model"),
            ("--native-runtime-root", "native"),
            ("--node-modules", "modules"),
            ("--state-parent", "."),
            ("--output", "output"),
        ] {
            args.extend([flag.to_owned(), root.join(leaf).display().to_string()]);
        }
        args.extend(
            [
                "--device",
                "CUDA0",
                "--api-port",
                "19370",
                "--console-port",
                "19371",
                "--cuda-visible-devices",
                "GPU-6b7fe24c-5f15-4ac5-88d6-c8934135a4ea",
            ]
            .map(str::to_owned),
        );
        let input = admission::Input::admit(&root, &args).unwrap();
        Self { owned, input }
    }
}
fn limits() -> Limits {
    Limits {
        execution: Duration::from_secs(2),
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(1),
        retained_bytes_per_stream: 4096,
        readiness: Readiness::None,
        completion: Completion::Exit,
    }
}
#[test]
fn closed_child_home_is_private_and_uuid_is_redacted_without_parent_mutation() {
    let fixture = Fixture::new();
    let foreign = fixture.owned.path().join("foreign");
    fs::create_dir(&foreign).unwrap();
    fs::write(foreign.join("sentinel"), b"untouched").unwrap();
    let state = PrivateState::create(&fixture.input.parent, "sdk-test").unwrap();
    state.prepare().unwrap();
    let private = state.root().to_owned();
    let mut spec = specification(fixture.owned.path(), &fixture.input, &state, &private).unwrap();
    spec.executable = "/bin/sh".into();
    spec.arguments = vec![Value::Public("-c".into()), Value::Public("test \"$HOME\" = \"$1/home\" && test \"$CUDA_VISIBLE_DEVICES\" = \"$2\" && test -z \"${SDK_FOREIGN_SENTINEL+x}\" && mkdir \"$HOME/.pi\" && printf '%s' \"$CUDA_VISIBLE_DEVICES\"".into()), Value::Public("sdk-peer".into()), Value::Public(private.clone().into()), Value::Secret(fixture.input.cuda.as_ref().unwrap().into())];
    let report = process::supervise(
        &spec,
        &limits(),
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert!(report.success() && report.cleanup.complete);
    assert!(
        !String::from_utf8_lossy(&report.stdout.bytes_retained)
            .contains(fixture.input.cuda.as_ref().unwrap())
    );
    assert!(private.join("home/.pi").is_dir());
    state.finish(Ok::<_, ()>(())).unwrap();
    assert!(!private.exists());
    assert_eq!(fs::read(foreign.join("sentinel")).unwrap(), b"untouched");
}
#[test]
fn actual_admitted_sources_refuse_replacement_before_execution() {
    let fixture = Fixture::new();
    fs::write(&fixture.input.model, b"changed fixture").unwrap();
    assert!(fixture.input.custody().is_err());
    assert!(!fixture.input.output.exists());
    assert!(
        admission::Input::admit(
            fixture.owned.path(),
            &["run".into(), "--arbitrary-script".into(), "/bin/sh".into()]
        )
        .is_err()
    );
}
#[test]
fn owned_peer_timeout_still_reaps_and_private_state_is_removed() {
    let fixture = Fixture::new();
    let state = PrivateState::create(&fixture.input.parent, "sdk-test").unwrap();
    state.prepare().unwrap();
    let private = state.root().to_owned();
    let mut spec = specification(fixture.owned.path(), &fixture.input, &state, &private).unwrap();
    spec.executable = "/bin/sh".into();
    spec.arguments = vec![
        Value::Public("-c".into()),
        Value::Public("sleep 30 & wait".into()),
    ];
    let report = process::supervise(
        &spec,
        &limits(),
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert!(!report.success());
    assert!(report.cleanup.complete);
    state.finish(Ok::<_, ()>(())).unwrap();
    assert!(!private.exists());
}

#[test]
fn locked_interpreter_invocation_path_survives_symlink_admission() {
    let fixture = Fixture::with_python_symlink(true);
    assert_eq!(
        fixture.input.python,
        fixture.owned.path().canonicalize().unwrap().join("python")
    );
    assert_ne!(
        fixture.input.python,
        fixture.input.python.canonicalize().unwrap()
    );
    fixture.input.custody().unwrap();
    fs::remove_file(&fixture.input.python).unwrap();
    std::os::unix::fs::symlink("/bin/ls", &fixture.input.python).unwrap();
    assert!(fixture.input.custody().is_err());
}
