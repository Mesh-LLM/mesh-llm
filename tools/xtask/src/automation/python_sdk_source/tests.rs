use super::*;
fn fixture() -> (tempfile::TempDir, Pin) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    git(&root, &["init"]).unwrap();
    std::fs::write(root.join("client.py"), "inert SDK source fixture").unwrap();
    std::fs::write(root.join(".gitignore"), ".venv/\n").unwrap();
    let value = serde_json::json!({"schema_version":1,"repository":"Mesh-LLM/mesh-llm-python-sdk","mesh_source":"fixture", "generator":{"uniffi":"0.32.0"},"files":{"client.py":digest(b"inert SDK source fixture"),".gitignore":digest(b".venv/\n")}});
    let bytes = serde_json::to_vec(&value).unwrap();
    std::fs::write(root.join("sdk-inputs.json"), &bytes).unwrap();
    git(&root, &["add", "."]).unwrap();
    git(
        &root,
        &[
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-m",
            "fixture",
        ],
    )
    .unwrap();
    let revision = String::from_utf8(git(&root, &["rev-parse", "HEAD"]).unwrap())
        .unwrap()
        .trim()
        .to_owned();
    (
        dir,
        Pin {
            schema_version: 1,
            repository: "Mesh-LLM/mesh-llm-python-sdk".into(),
            revision,
            manifest_sha256: digest(&bytes),
        },
    )
}
#[test]
fn immutable_sdk_source_accepts_environment_outputs_but_refuses_source_and_commit_drift() {
    let (dir, pin) = fixture();
    let root = dir.path().canonicalize().unwrap();
    assert!(admit_pin(&root, &pin).is_ok());
    std::fs::create_dir(root.join(".venv")).unwrap();
    std::fs::write(root.join(".venv/output"), "untracked environment").unwrap();
    assert!(admit_pin(&root, &pin).is_ok());
    std::fs::write(root.join("openai.py"), "unbound source").unwrap();
    assert!(admit_pin(&root, &pin).is_err());
    std::fs::remove_file(root.join("openai.py")).unwrap();
    std::fs::write(root.join("client.py"), "changed").unwrap();
    assert!(admit_pin(&root, &pin).is_err());
    git(&root, &["add", "client.py"]).unwrap();
    git(
        &root,
        &[
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-m",
            "changed",
        ],
    )
    .unwrap();
    assert!(admit_pin(&root, &pin).is_err());
}
#[test]
fn manifest_digest_roster_and_symlink_substitution_fail_closed() {
    let (dir, mut pin) = fixture();
    let root = dir.path().canonicalize().unwrap();
    pin.manifest_sha256 = "0".repeat(64);
    assert!(admit_pin(&root, &pin).is_err());
    let bytes = std::fs::read(root.join("sdk-inputs.json")).unwrap();
    pin.manifest_sha256 = digest(&bytes);
    std::fs::write(root.join("unbound.py"), "inert unbound fixture").unwrap();
    git(&root, &["add", "unbound.py"]).unwrap();
    assert!(admit_pin(&root, &pin).is_err());
    git(&root, &["reset", "--", "unbound.py"]).unwrap();
    std::fs::remove_file(root.join("unbound.py")).unwrap();
    std::fs::remove_file(root.join("client.py")).unwrap();
    std::os::unix::fs::symlink("sdk-inputs.json", root.join("client.py")).unwrap();
    assert!(admit_pin(&root, &pin).is_err());
}
#[test]
fn setup_restore_and_runtime_share_the_compiled_external_pin() {
    let pin: Pin = serde_json::from_slice(include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../ci/required-sdk-python/sdk-source.json"
    )))
    .unwrap();
    let action = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../.github/actions/prepare-python-sdk-source/action.yml"
    ));
    assert!(action.contains(&format!("repository: {}", pin.repository)));
    assert!(action.contains(&format!("ref: {}", pin.revision)));
    assert!(!action.contains("-I -c"));
    let setup = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../.github/actions/setup-canary-python/action.yml"
    ));
    assert!(setup.contains("./.github/actions/prepare-python-sdk-source"));
    assert!(!setup.contains("-I -c"));
    assert!(action.contains("sdk-source --kind root"));
    for workflow in [
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../.github/workflows/ci-quality-slice.yml"
        )),
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../.github/workflows/sdk-smoke.yml"
        )),
    ] {
        assert!(workflow.contains("./.github/actions/prepare-python-sdk-source"));
    }
}

#[test]
fn physical_source_admission_does_not_execute_local_git_clean_filters() {
    let (dir, pin) = fixture();
    let root = dir.path().canonicalize().unwrap();
    let sentinel = root.join("filter-executed");
    std::fs::write(
        root.join(".git/info/attributes"),
        "client.py filter=poison\n",
    )
    .unwrap();
    git(
        &root,
        &[
            "config",
            "filter.poison.clean",
            &format!("/bin/sh -c 'touch \"{}\"; cat'", sentinel.display()),
        ],
    )
    .unwrap();
    assert!(admit_pin(&root, &pin).is_ok());
    assert!(!sentinel.exists());
}
