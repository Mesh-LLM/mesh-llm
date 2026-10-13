use super::super::immutable_sdk_checkout::git;
use super::*;
fn fixture() -> (tempfile::TempDir, Pin) {
    fixture_bytes(b"inert research source fixture")
}
fn fixture_bytes(client: &[u8]) -> (tempfile::TempDir, Pin) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    git(&root, &["init"]).unwrap();
    std::fs::write(root.join("client.py"), client).unwrap();
    std::fs::write(root.join(".gitignore"), b".venv/\n").unwrap();
    let manifest = serde_json::json!({"schema_version":1,"repository":REPOSITORY,"files":{"client.py":digest(client),".gitignore":digest(b".venv/\n")}});
    let bytes = serde_json::to_vec(&manifest).unwrap();
    std::fs::write(root.join("project-inputs.json"), &bytes).unwrap();
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
            repository: REPOSITORY.into(),
            revision,
            manifest_sha256: digest(&bytes),
        },
    )
}

fn roster_fixture(count: usize, padding: usize) -> (tempfile::TempDir, Pin) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    git(&root, &["init"]).unwrap();
    let mut files = BTreeMap::new();
    for index in 0..count {
        let name = format!("tokenizer-{index:04}{}.fixture", "x".repeat(padding));
        std::fs::write(root.join(&name), b"source data").unwrap();
        files.insert(name, digest(b"source data"));
    }
    let bytes = serde_json::to_vec(&serde_json::json!({
        "schema_version": 1, "repository": REPOSITORY, "files": files
    }))
    .unwrap();
    std::fs::write(root.join("project-inputs.json"), &bytes).unwrap();
    git(&root, &["add", "."]).unwrap();
    commit_roster(&root);
    let revision = String::from_utf8(git(&root, &["rev-parse", "HEAD"]).unwrap())
        .unwrap()
        .trim()
        .to_owned();
    (
        dir,
        Pin {
            schema_version: 1,
            repository: REPOSITORY.into(),
            revision,
            manifest_sha256: digest(&bytes),
        },
    )
}

fn commit_roster(root: &Path) {
    git(
        root,
        &[
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "roster",
        ],
    )
    .unwrap();
}

#[test]
fn research_raw_roster_admits_398_paths_including_redacted_diagnostic_names() {
    let (dir, pin) = roster_fixture(397, 0);
    let root = dir.path().canonicalize().unwrap();
    // The SDK diagnostic-only reader still refuses transformed source output.
    assert!(git(&root, &["ls-files"]).is_err());
    assert_eq!(admit_pin(&root, &pin).unwrap(), root);
    std::fs::write(root.join("tokenizer-0000.fixture"), b"tampered data").unwrap();
    assert!(admit_pin(&root, &pin).is_err());
}

#[test]
fn research_raw_roster_refuses_extra_tracked_paths_and_byte_overflow() {
    let (dir, mut pin) = roster_fixture(397, 0);
    let root = dir.path().canonicalize().unwrap();
    std::fs::write(root.join("unbound.fixture"), b"unbound data").unwrap();
    git(&root, &["add", "unbound.fixture"]).unwrap();
    commit_roster(&root);
    pin.revision = String::from_utf8(git(&root, &["rev-parse", "HEAD"]).unwrap())
        .unwrap()
        .trim()
        .to_owned();
    assert!(
        admit_pin(&root, &pin)
            .unwrap_err()
            .to_string()
            .contains("tracked roster refused")
    );
    // Every individual path is legal and short; the complete raw roster exceeds 64 KiB.
    let (dir, pin) = roster_fixture(397, 180);
    assert!(admit_pin(&dir.path().canonicalize().unwrap(), &pin).is_err());
}

#[test]
fn research_raw_roster_preserves_the_finite_path_count_bound() {
    let (dir, _) = fixture();
    let expected = (0..4097)
        .map(|index| (format!("source-{index}.fixture"), digest(b"source")))
        .collect();
    assert!(
        immutable_sdk_checkout::research_files(&dir.path().canonicalize().unwrap(), &expected)
            .unwrap_err()
            .to_string()
            .contains("tracked roster bound refused")
    );
}
#[test]
fn research_source_preserves_ignored_preparation_outputs_and_rejects_source_or_commit_drift() {
    let (dir, pin) = fixture();
    let root = dir.path().canonicalize().unwrap();
    assert_eq!(admit_pin(&root, &pin).unwrap(), root);
    std::fs::create_dir(root.join(".venv")).unwrap();
    std::fs::write(root.join(".venv/output"), "prepared environment").unwrap();
    assert!(admit_pin(&root, &pin).is_ok());
    std::fs::write(root.join("client.py"), "changed source").unwrap();
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
fn research_source_refuses_manifest_roster_and_untracked_shadowing() {
    let (dir, mut pin) = fixture();
    let root = dir.path().canonicalize().unwrap();
    pin.manifest_sha256 = "0".repeat(64);
    assert!(admit_pin(&root, &pin).is_err());
    pin.manifest_sha256 = digest(&std::fs::read(root.join("project-inputs.json")).unwrap());
    std::fs::write(root.join("sitecustomize.py"), "unbound source").unwrap();
    assert!(admit_pin(&root, &pin).is_err());
    git(&root, &["add", "sitecustomize.py"]).unwrap();
    assert!(admit_pin(&root, &pin).is_err());
    git(
        &root,
        &[
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-m",
            "unbound tracked source",
        ],
    )
    .unwrap();
    pin.revision = String::from_utf8(git(&root, &["rev-parse", "HEAD"]).unwrap())
        .unwrap()
        .trim()
        .to_owned();
    assert!(
        admit_pin(&root, &pin)
            .unwrap_err()
            .to_string()
            .contains("tracked roster refused")
    );
}
#[test]
fn research_source_link_substitution_refused() {
    let (dir, pin) = fixture();
    let root = dir.path().canonicalize().unwrap();
    assert!(admit_pin(&root, &pin).is_ok());
    std::fs::remove_file(root.join("client.py")).unwrap();
    std::os::unix::fs::symlink("project-inputs.json", root.join("client.py")).unwrap();
    assert!(
        admit_pin(&root, &pin)
            .unwrap_err()
            .to_string()
            .contains("file custody refused")
    );
}

#[test]
fn research_vocab_bound_admits_large_pinned_source_and_refuses_tampering_and_oversize() {
    let bytes = vec![b'v'; 5 * 1048576];
    let (dir, pin) = fixture_bytes(&bytes);
    let root = dir.path().canonicalize().unwrap();
    assert!(admit_pin(&root, &pin).is_ok());
    let path = root.join("client.py");
    let mut changed = bytes;
    changed[0] = b'x';
    std::fs::write(&path, changed).unwrap();
    assert!(admit_pin(&root, &pin).is_err());
    let (oversize, pin) = fixture_bytes(&vec![b'v'; 16 * 1048576 + 1]);
    assert!(admit_pin(&oversize.path().canonicalize().unwrap(), &pin).is_err());
}

#[test]
fn preparation_and_existing_test_cadence_use_the_compiled_research_pin() {
    let pin: Pin = serde_json::from_slice(include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../ci/python-research-source.json"
    )))
    .unwrap();
    let action = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../.github/actions/prepare-python-research-source/action.yml"
    ));
    assert_eq!(pin.repository, REPOSITORY);
    assert!(action.contains(&format!("repository: {}", pin.repository)));
    assert!(action.contains(&format!("ref: {}", pin.revision)));
    assert!(action.contains("persist-credentials: false"));
    assert!(action.contains("research-source --kind root"));
    assert!(action.find("research_root=\"").unwrap() < action.find(">> \"$GITHUB_ENV\"").unwrap());
    let workflow = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../.github/workflows/ci-rust-tests-slice.yml"
    ));
    let preparation = workflow
        .find("uses: ./.github/actions/prepare-python-research-source")
        .unwrap();
    assert!(workflow[..preparation].contains(
        "if: ${{ contains(fromJson(steps.resolve_batch_crates.outputs.crates), 'skippy-bench') }}"
    ));
    assert!(
        preparation
            < workflow
                .find("name: Run isolated Cargo tests for the batch")
                .unwrap()
    );
    let recipe = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../just/ci.just"));
    assert!(
        recipe
            .find("automation smoke-observation research-source --kind root")
            .unwrap()
            < recipe.find("echo \"=== 0/11 Test-all").unwrap()
    );
}
