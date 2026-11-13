use super::*;
use std::fs;

fn write_manifest(root: &Path) {
    let mut files = BTreeMap::new();
    for relative in [
        ".gitignore",
        "README.md",
        "ci/required-sdk-python/requirements.lock",
    ] {
        let path = root.join(relative);
        files.insert(relative, digest(&read(&path, MAX_FILE_BYTES).unwrap()));
    }
    let manifest = serde_json::json!({
        "schema_version": 1,
        "source_checkpoint": "0357f19f02cc3c2769ab46b0299af027e07049c1",
        "mesh_source": "e819fd4fe8d3d87517708faa0f2589c05aa48d80",
        "generator": {"uniffi": "0.32.0"},
        "files": files,
    });
    fs::write(root.join(MANIFEST), serde_json::to_vec(&manifest).unwrap()).unwrap();
}

fn fixture() -> tempfile::TempDir {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join(SDK_ROOT);
    fs::create_dir_all(root.join("ci/required-sdk-python")).unwrap();
    fs::write(root.join(".gitignore"), ".venv/\n").unwrap();
    fs::write(root.join("README.md"), "in-tree SDK fixture\n").unwrap();
    fs::write(
        root.join("ci/required-sdk-python/requirements.lock"),
        "openai==1.0.0 --hash=sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n",
    )
    .unwrap();
    write_manifest(&root);
    temporary
}

#[test]
fn local_sdk_manifest_admits_only_its_exact_source_roster_and_hashes() {
    let temporary = fixture();
    let mesh = temporary.path().canonicalize().unwrap();
    let source = admit(&mesh).unwrap();
    assert_eq!(source.root, mesh.join(SDK_ROOT));
    assert_eq!(source.pins.len(), 4);

    let ignored = source.root.join(".venv");
    fs::create_dir(&ignored).unwrap();
    fs::write(ignored.join("runtime-output"), "environment output").unwrap();
    fs::create_dir(source.root.join("dist")).unwrap();
    fs::write(source.root.join("dist/wheel.whl"), "build output").unwrap();
    assert!(admit(&mesh).is_ok());

    fs::write(source.root.join("README.md"), "changed").unwrap();
    assert!(admit(&mesh).is_err());
}

#[cfg(unix)]
#[test]
fn local_sdk_manifest_refuses_unlisted_files_and_symlink_substitution() {
    let temporary = fixture();
    let mesh = temporary.path().canonicalize().unwrap();
    let root = mesh.join(SDK_ROOT);
    fs::write(root.join("unlisted.py"), "new Python source\n").unwrap();
    assert!(admit(&mesh).is_err());
    fs::remove_file(root.join("unlisted.py")).unwrap();

    fs::remove_file(root.join("README.md")).unwrap();
    std::os::unix::fs::symlink(".gitignore", root.join("README.md")).unwrap();
    assert!(admit(&mesh).is_err());
}

#[test]
fn checked_in_sdk_snapshot_is_admitted_from_the_mesh_checkout() {
    let mesh = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let source = admit(&mesh).unwrap();
    assert!(source.root.join("src/meshllm/client.py").is_file());
    assert!(
        source
            .root
            .join("ci/required-sdk-python/requirements.lock")
            .is_file()
    );
    assert!(!source.root.join("sdk").exists());
}

#[test]
fn setup_uses_in_tree_sdk_and_no_external_checkout() {
    let setup = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../.github/actions/setup-canary-python/action.yml"
    ));
    assert!(setup.contains("sdk-source --kind root"));
    assert!(setup.contains("sdk-source --kind \"$SDK_KIND\""));
    assert!(!setup.contains("prepare-python-sdk-source"));

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
        assert!(!workflow.contains("prepare-python-sdk-source"));
    }
}

#[cfg(windows)]
#[test]
fn in_tree_sdk_relative_paths_keep_native_separators() {
    let temporary = fixture();
    let source = admit(&temporary.path().canonicalize().unwrap()).unwrap();
    let project = source_path(&source.root, "ci/canary-python").unwrap();
    assert!(!project.to_string_lossy().ends_with("/"));
}
