use super::*;
const COMMIT: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
fn cache(root: &Path, revision: &str) -> PathBuf {
    let repo = root.join("models--org--repo");
    let snapshot = repo.join("snapshots").join(COMMIT);
    fs::create_dir_all(&snapshot).unwrap();
    fs::write(snapshot.join("model-package.json"), b"{}").unwrap();
    let reference = repo.join("refs").join(revision);
    fs::create_dir_all(reference.parent().unwrap()).unwrap();
    fs::write(reference, COMMIT).unwrap();
    let newer = repo
        .join("snapshots")
        .join("ffffffffffffffffffffffffffffffffffffffff");
    fs::create_dir_all(newer).unwrap();
    snapshot
}
#[test]
fn exact_cache_binding_preserves_literal_revision_and_never_selects_latest() {
    for revision in [
        "main",
        "release=v2",
        "release/étape",
        "release#v2",
        "100%done",
    ] {
        let root = tempfile::tempdir().unwrap();
        let snapshot = cache(root.path(), revision);
        let reference = PackageReference::parse(&format!("hf://org/repo@{revision}")).unwrap();
        let bound = resolve(&reference, root.path()).unwrap();
        assert_eq!(bound.repo(), "org/repo");
        assert_eq!(bound.requested_revision(), revision);
        assert_eq!(bound.commit(), COMMIT);
        assert_eq!(bound.snapshot_path(), snapshot.canonicalize().unwrap());
        let missing = PackageReference::parse("hf://org/repo@missing").unwrap();
        assert!(resolve(&missing, root.path()).is_err());
    }
}
#[test]
fn immutable_commit_does_not_need_a_branch_ref() {
    let root = tempfile::tempdir().unwrap();
    cache(root.path(), "main");
    fs::remove_dir_all(root.path().join("models--org--repo/refs")).unwrap();
    let reference = PackageReference::parse(&format!("hf://org/repo@{COMMIT}")).unwrap();
    assert_eq!(resolve(&reference, root.path()).unwrap().commit(), COMMIT);
}
#[test]
fn malformed_ref_and_missing_manifest_refuse_before_success() {
    let root = tempfile::tempdir().unwrap();
    let snapshot = cache(root.path(), "main");
    let reference = PackageReference::parse("hf://org/repo@main").unwrap();
    let path = root.path().join("models--org--repo/refs/main");
    for value in ["../foreign", "/absolute", "short", "", "aa\nbb"] {
        fs::write(&path, value).unwrap();
        assert!(resolve(&reference, root.path()).is_err());
    }
    fs::write(&path, COMMIT).unwrap();
    fs::remove_file(snapshot.join("model-package.json")).unwrap();
    assert!(resolve(&reference, root.path()).is_err());
    assert!(
        resolve(
            &PackageReference::parse("hf://other/repo").unwrap(),
            root.path()
        )
        .is_err()
    );
}
#[test]
fn sdk_cache_only_itself_rejects_malformed_ref_without_snapshot_access() {
    let root = tempfile::tempdir().unwrap();
    cache(root.path(), "main");
    fs::write(
        root.path().join("models--org--repo/refs/main"),
        "../foreign",
    )
    .unwrap();
    let api = hf_hub::HFClientBuilder::new()
        .cache_dir(root.path())
        .endpoint("http://127.0.0.1:9")
        .token("fixture")
        .build()
        .unwrap();
    let sync = hf_hub::HFClientSync::from_inner(api).unwrap();
    let error = sync
        .model("org", "repo")
        .snapshot_download()
        .revision("main")
        .local_files_only(true)
        .send()
        .unwrap_err();
    assert!(matches!(error, hf_hub::HFError::InvalidParameter(_)));
}
#[cfg(unix)]
#[test]
fn ordinary_blob_manifest_pointer_allowed_but_foreign_and_directory_pointers_refuse() {
    use std::os::unix::fs::symlink;
    let root = tempfile::tempdir().unwrap();
    let snapshot = cache(root.path(), "main");
    let repo = root.path().join("models--org--repo");
    let blobs = repo.join("blobs");
    fs::create_dir(&blobs).unwrap();
    fs::write(blobs.join("metadata"), b"{}").unwrap();
    let manifest = snapshot.join("model-package.json");
    fs::remove_file(&manifest).unwrap();
    symlink("../../blobs/metadata", &manifest).unwrap();
    let reference = PackageReference::parse("hf://org/repo").unwrap();
    assert!(resolve(&reference, root.path()).is_ok());
    fs::remove_file(&manifest).unwrap();
    let outside = tempfile::tempdir().unwrap();
    fs::write(outside.path().join("foreign"), b"{}").unwrap();
    symlink(outside.path().join("foreign"), &manifest).unwrap();
    assert!(resolve(&reference, root.path()).is_err());
    fs::remove_file(&manifest).unwrap();
    fs::write(&manifest, b"{}").unwrap();
    fs::remove_dir_all(&snapshot).unwrap();
    symlink(outside.path(), &snapshot).unwrap();
    assert!(resolve(&reference, root.path()).is_err());
}
