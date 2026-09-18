use super::*;
use std::path::PathBuf;

fn hf_identity(repo_id: &str, revision: &str) -> model_hf::store::local::HuggingFaceModelIdentity {
    model_hf::store::local::HuggingFaceModelIdentity {
        repo_id: repo_id.to_string(),
        revision: revision.to_string(),
        file: "model-00001-of-00002.gguf".to_string(),
        canonical_ref: format!("{repo_id}@{revision}/model-00001-of-00002.gguf"),
        local_file_name: "model-00001-of-00002.gguf".to_string(),
    }
}

fn hf_identity_files() -> Vec<HuggingFaceGgufIdentityFile> {
    vec![
        HuggingFaceGgufIdentityFile {
            path: "model-00001-of-00002.gguf".to_string(),
            bytes: 100,
            sha256: "a".repeat(64),
        },
        HuggingFaceGgufIdentityFile {
            path: "model-00002-of-00002.gguf".to_string(),
            bytes: 200,
            sha256: "b".repeat(64),
        },
    ]
}

#[test]
fn hugging_face_identity_is_stable_across_local_cache_paths() {
    let first_dir = tempfile::tempdir().unwrap();
    let second_dir = tempfile::tempdir().unwrap();
    let first_path = first_dir.path().join("model.gguf");
    let second_path = second_dir.path().join("model.gguf");
    write_test_metadata_gguf(&first_path, 4096);
    std::fs::copy(&first_path, &second_path).unwrap();
    let identity = hf_identity("owner/model", &"c".repeat(40));
    let files = hf_identity_files();
    let identity_sha256 = huggingface_identity_sha256(&identity, &files).unwrap();
    let bytes = first_path.metadata().unwrap().len();
    let source_file = |path| SkippyPackageSourceFile {
        path,
        bytes,
        sha256: "a".repeat(64),
    };
    let first = super::super::synthetic_gguf_package_from_source_files(
        vec![source_file(first_path)],
        None,
        Some(identity_sha256.clone()),
        true,
    )
    .unwrap();
    let second = super::super::synthetic_gguf_package_from_source_files(
        vec![source_file(second_path)],
        None,
        Some(identity_sha256),
        true,
    )
    .unwrap();

    assert_eq!(first.package_ref, second.package_ref);
    assert_eq!(first.manifest_sha256, second.manifest_sha256);
    assert_eq!(first.source_model_sha256, second.source_model_sha256);
    assert_ne!(first.source_model_path, second.source_model_path);
    assert_eq!(
        crate::source_registry::verify_registered_content_source(
            "owner/model",
            &first.package_ref,
            &first.manifest_sha256,
            &first.source_model_sha256,
        )
        .unwrap()
        .package_ref,
        first.package_ref
    );
}

#[test]
fn hugging_face_identity_changes_with_commit_or_ordered_file_set() {
    let first = hf_identity("owner/model", &"c".repeat(40));
    let second = hf_identity("owner/model", &"d".repeat(40));
    let files = hf_identity_files();
    let first_digest = huggingface_identity_sha256(&first, &files).unwrap();

    assert_ne!(
        first_digest,
        huggingface_identity_sha256(&second, &files).unwrap()
    );

    let mut reordered = files.clone();
    reordered.reverse();
    assert_ne!(
        first_digest,
        huggingface_identity_sha256(&first, &reordered).unwrap()
    );

    let mut renamed = files;
    renamed[0].path = "other-00001-of-00002.gguf".to_string();
    assert_ne!(
        first_digest,
        huggingface_identity_sha256(&first, &renamed).unwrap()
    );
}

fn push_test_gguf_string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as i64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn write_test_metadata_gguf(path: &Path, context_length: u32) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0i64.to_le_bytes());
    bytes.extend_from_slice(&4i64.to_le_bytes());
    push_test_gguf_string(&mut bytes, "general.architecture");
    bytes.extend_from_slice(&8u32.to_le_bytes());
    push_test_gguf_string(&mut bytes, "llama");
    for (key, value) in [
        ("llama.block_count", 2),
        ("llama.embedding_length", 128),
        ("llama.context_length", context_length),
    ] {
        push_test_gguf_string(&mut bytes, key);
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, bytes).unwrap();
}

#[cfg(unix)]
#[test]
fn immutable_hf_source_uses_supplied_snapshot_without_constructing_client() {
    use std::os::unix::fs::symlink;
    let cache = tempfile::tempdir().unwrap();
    let repo = cache.path().join("models--owner--model");
    let snapshot = repo.join("snapshots").join("c".repeat(40));
    let blobs = repo.join("blobs");
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::create_dir_all(&blobs).unwrap();
    let payload = blobs.join("payload");
    write_test_metadata_gguf(&payload, 4096);
    let digest = crate::package::file_sha256(&payload).unwrap();
    let blob = blobs.join(&digest);
    std::fs::rename(&payload, &blob).unwrap();
    let mut identity = hf_identity("owner/model", &"c".repeat(40));
    identity.file = "model.gguf".to_string();
    symlink(&blob, snapshot.join(&identity.file)).unwrap();
    let result =
        super::super::synthetic_huggingface_gguf_package("model", &identity, &snapshot, || {
            anyhow::bail!("cached immutable source must not construct an HF client")
        })
        .unwrap();
    assert_eq!(result.source_files[0].sha256, digest);
    assert_eq!(result.layer_count, 2);
    assert_eq!(result.source_model_path, blob.canonicalize().unwrap());
    assert_eq!(
        crate::source_registry::verify_registered_content_source(
            "model",
            &result.package_ref,
            &result.manifest_sha256,
            &result.source_model_sha256,
        )
        .unwrap(),
        result
    );
}

#[test]
fn identity_ignores_alias_and_absolute_path() {
    let first_dir = tempfile::tempdir().unwrap();
    let second_dir = tempfile::tempdir().unwrap();
    let first_path = first_dir.path().join("first-name.gguf");
    let second_path = second_dir.path().join("other-name.gguf");
    write_test_metadata_gguf(&first_path, 4096);
    std::fs::copy(&first_path, &second_path).unwrap();

    let first = synthetic_content_addressed_gguf_package("alias-a", &first_path).unwrap();
    let second = synthetic_content_addressed_gguf_package("alias-b", &second_path).unwrap();

    assert_eq!(first.package_ref, second.package_ref);
    assert_eq!(first.manifest_sha256, second.manifest_sha256);
    assert_eq!(first.source_model_sha256, second.source_model_sha256);
    assert_ne!(first.source_model_path, second.source_model_path);
    assert!(crate::source_registry::is_content_addressed_gguf_ref(
        &first.package_ref
    ));
}

#[test]
fn identity_changes_with_source_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let first_path = dir.path().join("first.gguf");
    let second_path = dir.path().join("second.gguf");
    write_test_metadata_gguf(&first_path, 4096);
    write_test_metadata_gguf(&second_path, 8192);

    let first = synthetic_content_addressed_gguf_package("same-alias", &first_path).unwrap();
    let second = synthetic_content_addressed_gguf_package("same-alias", &second_path).unwrap();

    assert_ne!(first.package_ref, second.package_ref);
    assert_ne!(first.manifest_sha256, second.manifest_sha256);
    assert_ne!(first.source_model_sha256, second.source_model_sha256);
}

#[test]
fn split_digest_is_path_independent_and_order_sensitive() {
    let first = vec![
        SkippyPackageSourceFile {
            path: PathBuf::from("/node-a/model-00001-of-00002.gguf"),
            bytes: 10,
            sha256: "a".repeat(64),
        },
        SkippyPackageSourceFile {
            path: PathBuf::from("/node-a/model-00002-of-00002.gguf"),
            bytes: 20,
            sha256: "b".repeat(64),
        },
    ];
    let relocated = vec![
        SkippyPackageSourceFile {
            path: PathBuf::from("/other/first.gguf"),
            ..first[0].clone()
        },
        SkippyPackageSourceFile {
            path: PathBuf::from("/other/second.gguf"),
            ..first[1].clone()
        },
    ];
    let reversed = vec![first[1].clone(), first[0].clone()];

    assert_eq!(
        aggregate_source_sha256(&first),
        aggregate_source_sha256(&relocated)
    );
    assert_ne!(
        aggregate_source_sha256(&first),
        aggregate_source_sha256(&reversed)
    );
}

#[test]
fn split_identity_matches_across_paths_and_filename_prefixes() {
    let first_dir = tempfile::tempdir().unwrap();
    let second_dir = tempfile::tempdir().unwrap();
    let first_primary = first_dir.path().join("alpha-00001-of-00002.gguf");
    let first_secondary = first_dir.path().join("alpha-00002-of-00002.gguf");
    let second_primary = second_dir.path().join("beta-00001-of-00002.gguf");
    let second_secondary = second_dir.path().join("beta-00002-of-00002.gguf");
    write_test_metadata_gguf(&first_primary, 4096);
    write_test_metadata_gguf(&first_secondary, 4096);
    std::fs::copy(&first_primary, &second_primary).unwrap();
    std::fs::copy(&first_secondary, &second_secondary).unwrap();

    let first = synthetic_content_addressed_gguf_package("logical/model", &first_primary).unwrap();
    let second =
        synthetic_content_addressed_gguf_package("logical/model", &second_primary).unwrap();

    assert_eq!(first.source_files.len(), 2);
    assert_eq!(second.source_files.len(), 2);
    assert_eq!(first.package_ref, second.package_ref);
    assert_eq!(first.manifest_sha256, second.manifest_sha256);
    assert_eq!(first.source_model_sha256, second.source_model_sha256);
    assert_ne!(first.source_model_path, second.source_model_path);
}

#[test]
fn identity_requires_an_absolute_path() {
    let error =
        synthetic_content_addressed_gguf_package("logical/model", Path::new("relative-model.gguf"))
            .unwrap_err()
            .to_string();

    assert!(error.contains("must be absolute"));
}

#[cfg(unix)]
#[test]
fn identity_rejects_user_supplied_symlink() {
    use std::os::unix::fs::symlink;

    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("model-target.gguf");
    let link = dir.path().join("model.gguf");
    write_test_metadata_gguf(&target, 4096);
    symlink(&target, &link).unwrap();

    let error = synthetic_content_addressed_gguf_package("logical/model", &link)
        .unwrap_err()
        .to_string();

    assert!(error.contains("non-symlink file"));
    assert!(error.contains("model.gguf"));
}

#[cfg(unix)]
#[test]
fn split_rejects_symlinked_secondary_shard() {
    use std::os::unix::fs::symlink;

    let dir = tempfile::tempdir().unwrap();
    let primary = dir.path().join("model-00001-of-00002.gguf");
    let secondary = dir.path().join("model-00002-of-00002.gguf");
    let target = dir.path().join("secondary-target.gguf");
    write_test_metadata_gguf(&primary, 4096);
    write_test_metadata_gguf(&target, 4096);
    symlink(&target, &secondary).unwrap();

    let error = synthetic_content_addressed_gguf_package("logical/model", &primary)
        .unwrap_err()
        .to_string();

    assert!(error.contains("non-symlink file"));
    assert!(error.contains("00002-of-00002"));
}

#[cfg(unix)]
#[test]
fn split_accepts_hugging_face_snapshot_links_and_builds_regular_named_view() {
    use std::os::unix::fs::symlink;

    let cache = tempfile::tempdir().unwrap();
    let repo = cache.path().join("models--example--model");
    let blobs = repo.join("blobs");
    let snapshot = repo.join("snapshots").join("revision");
    std::fs::create_dir_all(&blobs).unwrap();
    std::fs::create_dir_all(&snapshot).unwrap();

    let first_blob = blobs.join("first-blob");
    let second_blob = blobs.join("second-blob");
    write_test_metadata_gguf(&first_blob, 4096);
    write_test_metadata_gguf(&second_blob, 4096);

    let first = snapshot.join("model-00001-of-00002.gguf");
    let second = snapshot.join("model-00002-of-00002.gguf");
    symlink(Path::new("../../blobs/first-blob"), &first).unwrap();
    symlink(Path::new("../../blobs/second-blob"), &second).unwrap();

    validate_source_set(&first).unwrap();
    let source_paths = super::super::direct_gguf_source_paths(&first).unwrap();

    assert_eq!(
        source_paths
            .iter()
            .filter_map(|path| path.file_name())
            .collect::<Vec<_>>(),
        vec![first.file_name().unwrap(), second.file_name().unwrap()]
    );
    assert!(source_paths.iter().all(|path| {
        let metadata = std::fs::symlink_metadata(path).unwrap();
        metadata.is_file() && !metadata.file_type().is_symlink()
    }));
    assert!(same_file::is_same_file(&first_blob, &source_paths[0]).unwrap());
    assert!(same_file::is_same_file(&second_blob, &source_paths[1]).unwrap());

    let reused = super::super::direct_gguf_source_paths(&first).unwrap();
    assert_eq!(reused, source_paths);

    std::fs::remove_file(&source_paths[1]).unwrap();
    std::fs::write(&source_paths[1], b"tampered").unwrap();
    let error = super::super::direct_gguf_source_paths(&first)
        .unwrap_err()
        .to_string();
    assert!(error.contains("does not reference its verified Hugging Face blob"));
}

#[cfg(unix)]
#[test]
fn hugging_face_snapshot_link_rejects_blob_store_escape() {
    use std::os::unix::fs::symlink;

    let cache = tempfile::tempdir().unwrap();
    let repo = cache.path().join("models--example--model");
    let snapshot = repo.join("snapshots").join("revision");
    std::fs::create_dir_all(repo.join("blobs")).unwrap();
    std::fs::create_dir_all(&snapshot).unwrap();

    let outside = cache.path().join("outside.gguf");
    let model = snapshot.join("model.gguf");
    write_test_metadata_gguf(&outside, 4096);
    symlink(&outside, &model).unwrap();

    let error = validate_source_set(&model).unwrap_err().to_string();

    assert!(error.contains("escapes its blob store"), "{error}");
}

// APFS rejects invalid UTF-8 path bytes at creation time; Linux permits
// them and therefore exercises the canonical-parent edge directly.
#[cfg(target_os = "linux")]
#[test]
fn identity_rejects_non_utf8_canonical_parent() {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;
    use std::os::unix::fs::symlink;

    let dir = tempfile::tempdir().unwrap();
    let target_dir = dir.path().join(OsString::from_vec(vec![b'm', 0xff]));
    std::fs::create_dir(&target_dir).unwrap();
    let target_model = target_dir.join("model.gguf");
    write_test_metadata_gguf(&target_model, 4096);
    let utf8_parent = dir.path().join("models");
    symlink(&target_dir, &utf8_parent).unwrap();

    let error =
        synthetic_content_addressed_gguf_package("logical/model", &utf8_parent.join("model.gguf"))
            .unwrap_err()
            .to_string();

    assert!(error.contains("canonical content-addressed GGUF path"));
    assert!(error.contains("valid UTF-8"));
}
