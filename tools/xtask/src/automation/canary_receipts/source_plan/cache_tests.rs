use super::admission::verify;
use super::contract::{CacheAdmission, CachePolicy};
use crate::automation::canary_receipts::Digest;
use serde_json::json;

#[test]
fn regular_snapshots_admit_target_draft_and_projector_without_a_blob_directory() {
    let state = tempfile::tempdir().unwrap();
    let root = state.path().join("hub");
    let revision = "a".repeat(40);
    let snapshot = root
        .join("models--fixture--model/snapshots")
        .join(&revision);
    std::fs::create_dir_all(&snapshot).unwrap();
    let mut target = super::gguf_tests::fixture(
        "llama",
        &[("llama.block_count", 32), ("llama.embedding_length", 2048)],
    );
    target.extend_from_slice(b"weights");
    let mut artifacts = Vec::new();
    for (name, bytes) in [
        ("target.gguf", target.clone()),
        ("draft.gguf", target),
        ("projector.gguf", b"projector sidecar".to_vec()),
    ] {
        std::fs::write(snapshot.join(name), &bytes).unwrap();
        artifacts.push(json!({"repo":"fixture/model","revision":revision,"files":[name],"file_integrity":{name:{"size_bytes":bytes.len(),"blob_id":Digest::of_bytes(&bytes).as_str()}}}));
    }
    let mut plan = json!({"selected_models":[{"architecture":"llama","execution":{"layer_end":32,"activation_width":2048,"mtp_layers":0},"artifact":artifacts[0],"draft_artifact":artifacts[1],"mmproj_artifact":artifacts[2]}]});
    let bytes = serde_json::to_vec(&plan).unwrap();
    let policy = CachePolicy::GgufMetadata { root: root.clone() };
    assert!(matches!(
        verify(&bytes, &policy).unwrap(),
        CacheAdmission::GgufMetadata
    ));
    for name in ["target.gguf", "draft.gguf", "projector.gguf"] {
        let path = snapshot.join(name);
        let original = std::fs::read(&path).unwrap();
        let mut corrupt = original.clone();
        *corrupt.last_mut().unwrap() ^= 1;
        std::fs::write(&path, corrupt).unwrap();
        assert!(
            verify(&bytes, &policy)
                .err()
                .unwrap()
                .to_string()
                .contains("SHA-256")
        );
        std::fs::write(path, original).unwrap();
    }
    plan["selected_models"][0]["architecture"] = json!("different");
    assert!(
        verify(&serde_json::to_vec(&plan).unwrap(), &policy)
            .err()
            .unwrap()
            .to_string()
            .contains("disagrees with planned")
    );
}

#[cfg(unix)]
#[test]
fn layout_independent_admission_rejects_escaped_paths_and_special_entries() {
    let state = tempfile::tempdir().unwrap();
    let root = state.path().join("hub");
    let revision = "a".repeat(40);
    let snapshot = root
        .join("models--fixture--model/snapshots")
        .join(&revision);
    std::fs::create_dir_all(&snapshot).unwrap();
    let outside = state.path().join("outside.gguf");
    std::fs::write(&outside, b"good").unwrap();
    let entry = snapshot.join("target.gguf");
    std::os::unix::fs::symlink(&outside, &entry).unwrap();
    let plan = json!({"selected_models":[{"architecture":"llama","execution":{"layer_end":32,"activation_width":2048,"mtp_layers":0},"artifact":{"repo":"fixture/model","revision":revision,"files":["target.gguf"],"file_integrity":{"target.gguf":{"size_bytes":4,"blob_id":Digest::of_bytes(b"good").as_str()}}}}]});
    let bytes = serde_json::to_vec(&plan).unwrap();
    let policy = CachePolicy::BlobIdentity { root: root.clone() };
    assert!(
        verify(&bytes, &policy)
            .err()
            .unwrap()
            .to_string()
            .contains("escapes cache root")
    );
    std::fs::remove_file(&entry).unwrap();
    std::fs::create_dir(&entry).unwrap();
    assert!(
        verify(&bytes, &policy)
            .err()
            .unwrap()
            .to_string()
            .contains("not a regular file")
    );
    std::fs::remove_dir(&entry).unwrap();
    let inside = root.join("copied-model.gguf");
    std::fs::write(&inside, b"good").unwrap();
    std::os::unix::fs::symlink(&inside, &entry).unwrap();
    assert!(matches!(
        verify(&bytes, &policy).unwrap(),
        CacheAdmission::BlobIdentity
    ));
}

#[cfg(unix)]
#[test]
fn projector_when_identity_is_valid_does_not_require_trunk_dimensions() {
    let state = tempfile::tempdir().unwrap();
    let root = state.path().join("hub");
    let repository = root.join("models--fixture--model");
    let revision = "a".repeat(40);
    let snapshot = repository.join("snapshots").join(&revision);
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::create_dir_all(repository.join("blobs")).unwrap();
    let target = super::gguf_tests::fixture(
        "llama",
        &[("llama.block_count", 32), ("llama.embedding_length", 2048)],
    );
    let projector = b"projector sidecar".to_vec();
    let mut artifacts = Vec::new();
    for (name, bytes) in [("target.gguf", target), ("projector.gguf", projector)] {
        let digest = Digest::of_bytes(&bytes);
        let blob = repository.join("blobs").join(digest.as_str());
        std::fs::write(&blob, &bytes).unwrap();
        std::os::unix::fs::symlink(&blob, snapshot.join(name)).unwrap();
        artifacts.push(json!({"repo":"fixture/model","revision":revision,"files":[name],"file_integrity":{name:{"size_bytes":bytes.len(),"blob_id":digest.as_str()}}}));
    }
    let bytes = serde_json::to_vec(&json!({"selected_models":[{"architecture":"llama","execution":{"layer_end":32,"activation_width":2048,"mtp_layers":0},"artifact":artifacts[0],"draft_artifact":null,"mmproj_artifact":artifacts[1]}]})).unwrap();

    let result = verify(&bytes, &CachePolicy::GgufMetadata { root }).unwrap();

    assert!(matches!(result, CacheAdmission::GgufMetadata));
}

#[cfg(unix)]
#[test]
fn nas_admission_checks_pinned_identity_and_size_without_rehashing_payloads() {
    let state = tempfile::tempdir().unwrap();
    let root = state.path().join("hub");
    let repository = root.join("models--fixture--model");
    let revision = "a".repeat(40);
    let snapshot = repository.join("snapshots").join(&revision);
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::create_dir_all(repository.join("blobs")).unwrap();
    let digest = Digest::of_bytes(b"good");
    let blob = repository.join("blobs").join(digest.as_str());
    std::fs::write(&blob, b"evil").unwrap();
    std::os::unix::fs::symlink(&blob, snapshot.join("projector.gguf")).unwrap();
    let artifact = json!({"repo":"fixture/model","revision":revision,"files":["projector.gguf"],"file_integrity":{"projector.gguf":{"size_bytes":4,"blob_id":digest.as_str()}}});
    let mut target = super::gguf_tests::fixture(
        "llama",
        &[("llama.block_count", 32), ("llama.embedding_length", 2048)],
    );
    target.extend_from_slice(b"weights");
    let target_digest = Digest::of_bytes(&target);
    *target.last_mut().unwrap() = b'!';
    let target_blob = repository.join("blobs").join(target_digest.as_str());
    std::fs::write(&target_blob, &target).unwrap();
    std::os::unix::fs::symlink(&target_blob, snapshot.join("target.gguf")).unwrap();
    let target_artifact = json!({"repo":"fixture/model","revision":revision,"files":["target.gguf"],"file_integrity":{"target.gguf":{"size_bytes":target.len(),"blob_id":target_digest.as_str()}}});
    let bytes = serde_json::to_vec(&json!({"selected_models":[{"architecture":"llama","execution":{"layer_end":32,"activation_width":2048,"mtp_layers":0},"artifact":target_artifact,"draft_artifact":null,"mmproj_artifact":artifact}]})).unwrap();

    // Admission deliberately trusts the immutable cache's pinned blob name.
    // Same-size payload changes in either target or projector are not rehashed.
    for policy in [
        CachePolicy::BlobIdentity { root: root.clone() },
        CachePolicy::GgufMetadata { root: root.clone() },
    ] {
        assert!(verify(&bytes, &policy).is_ok());
        std::fs::write(&blob, b"truncated").unwrap();
        assert!(verify(&bytes, &policy).is_err());
        std::fs::write(&blob, b"evil").unwrap();

        let other_blob = repository.join("blobs").join("b".repeat(64));
        std::fs::write(&other_blob, b"evil").unwrap();
        let projector_path = snapshot.join("projector.gguf");
        std::fs::remove_file(&projector_path).unwrap();
        std::os::unix::fs::symlink(&other_blob, &projector_path).unwrap();
        assert!(verify(&bytes, &policy).is_err());
        std::fs::remove_file(&projector_path).unwrap();
        std::os::unix::fs::symlink(&blob, &projector_path).unwrap();
    }
}

#[cfg(unix)]
#[test]
fn every_current_non_chat_model_keeps_workload_lanes_and_checks_cached_architecture() {
    let source = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let manifest = source.join("ci/llama-canary/family-certified.json");
    let mut plan: serde_json::Value = serde_json::from_slice(
        &crate::ci_plan::family::controller_plan(&source, &manifest).unwrap(),
    )
    .unwrap();
    let mut checked = 0;
    for model in plan["selected_models"].as_array_mut().unwrap() {
        if model["class"] == "causal_generation" {
            continue;
        }
        checked += 1;
        let state = tempfile::tempdir().unwrap();
        let root = state.path().join("hub");
        let repository = root.join("models--fixture--model");
        let revision = "a".repeat(40);
        let snapshot = repository.join("snapshots").join(&revision);
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::create_dir_all(repository.join("blobs")).unwrap();
        let architecture = model["architecture"].as_str().unwrap();
        let count_key = format!("{architecture}.block_count");
        let width_key = format!("{architecture}.embedding_length");
        let target = super::gguf_tests::fixture(
            architecture,
            &[
                (
                    &count_key,
                    model["execution"]["layer_end"].as_u64().unwrap(),
                ),
                (
                    &width_key,
                    model["execution"]["activation_width"].as_u64().unwrap(),
                ),
            ],
        );
        let target_digest = Digest::of_bytes(&target);
        let blob = repository.join("blobs").join(target_digest.as_str());
        std::fs::write(&blob, &target).unwrap();
        std::os::unix::fs::symlink(&blob, snapshot.join("target.gguf")).unwrap();
        model["artifact"] = json!({"repo":"fixture/model","revision":revision,"files":["target.gguf"],"file_integrity":{"target.gguf":{"size_bytes":target.len(),"blob_id":target_digest.as_str()}}});
        if !model["mmproj_artifact"].is_null() {
            let projector = b"opaque projector sidecar";
            let digest = Digest::of_bytes(projector);
            let blob = repository.join("blobs").join(digest.as_str());
            std::fs::write(&blob, projector).unwrap();
            std::os::unix::fs::symlink(&blob, snapshot.join("projector.gguf")).unwrap();
            model["mmproj_artifact"] = json!({"repo":"fixture/model","revision":revision,"files":["projector.gguf"],"file_integrity":{"projector.gguf":{"size_bytes":projector.len(),"blob_id":digest.as_str()}}});
        }
        let before = model.clone();
        let document = serde_json::to_vec(&json!({"selected_models":[model]})).unwrap();
        assert!(matches!(
            verify(&document, &CachePolicy::GgufMetadata { root: root.clone() }).unwrap(),
            CacheAdmission::GgufMetadata
        ));
        assert_eq!(*model, before);
        let lanes = model["certification_lanes"].as_array().unwrap();
        assert_eq!(lanes.len(), 2);
        assert!(lanes.iter().all(|lane| {
            let lane = lane.as_str().unwrap();
            lane.ends_with("-smoke") || lane.ends_with("-oracle")
        }));
        model["architecture"] = json!("different");
        let document = serde_json::to_vec(&json!({"selected_models":[model]})).unwrap();
        assert!(verify(&document, &CachePolicy::GgufMetadata { root }).is_err());
    }
    assert!(
        checked > 0,
        "current roster must exercise non-chat cache admission"
    );
}
