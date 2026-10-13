use super::admission::verify;
use super::contract::{CacheAdmission, CachePolicy};
use crate::automation::canary_receipts::Digest;
use serde_json::json;

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
fn projector_when_blob_digest_is_corrupt_still_fails_admission() {
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
    let target = super::gguf_tests::fixture(
        "llama",
        &[("llama.block_count", 32), ("llama.embedding_length", 2048)],
    );
    let target_digest = Digest::of_bytes(&target);
    let target_blob = repository.join("blobs").join(target_digest.as_str());
    std::fs::write(&target_blob, &target).unwrap();
    std::os::unix::fs::symlink(&target_blob, snapshot.join("target.gguf")).unwrap();
    let target_artifact = json!({"repo":"fixture/model","revision":revision,"files":["target.gguf"],"file_integrity":{"target.gguf":{"size_bytes":target.len(),"blob_id":target_digest.as_str()}}});
    let bytes = serde_json::to_vec(&json!({"selected_models":[{"architecture":"llama","execution":{"layer_end":32,"activation_width":2048,"mtp_layers":0},"artifact":target_artifact,"draft_artifact":null,"mmproj_artifact":artifact}]})).unwrap();

    let result = verify(&bytes, &CachePolicy::GgufMetadata { root });

    assert!(result.is_err());
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
