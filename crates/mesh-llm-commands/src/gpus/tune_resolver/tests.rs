use super::*;
use hf_hub::RepoTypeModel;
use model_hf::store::{
    huggingface_hub_cache_dir, huggingface_identity_for_path, huggingface_repo_folder_name,
    model_ref_for_path,
};
use rand::{RngExt, distr::Alphanumeric, rng};
use std::fs;
use tempfile::{TempDir, tempdir};

struct CacheFixtureGuard {
    repo_root: PathBuf,
}

impl Drop for CacheFixtureGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.repo_root);
    }
}

fn write_local_gguf_file(dir: &TempDir, name: &str) -> PathBuf {
    let path = dir.path().join(name);
    fs::write(&path, b"GGUF").unwrap();
    path
}

fn random_suffix() -> String {
    rng()
        .sample_iter(Alphanumeric)
        .take(10)
        .map(char::from)
        .collect()
}

fn write_hf_cache_gguf(revision: &str, file: &str) -> (CacheFixtureGuard, String, PathBuf) {
    let repo = format!("meshllm-gpu-tune-tests-{}", random_suffix());
    let repo_id = format!("meshllm/{repo}");
    let repo_root =
        huggingface_hub_cache_dir().join(huggingface_repo_folder_name(&repo_id, RepoTypeModel));
    let snapshot_dir = repo_root.join("snapshots").join(revision);
    let path = snapshot_dir.join(file);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, b"GGUF").unwrap();
    (CacheFixtureGuard { repo_root }, repo_id, path)
}

fn mesh_config_with_models(models: &[String]) -> MeshConfig {
    MeshConfig {
        models: models
            .iter()
            .cloned()
            .map(|model| mesh_llm_config::ModelConfigEntry {
                model,
                ..mesh_llm_config::ModelConfigEntry::default()
            })
            .collect(),
        ..MeshConfig::default()
    }
}

#[test]
fn gpu_tune_local_resolver_handles_paths_cache_refs_configured_misses_and_duplicates() {
    let temp = tempdir().unwrap();
    let local_path = write_local_gguf_file(&temp, "sample.gguf");
    let (_cache_guard, repo_id, cached_path) =
        write_hf_cache_gguf("rev-123", "Q4_K_M/example-model-Q4_K_M.gguf");
    let cached_ref = model_ref_for_path(&cached_path);
    let expected_identity = huggingface_identity_for_path(&cached_path).unwrap();
    let config =
        mesh_config_with_models(&[cached_ref.clone(), "missing-configured-model".to_string()]);

    let explicit = resolve_explicit_tune_targets(
        &config,
        &[
            local_path.display().to_string(),
            cached_ref.clone(),
            cached_path.display().to_string(),
            local_path.display().to_string(),
        ],
    );
    let configured = resolve_configured_tune_targets(&config);

    assert_eq!(explicit.resolved.len(), 2);
    assert_eq!(explicit.duplicates.len(), 2);
    assert!(explicit.errors.is_empty());
    assert_eq!(configured.resolved.len(), 1);
    assert_eq!(configured.errors.len(), 1);
    assert_eq!(configured.errors[0].input, "missing-configured-model");

    let filesystem_target = explicit
        .resolved
        .iter()
        .find(|target| target.resolved_path == local_path.canonicalize().unwrap())
        .unwrap();
    assert_eq!(
        filesystem_target.selection,
        TuneTargetSelection::Explicit { configured: false }
    );

    let cache_target = explicit
        .resolved
        .iter()
        .find(|target| {
            matches!(
                target.local_source,
                LocalTargetSource::HuggingFaceCache { .. }
            )
        })
        .unwrap();
    assert!(cache_target.canonical_model_ref.starts_with(&repo_id));
    assert_eq!(cache_target.config_matches.len(), 1);
    match &cache_target.local_source {
        LocalTargetSource::HuggingFaceCache { canonical_ref } => {
            assert_eq!(canonical_ref, &expected_identity.canonical_ref);
        }
        LocalTargetSource::FilesystemPath { .. } => panic!("expected cache target"),
    }
}

#[test]
fn gpu_tune_rejects_remote_only_refs_without_download() {
    let resolution = resolve_explicit_tune_targets_with_probe_for_tests(
        &MeshConfig::default(),
        &[
            "hf://meshllm/example@rev/Q4_K_M/model.gguf".to_string(),
            "missing-bare-name".to_string(),
        ],
        &|input| panic!("remote resolution should not be attempted for {input}"),
    );

    assert!(resolution.resolved.is_empty());
    assert!(resolution.duplicates.is_empty());
    assert_eq!(resolution.errors.len(), 2);
    assert_eq!(
        resolution.errors[0].reason,
        TuneTargetResolveReason::RemoteRefRequiresDownload
    );
    assert_eq!(
        resolution.errors[1].reason,
        TuneTargetResolveReason::NotFoundLocally
    );
    assert!(resolution.errors[0].to_string().contains("local-only"));
    assert!(
        resolution.errors[1]
            .to_string()
            .contains("missing-bare-name")
    );
}
