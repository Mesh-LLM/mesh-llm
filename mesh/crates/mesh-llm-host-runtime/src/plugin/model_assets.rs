//! Host-owned acquisition of provider-planned model bundles.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use mesh_llm_plugin::models::{AssetPlan, ManagedModelConfig, PreparedModelAssets};
use tokio::sync::{Mutex, mpsc};

use super::{PROTOCOL_VERSION, proto, runtime::PluginRuntime};

pub(super) fn managed_models_from_env(
    env: &BTreeMap<String, String>,
) -> Result<Vec<ManagedModelConfig>> {
    env.get(mesh_llm_plugin::models::CONFIG_ENV)
        .map(|json| serde_json::from_str(json).context("invalid managed model configuration"))
        .transpose()
        .map(Option::unwrap_or_default)
}

/// Keep downloading and disk IO off the connection's control loop.
pub(super) fn prepare_for_connection(
    request_id: u64,
    request: proto::PrepareModelAssetsRequest,
    plugin_name: String,
    generation: u64,
    runtime: Arc<Mutex<Option<PluginRuntime>>>,
    outbound: mpsc::Sender<proto::Envelope>,
) {
    tokio::spawn(async move {
        let result = prepare_request(&request.plan_json, generation, &runtime).await;
        let payload = match result {
            Ok(assets) => match serde_json::to_string(&assets) {
                Ok(assets_json) => proto::envelope::Payload::PrepareModelAssetsResponse(
                    proto::PrepareModelAssetsResponse { assets_json },
                ),
                Err(error) => error_payload(error),
            },
            Err(error) => error_payload(error),
        };
        // A replaced connection must never receive an old job completion.
        if runtime
            .lock()
            .await
            .as_ref()
            .is_some_and(|rt| rt.generation == generation)
        {
            let _ = outbound
                .send(proto::Envelope {
                    protocol_version: PROTOCOL_VERSION,
                    plugin_id: plugin_name,
                    request_id,
                    payload: Some(payload),
                })
                .await;
        }
    });
}

fn error_payload(error: impl std::fmt::Display) -> proto::envelope::Payload {
    proto::envelope::Payload::ErrorResponse(proto::ErrorResponse {
        code: -32602,
        message: format!("model assets: {error}"),
        data_json: String::new(),
    })
}

async fn prepare_request(
    json: &str,
    generation: u64,
    runtime: &Mutex<Option<PluginRuntime>>,
) -> Result<PreparedModelAssets> {
    if json.len() > 256 * 1024 {
        bail!("asset plan exceeds size limit");
    }
    let plan: AssetPlan = serde_json::from_str(json).context("invalid asset plan")?;
    plan.validate()?;
    let selected = {
        let guard = runtime.lock().await;
        let connection = guard
            .as_ref()
            .filter(|rt| rt.generation == generation)
            .context("provider connection is no longer current")?;
        connection
            .managed_models
            .iter()
            .find(|config| config.model == plan.model)
            .cloned()
            .context("model is not configured for this provider")?
    };
    authorize_plan(&selected, &plan)?;
    if let Some(local_dir) = selected.source.local_dir {
        return prepare_local(&local_dir, &plan).await;
    }
    prepare_huggingface(&plan).await
}

fn authorize_plan(config: &ManagedModelConfig, plan: &AssetPlan) -> Result<()> {
    let (repo, revision) = config
        .model
        .rsplit_once('@')
        .map_or((config.model.as_str(), None), |(repo, rev)| {
            (repo, Some(rev))
        });
    let allowed_repo = config.source.repo.as_deref().unwrap_or(repo);
    let allowed_revision = config.source.revision.as_deref().or(revision);
    if plan.repo != allowed_repo {
        bail!("plan repository does not match the configured model source");
    }
    if allowed_revision.is_some_and(|revision| revision != plan.revision) {
        bail!("plan revision does not match the configured model source");
    }
    Ok(())
}

async fn prepare_local(directory: &Path, plan: &AssetPlan) -> Result<PreparedModelAssets> {
    let directory = tokio::fs::canonicalize(directory)
        .await
        .context("open local model directory")?;
    let mut files = BTreeMap::new();
    for name in &plan.files {
        let path = tokio::fs::canonicalize(directory.join(name))
            .await
            .with_context(|| format!("required model asset missing: {name}"))?;
        if !path.starts_with(&directory) || !tokio::fs::metadata(&path).await?.is_file() {
            bail!("model asset escapes its directory or is not a file: {name}");
        }
        files.insert(name.clone(), path);
    }
    Ok(PreparedModelAssets {
        directory,
        files,
        revision: plan.revision.clone(),
    })
}

async fn prepare_huggingface(plan: &AssetPlan) -> Result<PreparedModelAssets> {
    let api = crate::models::build_hf_tokio_api(true)?;
    let (owner, name) = plan.repo.split_once('/').context("invalid repository")?;
    let mut revision = plan.revision.clone();
    let mut directory: Option<PathBuf> = None;
    let mut files = BTreeMap::new();
    for file in &plan.files {
        tracing::info!(model = %plan.model, file, "preparing managed model asset");
        let repository = api.model(owner, name);
        let cached = repository
            .download_file()
            .filename(file.clone())
            .revision(revision.clone())
            .local_files_only(true)
            .send()
            .await;
        let path = match cached {
            Ok(path) => path,
            Err(error) if std::env::var("HF_HUB_OFFLINE").is_ok_and(|value| value == "1") => {
                return Err(error)
                    .with_context(|| format!("model asset {file} is not cached in offline mode"));
            }
            Err(_) => repository
                .download_file()
                .filename(file.clone())
                .revision(revision.clone())
                .send()
                .await
                .with_context(|| format!("download model asset {file}"))?,
        };
        let root = bundle_root(&path, file)?;
        // A cached HF snapshot directory is named by its immutable commit.
        // Resolve a moving tag once, then use that commit for every graph.
        if directory.is_none() {
            let commit = root
                .file_name()
                .and_then(|name| name.to_str())
                .filter(|name| name.len() == 40 && name.bytes().all(|b| b.is_ascii_hexdigit()))
                .context("HF download did not return an immutable snapshot path")?;
            revision = commit.to_string();
            directory = Some(root.clone());
        }
        if directory.as_ref() != Some(&root) {
            bail!("model bundle assets resolved to different revisions");
        }
        files.insert(file.clone(), path);
    }
    Ok(PreparedModelAssets {
        directory: directory.context("empty model asset plan")?,
        files,
        revision,
    })
}

fn bundle_root(path: &Path, relative: &str) -> Result<PathBuf> {
    let mut root = path;
    for _ in relative.split('/') {
        root = root.parent().context("invalid cached model asset path")?;
    }
    Ok(root.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn managed_configuration_rejects_malformed_data_instead_of_empty_inventory() {
        assert!(
            managed_models_from_env(&BTreeMap::new())
                .unwrap()
                .is_empty()
        );
        let env = BTreeMap::from([(mesh_llm_plugin::models::CONFIG_ENV.into(), "{broken".into())]);
        assert!(
            managed_models_from_env(&env)
                .unwrap_err()
                .to_string()
                .contains("invalid managed model configuration")
        );
    }

    fn plan() -> AssetPlan {
        AssetPlan {
            model: "owner/model@pin".into(),
            repo: "owner/model".into(),
            revision: "pin".into(),
            files: vec!["onnx/model.onnx".into(), "onnx/model.onnx_data".into()],
        }
    }

    #[tokio::test]
    async fn local_external_data_bundle_requires_every_asset() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("onnx")).unwrap();
        std::fs::write(dir.path().join("onnx/model.onnx"), b"graph").unwrap();
        assert!(prepare_local(dir.path(), &plan()).await.is_err());
        std::fs::write(dir.path().join("onnx/model.onnx_data"), b"weights").unwrap();
        let assets = prepare_local(dir.path(), &plan()).await.unwrap();
        assert_eq!(assets.files.len(), 2);
        assert_eq!(assets.directory, dir.path().canonicalize().unwrap());
    }

    #[test]
    fn bundle_root_preserves_nested_paths() {
        assert_eq!(
            bundle_root(
                Path::new("/cache/snapshots/pin/onnx/model.onnx"),
                "onnx/model.onnx"
            )
            .unwrap(),
            Path::new("/cache/snapshots/pin")
        );
    }

    #[test]
    fn acquisition_cannot_change_the_selected_source() {
        let config: ManagedModelConfig = serde_json::from_value(serde_json::json!({
            "model": "owner/model@pin", "provider": "onnx"
        }))
        .unwrap();
        assert!(authorize_plan(&config, &plan()).is_ok());
        let mut other = plan();
        other.repo = "other/model".into();
        assert!(authorize_plan(&config, &other).is_err());
        other = plan();
        other.revision = "other".into();
        assert!(authorize_plan(&config, &other).is_err());
    }
}
