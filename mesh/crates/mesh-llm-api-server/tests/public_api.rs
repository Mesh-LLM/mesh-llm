#![allow(unused)]

use mesh_llm_api_server::{
    CapabilityLevel, CleanupPolicy, ClientBuilder, DeleteModelOptions, DownloadOptions,
    InviteToken, LoadModelOptions, MeshClient, MeshNode, Model, ModelKind, ModelSearchQuery,
    ModelSource, OwnerKeypair, PrunePolicy, PublicMesh, ServingController, ServingModelState,
    Status, UnloadModelOptions,
};
use std::str::FromStr;
use std::sync::{Arc, Mutex};

#[test]
fn client_builder_with_keypair_and_token() {
    let kp = OwnerKeypair::generate();
    let token = InviteToken::from_str("mesh-test:abc123").expect("valid token");
    let _builder = ClientBuilder::new(kp, token);
}

#[test]
fn client_builder_builds_mesh_client() {
    let kp = OwnerKeypair::generate();
    let token = InviteToken::from_str("mesh-test:abc123").expect("valid token");
    let builder = ClientBuilder::new(kp, token);
    let _client: MeshClient = builder.build().expect("build");
}

#[test]
fn mesh_client_has_reconnect_method() {
    fn _assert_reconnect(c: &mut MeshClient) {
        drop(c.reconnect());
    }
}

#[test]
fn mesh_node_builder_builds_node_with_namespaces() {
    let kp = OwnerKeypair::generate();
    let token = InviteToken::from_str("mesh-test:abc123").expect("valid token");
    let node = MeshNode::builder()
        .identity(kp)
        .join(token)
        .serving_enabled(true)
        .build()
        .expect("build node");

    let _inference = node.inference();
    let _models = node.models();
    let _serving = node.serving();
    let _status = node.status();
    let _events = node.events();
}

#[test]
fn public_mesh_builds_node_and_client_builders() {
    let mesh = PublicMesh {
        invite_token: "mesh-test:abc123".to_string(),
        serving: vec!["Qwen".to_string()],
        wanted: vec![],
        on_disk: vec![],
        total_vram_bytes: 24_000_000_000,
        node_count: 2,
        client_count: 0,
        max_clients: 8,
        name: Some("public".to_string()),
        region: Some("AU".to_string()),
        mesh_id: Some("mesh-1".to_string()),
        publisher_npub: "npub1test".to_string(),
        published_at: 1,
        expires_at: None,
    };

    let node = MeshNode::builder()
        .identity(OwnerKeypair::generate())
        .join(mesh.invite_token.parse().expect("valid token"))
        .build();
    assert!(node.is_ok());

    let client_builder = mesh.client_builder(OwnerKeypair::generate());
    assert!(client_builder.is_ok());
}

#[tokio::test]
async fn mesh_node_exposes_config_backed_statuses() {
    let kp = OwnerKeypair::generate();
    let token = InviteToken::from_str("mesh-test:abc123").expect("valid token");
    let cache_dir = std::env::temp_dir().join("mesh-llm-api-server-node-public-api-test");
    let node = MeshNode::builder()
        .identity(kp)
        .join(token)
        .cache_dir(cache_dir.clone())
        .serving_enabled(true)
        .build()
        .expect("build node");

    let cache_status = node.models().cache_status().await.expect("cache status");
    assert_eq!(cache_status.cache_dir.as_deref(), Some(cache_dir.as_path()));

    let serving_status = node.serving().status().await.expect("serving status");
    assert!(serving_status.enabled);
}

#[tokio::test]
async fn mesh_node_serving_uses_in_process_controller() {
    let kp = OwnerKeypair::generate();
    let token = InviteToken::from_str("mesh-test:abc123").expect("valid token");
    let controller = Arc::new(FakeServingController::default());
    let node = MeshNode::builder()
        .identity(kp)
        .join(token)
        .serving_controller(controller.clone())
        .build()
        .expect("build node");

    let loaded = node
        .serving()
        .load("org/model:Q4_K_M", LoadModelOptions::default())
        .await
        .expect("load model");
    assert_eq!(loaded.model_id, "org/model:Q4_K_M");
    assert_eq!(loaded.model_ref, "org/model:Q4_K_M");
    assert_eq!(loaded.instance_id.as_deref(), Some("instance-1"));
    assert!(matches!(loaded.state, ServingModelState::Ready));

    let status = node.serving().status().await.expect("serving status");
    assert!(status.enabled);
    assert_eq!(status.models.len(), 1);

    node.serving()
        .unload_model("org/model:Q4_K_M", UnloadModelOptions::default())
        .await
        .expect("unload model");
    assert!(controller.models.lock().unwrap().is_empty());
}

#[tokio::test]
async fn mesh_node_serving_forwards_unload_options() {
    let kp = OwnerKeypair::generate();
    let token = InviteToken::from_str("mesh-test:abc123").expect("valid token");
    let controller = Arc::new(FakeServingController::default());
    let node = MeshNode::builder()
        .identity(kp)
        .join(token)
        .serving_controller(controller.clone())
        .build()
        .expect("build node");

    let options = UnloadModelOptions {
        drain_timeout: std::time::Duration::from_millis(1_250),
        force: true,
    };
    node.serving()
        .unload_instance("instance-1", options)
        .await
        .expect("unload instance");

    let requests = controller.unload_requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0].target,
        mesh_llm_node::serving::UnloadTarget::Instance("instance-1".to_string())
    );
    assert_eq!(
        requests[0].options.drain_timeout,
        std::time::Duration::from_millis(1_250)
    );
    assert!(requests[0].options.force);
}

#[tokio::test]
async fn mesh_node_models_installed_scans_configured_cache() {
    let kp = OwnerKeypair::generate();
    let token = InviteToken::from_str("mesh-test:abc123").expect("valid token");
    let cache_dir = unique_temp_dir("mesh-llm-api-server-installed-cache");
    let model = cache_dir
        .join("models--org--repo-GGUF")
        .join("snapshots")
        .join("abc")
        .join("Repo-Q4_K_M.gguf");
    std::fs::create_dir_all(model.parent().unwrap()).unwrap();
    std::fs::write(&model, b"gguf").unwrap();

    let node = MeshNode::builder()
        .identity(kp)
        .join(token)
        .cache_dir(cache_dir.clone())
        .build()
        .expect("build node");

    let installed = node.models().installed().await.expect("installed models");
    assert_eq!(installed.len(), 1);
    assert_eq!(installed[0].model_ref, "org/repo-GGUF:Q4_K_M");
    assert_eq!(installed[0].path, model);
    assert_eq!(installed[0].size_bytes, Some(4));
    assert_eq!(installed[0].capabilities.vision, CapabilityLevel::None);

    let _ = std::fs::remove_dir_all(cache_dir);
}

#[tokio::test]
async fn mesh_node_models_recommended_and_show_include_capabilities() {
    let kp = OwnerKeypair::generate();
    let token = InviteToken::from_str("mesh-test:abc123").expect("valid token");
    let node = MeshNode::builder()
        .identity(kp)
        .join(token)
        .build()
        .expect("build node");

    let recommended = node.models().recommended().await.expect("recommended");
    assert!(!recommended.is_empty());
    assert!(
        recommended
            .iter()
            .any(|model| model.id == "Qwen3-4B-Q4_K_M")
    );

    let search = node
        .models()
        .search(ModelSearchQuery {
            query: "qwen3".to_string(),
            limit: Some(3),
        })
        .await
        .expect("search");
    assert!(!search.is_empty());
    assert!(
        search
            .iter()
            .any(|model| model.capabilities.reasoning == CapabilityLevel::Supported)
    );

    let details = node
        .models()
        .show("Qwen3-4B-Q4_K_M")
        .await
        .expect("show catalog model");
    assert_eq!(details.source, ModelSource::Catalog);
    assert_eq!(details.kind, ModelKind::Gguf);
    assert_eq!(details.id, "Qwen3-4B-Q4_K_M");
    assert_eq!(details.capabilities.reasoning, CapabilityLevel::Supported);
}

#[tokio::test]
async fn mesh_node_models_download_returns_installed_model_without_network() {
    let kp = OwnerKeypair::generate();
    let token = InviteToken::from_str("mesh-test:abc123").expect("valid token");
    let cache_dir = unique_temp_dir("mesh-llm-api-server-download-installed");
    let model = cache_dir
        .join("models--org--repo-GGUF")
        .join("snapshots")
        .join("abc")
        .join("Repo-Q4_K_M.gguf");
    std::fs::create_dir_all(model.parent().unwrap()).unwrap();
    std::fs::write(&model, b"gguf").unwrap();

    let node = MeshNode::builder()
        .identity(kp)
        .join(token)
        .cache_dir(cache_dir.clone())
        .build()
        .expect("build node");

    let downloaded = node
        .models()
        .download("org/repo-GGUF:Q4_K_M", DownloadOptions)
        .await
        .expect("download installed");
    assert_eq!(downloaded.model_ref, "org/repo-GGUF:Q4_K_M");
    assert_eq!(downloaded.primary_path.as_deref(), Some(model.as_path()));
    assert!(
        downloaded
            .details
            .as_ref()
            .is_some_and(|details| details.installed)
    );

    let _ = std::fs::remove_dir_all(cache_dir);
}

#[tokio::test]
async fn mesh_node_models_delete_cleanup_and_prune_work_on_configured_roots() {
    let kp = OwnerKeypair::generate();
    let token = InviteToken::from_str("mesh-test:abc123").expect("valid token");
    let cache_dir = unique_temp_dir("mesh-llm-api-server-delete-cleanup");
    let runtime_dir = unique_temp_dir("mesh-llm-api-server-prune-derived");
    let model = cache_dir
        .join("models--org--repo-GGUF")
        .join("snapshots")
        .join("abc")
        .join("Repo-Q4_K_M.gguf");
    let cleanup_model = cache_dir
        .join("models--org--cleanup-GGUF")
        .join("snapshots")
        .join("abc")
        .join("Cleanup-Q4_K_M.gguf");
    let derived = runtime_dir.join("materialized").join("stage.gguf");
    std::fs::create_dir_all(model.parent().unwrap()).unwrap();
    std::fs::create_dir_all(cleanup_model.parent().unwrap()).unwrap();
    std::fs::create_dir_all(derived.parent().unwrap()).unwrap();
    std::fs::write(&model, b"gguf").unwrap();
    std::fs::write(&cleanup_model, b"clean").unwrap();
    std::fs::write(&derived, b"stage").unwrap();
    let expected_model = model.canonicalize().unwrap();
    let expected_cleanup_model = cleanup_model.canonicalize().unwrap();
    let expected_derived = derived.canonicalize().unwrap();

    let node = MeshNode::builder()
        .identity(kp)
        .join(token)
        .cache_dir(cache_dir.clone())
        .runtime_dir(runtime_dir.clone())
        .build()
        .expect("build node");

    let deleted = node
        .models()
        .delete("org/repo-GGUF:Q4_K_M", DeleteModelOptions::default())
        .await
        .expect("delete model");
    assert_eq!(deleted.deleted_paths, vec![expected_model]);
    assert!(!model.exists());

    let preview = node
        .models()
        .cleanup(CleanupPolicy::default())
        .await
        .expect("cleanup preview");
    assert!(preview.deleted_paths.is_empty());
    assert_eq!(preview.skipped_paths, vec![cleanup_model.clone()]);

    let cleanup = node
        .models()
        .cleanup(CleanupPolicy { remove_all: true })
        .await
        .expect("cleanup delete");
    assert_eq!(cleanup.deleted_paths, vec![expected_cleanup_model]);
    assert!(!cleanup_model.exists());

    let pruned = node
        .models()
        .prune_derived_cache(PrunePolicy { remove_all: true })
        .await
        .expect("prune derived");
    assert_eq!(pruned.deleted_paths, vec![expected_derived]);
    assert!(!derived.exists());

    let _ = std::fs::remove_dir_all(cache_dir);
    let _ = std::fs::remove_dir_all(runtime_dir);
}

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "{prefix}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

#[derive(Default)]
struct FakeServingController {
    models: Mutex<Vec<mesh_llm_node::serving::ServedModel>>,
    unload_requests: Mutex<Vec<mesh_llm_node::serving::UnloadModelRequest>>,
}

impl ServingController for FakeServingController {
    fn load<'a>(
        &'a self,
        request: mesh_llm_node::serving::LoadModelRequest,
    ) -> mesh_llm_node::serving::ServingFuture<'a, mesh_llm_node::serving::ServedModel> {
        Box::pin(async move {
            let model_ref = request.model_ref;
            let model = mesh_llm_node::serving::ServedModel {
                model_ref: model_ref.clone(),
                profile: String::new(),
                model_id: model_ref,
                instance_id: Some("instance-1".to_string()),
                state: mesh_llm_node::serving::ServingModelState::Ready,
                backend: Some("fake".to_string()),
                capabilities: Default::default(),
                context_length: Some(4096),
                error: None,
            };
            self.models.lock().unwrap().push(model.clone());
            Ok(model)
        })
    }

    fn unload<'a>(
        &'a self,
        request: mesh_llm_node::serving::UnloadModelRequest,
    ) -> mesh_llm_node::serving::ServingFuture<'a, ()> {
        Box::pin(async move {
            let target = request.target.as_runtime_target();
            self.unload_requests.lock().unwrap().push(request.clone());
            self.models.lock().unwrap().retain(|model| {
                model.model_id != target && model.instance_id.as_deref() != Some(target)
            });
            Ok(())
        })
    }

    fn served_models<'a>(
        &'a self,
    ) -> mesh_llm_node::serving::ServingFuture<'a, Vec<mesh_llm_node::serving::ServedModel>> {
        Box::pin(async move { Ok(self.models.lock().unwrap().clone()) })
    }

    fn status<'a>(
        &'a self,
    ) -> mesh_llm_node::serving::ServingFuture<'a, mesh_llm_node::serving::ServingStatus> {
        Box::pin(async move {
            Ok(mesh_llm_node::serving::ServingStatus {
                enabled: true,
                models: self.models.lock().unwrap().clone(),
            })
        })
    }

    fn set_device_policy<'a>(
        &'a self,
        _policy: mesh_llm_node::serving::DevicePolicy,
    ) -> mesh_llm_node::serving::ServingFuture<'a, ()> {
        Box::pin(async { Ok(()) })
    }
}
