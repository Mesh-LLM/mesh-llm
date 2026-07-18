use super::coordinator::*;
use super::loading::*;
use super::*;
use crate::runtime::RuntimeResourcePlanningProfile;
use crate::runtime::local::*;
use crate::runtime::local_package::*;
use crate::runtime::split_planning::RuntimeSliceStagePlan;
use crate::runtime::survey;
use crate::{mesh::NodeRole, plugin};
use iroh::SecretKey;
use sha2::{Digest, Sha256};
use skippy_protocol::{FlashAttentionType, LoadMode};
use std::fs;
use std::path::{Path, PathBuf};

pub(super) fn make_id(seed: u8) -> iroh::EndpointId {
    let mut bytes = [0u8; 32];
    bytes[0] = seed;
    SecretKey::from_bytes(&bytes).public()
}

pub(super) fn package(layer_count: u32) -> skippy::SkippyPackageIdentity {
    skippy::SkippyPackageIdentity {
        package_ref: "gguf:///models/qwen.gguf".to_string(),
        manifest_sha256: "manifest".to_string(),
        source_model_path: PathBuf::from("/models/qwen.gguf"),
        source_model_sha256: "source".to_string(),
        source_model_bytes: u64::from(layer_count) * 1_000_000,
        source_files: Vec::new(),
        layer_weight_bytes: Vec::new(),
        layer_count,
        activation_width: 2048,
        tensor_count: 100,
        generation: None,
    }
}

pub(super) fn stage_load_request(load_mode: LoadMode) -> skippy::StageLoadRequest {
    skippy::StageLoadRequest {
        topology_id: "topology-a".to_string(),
        run_id: "run-a".to_string(),
        model_id: "model-a".to_string(),
        backend: "skippy".to_string(),
        package_ref: match load_mode {
            LoadMode::LayerPackage => "hf://meshllm/Qwen3-8B-Q4_K_M-layers".to_string(),
            LoadMode::RuntimeSlice | LoadMode::ArtifactSlice => {
                "gguf:///models/qwen.gguf".to_string()
            }
        },
        manifest_sha256: "a".repeat(64),
        stage_id: "stage-1".to_string(),
        stage_index: 1,
        layer_start: 18,
        layer_end: 36,
        model_path: Some("/models/qwen.gguf".to_string()),
        source_model_bytes: Some(4_900_000_000),
        projector_path: None,
        selected_device: None,
        bind_addr: "127.0.0.1:0".to_string(),
        activation_width: 4096,
        wire_dtype: skippy::StageWireDType::F16,
        ctx_size: 8192,
        lane_count: 4,
        n_batch: Some(2048),
        n_ubatch: Some(512),
        n_gpu_layers: -1,
        mmap: None,
        mlock: false,
        cache_type_k: "f16".to_string(),
        cache_type_v: "f16".to_string(),
        flash_attn_type: FlashAttentionType::Auto,
        native_mtp_enabled: true,
        shutdown_generation: 1,
        coordinator_term: 1,
        coordinator_id: None,
        lease_until_unix_ms: u64::MAX,
        load_mode,
        upstream: None,
        downstream: None,
    }
}

pub(super) fn split_test_peer(
    seed: u8,
    model_name: &str,
    stage_protocol_generation_supported: bool,
) -> mesh::PeerInfo {
    let id = make_id(seed);
    mesh::PeerInfo {
        id,
        addr: iroh::EndpointAddr {
            id,
            addrs: Default::default(),
        },
        mesh_id: None,
        mesh_policy_hash: None,
        genesis_policy: None,
        role: NodeRole::Worker,
        first_joined_mesh_ts: None,
        models: Vec::new(),
        vram_bytes: 24_000_000_000,
        rtt_ms: None,
        model_source: None,
        admitted: true,
        serving_models: Vec::new(),
        hosted_models: Vec::new(),
        hosted_models_known: false,
        available_models: Vec::new(),
        requested_models: vec![model_name.to_string()],
        explicit_model_interests: Vec::new(),
        last_seen: std::time::Instant::now(),
        last_mentioned: std::time::Instant::now(),
        version: None,
        gpu_name: None,
        hostname: None,
        is_soc: None,
        gpu_vram: None,
        gpu_reserved_bytes: None,
        gpu_mem_bandwidth_gbps: None,
        gpu_compute_tflops_fp32: None,
        gpu_compute_tflops_fp16: None,
        available_model_metadata: Vec::new(),
        experts_summary: None,
        available_model_sizes: std::collections::HashMap::new(),
        served_model_descriptors: Vec::new(),
        served_model_runtime: Vec::new(),
        owner_attestation: None,
        release_attestation_summary: crate::ReleaseAttestationSummary::default(),
        artifact_transfer_supported: false,
        stage_protocol_generation_supported,
        stage_status_list_supported: false,
        advertised_model_throughput: vec![],

        display_rtt: None,
        selected_path: None,
        propagated_latency: None,
        owner_summary: crate::crypto::OwnershipSummary::default(),
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn push_gguf_string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn push_u32_kv(bytes: &mut Vec<u8>, key: &str, value: u32) {
    push_gguf_string(bytes, key);
    bytes.extend_from_slice(&4u32.to_le_bytes());
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_string_kv(bytes: &mut Vec<u8>, key: &str, value: &str) {
    push_gguf_string(bytes, key);
    bytes.extend_from_slice(&8u32.to_le_bytes());
    push_gguf_string(bytes, value);
}

pub(super) fn write_fake_gguf_model(path: &Path) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&0i64.to_le_bytes());
    bytes.extend_from_slice(&8i64.to_le_bytes());
    push_string_kv(&mut bytes, "general.architecture", "llama");
    push_string_kv(&mut bytes, "tokenizer.ggml.model", "gpt2");
    push_u32_kv(&mut bytes, "llama.context_length", 8192);
    push_u32_kv(&mut bytes, "llama.embedding_length", 4096);
    push_u32_kv(&mut bytes, "llama.block_count", 24);
    push_u32_kv(&mut bytes, "llama.attention.head_count", 32);
    push_u32_kv(&mut bytes, "llama.attention.head_count_kv", 8);
    push_u32_kv(&mut bytes, "llama.attention.key_length", 128);
    fs::write(path, bytes).unwrap();
}

#[test]
fn split_metadata_reads_a_synthetic_direct_gguf_source() {
    let temp = tempfile::tempdir().unwrap();
    let model_path = temp.path().join("model.gguf");
    write_fake_gguf_model(&model_path);
    let package = skippy::synthetic_direct_gguf_package("test/model", &model_path).unwrap();

    let metadata = scan_layer_package_metadata(&package).expect("direct GGUF metadata");

    assert_eq!(metadata.context_length, 8192);
    assert_eq!(metadata.embedding_size, 4096);
}

pub(super) fn write_test_layer_package(dir: &Path, source_model_bytes: u64) {
    fs::create_dir_all(dir.join("layers")).unwrap();
    fs::write(dir.join("metadata.gguf"), b"metadata").unwrap();
    fs::write(dir.join("embeddings.gguf"), b"embeddings").unwrap();
    fs::write(dir.join("output.gguf"), b"output").unwrap();
    fs::write(dir.join("layers/00000.gguf"), b"layer0").unwrap();
    let manifest = serde_json::json!({
        "schema_version": 1,
        "model_id": "meshllm/test-layer-package",
        "source_model": {
            "path": "/models/test-layer-package.gguf",
            "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "files": [{
                "path": "/models/test-layer-package.gguf",
                "size_bytes": source_model_bytes,
                "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            }]
        },
        "format": "layer-package",
        "layer_count": 1,
        "activation_width": 4096,
        "shared": {
            "metadata": {
                "path": "metadata.gguf",
                "tensor_count": 1,
                "tensor_bytes": 1,
                "artifact_bytes": 8,
                "sha256": sha256_hex(b"metadata")
            },
            "embeddings": {
                "path": "embeddings.gguf",
                "tensor_count": 1,
                "tensor_bytes": 1,
                "artifact_bytes": 10,
                "sha256": sha256_hex(b"embeddings")
            },
            "output": {
                "path": "output.gguf",
                "tensor_count": 1,
                "tensor_bytes": 1,
                "artifact_bytes": 6,
                "sha256": sha256_hex(b"output")
            }
        },
        "layers": [{
            "layer_index": 0,
            "path": "layers/00000.gguf",
            "tensor_count": 1,
            "tensor_bytes": 1,
            "artifact_bytes": 6,
            "sha256": sha256_hex(b"layer0")
        }],
        "skippy_abi_version": "0.1.0",
    });
    fs::write(
        dir.join("model-package.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
}

pub(super) fn participant(seed: u8) -> SplitParticipant {
    SplitParticipant::new(make_id(seed), 24_000_000_000, None)
}

pub(super) fn stage(
    seed: u8,
    stage_index: u32,
    layer_start: u32,
    layer_end: u32,
) -> RuntimeSliceStagePlan {
    RuntimeSliceStagePlan {
        stage_id: format!("stage-{stage_index}"),
        stage_index,
        node_id: make_id(seed),
        layer_start,
        layer_end,
        parameter_bytes: u64::from(layer_end.saturating_sub(layer_start)) * 1_000_000,
    }
}

pub(super) fn runtime_status_for_stage(
    generation: &SplitTopologyGeneration,
    stage: &RuntimeSliceStagePlan,
    state: skippy::StageRuntimeState,
) -> mesh::StageRuntimeStatus {
    mesh::StageRuntimeStatus {
        topology_id: generation.topology_id.clone(),
        run_id: generation.run_id.clone(),
        model_id: "model-a".to_string(),
        backend: "skippy".to_string(),
        package_ref: Some("gguf:///model.gguf".to_string()),
        manifest_sha256: Some("direct-gguf:1:model.gguf".to_string()),
        source_model_path: Some("/model.gguf".to_string()),
        source_model_sha256: None,
        source_model_bytes: Some(1),
        materialized_path: None,
        materialized_pinned: false,
        projector_path: None,
        stage_id: stage.stage_id.clone(),
        stage_index: stage.stage_index,
        node_id: Some(stage.node_id),
        layer_start: stage.layer_start,
        layer_end: stage.layer_end,
        state,
        bind_addr: "127.0.0.1:31000".to_string(),
        activation_width: 896,
        wire_dtype: skippy::StageWireDType::F16,
        selected_device: None,
        ctx_size: 512,
        lane_count: 4,
        n_batch: None,
        n_ubatch: None,
        flash_attn_type: FlashAttentionType::Auto,
        error: None,
        shutdown_generation: generation.generation,
    }
}

pub(super) fn local_stage(
    node_id: iroh::EndpointId,
    stage_index: u32,
    layer_start: u32,
    layer_end: u32,
) -> RuntimeSliceStagePlan {
    RuntimeSliceStagePlan {
        stage_id: format!("stage-{stage_index}"),
        stage_index,
        node_id,
        layer_start,
        layer_end,
        parameter_bytes: u64::from(layer_end.saturating_sub(layer_start)) * 1_000_000,
    }
}

#[tokio::test]
async fn split_generation_load_settings_consumes_resolved_skippy_config() {
    let node = mesh::Node::new_for_tests(NodeRole::Host { http_port: 9337 })
        .await
        .unwrap();
    let temp_dir = tempfile::tempdir().unwrap();
    let model_path = temp_dir.path().join("qwen.gguf");
    let projector_path = temp_dir.path().join("config-mmproj.gguf");
    write_fake_gguf_model(&model_path);
    fs::write(&projector_path, b"mmproj").unwrap();
    let mesh_config: plugin::MeshConfig = toml::from_str(&format!(
        r#"
[[models]]
model = "Qwen"

[models.model_fit]
ctx_size = 2048
batch = 768
ubatch = 192
cache_type_k = "q4_0"
cache_type_v = "q5_0"

[models.hardware]
model_path = "{model_path}"
device = "CUDA0"
gpu_layers = 77
mmproj = "{projector_path}"

[models.throughput]
parallel = 2
threads = 6
threads_batch = 3

[models.skippy]
activation_wire_dtype = "q8"
prefill_chunking = "fixed"
prefill_chunk_size = 96

[models.speculative]
strategy = "disabled"
mode = "draft"
draft_model_path = "/models/draft.gguf"
draft_max_tokens = 7
draft_gpu_layers = 11

[models.request_defaults]
max_tokens = 321
temperature = 0.35
stop = ["END"]
"#,
        model_path = model_path.display(),
        projector_path = projector_path.display()
    ))
    .expect("test mesh config should parse");
    let mut package = package(40);
    package.package_ref = "hf://Mesh-LLM/test-split-package".to_string();
    let temp_dir = tempfile::tempdir().unwrap();
    let model_path = temp_dir.path().join("qwen.gguf");
    write_fake_gguf_model(&model_path);
    let local_id = node.id();
    let generation = SplitTopologyGeneration::new(
        "resolver-topology".into(),
        "resolver-run".into(),
        1,
        vec![SplitParticipant::new(local_id, 24_000_000_000, None)],
        vec![
            local_stage(local_id, 0, 0, 12),
            local_stage(local_id, 1, 12, 40),
        ],
    );

    let spec = SplitGenerationLoadSpec {
        node: &node,
        mesh_config: &mesh_config,
        model_ref: "Qwen",
        model_path: &model_path,
        package: &package,
        generation: &generation,
        projector_path: Some("/models/fallback-mmproj.gguf".to_string()),
        ctx_size: 8192,
        pinned_gpu: None,
        slots: 4,
        cache_type_k_override: None,
        cache_type_v_override: None,
        n_batch_override: None,
        n_ubatch_override: None,
        flash_attention_override: FlashAttentionType::Auto,
        openai_guardrail_policy: openai_guardrail_policy_handle(
            openai_frontend::GuardrailMode::Disabled,
        ),
        skippy_telemetry: skippy::SkippyTelemetryOptions::off(),
        survey_telemetry: survey::SurveyTelemetry::disabled(),
    };
    let settings = split_generation_load_settings(&spec).expect("split settings should resolve");

    assert_eq!(settings.load_mode, LoadMode::LayerPackage);
    assert_eq!(settings.activation_width, 2048);
    assert_eq!(settings.activation_wire_dtype, skippy::StageWireDType::Q8);
    assert_eq!(settings.runtime_options.n_threads, Some(6));
    assert_eq!(settings.runtime_options.n_threads_batch, Some(3));
    assert_eq!(settings.runtime_options.config.ctx_size, 8192);
    assert_eq!(settings.runtime_options.config.lane_count, 4);
    assert_eq!(settings.runtime_options.config.n_batch, Some(768));
    assert_eq!(settings.runtime_options.config.n_ubatch, Some(192));
    assert_eq!(settings.runtime_options.config.n_gpu_layers, 77);
    assert_eq!(
        settings
            .runtime_options
            .config
            .selected_device
            .as_ref()
            .map(|device| device.backend_device.as_str()),
        Some("CUDA0")
    );
    assert_eq!(settings.runtime_options.config.cache_type_k, "q4_0");
    assert_eq!(settings.runtime_options.config.cache_type_v, "q5_0");
    assert_eq!(
        settings.runtime_options.config.projector_path.as_deref(),
        Some(projector_path.to_string_lossy().as_ref())
    );
    assert!(!settings.runtime_options.config.native_mtp_enabled);
    assert!(!settings.embedded_openai.native_mtp_enabled);
    assert_eq!(settings.embedded_openai.generation_concurrency, 4);
    assert_eq!(settings.embedded_openai.default_max_tokens, 321);
    assert_eq!(
        settings.embedded_openai.request_defaults.temperature,
        Some(0.35)
    );
    assert_eq!(
        settings.embedded_openai.request_defaults.stop.as_deref(),
        Some(["END".to_string()].as_slice())
    );
    assert_eq!(settings.embedded_openai.prefill_chunk_policy, "fixed");
    assert_eq!(settings.embedded_openai.prefill_chunk_size, 96);
    assert_eq!(
        settings.embedded_openai.draft_model_path.as_deref(),
        Some(Path::new("/models/draft.gguf"))
    );
    assert_eq!(settings.embedded_openai.speculative_window, 7);
    assert_eq!(settings.embedded_openai.draft_n_gpu_layers, Some(11));
}

#[tokio::test]
async fn runtime_resolver_uses_config_model_id_but_preserves_served_model_id() {
    let node = mesh::Node::new_for_tests(NodeRole::Host { http_port: 9337 })
        .await
        .unwrap();
    let temp_dir = tempfile::tempdir().unwrap();
    let model_path = temp_dir.path().join("alias-target.gguf");
    write_fake_gguf_model(&model_path);
    let mesh_config: plugin::MeshConfig = toml::from_str(&format!(
        r#"
[[models]]
model = "configured/model-ref"

[models.hardware]
model_path = "{model_path}"

[models.throughput]
threads = 9
threads_batch = 5

[models.request_defaults]
max_tokens = 222
"#,
        model_path = model_path.display()
    ))
    .expect("test mesh config should parse");
    let model_bytes = fs::metadata(&model_path).unwrap().len();
    let spec = LocalRuntimeModelStartSpec {
        node: &node,
        mesh_config: &mesh_config,
        config_model_id: Some("configured/model-ref"),
        model_path: &model_path,
        model_bytes,
        mmproj_override: None,
        ctx_size_override: None,
        pinned_gpu: None,
        capacity_budget_bytes: None,
        cache_type_k_override: None,
        cache_type_v_override: None,
        n_batch_override: None,
        n_ubatch_override: None,
        flash_attention_override: FlashAttentionType::Auto,
        parallel_override: None,
        planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
        openai_guardrail_policy: openai_guardrail_policy_handle(
            openai_frontend::GuardrailMode::Disabled,
        ),
        skippy_telemetry: skippy::SkippyTelemetryOptions::off(),
        survey_telemetry: survey::SurveyTelemetry::disabled(),
    };

    let resolved =
        resolve_runtime_skippy_config(&spec, "runtime/served-name", model_bytes, 4096, 3, None)
            .expect("runtime config should resolve through configured model id");

    assert_eq!(resolved.model_id, "runtime/served-name");
    assert_eq!(resolved.throughput.threads, Some(9));
    assert_eq!(resolved.throughput.threads_batch, Some(5));
    assert_eq!(resolved.request_defaults.max_tokens, 222);
    assert_eq!(resolved.model_fit.ctx_size, 4096);
    assert_eq!(resolved.throughput.parallel, 3);
}

#[test]
fn runtime_verified_served_model_descriptor_preserves_identity_and_updates_capabilities() {
    let existing = mesh::ServedModelDescriptor {
        identity: mesh::ServedModelIdentity {
            model_name: "Qwen3VL-2B-Instruct-Q4_K_M".into(),
            is_primary: false,
            source_kind: mesh::ModelSourceKind::HuggingFace,
            repository: Some("Qwen/Qwen3-VL-2B-Instruct-GGUF".into()),
            artifact: Some("Qwen3VL-2B-Instruct-Q4_K_M.gguf".into()),
            ..Default::default()
        },
        capabilities_known: false,
        capabilities: models::ModelCapabilities::default(),
        topology: None,
        metadata: None,
    };
    let capabilities = models::ModelCapabilities {
        multimodal: true,
        vision: models::CapabilityLevel::Supported,
        ..Default::default()
    };

    let descriptor = runtime_verified_served_model_descriptor(
        Some(existing),
        "Qwen3VL-2B-Instruct-Q4_K_M",
        "Qwen3VL-2B-Instruct-Q4_K_M",
        capabilities,
    );

    assert_eq!(
        descriptor.identity.source_kind,
        mesh::ModelSourceKind::HuggingFace
    );
    assert_eq!(
        descriptor.identity.repository.as_deref(),
        Some("Qwen/Qwen3-VL-2B-Instruct-GGUF")
    );
    assert!(descriptor.identity.is_primary);
    assert!(descriptor.capabilities_known);
    assert_eq!(descriptor.capabilities, capabilities);
}

#[test]
fn runtime_verified_served_model_descriptor_builds_fallback_identity() {
    let descriptor = runtime_verified_served_model_descriptor(
        None,
        "Primary",
        "Runtime",
        models::ModelCapabilities::default(),
    );

    assert_eq!(descriptor.identity.model_name, "Runtime");
    assert!(!descriptor.identity.is_primary);
    assert_eq!(
        descriptor.identity.source_kind,
        mesh::ModelSourceKind::Unknown
    );
    assert_eq!(
        descriptor.identity.local_file_name.as_deref(),
        Some("Runtime.gguf")
    );
    assert_eq!(
        descriptor.capabilities,
        models::ModelCapabilities::default()
    );
    assert!(descriptor.capabilities_known);
}

pub(super) fn test_stage_status_from_load(
    load: &skippy::StageLoadRequest,
    state: skippy::StageRuntimeState,
) -> skippy::StageStatusSnapshot {
    skippy::StageStatusSnapshot {
        topology_id: load.topology_id.clone(),
        run_id: load.run_id.clone(),
        model_id: load.model_id.clone(),
        backend: load.backend.clone(),
        package_ref: Some(load.package_ref.clone()),
        manifest_sha256: Some(load.manifest_sha256.clone()),
        source_model_path: load.model_path.clone(),
        source_model_sha256: None,
        source_model_bytes: load.source_model_bytes,
        materialized_path: None,
        materialized_pinned: false,
        projector_path: load.projector_path.clone(),
        stage_id: load.stage_id.clone(),
        stage_index: load.stage_index,
        layer_start: load.layer_start,
        layer_end: load.layer_end,
        state,
        bind_addr: "127.0.0.1:31000".to_string(),
        activation_width: load.activation_width as u32,
        wire_dtype: load.wire_dtype,
        selected_device: load.selected_device.clone(),
        ctx_size: load.ctx_size,
        lane_count: load.lane_count,
        n_batch: load.n_batch,
        n_ubatch: load.n_ubatch,
        flash_attn_type: load.flash_attn_type,
        error: None,
        shutdown_generation: load.shutdown_generation,
        coordinator_term: load.coordinator_term,
        coordinator_id: load.coordinator_id,
        lease_until_unix_ms: load.lease_until_unix_ms,
    }
}

pub(super) fn test_stage_status_from_stop(
    stop: &skippy::StageStopRequest,
) -> skippy::StageStatusSnapshot {
    skippy::StageStatusSnapshot {
        topology_id: stop.topology_id.clone(),
        run_id: stop.run_id.clone(),
        model_id: String::new(),
        backend: "skippy".to_string(),
        package_ref: None,
        manifest_sha256: None,
        source_model_path: None,
        source_model_sha256: None,
        source_model_bytes: None,
        materialized_path: None,
        materialized_pinned: false,
        projector_path: None,
        stage_id: stop.stage_id.clone(),
        stage_index: 0,
        layer_start: 0,
        layer_end: 0,
        state: skippy::StageRuntimeState::Stopped,
        bind_addr: String::new(),
        activation_width: 0,
        wire_dtype: skippy::StageWireDType::F16,
        selected_device: None,
        ctx_size: 0,
        lane_count: 0,
        n_batch: None,
        n_ubatch: None,
        flash_attn_type: FlashAttentionType::Auto,
        error: None,
        shutdown_generation: stop.shutdown_generation,
        coordinator_term: stop.coordinator_term,
        coordinator_id: None,
        lease_until_unix_ms: 0,
    }
}

pub(super) fn test_preparation_status_from_load(
    load: &skippy::StageLoadRequest,
) -> skippy::StagePreparationStatus {
    skippy::StagePreparationStatus {
        topology_id: load.topology_id.clone(),
        run_id: load.run_id.clone(),
        model_id: load.model_id.clone(),
        backend: load.backend.clone(),
        package_ref: load.package_ref.clone(),
        manifest_sha256: load.manifest_sha256.clone(),
        stage_id: load.stage_id.clone(),
        stage_index: load.stage_index,
        layer_start: load.layer_start,
        layer_end: load.layer_end,
        state: skippy::StagePreparationState::Available,
        bytes_done: load.source_model_bytes,
        bytes_total: load.source_model_bytes,
        bind_addr: None,
        error: None,
        shutdown_generation: load.shutdown_generation,
        coordinator_term: load.coordinator_term,
        coordinator_id: load.coordinator_id,
        lease_until_unix_ms: load.lease_until_unix_ms,
    }
}

pub(super) fn test_inventory_from_request(
    request: &skippy::StageInventoryRequest,
) -> skippy::StageLayerInventory {
    skippy::StageLayerInventory {
        model_id: request.model_id.clone(),
        package_ref: request.package_ref.clone(),
        manifest_sha256: request.manifest_sha256.clone(),
        layer_count: 40,
        ready_ranges: Vec::new(),
        available_ranges: vec![skippy::LayerRange {
            layer_start: 0,
            layer_end: 40,
        }],
        missing_ranges: Vec::new(),
        preparing_ranges: Vec::new(),
        source_model_path: Some("/models/qwen.gguf".to_string()),
        source_model_bytes: Some(40_000_000),
        source_model_kind: skippy::SourceModelKind::LayerPackage,
    }
}
