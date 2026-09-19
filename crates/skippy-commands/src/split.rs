use std::{net::SocketAddr, path::PathBuf};

use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use skippy_api::{source, split_certification, stage_admission, stage_load};
use skippy_protocol::{PeerConfig, StageConfig};
use skippy_topology::{
    NodeSpec, PlannerPolicy, TopologyPlanRequest, dense_attention_layers, infer_family_capability,
    plan_balanced_accepted_contiguous,
};

/// Parsed `plan-split` inputs, decoupled from clap.
#[derive(Debug, Clone)]
pub struct PlanSplitCommand {
    pub model_path: PathBuf,
    pub model_id: Option<String>,
    /// Ordered worker listen endpoints, one per stage. Use routable addresses across machines.
    pub workers: Vec<SocketAddr>,
    pub ctx_size: u32,
    pub lanes: u32,
    pub n_gpu_layers: i32,
    /// New directory for stage configs and their admission descriptors; never overwritten.
    pub output_dir: PathBuf,
}

fn validate(args: &PlanSplitCommand) -> Result<()> {
    ensure!(
        args.workers.len() >= 2,
        "a split needs at least two --worker endpoints"
    );
    ensure!(
        args.ctx_size > 0 && args.lanes > 0,
        "context size and lanes must be positive"
    );
    let unique = args
        .workers
        .iter()
        .collect::<std::collections::BTreeSet<_>>();
    ensure!(
        unique.len() == args.workers.len(),
        "worker endpoints must be distinct"
    );
    ensure!(
        args.workers
            .iter()
            .all(|a| a.port() != 0 && !a.ip().is_unspecified()),
        "workers require a nonzero port and a routable or loopback address"
    );
    ensure!(
        !args.output_dir.try_exists()?,
        "output directory already exists"
    );
    Ok(())
}

pub fn run(args: PlanSplitCommand) -> Result<()> {
    validate(&args)?;
    let path = if args.model_path.is_absolute() {
        args.model_path.clone()
    } else {
        std::env::current_dir()?.join(&args.model_path)
    };
    ensure!(
        path.is_file(),
        "plan-split requires a GGUF file (the first shard for multipart models); safetensors directories are not supported"
    );
    let model_id = args.model_id.clone().unwrap_or_else(|| {
        path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("local-model")
            .into()
    });
    let identity = source::synthetic_direct_gguf_package(&model_id, &path, None)?;
    ensure!(
        identity.layer_weight_bytes.len() == identity.layer_count as usize,
        "split planning requires weight sizes for every model layer"
    );
    let (manifest, _) =
        source::planning::direct_gguf_planning_manifest_from_identity(&model_id, &identity)?;
    let architecture = manifest
        .model_metadata
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .context("model architecture missing")?;
    // Standalone enforces the same release-bound certification roster as Mesh.
    split_certification::require_split_certification(&identity, architecture, false)?;
    let run_id = format!("skippy-{}", uuid::Uuid::new_v4());
    let mut layers = dense_attention_layers(identity.layer_count, 1);
    for (layer, bytes) in layers.iter_mut().zip(&identity.layer_weight_bytes) {
        layer.parameter_bytes = *bytes;
    }
    let request = TopologyPlanRequest {
        topology_id: run_id.clone(),
        model_id: model_id.clone(),
        layers,
        nodes: args
            .workers
            .iter()
            .enumerate()
            .map(|(i, _)| NodeSpec {
                node_id: format!("worker-{i}"),
                cached_slice_bytes: 0,
                vram_bytes: 0,
            })
            .collect(),
        family: infer_family_capability(&model_id, identity.layer_count, identity.activation_width),
        policy: PlannerPolicy::default(),
    };
    let topology = plan_balanced_accepted_contiguous(&request, args.workers.len())?;
    let ranges = topology
        .stages
        .iter()
        .map(|s| (s.layer_start, s.layer_end))
        .collect::<Vec<_>>();
    let batched = args
        .lanes
        .checked_mul(8)
        .context("planning batch overflow")?;
    let profiles = [
        ("batched", batched, args.lanes),
        ("decode", args.lanes, args.lanes),
        ("prefill", 8, 1),
    ]
    .map(
        |(id, tokens, sequences)| stage_admission::StagePlannerProfile {
            profile_id: id.into(),
            n_tokens: tokens,
            n_sequences: sequences,
            n_outputs: tokens,
            n_recurrent_rollback_sequences: 0,
        },
    );
    let backend = format!(
        "skippy-backend:requested-gpu-layers:{}:v1",
        args.n_gpu_layers
    );
    let mut graph = Sha256::new();
    graph.update(b"skippy-graph-configuration:v1\0");
    graph.update(args.ctx_size.to_le_bytes());
    graph.update(args.lanes.to_le_bytes());
    graph.update((backend.len() as u64).to_le_bytes());
    graph.update(backend.as_bytes());
    let graph_id = format!(
        "skippy-graph-configuration:v1:{}",
        hex::encode(graph.finalize())
    );
    let admissions = stage_admission::realize_direct_gguf_stage_admissions(
        &model_id, &identity, &ranges, &profiles, &graph_id, &backend,
    )?;
    ensure!(
        admissions.len() == args.workers.len(),
        "admission stage count mismatch"
    );
    validate_frontiers(&admissions)?;
    let peer = |index: usize| PeerConfig {
        stage_id: topology.stages[index].stage_id.clone(),
        stage_index: index as u32,
        endpoint: args.workers[index].to_string(),
    };
    let mut configs = Vec::<StageConfig>::new();
    for (index, (stage, admission)) in topology.stages.iter().zip(&admissions).enumerate() {
        let names = stage_load::admitted_resident_tensor_names(admission, &manifest)?;
        let load = stage_load::AdmittedStageOptions {
            topology_id: run_id.clone(),
            run_id: run_id.clone(),
            model_id: model_id.clone(),
            stage_id: stage.stage_id.clone(),
            layer_start: stage.layer_start,
            layer_end: stage.layer_end,
            ctx_size: args.ctx_size,
            lane_count: args.lanes,
            selected_device: None,
            package_ref: identity.package_ref.clone(),
            manifest_sha256: identity.manifest_sha256.clone(),
            model_path: Some(identity.source_model_path.to_string_lossy().into_owned()),
            source_model_sha256: Some(identity.source_model_sha256.clone()),
            source_model_bytes: Some(identity.source_model_bytes),
            projector_path: None,
            projector_use_gpu: None,
            media_marker: None,
            image_min_tokens: None,
            image_max_tokens: None,
            batch_max_tokens: None,
            glm_dsa_policy: Default::default(),
            generation_signal_window: None,
            activation_codec: Default::default(),
            activation_codec_policy: Default::default(),
            stage_index: index as u32,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: args.n_gpu_layers,
            mmap: None,
            mlock: false,
            runtime_settings: Default::default(),
            cache_type_k: "f16".into(),
            cache_type_v: "f16".into(),
            flash_attn_type: Default::default(),
            load_mode: skippy_protocol::LoadMode::RuntimeSlice,
            native_mtp_enabled: false,
            bind_addr: args.workers[index].to_string(),
            upstream: index.checked_sub(1).map(peer),
            downstream: (index + 1 < args.workers.len()).then(|| peer(index + 1)),
            admission: admission.clone(),
        };
        configs.push(stage_load::admitted_stage_config(&load, None, names)?);
    }
    // All stages must be admitted before any output is published. Exclusive directory
    // creation prevents accidental replacement of a previous deployment plan.
    std::fs::create_dir(&args.output_dir).context("create new split plan directory")?;
    for (index, config) in configs.iter().enumerate() {
        std::fs::write(
            args.output_dir.join(format!("stage-{index}.json")),
            serde_json::to_vec_pretty(config)?,
        )?;
    }
    std::fs::write(
        args.output_dir.join("admissions.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "certification": "certified",
            "stages": admissions.iter().map(admission_json).collect::<Vec<_>>()
        }))?,
    )?;
    crate::console::write_json(
        &serde_json::json!({"certification":"certified","run_id":run_id,"model_id":model_id,"output_dir":args.output_dir,"topology":topology,"source_sha256":identity.source_model_sha256}),
    )
}

fn validate_frontiers(admissions: &[skippy_protocol::StageAdmissionDescriptor]) -> Result<()> {
    for pair in admissions.windows(2) {
        let upstream = stage_load::admitted_activation_frontier(&pair[0])?;
        let downstream = stage_load::admitted_activation_frontier(&pair[1])?;
        ensure!(
            upstream.activation_exports == downstream.activation_imports
                && upstream.activation_export_bindings == downstream.activation_import_bindings,
            "adjacent stage activation frontiers do not match",
        );
    }
    Ok(())
}

// Human-readable evidence only; worker transport continues to use the canonical
// protobuf admission representation owned by skippy-protocol.
fn admission_json(a: &skippy_protocol::StageAdmissionDescriptor) -> serde_json::Value {
    serde_json::json!({
        "version": a.version, "package_id": a.package_id, "plan_id": a.plan_id,
        "layer_start": a.layer_start, "layer_end": a.layer_end,
        "resident_tensor_ids": a.resident_tensor_ids,
        "sidecars": a.sidecars.iter().map(|s| serde_json::json!({
            "kind": match s.kind { skippy_protocol::StageAdmissionSidecarKind::Mmproj => "mmproj" },
            "artifact_id":s.artifact_id, "name":s.name,
        })).collect::<Vec<_>>(),
        "profiles": a.profiles.iter().map(|p| serde_json::json!({
            "profile_id":p.profile_id,"graph_identity":p.graph_identity,"profile_identity":p.profile_identity,
            "slice_identity":p.slice_identity,"source_snapshot_identity":p.source_snapshot_identity,
            "graph_configuration_id":p.graph_configuration_id,"backend_id":p.backend_id,
            "activation_imports":p.activation_imports,"activation_exports":p.activation_exports,
            "activation_import_bindings":p.activation_import_bindings,"activation_export_bindings":p.activation_export_bindings,
        })).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn split_publication_rejects_a_neighbor_binding_mismatch() {
        use skippy_protocol::{StageAdmissionDescriptor, StageAdmissionProfile};
        let profile = StageAdmissionProfile {
            profile_id: "decode".into(),
            graph_identity: "graph".into(),
            profile_identity: "profile".into(),
            slice_identity: "slice".into(),
            source_snapshot_identity: "source".into(),
            graph_configuration_id: "config".into(),
            backend_id: "backend".into(),
            activation_imports: vec!["activation".into()],
            activation_exports: vec!["activation".into()],
            activation_import_bindings: vec!["binding".into()],
            activation_export_bindings: vec!["binding".into()],
        };
        let first = StageAdmissionDescriptor {
            version: skippy_protocol::STAGE_ADMISSION_DESCRIPTOR_VERSION,
            package_id: "package".into(),
            plan_id: "plan".into(),
            layer_start: 0,
            layer_end: 1,
            resident_tensor_ids: vec![],
            sidecars: vec![],
            profiles: vec![profile],
        };
        let mut second = first.clone();
        second.layer_start = 1;
        second.layer_end = 2;
        validate_frontiers(&[first.clone(), second.clone()]).unwrap();
        second.profiles[0].activation_import_bindings[0] = "different-source-binding".into();
        assert!(
            validate_frontiers(&[first, second])
                .unwrap_err()
                .to_string()
                .contains("frontiers")
        );
    }

    #[test]
    fn checkpoint_directory_fails_before_native_planning_or_publication() {
        let root = tempfile::tempdir().unwrap();
        let args = PlanSplitCommand {
            model_path: root.path().to_owned(),
            model_id: None,
            workers: vec![
                "127.0.0.1:9100".parse().unwrap(),
                "127.0.0.1:9101".parse().unwrap(),
            ],
            ctx_size: 512,
            lanes: 1,
            n_gpu_layers: 0,
            output_dir: root.path().join("plan"),
        };
        assert!(
            run(args)
                .unwrap_err()
                .to_string()
                .contains("requires a GGUF file")
        );
        assert!(!root.path().join("plan").exists());
    }

    #[test]
    fn invalid_workers_fail_before_reading_model_or_writing_output() {
        let root = tempfile::tempdir().unwrap();
        let args = PlanSplitCommand {
            model_path: root.path().join("missing.gguf"),
            model_id: None,
            workers: vec!["127.0.0.1:9100".parse().unwrap(); 2],
            ctx_size: 512,
            lanes: 1,
            n_gpu_layers: 0,
            output_dir: root.path().join("plan"),
        };
        assert!(run(args).unwrap_err().to_string().contains("distinct"));
        assert!(!root.path().join("plan").exists());
    }
}
