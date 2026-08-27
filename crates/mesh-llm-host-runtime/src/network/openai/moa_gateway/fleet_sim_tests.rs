//! Synthetic fleet sizing for the MoA worker pool.
//!
//! Fabricates N admitted mesh peers directly in `state.peers` (no processes,
//! no sockets, no inference) and runs the *real* `assemble_worker_pool` and
//! `compute_actor_candidates` over them. Every input those functions read —
//! served model descriptors, gossiped `parameter_count_b`, advertised context,
//! `tool_use` capability — is a plain field on `PeerInfo`, so a 2000-node
//! fleet is 2000 structs.
//!
//! Purpose: measure how admitted pool width scales with fleet size, model
//! diversity, and tier mix, before spending anything on real inference.

use super::pool::{assemble_worker_pool, compute_actor_candidates};
use crate::inference::election;
use crate::mesh;
use crate::models::{CapabilityLevel, ModelCapabilities};
use iroh::{EndpointAddr, EndpointId, SecretKey};
use std::collections::{BTreeMap, HashMap};

/// A model as a fleet node would advertise it.
#[derive(Clone, Copy)]
struct FleetModel {
    name: &'static str,
    parameter_count_b: f64,
    context_length: u32,
}

const SMALL_MODELS: &[FleetModel] = &[
    FleetModel {
        name: "gemma-4-E4B-it-Q4_K_M",
        parameter_count_b: 4.0,
        context_length: 32768,
    },
    FleetModel {
        name: "Qwen3.5-9B-Q4_K_M",
        parameter_count_b: 9.0,
        context_length: 32768,
    },
    FleetModel {
        name: "Llama-3.2-3B-Instruct-Q4_K_M",
        parameter_count_b: 3.0,
        context_length: 32768,
    },
];

const BIG_MODELS: &[FleetModel] = &[
    FleetModel {
        name: "Qwen3.8-27B-Q4_K_M",
        parameter_count_b: 27.0,
        context_length: 32768,
    },
    FleetModel {
        name: "Qwen3-32B-Q4_K_M",
        parameter_count_b: 32.0,
        context_length: 32768,
    },
    FleetModel {
        name: "Gemma-3-27B-it-Q4_K_M",
        parameter_count_b: 27.0,
        context_length: 32768,
    },
];

fn endpoint_id(seed: u32) -> EndpointId {
    let mut bytes = [0u8; 32];
    bytes[..4].copy_from_slice(&seed.to_le_bytes());
    // Seed 0 is a valid scalar but keep it distinct from an all-zero key.
    bytes[31] = 1;
    EndpointId::from(SecretKey::from_bytes(&bytes).public())
}

/// One admitted host peer serving exactly one model.
fn fleet_peer(seed: u32, model: FleetModel) -> mesh::PeerInfo {
    let id = endpoint_id(seed);
    mesh::PeerInfo {
        id,
        addr: EndpointAddr {
            id,
            addrs: Default::default(),
        },
        mesh_id: None,
        mesh_policy_hash: None,
        genesis_policy: None,
        role: mesh::NodeRole::Host { http_port: 9337 },
        first_joined_mesh_ts: None,
        models: vec![model.name.to_string()],
        vram_bytes: 0,
        rtt_ms: None,
        model_source: None,
        admitted: true,
        serving_models: vec![model.name.to_string()],
        hosted_models: vec![model.name.to_string()],
        hosted_models_known: true,
        available_models: vec![],
        requested_models: vec![],
        explicit_model_interests: vec![],
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
        available_model_metadata: vec![],
        experts_summary: None,
        available_model_sizes: HashMap::new(),
        served_model_descriptors: vec![mesh::ServedModelDescriptor {
            identity: mesh::ServedModelIdentity {
                model_name: model.name.to_string(),
                is_primary: true,
                ..Default::default()
            },
            capabilities_known: true,
            capabilities: ModelCapabilities {
                tool_use: CapabilityLevel::Supported,
                ..Default::default()
            },
            topology: None,
            metadata: Some(mesh::ServedModelMetadata {
                parameter_count_b: Some(model.parameter_count_b),
                native_context_length: Some(model.context_length),
                ..Default::default()
            }),
        }],
        served_model_runtime: vec![mesh::ModelRuntimeDescriptor {
            model_name: model.name.to_string(),
            identity_hash: None,
            context_length: Some(model.context_length),
            ready: true,
        }],
        owner_attestation: None,
        release_attestation_summary: crate::ReleaseAttestationSummary::default(),
        artifact_transfer_supported: false,
        stage_protocol_generation_supported: false,
        stage_status_list_supported: false,
        owner_summary: Default::default(),
        advertised_model_throughput: vec![],
        inference_admission_state: None,
        display_rtt: None,
        selected_path: None,
        propagated_latency: None,
    }
}

/// Build a node whose mesh view is `fleet`: (model, replica count) pairs.
async fn node_with_fleet(fleet: &[(FleetModel, usize)]) -> mesh::Node {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("test node");
    let mut seed = 1u32;
    for (model, replicas) in fleet {
        for _ in 0..*replicas {
            node.insert_test_peer(fleet_peer(seed, *model)).await;
            seed += 1;
        }
    }
    node
}

/// Admitted worker names for a fleet, using the real assembly path.
async fn admitted_pool(fleet: &[(FleetModel, usize)]) -> Vec<String> {
    let node = node_with_fleet(fleet).await;
    let targets = election::ModelTargets::default();
    let http = reqwest::Client::new();
    let (_backends, models) =
        assemble_worker_pool(&node, Some(&targets), Some(13_000), &http).await;
    models.into_iter().map(|m| m.name).collect()
}

fn total_nodes(fleet: &[(FleetModel, usize)]) -> usize {
    fleet.iter().map(|(_, n)| *n).sum()
}

/// Pool width is a function of distinct model count, not fleet size.
///
/// This is the claim the whole fleet plan rests on: adding identical nodes
/// cannot widen the committee, because `assemble_worker_pool` resolves exactly
/// one worker per canonical model name.
#[tokio::test]
async fn pool_width_is_flat_in_fleet_size() {
    let mut rows: Vec<(usize, usize, usize)> = Vec::new();
    for replicas in [1usize, 10, 100, 1000] {
        let fleet = vec![
            (BIG_MODELS[0], replicas),
            (SMALL_MODELS[0], replicas),
            (SMALL_MODELS[1], replicas),
        ];
        let pool = admitted_pool(&fleet).await;
        rows.push((total_nodes(&fleet), 3, pool.len()));
    }

    for (nodes, distinct_models, admitted) in &rows {
        println!("nodes={nodes:>5} distinct_models={distinct_models} admitted={admitted}");
    }

    let widths: Vec<usize> = rows.iter().map(|(_, _, w)| *w).collect();
    assert!(
        widths.windows(2).all(|w| w[0] == w[1]),
        "admitted pool width changed with fleet size: {widths:?}"
    );
}

/// Sweep distinct model count with the fleet size held constant.
#[tokio::test]
async fn pool_width_tracks_model_diversity() {
    let mut rows: Vec<(usize, usize)> = Vec::new();
    for distinct in 1usize..=6 {
        let mut fleet = Vec::new();
        let all: Vec<FleetModel> = BIG_MODELS
            .iter()
            .zip(SMALL_MODELS.iter())
            .flat_map(|(b, s)| [*b, *s])
            .collect();
        for model in all.iter().take(distinct) {
            fleet.push((*model, 200usize));
        }
        let pool = admitted_pool(&fleet).await;
        rows.push((distinct, pool.len()));
    }
    for (distinct, admitted) in &rows {
        println!("distinct_models={distinct} nodes=1200 admitted={admitted}");
    }
}

/// The bimodal fleet Mic described: 1200 small nodes, 800 big nodes.
#[tokio::test]
async fn bimodal_fleet_admission_and_actor() {
    let fleet = vec![
        (SMALL_MODELS[0], 600usize),
        (SMALL_MODELS[1], 600),
        (BIG_MODELS[0], 400),
        (BIG_MODELS[1], 400),
    ];
    let node = node_with_fleet(&fleet).await;
    let targets = election::ModelTargets::default();
    let http = reqwest::Client::new();
    let (_backends, models) =
        assemble_worker_pool(&node, Some(&targets), Some(13_000), &http).await;
    let actors = compute_actor_candidates(&node, &models).await;

    println!("fleet nodes = {}", total_nodes(&fleet));
    println!(
        "admitted workers = {:?}",
        models.iter().map(|m| m.name.as_str()).collect::<Vec<_>>()
    );
    println!(
        "actor order = {:?}",
        actors
            .iter()
            .filter_map(|&i| models.get(i).map(|m| m.name.as_str()))
            .collect::<Vec<_>>()
    );

    let by_name: BTreeMap<&str, ()> = models.iter().map(|m| (m.name.as_str(), ())).collect();
    println!("distinct admitted names = {}", by_name.len());
}

/// How many of the fleet's nodes can ever receive a call for one turn?
#[tokio::test]
async fn calls_per_turn_vs_fleet_size() {
    let mut summary: HashMap<usize, usize> = HashMap::new();
    for replicas in [1usize, 50, 500] {
        let fleet = vec![
            (BIG_MODELS[0], replicas),
            (BIG_MODELS[1], replicas),
            (SMALL_MODELS[0], replicas),
            (SMALL_MODELS[1], replicas),
        ];
        let pool = admitted_pool(&fleet).await;
        summary.insert(total_nodes(&fleet), pool.len());
        println!(
            "nodes={:>5} admitted={} names={pool:?}",
            total_nodes(&fleet),
            pool.len()
        );
    }
}
