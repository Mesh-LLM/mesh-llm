use super::*;
use crate::protocol::local_ann_to_proto_ann;
use iroh::SecretKey;

fn endpoint_id(seed: u8) -> EndpointId {
    EndpointId::from(SecretKey::from_bytes(&[seed; 32]).public())
}

fn advertised_addr(seed: u8) -> EndpointAddr {
    EndpointAddr {
        id: endpoint_id(seed),
        addrs: [
            TransportAddr::Relay("https://relay.example.org./".parse().unwrap()),
            TransportAddr::Ip("203.0.113.5:4433".parse().unwrap()),
            TransportAddr::Ip("[2001:db8::1]:4433".parse().unwrap()),
        ]
        .into_iter()
        .collect(),
    }
}

/// A host announcement as a peer would decode it, carrying self-asserted
/// fields alongside hop measurements and short-lived data.
fn decoded_host_announcement(seed: u8) -> PeerAnnouncement {
    let addr = advertised_addr(seed);
    let wire_ann = wire::PeerAnnouncement {
        endpoint_id: addr.id.as_bytes().to_vec(),
        serialized_addr: serde_json::to_vec(&addr).unwrap(),
        role: wire::NodeRole::Host as i32,
        http_port: Some(9337),
        version: Some("0.78.0".to_string()),
        vram_bytes: 24_000_000_000,
        mesh_id: Some("mesh-1".to_string()),
        mesh_policy_hash: Some("policy-1".to_string()),
        serving_models: vec!["Qwen3-8B".to_string()],
        hosted_models: vec!["Qwen3-8B".to_string()],
        hosted_models_known: Some(true),
        served_model_identities: vec![wire::ServedModelIdentity {
            model_name: "Qwen3-8B".to_string(),
            is_primary: true,
            ..Default::default()
        }],
        requested_models: vec!["Llama-3-70B".to_string()],
        catalog_models: vec!["Qwen3-8B".to_string()],
        model_source: Some("hf://Qwen/Qwen3-8B".to_string()),
        first_joined_mesh_ts: Some(1_700_000_000),
        hardware: Some(wire::HardwareInfo {
            hostname: Some("studio".to_string()),
            is_soc: Some(true),
            ..Default::default()
        }),
        inference_admission_state: Some(wire::InferenceAdmissionState::Accepting as i32),
        demand: vec![wire::ModelDemandEntry {
            model_name: "Qwen3-8B".to_string(),
            last_active: 1,
            request_count: 3,
        }],
        latency_ms: Some(999),
        latency_source: Some(wire::LatencySource::Direct as i32),
        ..Default::default()
    };
    proto_ann_to_local(&wire_ann).expect("valid announcement").1
}

#[test]
fn record_round_trip_keeps_what_the_node_asserts_about_itself() {
    let original = decoded_host_announcement(0x41);
    let record = node_record_from_wire_ann(&local_ann_to_proto_ann(&original), &original.addr);

    let (addr, restored) =
        node_record_to_local(original.addr.id, &record, None).expect("record converts");

    assert_eq!(addr, original.addr);
    assert_eq!(restored.addr, original.addr);
    assert!(matches!(
        restored.role,
        crate::mesh::NodeRole::Host { http_port: 9337 }
    ));
    assert_eq!(restored.version, original.version);
    assert_eq!(restored.vram_bytes, original.vram_bytes);
    assert_eq!(restored.hostname.as_deref(), Some("studio"));
    assert_eq!(restored.is_soc, Some(true));
    assert_eq!(restored.mesh_id, original.mesh_id);
    assert_eq!(restored.mesh_policy_hash, original.mesh_policy_hash);
    assert_eq!(restored.serving_models, original.serving_models);
    assert_eq!(restored.hosted_models, original.hosted_models);
    assert_eq!(
        restored.served_model_descriptors.len(),
        original.served_model_descriptors.len()
    );
    assert_eq!(restored.requested_models, original.requested_models);
    assert_eq!(restored.models, original.models);
    assert_eq!(restored.model_source, original.model_source);
    assert_eq!(restored.first_joined_mesh_ts, original.first_joined_mesh_ts);
    assert_eq!(
        restored.inference_admission_state,
        original.inference_admission_state
    );
    assert_eq!(
        restored.stage_status_list_supported,
        original.stage_status_list_supported
    );
}

#[test]
fn record_leaves_out_hop_measurements_and_short_lived_data() {
    let original = decoded_host_announcement(0x42);
    assert!(!original.model_demand.is_empty());
    assert_eq!(original.latency_ms, Some(999));
    let record = node_record_from_wire_ann(&local_ann_to_proto_ann(&original), &original.addr);

    let (_, restored) =
        node_record_to_local(original.addr.id, &record, None).expect("record converts");

    assert!(restored.model_demand.is_empty());
    assert_eq!(restored.latency_ms, None);
    assert!(restored.cache_affinity.is_none());
    assert!(restored.direct_admission_proof.is_none());
}

#[test]
fn hop_observation_supplies_the_relays_latency_view() {
    let original = decoded_host_announcement(0x43);
    let record = node_record_from_wire_ann(&local_ann_to_proto_ann(&original), &original.addr);
    let observer = endpoint_id(0x77);
    let mut relay_view = original.clone();
    relay_view.latency_ms = Some(42);
    relay_view.latency_source = Some(wire::LatencySource::Estimated);
    relay_view.latency_age_ms = Some(1_500);
    relay_view.latency_observer_id = Some(observer);
    let hop = hop_observation_from_ann(&relay_view);

    let (_, restored) =
        node_record_to_local(original.addr.id, &record, Some(&hop)).expect("record converts");

    assert_eq!(restored.latency_ms, Some(42));
    assert_eq!(
        restored.latency_source,
        Some(wire::LatencySource::Estimated)
    );
    assert_eq!(restored.latency_age_ms, Some(1_500));
    assert_eq!(restored.latency_observer_id, Some(observer));
}
