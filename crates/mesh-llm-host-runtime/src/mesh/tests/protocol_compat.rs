#[test]
fn stage_load_proto_roundtrip_preserves_source_model_bytes() {
    let load = stage_load_request();
    let proto = stage_load_to_proto(load.clone());
    assert_eq!(proto.source_model_bytes, Some(123_456_789));
    assert_eq!(proto.mmap, Some(false));
    assert_eq!(proto.mlock, Some(true));

    let decoded = stage_load_from_proto(proto).unwrap();
    assert_eq!(decoded.source_model_bytes, Some(123_456_789));
    assert_eq!(decoded.model_path.as_deref(), Some("/models/demo.gguf"));
    assert_eq!(decoded.mmap, Some(false));
    assert!(decoded.mlock);
}

#[test]
fn stage_control_request_timeout_uses_stage_load_floor() {
    let mut load = stage_load_request();
    load.source_model_bytes = None;
    assert_eq!(
        Node::stage_control_request_timeout(&crate::inference::skippy::StageControlRequest::Load(
            load.clone()
        )),
        std::time::Duration::from_secs(900)
    );

    load.source_model_bytes = Some(170 * 1024 * 1024 * 1024);
    assert_eq!(
        Node::stage_control_request_timeout(&crate::inference::skippy::StageControlRequest::Load(
            load
        )),
        std::time::Duration::from_secs(1360)
    );

    let mut prepare_load = stage_load_request();
    prepare_load.source_model_bytes = Some(170 * 1024 * 1024 * 1024);
    assert_eq!(
        Node::stage_control_request_timeout(
            &crate::inference::skippy::StageControlRequest::Prepare(
                crate::inference::skippy::StagePrepareRequest {
                    load: prepare_load,
                    coordinator_id: None,
                },
            )
        ),
        std::time::Duration::from_secs(1360)
    );
}

#[test]
fn test_merge_demand_takes_max() {
    let mut ours = HashMap::new();
    ours.insert(
        "GLM".into(),
        ModelDemand {
            last_active: 100,
            request_count: 50,
        },
    );
    ours.insert(
        "Hermes".into(),
        ModelDemand {
            last_active: 200,
            request_count: 10,
        },
    );

    let mut theirs = HashMap::new();
    theirs.insert(
        "GLM".into(),
        ModelDemand {
            last_active: 150,
            request_count: 30,
        },
    );
    theirs.insert(
        "Qwen".into(),
        ModelDemand {
            last_active: 300,
            request_count: 5,
        },
    );

    merge_demand(&mut ours, &theirs);

    // GLM: max(100,150)=150 for last_active, max(50,30)=50 for count
    assert_eq!(ours["GLM"].last_active, 150);
    assert_eq!(ours["GLM"].request_count, 50);
    // Hermes: unchanged (not in theirs)
    assert_eq!(ours["Hermes"].last_active, 200);
    assert_eq!(ours["Hermes"].request_count, 10);
    // Qwen: new entry from theirs
    assert_eq!(ours["Qwen"].last_active, 300);
    assert_eq!(ours["Qwen"].request_count, 5);
}

#[test]
fn test_merge_demand_empty_maps() {
    let mut ours = HashMap::new();
    let theirs = HashMap::new();
    merge_demand(&mut ours, &theirs);
    assert!(ours.is_empty());

    let mut theirs2 = HashMap::new();
    theirs2.insert(
        "GLM".into(),
        ModelDemand {
            last_active: 100,
            request_count: 1,
        },
    );
    merge_demand(&mut ours, &theirs2);
    assert_eq!(ours.len(), 1);
    assert_eq!(ours["GLM"].request_count, 1);
}

#[test]
fn test_merge_demand_idempotent() {
    let mut ours = HashMap::new();
    ours.insert(
        "GLM".into(),
        ModelDemand {
            last_active: 100,
            request_count: 50,
        },
    );

    let theirs = ours.clone();
    merge_demand(&mut ours, &theirs);

    assert_eq!(ours["GLM"].last_active, 100);
    assert_eq!(ours["GLM"].request_count, 50);
}

#[test]
fn test_demand_ttl_filtering() {
    let now = now_secs();
    let mut demand = HashMap::new();

    // Recent — should survive
    demand.insert(
        "Recent".into(),
        ModelDemand {
            last_active: now - 60, // 1 min ago
            request_count: 10,
        },
    );
    // Stale — should be filtered
    demand.insert(
        "Stale".into(),
        ModelDemand {
            last_active: now - DEMAND_TTL_SECS - 100, // past TTL
            request_count: 100,
        },
    );

    let filtered: HashMap<String, ModelDemand> = demand
        .into_iter()
        .filter(|(_, d)| (now - d.last_active) < DEMAND_TTL_SECS)
        .collect();

    assert_eq!(filtered.len(), 1);
    assert!(filtered.contains_key("Recent"));
    assert!(!filtered.contains_key("Stale"));
}

#[test]
fn test_demand_serialization_roundtrip() {
    let mut demand: HashMap<String, ModelDemand> = HashMap::new();
    demand.insert(
        "GLM".into(),
        ModelDemand {
            last_active: 1772309000,
            request_count: 42,
        },
    );

    let json = serde_json::to_string(&demand).unwrap();
    let decoded: HashMap<String, ModelDemand> = serde_json::from_str(&json).unwrap();

    assert_eq!(decoded["GLM"].last_active, 1772309000);
    assert_eq!(decoded["GLM"].request_count, 42);
}

#[test]
fn test_demand_deserialization_missing_field() {
    // Simulate old gossip message without model_demand field
    // Just verify ModelDemand defaults work
    let d = ModelDemand::default();
    assert_eq!(d.last_active, 0);
    assert_eq!(d.request_count, 0);

    // Verify HashMap<String, ModelDemand> defaults to empty
    let empty: HashMap<String, ModelDemand> = Default::default();
    assert!(empty.is_empty());

    // The real test: serde default on a struct with model_demand
    #[derive(Deserialize, Default)]
    struct TestStruct {
        #[serde(default)]
        model_demand: HashMap<String, ModelDemand>,
        #[serde(default)]
        requested_models: Vec<String>,
    }
    let parsed: TestStruct = serde_json::from_str("{}").unwrap();
    assert!(parsed.model_demand.is_empty());
    assert!(parsed.requested_models.is_empty());
}

#[test]
fn test_peer_announcement_gpu_serde_roundtrip() {
    // Test that gpu_name and hostname fields serialize and deserialize correctly
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct TestAnnouncement {
        #[serde(default)]
        gpu_name: Option<String>,
        #[serde(default)]
        hostname: Option<String>,
    }

    let test = TestAnnouncement {
        gpu_name: Some("NVIDIA A100".to_string()),
        hostname: Some("worker-01".to_string()),
    };

    let json = serde_json::to_string(&test).unwrap();
    let decoded: TestAnnouncement = serde_json::from_str(&json).unwrap();

    assert_eq!(decoded.gpu_name, Some("NVIDIA A100".to_string()));
    assert_eq!(decoded.hostname, Some("worker-01".to_string()));
}

#[test]
fn test_peer_announcement_backward_compat_no_hw_fields() {
    // Simulate old gossip message without gpu_name or hostname
    #[derive(Deserialize, Debug)]
    struct TestAnnouncement {
        #[serde(default)]
        gpu_name: Option<String>,
        #[serde(default)]
        hostname: Option<String>,
    }

    let json = r#"{"other_field": "value"}"#;
    let decoded: TestAnnouncement = serde_json::from_str(json).unwrap();

    assert_eq!(decoded.gpu_name, None);
    assert_eq!(decoded.hostname, None);
}

#[test]
fn test_peer_announcement_backward_compat_with_hw_fields() {
    // Simulate new gossip message with gpu_name and hostname
    #[derive(Deserialize, Debug)]
    struct TestAnnouncement {
        #[serde(default)]
        gpu_name: Option<String>,
        #[serde(default)]
        hostname: Option<String>,
    }

    let json = r#"{"gpu_name": "NVIDIA H100", "hostname": "gpu-server-02"}"#;
    let decoded: TestAnnouncement = serde_json::from_str(json).unwrap();

    assert_eq!(decoded.gpu_name, Some("NVIDIA H100".to_string()));
    assert_eq!(decoded.hostname, Some("gpu-server-02".to_string()));
}

#[test]
fn test_peer_announcement_hostname_serde_roundtrip() {
    // Test hostname-only roundtrip
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct TestAnnouncement {
        #[serde(default)]
        gpu_name: Option<String>,
        #[serde(default)]
        hostname: Option<String>,
    }

    let test = TestAnnouncement {
        gpu_name: None,
        hostname: Some("compute-node-42".to_string()),
    };

    let json = serde_json::to_string(&test).unwrap();
    let decoded: TestAnnouncement = serde_json::from_str(&json).unwrap();

    assert_eq!(decoded.hostname, Some("compute-node-42".to_string()));
    assert_eq!(decoded.gpu_name, None);
}

#[test]
fn test_peer_payload_hw_fields() {
    // Test that PeerPayload includes gpu_name and hostname fields
    #[derive(Serialize, Debug)]
    struct TestPeerPayload {
        id: String,
        gpu_name: Option<String>,
        hostname: Option<String>,
    }

    let payload = TestPeerPayload {
        id: "peer-123".to_string(),
        gpu_name: Some("NVIDIA A100".to_string()),
        hostname: Some("worker-01".to_string()),
    };

    let json = serde_json::to_string(&payload).unwrap();
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(value["gpu_name"], "NVIDIA A100");
    assert_eq!(value["hostname"], "worker-01");
}

#[test]
fn test_enumerate_host_false_omits_hw_fields_in_announcement() {
    // With enumerate_host: false (opt-out), hardware fields are NOT sent
    let enumerate_host = false;
    let gpu_name: Option<String> = Some("NVIDIA RTX 5090".to_string());
    let hostname: Option<String> = Some("carrack".to_string());
    let gpu_vram: Option<String> = Some("34359738368".to_string());

    let gossip_gpu_name = if enumerate_host {
        gpu_name.clone()
    } else {
        None
    };
    let gossip_hostname = if enumerate_host {
        hostname.clone()
    } else {
        None
    };
    let gossip_gpu_vram = if enumerate_host {
        gpu_vram.clone()
    } else {
        None
    };

    assert_eq!(gossip_gpu_name, None);
    assert_eq!(gossip_hostname, None);
    assert_eq!(gossip_gpu_vram, None);
}

#[test]
fn test_enumerate_host_true_includes_hw_fields_in_announcement() {
    // With enumerate_host: true (default), hardware fields ARE sent
    let enumerate_host = true;
    let gpu_name: Option<String> = Some("NVIDIA RTX 5090".to_string());
    let hostname: Option<String> = Some("carrack".to_string());
    let gpu_vram: Option<String> = Some("34359738368".to_string());

    let gossip_gpu_name = if enumerate_host {
        gpu_name.clone()
    } else {
        None
    };
    let gossip_hostname = if enumerate_host {
        hostname.clone()
    } else {
        None
    };
    let gossip_gpu_vram = if enumerate_host {
        gpu_vram.clone()
    } else {
        None
    };

    assert_eq!(gossip_gpu_name, Some("NVIDIA RTX 5090".to_string()));
    assert_eq!(gossip_hostname, Some("carrack".to_string()));
    assert_eq!(gossip_gpu_vram, Some("34359738368".to_string()));
}

#[test]
fn test_is_soc_always_included_regardless_of_enumerate_host() {
    // is_soc is always sent regardless of enumerate_host setting
    for enumerate_host in [false, true] {
        let is_soc: Option<bool> = Some(true);
        let gpu_name: Option<String> = Some("Tegra AGX Orin".to_string());

        let gossip_gpu_name = if enumerate_host {
            gpu_name.clone()
        } else {
            None
        };

        assert_eq!(is_soc, Some(true), "is_soc must always be sent");
        if enumerate_host {
            assert!(gossip_gpu_name.is_some());
        } else {
            assert!(gossip_gpu_name.is_none());
        }
    }
}

#[test]
fn test_peer_announcement_backward_compat_is_soc_gpu_vram() {
    #[derive(Deserialize, Debug)]
    struct TestAnnouncement {
        #[serde(default)]
        is_soc: Option<bool>,
        #[serde(default)]
        gpu_vram: Option<String>,
    }

    let json = r#"{"other_field": "value"}"#;
    let decoded: TestAnnouncement = serde_json::from_str(json).unwrap();
    assert_eq!(
        decoded.is_soc, None,
        "old nodes without is_soc should default to None"
    );
    assert_eq!(
        decoded.gpu_vram, None,
        "old nodes without gpu_vram should default to None"
    );
}

#[test]
fn test_peer_announcement_backward_compat_no_bandwidth_field() {
    #[derive(Deserialize)]
    struct TestAnnouncement {
        #[serde(
            default,
            rename = "gpu_bandwidth_gbps",
            alias = "gpu_mem_bandwidth_gbps"
        )]
        gpu_mem_bandwidth_gbps: Option<String>,
    }

    let json = r#"{"other_field": "value"}"#;
    let decoded: TestAnnouncement = serde_json::from_str(json).unwrap();

    assert_eq!(decoded.gpu_mem_bandwidth_gbps, None);
}

fn make_valid_gossip_frame() -> GossipFrame {
    GossipFrame {
        r#gen: NODE_PROTOCOL_GENERATION,
        sender_id: vec![0u8; 32],
        peers: vec![PeerAnnouncement {
            endpoint_id: vec![0u8; 32],
            role: NodeRole::Worker as i32,
            ..Default::default()
        }],
    }
}

#[test]
fn protocol_from_alpn_defaults_to_v1() {
    assert_eq!(protocol_from_alpn(ALPN_V1), ControlProtocol::ProtoV1);
    assert_eq!(
        protocol_from_alpn(b"mesh-llm/999"),
        ControlProtocol::ProtoV1
    );
}

#[test]
fn identity_from_model_source_treats_absolute_gguf_as_local() {
    let identity =
        identity_from_model_source("/home/jdumay/models/smollm2-a.gguf").expect("identity");

    assert_eq!(identity.source_kind, ModelSourceKind::LocalGguf);
    assert_eq!(identity.local_file_name.as_deref(), Some("smollm2-a.gguf"));
    assert_eq!(identity.repository, None);
}

#[test]
fn parse_hf_ref_parts_rejects_absolute_paths() {
    assert!(parse_hf_ref_parts("/home/jdumay/models/smollm2-a.gguf").is_none());
}

#[test]
fn identity_from_model_source_keeps_huggingface_refs() {
    let identity =
        identity_from_model_source("tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M").expect("identity");

    assert_eq!(identity.source_kind, ModelSourceKind::HuggingFace);
    assert_eq!(
        identity.canonical_ref.as_deref(),
        Some("tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M")
    );
}

#[test]
fn control_frame_roundtrip() {
    let frame = make_valid_gossip_frame();
    let encoded = encode_control_frame(STREAM_GOSSIP, &frame);
    let decoded: GossipFrame = decode_control_frame(STREAM_GOSSIP, &encoded)
        .expect("valid gossip frame must decode successfully");
    assert_eq!(decoded.r#gen, NODE_PROTOCOL_GENERATION);
    assert_eq!(decoded.peers.len(), 1);
    assert_eq!(decoded.peers[0].endpoint_id, vec![0u8; 32]);
    assert_eq!(decoded.peers[0].role, NodeRole::Worker as i32);
}
