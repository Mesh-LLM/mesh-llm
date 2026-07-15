fn make_test_peer(id: EndpointId, rtt_ms: Option<u32>, vram_gb: u64) -> PeerInfo {
    PeerInfo {
        id,
        addr: EndpointAddr {
            id,
            addrs: Default::default(),
        },
        mesh_id: None,
        mesh_policy_hash: None,
        genesis_policy: None,
        role: super::NodeRole::Worker,
        first_joined_mesh_ts: None,
        models: vec![],
        vram_bytes: vram_gb * 1024 * 1024 * 1024,
        rtt_ms,
        model_source: None,
        admitted: true,
        serving_models: vec![],
        hosted_models: vec![],
        hosted_models_known: false,
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
        served_model_descriptors: vec![],
        served_model_runtime: vec![],
        owner_attestation: None,
        release_attestation_summary: crate::ReleaseAttestationSummary::default(),
        artifact_transfer_supported: false,
        stage_protocol_generation_supported: false,
        stage_status_list_supported: false,
        owner_summary: OwnershipSummary::default(),
        advertised_model_throughput: vec![],

        display_rtt: None,
        selected_path: None,
        propagated_latency: None,
    }
}

/// RTT re-election: when a peer's RTT drops from above the 80ms split
/// threshold to below it (e.g. relay → direct), update_peer_rtt must
/// trigger a peer_change event so the election loop re-runs and can
/// now include the peer in split mode.
#[tokio::test]
async fn test_rtt_drop_triggers_reelection() -> Result<()> {
    let node = make_test_node(super::NodeRole::Worker).await?;
    let peer_key = SecretKey::generate();
    let peer_id = EndpointId::from(peer_key.public());

    // Add a fake peer with high relay RTT
    {
        let mut state = node.state.lock().await;
        state
            .peers
            .insert(peer_id, make_test_peer(peer_id, Some(2600), 16));
    }

    let rx = node.peer_change_rx.clone();

    // Update RTT to still-high value — should NOT trigger
    node.update_peer_rtt(peer_id, 500).await;
    assert!(
        !rx.has_changed()
            .expect("peer_change_rx closed unexpectedly"),
        "RTT 2600→500 (both above threshold) should not trigger re-election"
    );

    // Update RTT to below threshold — SHOULD trigger
    node.update_peer_rtt(peer_id, 15).await;
    assert!(
        rx.has_changed()
            .expect("peer_change_rx closed unexpectedly"),
        "RTT 500→15 (crossing threshold) must trigger re-election"
    );

    Ok(())
}

/// RTT re-election should NOT trigger when RTT was already below threshold.
#[tokio::test]
async fn test_rtt_below_threshold_no_reelection() -> Result<()> {
    let node = make_test_node(super::NodeRole::Worker).await?;
    let peer_key = SecretKey::generate();
    let peer_id = EndpointId::from(peer_key.public());

    {
        let mut state = node.state.lock().await;
        state
            .peers
            .insert(peer_id, make_test_peer(peer_id, Some(20), 16));
    }

    let rx = node.peer_change_rx.clone();

    // Update RTT to another low value — should NOT trigger
    node.update_peer_rtt(peer_id, 15).await;
    assert!(
        !rx.has_changed()
            .expect("peer_change_rx closed unexpectedly"),
        "RTT 20→15 (both below threshold) should not trigger re-election"
    );

    Ok(())
}

/// RTT re-election should NOT trigger for unknown peers.
#[tokio::test]
async fn test_rtt_update_unknown_peer_no_panic() -> Result<()> {
    let node = make_test_node(super::NodeRole::Worker).await?;
    let peer_key = SecretKey::generate();
    let peer_id = EndpointId::from(peer_key.public());

    let rx = node.peer_change_rx.clone();

    // Update RTT for a peer that doesn't exist — should not panic or trigger
    node.update_peer_rtt(peer_id, 15).await;
    assert!(
        !rx.has_changed()
            .expect("peer_change_rx closed unexpectedly"),
        "RTT update for unknown peer should not trigger re-election"
    );

    Ok(())
}

/// RTT should never increase — relay gossip RTT must not overwrite
/// a known-good direct path measurement.
#[tokio::test]
async fn test_rtt_cannot_regress() -> Result<()> {
    let node = make_test_node(super::NodeRole::Worker).await?;
    let peer_key = SecretKey::generate();
    let peer_id = EndpointId::from(peer_key.public());

    {
        let mut state = node.state.lock().await;
        state
            .peers
            .insert(peer_id, make_test_peer(peer_id, Some(20), 16));
    }

    // Try to raise RTT — should be rejected
    node.update_peer_rtt(peer_id, 2600).await;
    {
        let state = node.state.lock().await;
        let rtt = state.peers.get(&peer_id).unwrap().rtt_ms;
        assert_eq!(rtt, Some(20), "RTT must not increase from 20 to 2600");
    }

    // Lower RTT — should be accepted
    node.update_peer_rtt(peer_id, 10).await;
    {
        let state = node.state.lock().await;
        let rtt = state.peers.get(&peer_id).unwrap().rtt_ms;
        assert_eq!(rtt, Some(10), "RTT must decrease from 20 to 10");
    }

    Ok(())
}

/// Discovered peers must still be dialed directly before admission.
#[tokio::test]
async fn test_connect_to_peer_attempts_direct_verification_for_known_unadmitted_peer() -> Result<()>
{
    let node = make_test_node(super::NodeRole::Client).await?;
    let peer_key = SecretKey::generate();
    let peer_id = EndpointId::from(peer_key.public());

    // Simulate a transitive peer: tracked as a hint but not yet admitted.
    {
        let mut state = node.state.lock().await;
        let mut peer = make_test_peer(peer_id, Some(50), 8);
        peer.admitted = false;
        state.peers.insert(peer_id, peer);
        assert!(
            !state.connections.contains_key(&peer_id),
            "setup: peer must not have a connection"
        );
    }

    // connect_to_peer must attempt direct verification instead of treating the
    // hint as already admitted.
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(1),
        node.connect_to_peer(super::EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        }),
    )
    .await;

    assert!(
        result.is_ok(),
        "connect_to_peer should complete quickly for a discovered-only peer"
    );
    assert!(
        result.unwrap().is_err(),
        "connect_to_peer must try direct verification instead of silently accepting a hint"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_on_demand_transitive_peer_connection_completes_gossip() -> Result<()> {
    let host = make_test_node(super::NodeRole::Host { http_port: 9337 }).await?;
    let bridge = make_test_node(super::NodeRole::Worker).await?;
    let client = make_test_node(super::NodeRole::Client).await?;

    host.set_hosted_models(vec!["remote-coding-model".to_string()])
        .await;
    host.start_accepting();
    bridge.start_accepting();
    client.start_accepting();

    bridge.sync_from_peer_for_tests(&host).await;
    assert!(bridge.peers().await.iter().any(|peer| peer.id == host.id()));

    client.sync_from_peer_for_tests(&bridge).await;
    assert!(
        client
            .peers()
            .await
            .iter()
            .any(|peer| peer.id == bridge.id())
    );

    {
        let state = client.state.lock().await;
        assert!(
            !state.connections.contains_key(&host.id()),
            "setup: host should be known transitively but not directly connected"
        );
    }
    assert!(
        !client
            .hosts_for_model("remote-coding-model")
            .await
            .contains(&host.id()),
        "setup: client must not route to the transitive host before direct verification"
    );

    let _conn = client.connection_to_peer(host.id()).await?;

    wait_for_peer(&client, host.id()).await;
    {
        let state = client.state.lock().await;
        assert!(
            state.connections.contains_key(&host.id()),
            "on-demand connection should be retained after gossip succeeds"
        );
    }
    assert!(
        client
            .hosts_for_model("remote-coding-model")
            .await
            .contains(&host.id()),
        "the host should become routable after direct gossip succeeds"
    );

    Ok(())
}

#[test]
fn legacy_config_stream_ids_are_reserved_and_require_admission() {
    assert!(
        !stream_allowed_before_admission(STREAM_CONFIG_SUBSCRIBE, TrustPolicy::Off),
        "reserved STREAM_CONFIG_SUBSCRIBE (0x0b) must not bypass admission"
    );
    assert!(
        !stream_allowed_before_admission(STREAM_CONFIG_PUSH, TrustPolicy::Off),
        "reserved STREAM_CONFIG_PUSH (0x0c) must not bypass admission"
    );
}

fn test_owner_keypair(signing_seed: u8, encryption_seed: u8) -> crate::crypto::OwnerKeypair {
    crate::crypto::OwnerKeypair::from_bytes(&[signing_seed; 32], &[encryption_seed; 32])
        .expect("test owner keypair must be valid")
}

fn requirement_policy_owner() -> crate::crypto::OwnerKeypair {
    test_owner_keypair(0xb1, 0xb2)
}

fn proto_signed_node_ownership(
    ownership: &crate::crypto::SignedNodeOwnership,
) -> crate::proto::node::SignedNodeOwnership {
    crate::proto::node::SignedNodeOwnership {
        version: ownership.claim.version,
        cert_id: ownership.claim.cert_id.clone(),
        owner_id: ownership.claim.owner_id.clone(),
        owner_sign_public_key: hex::decode(&ownership.claim.owner_sign_public_key)
            .expect("test owner_sign_public_key must decode"),
        node_endpoint_id: hex::decode(&ownership.claim.node_endpoint_id)
            .expect("test node_endpoint_id must decode"),
        issued_at_unix_ms: ownership.claim.issued_at_unix_ms,
        expires_at_unix_ms: ownership.claim.expires_at_unix_ms,
        node_label: ownership.claim.node_label.clone(),
        hostname_hint: ownership.claim.hostname_hint.clone(),
        signature: hex::decode(&ownership.signature).expect("test signature must decode"),
    }
}

async fn open_owner_control_stream(
    target: &Node,
    owner_keypair: &crate::crypto::OwnerKeypair,
) -> Result<(
    Endpoint,
    iroh::endpoint::SendStream,
    iroh::endpoint::RecvStream,
    EndpointId,
)> {
    let endpoint = Endpoint::builder(iroh::endpoint::presets::Minimal)
        .secret_key(SecretKey::generate())
        .alpns(vec![ALPN_CONTROL_V1.to_vec()])
        .relay_mode(iroh::endpoint::RelayMode::Disabled)
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))?
        .bind()
        .await?;
    let ownership = sign_node_ownership(
        owner_keypair,
        endpoint.id().as_bytes(),
        current_time_unix_ms() + DEFAULT_NODE_CERT_LIFETIME_SECS * 1000,
        None,
        None,
    )?;
    let control_addr = Node::decode_invite_token(
        &target
            .control_endpoint()
            .await
            .expect("control endpoint should be available for owner-control tests"),
    )?;
    let conn = endpoint.connect(control_addr, ALPN_CONTROL_V1).await?;
    let (mut send, recv) = conn.open_bi().await?;
    write_len_prefixed(
        &mut send,
        &crate::proto::node::OwnerControlEnvelope {
            r#gen: NODE_PROTOCOL_GENERATION,
            handshake: Some(crate::proto::node::OwnerControlHandshake {
                ownership: Some(proto_signed_node_ownership(&ownership)),
            }),
            request: None,
            response: None,
            error: None,
        }
        .encode_to_vec(),
    )
    .await?;
    let endpoint_id = endpoint.id();
    Ok((endpoint, send, recv, endpoint_id))
}

async fn read_owner_control_envelope(
    recv: &mut iroh::endpoint::RecvStream,
) -> Result<crate::proto::node::OwnerControlEnvelope> {
    let bytes = crate::protocol::read_len_prefixed(recv).await?;
    let envelope = crate::proto::node::OwnerControlEnvelope::decode(bytes.as_slice())?;
    envelope
        .validate_frame()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    Ok(envelope)
}

async fn start_owner_control_test_server(
    owner_keypair: &crate::crypto::OwnerKeypair,
    config_dir: &std::path::Path,
) -> Result<(Node, SecretKey, std::path::PathBuf)> {
    let (node, secret_key) =
        Node::new_for_tests_with_secret(super::NodeRole::Host { http_port: 9337 }).await?;
    let config_path = config_dir.join("config.toml");
    *node.config_state.lock().await =
        crate::runtime::config_state::ConfigState::load(&config_path).unwrap_or_default();

    let ownership = sign_node_ownership(
        owner_keypair,
        node.id().as_bytes(),
        current_time_unix_ms() + DEFAULT_NODE_CERT_LIFETIME_SECS * 1000,
        None,
        None,
    )?;
    let trust_store = TrustStore::default();
    let owner_summary = verify_node_ownership(
        Some(&ownership),
        node.id().as_bytes(),
        &trust_store,
        TrustPolicy::Off,
        current_time_unix_ms(),
    );
    *node.owner_attestation.lock().await = Some(ownership);
    *node.owner_summary.lock().await = owner_summary;
    *node.trust_store.lock().await = trust_store;
    node.maybe_start_control_listener(secret_key.clone(), None, None, None)
        .await?;
    Ok((node, secret_key, config_path))
}

/// Wait until `node` has `target` in its peers list. Times out after 5 s.
/// Poll `node.peers()` until `target` appears in the list.
///
/// Panics (via `expect`) if `target` is not admitted within 5 seconds.
async fn wait_for_peer(node: &Node, target: EndpointId) {
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            if node.peers().await.iter().any(|p| p.id == target) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("peer was not admitted within 5 s");
}

fn requirement_policy(trusted_signer: &str) -> crate::MeshGenesisPolicy {
    crate::MeshGenesisPolicy::new(
        requirement_policy_owner().owner_id(),
        1_717_171_717_000,
        crate::MeshRequirements {
            node_version: crate::NodeVersionBounds::default(),
            protocol_generation: crate::ProtocolGenerationBounds {
                min: Some(NODE_PROTOCOL_GENERATION),
                max: Some(NODE_PROTOCOL_GENERATION),
            },
            release_attestation: crate::ReleaseAttestationRequirement {
                required: true,
                allowed_signer_keys: vec![trusted_signer.to_string()],
            },
        },
    )
    .expect("test mesh policy should validate")
}

fn requirement_policy_without_release_attestation() -> crate::MeshGenesisPolicy {
    crate::MeshGenesisPolicy::new(
        requirement_policy_owner().owner_id(),
        1_717_171_717_000,
        crate::MeshRequirements {
            node_version: crate::NodeVersionBounds::default(),
            protocol_generation: crate::ProtocolGenerationBounds {
                min: Some(NODE_PROTOCOL_GENERATION),
                max: Some(NODE_PROTOCOL_GENERATION),
            },
            release_attestation: crate::ReleaseAttestationRequirement {
                required: false,
                allowed_signer_keys: vec![],
            },
        },
    )
    .expect("test mesh policy should validate")
}

fn test_release_signing_key(seed: u8) -> ed25519_dalek::SigningKey {
    ed25519_dalek::SigningKey::from_bytes(&[seed; 32])
}

fn test_release_signer_key_id(seed: u8) -> String {
    format!(
        "ed25519:{}",
        hex::encode(test_release_signing_key(seed).verifying_key().as_bytes())
    )
}

fn test_release_attestation_with_seed(seed: u8) -> crate::ReleaseBuildAttestation {
    let signing_key = test_release_signing_key(seed);
    let mut attestation = crate::ReleaseBuildAttestation {
        version: 1,
        node_version: crate::VERSION.to_string(),
        build_id: "test-build".into(),
        commit: "deadbeef".into(),
        target_triple: "x86_64-apple-darwin".into(),
        supported_protocol_generation_min: Some(NODE_PROTOCOL_GENERATION),
        supported_protocol_generation_max: Some(NODE_PROTOCOL_GENERATION),
        artifact_digest: Some("sha256:test".into()),
        signer_key_id: test_release_signer_key_id(seed),
        signature_algorithm: "ed25519".into(),
        signature: vec![0; 64],
    };
    attestation.signature = ed25519_dalek::Signer::sign(
        &signing_key,
        &attestation
            .canonical_bytes()
            .expect("canonical release attestation bytes"),
    )
    .to_bytes()
    .to_vec();
    attestation
}

fn test_release_attestation(signer_key_id: &str) -> crate::ReleaseBuildAttestation {
    let mut attestation = test_release_attestation_with_seed(9);
    attestation.signer_key_id = signer_key_id.into();
    attestation
}

fn direct_proof_signing_key(seed: u8) -> SecretKey {
    let mut bytes = [0u8; 32];
    bytes[0] = seed;
    SecretKey::from_bytes(&bytes)
}

fn direct_proof_for_announcement(
    sender_seed: u8,
    mesh_id: &str,
    policy_hash: &str,
    release_attestation: Option<&crate::ReleaseBuildAttestation>,
) -> crate::DirectNodeAdmissionProof {
    direct_proof_for_announcement_at(
        sender_seed,
        mesh_id,
        policy_hash,
        release_attestation,
        current_time_unix_ms(),
    )
}

fn direct_proof_for_announcement_at(
    sender_seed: u8,
    mesh_id: &str,
    policy_hash: &str,
    release_attestation: Option<&crate::ReleaseBuildAttestation>,
    timestamp_unix_ms: u64,
) -> crate::DirectNodeAdmissionProof {
    let signing_key =
        ed25519_dalek::SigningKey::from_bytes(&direct_proof_signing_key(sender_seed).to_bytes());
    let attestation_hash = release_attestation
        .map(|attestation| {
            attestation
                .canonical_hash_hex()
                .unwrap_or_else(|_| "invalid-release-attestation".to_string())
        })
        .unwrap_or_else(|| "missing-release-attestation".to_string());
    let mut proof = crate::DirectNodeAdmissionProof {
        version: 1,
        sender_id: make_test_endpoint_id(sender_seed).as_bytes().to_vec(),
        mesh_id: mesh_id.to_string(),
        policy_hash: policy_hash.to_string(),
        attestation_hash,
        timestamp_unix_ms,
        signature_algorithm: "ed25519".to_string(),
        signature: vec![],
    };
    proof.signature = ed25519_dalek::Signer::sign(
        &signing_key,
        &proof
            .canonical_bytes()
            .expect("canonical direct proof bytes"),
    )
    .to_bytes()
    .to_vec();
    proof
}

async fn install_requirement_policy(node: &Node, policy: &crate::MeshGenesisPolicy) -> Result<()> {
    let mesh_id = policy
        .policy_derived_mesh_id()
        .map_err(|reason| anyhow::anyhow!("invalid test mesh id: {reason:?}"))?;
    let policy_hash = policy
        .canonical_hash_hex()
        .map_err(|reason| anyhow::anyhow!("invalid test policy hash: {reason:?}"))?;
    let owner = requirement_policy_owner();
    let signed_policy = crate::SignedMeshGenesisPolicy::sign(policy.clone(), &owner)
        .map_err(|reason| anyhow::anyhow!("invalid test signed policy: {reason:?}"))?;
    let token = crate::SignedBootstrapToken::sign(
        vec![serde_json::to_vec(&node.endpoint_addr_for_advertisement())?],
        &signed_policy,
        Some(current_time_unix_ms() + SIGNED_BOOTSTRAP_TOKEN_LIFETIME_MS),
        &owner,
    )
    .map_err(|reason| anyhow::anyhow!("invalid test bootstrap token: {reason:?}"))?;
    node.install_requirement_aware_mesh_state(
        mesh_id,
        policy_hash,
        policy.clone(),
        Some(signed_policy),
        Some(token),
    )
    .await
}

async fn configure_requirement_node(
    node: &Node,
    policy: &crate::MeshGenesisPolicy,
    signer: Option<&str>,
) -> Result<()> {
    install_requirement_policy(node, policy).await?;
    *node.release_attestation.lock().await = signer.map(test_release_attestation);
    Ok(())
}

fn requirement_peer_announcement(
    sender_seed: u8,
    policy: &crate::MeshGenesisPolicy,
    release_attestation: Option<crate::ReleaseBuildAttestation>,
    direct_admission_proof: Option<crate::DirectNodeAdmissionProof>,
) -> super::PeerAnnouncement {
    super::PeerAnnouncement {
        addr: EndpointAddr {
            id: make_test_endpoint_id(sender_seed),
            addrs: Default::default(),
        },
        role: super::NodeRole::Worker,
        first_joined_mesh_ts: None,
        models: vec![],
        vram_bytes: 0,
        model_source: None,
        serving_models: vec![],
        hosted_models: None,
        available_models: vec![],
        requested_models: vec![],
        explicit_model_interests: vec![],
        version: Some(crate::VERSION.to_string()),
        model_demand: HashMap::new(),
        mesh_id: Some(policy.policy_derived_mesh_id().expect("mesh id")),
        mesh_policy_hash: Some(policy.canonical_hash_hex().expect("policy hash")),
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
        served_model_descriptors: vec![],
        served_model_runtime: vec![],
        owner_attestation: None,
        genesis_policy: None,
        release_attestation,
        direct_admission_proof,
        artifact_transfer_supported: true,
        stage_protocol_generation_supported: true,
        stage_status_list_supported: true,
        advertised_model_throughput: vec![],
        latency_ms: None,
        latency_source: None,
        latency_age_ms: None,
        latency_observer_id: None,
    }
}

async fn expect_no_route_table_response(requester: &Node, target: &Node) -> Result<()> {
    use prost::Message as _;

    let conn = connect_mesh(
        &requester.endpoint,
        target.endpoint_addr_for_advertisement(),
    )
    .await?;
    let (mut send, mut recv) = conn.open_bi().await?;
    send.write_all(&[STREAM_ROUTE_REQUEST]).await?;
    let request = RouteTableRequest {
        requester_id: requester.id().as_bytes().to_vec(),
        r#gen: NODE_PROTOCOL_GENERATION,
    };
    write_len_prefixed(&mut send, &request.encode_to_vec()).await?;
    send.finish()?;

    let result = tokio::time::timeout(
        std::time::Duration::from_millis(500),
        read_len_prefixed(&mut recv),
    )
    .await;
    assert!(
        result.is_err()
            || result
                .expect("route timeout should already be handled")
                .is_err(),
        "rejected peer must not receive a route table"
    );
    Ok(())
}

pub(crate) fn assert_mesh_requirements_outbound_admits_compliant_peer_after_requirements_pass() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let host = make_test_node(super::NodeRole::Host { http_port: 9337 })
            .await
            .expect("host node");
        let joiner = make_test_node(super::NodeRole::Worker)
            .await
            .expect("joiner node");
        let trusted_signer = test_release_signer_key_id(9);
        let policy = requirement_policy(&trusted_signer);

        configure_requirement_node(&host, &policy, Some(&trusted_signer))
            .await
            .expect("configure host policy");
        configure_requirement_node(&joiner, &policy, Some(&trusted_signer))
            .await
            .expect("configure joiner policy");

        host.start_accepting();
        joiner.start_accepting();
        joiner
            .join(&host.invite_token().await)
            .await
            .expect("join should succeed");

        wait_for_peer(&joiner, host.id()).await;
        wait_for_peer(&host, joiner.id()).await;
    });
}

pub(crate) fn assert_mesh_requirements_inbound_rejects_before_topology_announcement() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let host = make_test_node(super::NodeRole::Host { http_port: 9337 })
            .await
            .expect("host node");
        let joiner = make_test_node(super::NodeRole::Worker)
            .await
            .expect("joiner node");
        let trusted_signer = test_release_signer_key_id(9);
        let policy = requirement_policy(&trusted_signer);

        configure_requirement_node(&host, &policy, Some(&trusted_signer))
            .await
            .expect("configure host policy");
        configure_requirement_node(&joiner, &policy, None)
            .await
            .expect("configure joiner policy");

        host.start_accepting();
        joiner.start_accepting();

        let _error = joiner
            .join(&host.invite_token().await)
            .await
            .expect_err("join should fail");
        assert!(
            joiner.peers().await.iter().all(|peer| peer.id != host.id()),
            "inbound rejection must happen before the joiner receives host topology"
        );
        assert!(
            host.peers().await.iter().all(|peer| peer.id != joiner.id()),
            "host must not admit the rejected inbound peer"
        );
    });
}

pub(crate) fn assert_mesh_requirements_outbound_rejects_before_peer_promotion() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let initiator = make_test_node(super::NodeRole::Worker)
            .await
            .expect("initiator node");
        let remote = make_test_node(super::NodeRole::Worker)
            .await
            .expect("remote node");
        let trusted_signer = test_release_signer_key_id(9);
        let policy = requirement_policy(&trusted_signer);

        configure_requirement_node(&initiator, &policy, Some(&trusted_signer))
            .await
            .expect("configure initiator policy");
        configure_requirement_node(&remote, &policy, None)
            .await
            .expect("configure remote policy");

        initiator.start_accepting();
        remote.start_accepting();

        initiator
            .connect_to_peer(remote.endpoint_addr_for_advertisement())
            .await
            .expect_err("outbound connect should fail before promotion");
        assert!(
            initiator
                .peers()
                .await
                .iter()
                .all(|peer| peer.id != remote.id()),
            "noncompliant outbound peer must never become admitted/routable"
        );
    });
}

pub(crate) fn assert_mesh_requirements_add_peer_rejects_missing_direct_admission_proof() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let node = make_test_node(super::NodeRole::Worker)
            .await
            .expect("test node");
        let trusted_signer = test_release_signer_key_id(9);
        let policy = requirement_policy(&trusted_signer);
        configure_requirement_node(&node, &policy, Some(&trusted_signer))
            .await
            .expect("configure node policy");

        let ann = requirement_peer_announcement(
            0x8f,
            &policy,
            Some(test_release_attestation(&trusted_signer)),
            None,
        );
        let peer_id = ann.addr.id;

        node.add_peer(
            peer_id,
            ann.addr.clone(),
            &ann,
            Some(NODE_PROTOCOL_GENERATION),
        )
        .await;

        assert!(
            !is_peer_admitted(&node.state.lock().await.peers.clone(), &peer_id),
            "missing direct proof must reject before promotion"
        );
        let recent = node.recent_mesh_requirement_rejections().await;
        assert_eq!(
            recent[0].reason,
            crate::MeshRequirementRejectReason::DirectProofMissing
        );
    });
}

pub(crate) fn assert_mesh_requirements_add_peer_rejects_invalid_direct_admission_proof() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let node = make_test_node(super::NodeRole::Worker)
            .await
            .expect("test node");
        let trusted_signer = test_release_signer_key_id(9);
        let policy = requirement_policy(&trusted_signer);
        configure_requirement_node(&node, &policy, Some(&trusted_signer))
            .await
            .expect("configure node policy");

        let release_attestation = test_release_attestation(&trusted_signer);
        let mut direct_proof = direct_proof_for_announcement(
            0x8e,
            &policy.policy_derived_mesh_id().expect("mesh id"),
            &policy.canonical_hash_hex().expect("policy hash"),
            Some(&release_attestation),
        );
        direct_proof.signature[0] ^= 0x01;
        let ann = requirement_peer_announcement(
            0x8e,
            &policy,
            Some(release_attestation),
            Some(direct_proof),
        );
        let peer_id = ann.addr.id;

        node.add_peer(
            peer_id,
            ann.addr.clone(),
            &ann,
            Some(NODE_PROTOCOL_GENERATION),
        )
        .await;

        assert!(
            !is_peer_admitted(&node.state.lock().await.peers.clone(), &peer_id),
            "invalid direct proof must reject before promotion"
        );
        let recent = node.recent_mesh_requirement_rejections().await;
        assert_eq!(
            recent[0].reason,
            crate::MeshRequirementRejectReason::BuildProofInvalid
        );
    });
}

pub(crate) fn assert_mesh_requirements_add_peer_rejects_stale_direct_admission_proof() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let node = make_test_node(super::NodeRole::Worker)
            .await
            .expect("test node");
        let trusted_signer = test_release_signer_key_id(9);
        let policy = requirement_policy(&trusted_signer);
        configure_requirement_node(&node, &policy, Some(&trusted_signer))
            .await
            .expect("configure node policy");

        let release_attestation = test_release_attestation(&trusted_signer);
        let direct_proof = direct_proof_for_announcement_at(
            0x8d,
            &policy.policy_derived_mesh_id().expect("mesh id"),
            &policy.canonical_hash_hex().expect("policy hash"),
            Some(&release_attestation),
            current_time_unix_ms() - crate::DIRECT_NODE_ADMISSION_PROOF_MAX_CLOCK_SKEW_MS - 1,
        );
        let ann = requirement_peer_announcement(
            0x8d,
            &policy,
            Some(release_attestation),
            Some(direct_proof),
        );
        let peer_id = ann.addr.id;

        node.add_peer(
            peer_id,
            ann.addr.clone(),
            &ann,
            Some(NODE_PROTOCOL_GENERATION),
        )
        .await;

        let recent = node.recent_mesh_requirement_rejections().await;
        assert_eq!(
            recent[0].reason,
            crate::MeshRequirementRejectReason::DirectProofStale
        );
    });
}

pub(crate) fn assert_mesh_requirements_add_peer_rejects_direct_proof_sender_mismatch() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let node = make_test_node(super::NodeRole::Worker)
            .await
            .expect("test node");
        let trusted_signer = test_release_signer_key_id(9);
        let policy = requirement_policy(&trusted_signer);
        configure_requirement_node(&node, &policy, Some(&trusted_signer))
            .await
            .expect("configure node policy");

        let release_attestation = test_release_attestation(&trusted_signer);
        let direct_proof = direct_proof_for_announcement(
            0x8c,
            &policy.policy_derived_mesh_id().expect("mesh id"),
            &policy.canonical_hash_hex().expect("policy hash"),
            Some(&release_attestation),
        );
        let ann = requirement_peer_announcement(
            0x8b,
            &policy,
            Some(release_attestation),
            Some(direct_proof),
        );
        let peer_id = ann.addr.id;

        node.add_peer(
            peer_id,
            ann.addr.clone(),
            &ann,
            Some(NODE_PROTOCOL_GENERATION),
        )
        .await;

        let recent = node.recent_mesh_requirement_rejections().await;
        assert_eq!(
            recent[0].reason,
            crate::MeshRequirementRejectReason::DirectProofSenderIdMismatch
        );
    });
}

pub(crate) fn assert_requirement_aware_mesh_without_attestation_rejects_missing_direct_proof() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let node = make_test_node(super::NodeRole::Worker)
            .await
            .expect("test node");
        let policy = requirement_policy_without_release_attestation();
        configure_requirement_node(&node, &policy, None)
            .await
            .expect("configure node policy");

        let ann = requirement_peer_announcement(0x8a, &policy, None, None);
        let peer_id = ann.addr.id;
        node.add_peer(
            peer_id,
            ann.addr.clone(),
            &ann,
            Some(NODE_PROTOCOL_GENERATION),
        )
        .await;

        let recent = node.recent_mesh_requirement_rejections().await;
        assert_eq!(
            recent[0].reason,
            crate::MeshRequirementRejectReason::DirectProofMissing
        );
    });
}

/// On the fast auto-join probe, if `apply_gossip_announcements` fails after the
/// dispatcher has already been spawned, the winning candidate must be both
/// dropped from `state.connections` AND have its QUIC connection closed (so the
/// dispatcher unwinds and no orphaned, keep-alive'd connection lingers), and the
/// `Err` must propagate so the caller falls back to the serial join path.
pub(crate) fn assert_fast_join_apply_failure_closes_connection_and_propagates_err() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        // Joiner enforces a release-attestation requirement.
        let trusted_signer = test_release_signer_key_id(9);
        let policy = requirement_policy(&trusted_signer);
        let joiner = make_test_node(super::NodeRole::Worker)
            .await
            .expect("joiner test node");
        configure_requirement_node(&joiner, &policy, Some(&trusted_signer))
            .await
            .expect("configure joiner policy");

        // Bootstrap peer accepts a real QUIC connection from the joiner.
        let bootstrap = make_test_node(super::NodeRole::Worker)
            .await
            .expect("bootstrap test node");
        bootstrap.start_accepting();
        joiner.start_accepting();

        let bootstrap_id = bootstrap.id();
        let bootstrap_addr = bootstrap.endpoint_addr_for_advertisement();
        let conn = connect_mesh(&joiner.endpoint, bootstrap_addr.clone())
            .await
            .expect("joiner connects to bootstrap");

        // Self-announcement from the bootstrap peer carrying NO release
        // attestation. `apply_announced_peer` hits the `peer_id == remote`
        // branch, `validate_direct_peer_requirements` rejects it, and
        // `apply_gossip_announcements` returns `Err`.
        let mut self_ann = requirement_peer_announcement(0x00, &policy, None, None);
        self_ann.addr = super::EndpointAddr {
            id: bootstrap_id,
            addrs: Default::default(),
        };
        let announcements = vec![(self_ann.addr.clone(), self_ann.clone())];

        let success = super::gossip::JoinProbeSuccess::new_for_tests(
            joiner.invite_token().await,
            None,
            super::EndpointAddr {
                id: bootstrap_id,
                addrs: Default::default(),
            },
            conn.clone(),
            announcements,
            42,
        );

        let result = joiner.commit_join_probe_success(success).await;
        assert!(
            result.is_err(),
            "apply failure must propagate Err so the caller falls back to serial join"
        );

        // The tracked entry must be gone.
        assert!(
            !joiner
                .state
                .lock()
                .await
                .connections
                .contains_key(&bootstrap_id),
            "failed candidate must be removed from tracked connections"
        );

        // The QUIC connection must be closed, not merely untracked. If it were
        // only untracked, `closed()` would hang here because the keep-alive
        // would hold the orphaned connection open.
        let closed = tokio::time::timeout(std::time::Duration::from_secs(2), conn.closed()).await;
        assert!(
            closed.is_ok(),
            "QUIC connection must be closed on apply failure, not left orphaned"
        );
    });
}

pub(crate) fn assert_requirement_aware_mesh_without_attestation_rejects_invalid_direct_proof() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let node = make_test_node(super::NodeRole::Worker)
            .await
            .expect("test node");
        let policy = requirement_policy_without_release_attestation();
        configure_requirement_node(&node, &policy, None)
            .await
            .expect("configure node policy");

        let mut direct_proof = direct_proof_for_announcement(
            0x89,
            &policy.policy_derived_mesh_id().expect("mesh id"),
            &policy.canonical_hash_hex().expect("policy hash"),
            None,
        );
        direct_proof.signature[0] ^= 0x01;
        let ann = requirement_peer_announcement(0x89, &policy, None, Some(direct_proof));
        let peer_id = ann.addr.id;
        node.add_peer(
            peer_id,
            ann.addr.clone(),
            &ann,
            Some(NODE_PROTOCOL_GENERATION),
        )
        .await;

        let recent = node.recent_mesh_requirement_rejections().await;
        assert_eq!(
            recent[0].reason,
            crate::MeshRequirementRejectReason::BuildProofInvalid
        );
    });
}

pub(crate) fn assert_requirement_aware_mesh_without_attestation_rejects_stale_direct_proof() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let node = make_test_node(super::NodeRole::Worker)
            .await
            .expect("test node");
        let policy = requirement_policy_without_release_attestation();
        configure_requirement_node(&node, &policy, None)
            .await
            .expect("configure node policy");

        let direct_proof = direct_proof_for_announcement_at(
            0x88,
            &policy.policy_derived_mesh_id().expect("mesh id"),
            &policy.canonical_hash_hex().expect("policy hash"),
            None,
            current_time_unix_ms() - crate::DIRECT_NODE_ADMISSION_PROOF_MAX_CLOCK_SKEW_MS - 1,
        );
        let ann = requirement_peer_announcement(0x88, &policy, None, Some(direct_proof));
        let peer_id = ann.addr.id;
        node.add_peer(
            peer_id,
            ann.addr.clone(),
            &ann,
            Some(NODE_PROTOCOL_GENERATION),
        )
        .await;

        let recent = node.recent_mesh_requirement_rejections().await;
        assert_eq!(
            recent[0].reason,
            crate::MeshRequirementRejectReason::DirectProofStale
        );
    });
}

pub(crate) fn assert_requirement_aware_mesh_without_attestation_rejects_sender_mismatch_direct_proof()
 {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let node = make_test_node(super::NodeRole::Worker)
            .await
            .expect("test node");
        let policy = requirement_policy_without_release_attestation();
        configure_requirement_node(&node, &policy, None)
            .await
            .expect("configure node policy");

        let direct_proof = direct_proof_for_announcement(
            0x87,
            &policy.policy_derived_mesh_id().expect("mesh id"),
            &policy.canonical_hash_hex().expect("policy hash"),
            None,
        );
        let ann = requirement_peer_announcement(0x86, &policy, None, Some(direct_proof));
        let peer_id = ann.addr.id;
        node.add_peer(
            peer_id,
            ann.addr.clone(),
            &ann,
            Some(NODE_PROTOCOL_GENERATION),
        )
        .await;

        let recent = node.recent_mesh_requirement_rejections().await;
        assert_eq!(
            recent[0].reason,
            crate::MeshRequirementRejectReason::DirectProofSenderIdMismatch
        );
    });
}

pub(crate) fn assert_requirement_aware_mesh_without_attestation_accepts_valid_direct_proof() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let node = make_test_node(super::NodeRole::Worker)
            .await
            .expect("test node");
        let policy = requirement_policy_without_release_attestation();
        configure_requirement_node(&node, &policy, None)
            .await
            .expect("configure node policy");

        let direct_proof = direct_proof_for_announcement(
            0x85,
            &policy.policy_derived_mesh_id().expect("mesh id"),
            &policy.canonical_hash_hex().expect("policy hash"),
            None,
        );
        let ann = requirement_peer_announcement(0x85, &policy, None, Some(direct_proof));
        let peer_id = ann.addr.id;
        node.add_peer(
            peer_id,
            ann.addr.clone(),
            &ann,
            Some(NODE_PROTOCOL_GENERATION),
        )
        .await;

        assert!(is_peer_admitted(
            &node.state.lock().await.peers.clone(),
            &peer_id
        ));
    });
}

pub(crate) fn assert_mesh_requirements_add_peer_rejects_untrusted_release_signer() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let node = make_test_node(super::NodeRole::Worker)
            .await
            .expect("test node");
        let trusted_signer = test_release_signer_key_id(9);
        let policy = requirement_policy(&trusted_signer);
        configure_requirement_node(&node, &policy, Some(&trusted_signer))
            .await
            .expect("configure node policy");

        let peer_id = make_test_endpoint_id(0x91);
        let ann = super::PeerAnnouncement {
            addr: EndpointAddr {
                id: peer_id,
                addrs: Default::default(),
            },
            role: super::NodeRole::Worker,
            first_joined_mesh_ts: None,
            models: vec![],
            vram_bytes: 0,
            model_source: None,
            serving_models: vec![],
            hosted_models: None,
            available_models: vec![],
            requested_models: vec![],
            explicit_model_interests: vec![],
            version: Some(crate::VERSION.to_string()),
            model_demand: HashMap::new(),
            mesh_id: Some(policy.policy_derived_mesh_id().expect("mesh id")),
            mesh_policy_hash: Some(policy.canonical_hash_hex().expect("policy hash")),
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
            served_model_descriptors: vec![],
            served_model_runtime: vec![],
            owner_attestation: None,
            genesis_policy: None,
            release_attestation: Some(test_release_attestation_with_seed(10)),
            direct_admission_proof: Some(direct_proof_for_announcement(
                0x91,
                &policy.policy_derived_mesh_id().expect("mesh id"),
                &policy.canonical_hash_hex().expect("policy hash"),
                Some(&test_release_attestation_with_seed(10)),
            )),
            artifact_transfer_supported: true,
            stage_protocol_generation_supported: true,
            stage_status_list_supported: true,
            advertised_model_throughput: vec![],
            latency_ms: None,
            latency_source: None,
            latency_age_ms: None,
            latency_observer_id: None,
        };

        node.add_peer(
            peer_id,
            ann.addr.clone(),
            &ann,
            Some(NODE_PROTOCOL_GENERATION),
        )
        .await;

        let peers = node.state.lock().await.peers.clone();
        assert!(
            !is_peer_admitted(&peers, &peer_id),
            "add_peer must reject untrusted release signers before promotion"
        );
        let recent = node.recent_mesh_requirement_rejections().await;
        assert_eq!(recent.len(), 1);
        assert_eq!(
            recent[0].reason,
            crate::MeshRequirementRejectReason::ReleaseSignerUntrusted
        );
    });
}

pub(crate) fn assert_mesh_requirements_add_peer_rejects_invalid_release_attestation_signature() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let node = make_test_node(super::NodeRole::Worker)
            .await
            .expect("test node");
        let trusted_signer = test_release_signer_key_id(9);
        let policy = requirement_policy(&trusted_signer);
        configure_requirement_node(&node, &policy, Some(&trusted_signer))
            .await
            .expect("configure node policy");

        let peer_id = make_test_endpoint_id(0x90);
        let mut invalid_attestation = test_release_attestation_with_seed(9);
        invalid_attestation.signature[0] ^= 0x01;
        let invalid_direct_proof = direct_proof_for_announcement(
            0x90,
            &policy.policy_derived_mesh_id().expect("mesh id"),
            &policy.canonical_hash_hex().expect("policy hash"),
            Some(&invalid_attestation),
        );
        let ann = super::PeerAnnouncement {
            addr: EndpointAddr {
                id: peer_id,
                addrs: Default::default(),
            },
            role: super::NodeRole::Worker,
            first_joined_mesh_ts: None,
            models: vec![],
            vram_bytes: 0,
            model_source: None,
            serving_models: vec![],
            hosted_models: None,
            available_models: vec![],
            requested_models: vec![],
            explicit_model_interests: vec![],
            version: Some(crate::VERSION.to_string()),
            model_demand: HashMap::new(),
            mesh_id: Some(policy.policy_derived_mesh_id().expect("mesh id")),
            mesh_policy_hash: Some(policy.canonical_hash_hex().expect("policy hash")),
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
            served_model_descriptors: vec![],
            served_model_runtime: vec![],
            owner_attestation: None,
            genesis_policy: None,
            release_attestation: Some(invalid_attestation),
            direct_admission_proof: Some(invalid_direct_proof),
            artifact_transfer_supported: true,
            stage_protocol_generation_supported: true,
            stage_status_list_supported: true,
            advertised_model_throughput: vec![],
            latency_ms: None,
            latency_source: None,
            latency_age_ms: None,
            latency_observer_id: None,
        };

        node.add_peer(
            peer_id,
            ann.addr.clone(),
            &ann,
            Some(NODE_PROTOCOL_GENERATION),
        )
        .await;

        let peers = node.state.lock().await.peers.clone();
        assert!(
            !is_peer_admitted(&peers, &peer_id),
            "add_peer must reject cryptographically invalid release attestations before promotion"
        );
        let recent = node.recent_mesh_requirement_rejections().await;
        assert_eq!(recent.len(), 1);
        assert_eq!(
            recent[0].reason,
            crate::MeshRequirementRejectReason::BuildProofInvalid
        );
    });
}

pub(crate) fn assert_mesh_requirements_add_peer_rejects_wrong_mesh_id() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let node = make_test_node(super::NodeRole::Worker)
            .await
            .expect("test node");
        let trusted_signer = test_release_signer_key_id(9);
        let policy = requirement_policy(&trusted_signer);
        configure_requirement_node(&node, &policy, Some(&trusted_signer))
            .await
            .expect("configure node policy");

        let peer_id = make_test_endpoint_id(0x92);
        let ann = super::PeerAnnouncement {
            addr: EndpointAddr {
                id: peer_id,
                addrs: Default::default(),
            },
            role: super::NodeRole::Worker,
            first_joined_mesh_ts: None,
            models: vec![],
            vram_bytes: 0,
            model_source: None,
            serving_models: vec![],
            hosted_models: None,
            available_models: vec![],
            requested_models: vec![],
            explicit_model_interests: vec![],
            version: Some(crate::VERSION.to_string()),
            model_demand: HashMap::new(),
            mesh_id: Some("mesh-wrong".to_string()),
            mesh_policy_hash: Some(policy.canonical_hash_hex().expect("policy hash")),
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
            served_model_descriptors: vec![],
            served_model_runtime: vec![],
            owner_attestation: None,
            genesis_policy: None,
            release_attestation: Some(test_release_attestation(&test_release_signer_key_id(9))),
            direct_admission_proof: Some(direct_proof_for_announcement(
                0x92,
                "mesh-wrong",
                &policy.canonical_hash_hex().expect("policy hash"),
                Some(&test_release_attestation(&test_release_signer_key_id(9))),
            )),
            artifact_transfer_supported: true,
            stage_protocol_generation_supported: true,
            stage_status_list_supported: true,
            advertised_model_throughput: vec![],
            latency_ms: None,
            latency_source: None,
            latency_age_ms: None,
            latency_observer_id: None,
        };

        node.add_peer(
            peer_id,
            ann.addr.clone(),
            &ann,
            Some(NODE_PROTOCOL_GENERATION),
        )
        .await;

        let peers = node.state.lock().await.peers.clone();
        assert!(
            !is_peer_admitted(&peers, &peer_id),
            "direct peers advertising the wrong mesh must be rejected before promotion"
        );
        let recent = node.recent_mesh_requirement_rejections().await;
        assert_eq!(recent.len(), 1);
        assert_eq!(
            recent[0].reason,
            crate::MeshRequirementRejectReason::MeshPolicyMismatch
        );
    });
}

pub(crate) fn assert_mesh_requirements_transitive_gossip_never_admits_peer_without_direct_proof() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let host = make_test_node(super::NodeRole::Host { http_port: 9337 })
            .await
            .expect("host node");
        let bridge = make_test_node(super::NodeRole::Worker)
            .await
            .expect("bridge node");
        let client = make_test_node(super::NodeRole::Client)
            .await
            .expect("client node");
        let trusted_signer = test_release_signer_key_id(9);
        let policy = requirement_policy(&trusted_signer);

        host.set_hosted_models(vec!["remote-coding-model".to_string()])
            .await;
        configure_requirement_node(&host, &policy, Some(&trusted_signer))
            .await
            .expect("configure host policy");
        configure_requirement_node(&bridge, &policy, Some(&trusted_signer))
            .await
            .expect("configure bridge policy");
        configure_requirement_node(&client, &policy, Some(&trusted_signer))
            .await
            .expect("configure client policy");

        host.start_accepting();
        bridge.start_accepting();
        client.start_accepting();

        bridge.sync_from_peer_for_tests(&host).await;
        assert!(bridge.peers().await.iter().any(|peer| peer.id == host.id()));

        client.sync_from_peer_for_tests(&bridge).await;
        assert!(
            client
                .peers()
                .await
                .iter()
                .any(|peer| peer.id == bridge.id())
        );

        let peers = client.state.lock().await.peers.clone();
        assert!(
            peers.contains_key(&host.id()),
            "host should still be tracked as a hint"
        );
        assert!(
            !is_peer_admitted(&peers, &host.id()),
            "transitive gossip must not admit the host without a direct proof path"
        );
        assert!(
            !client
                .hosts_for_model("remote-coding-model")
                .await
                .contains(&host.id()),
            "transitive-only host must not be routable before direct verification"
        );

        let _conn = client
            .connection_to_peer(host.id())
            .await
            .expect("direct connection should promote the host");
        wait_for_peer(&client, host.id()).await;
        assert!(
            client
                .hosts_for_model("remote-coding-model")
                .await
                .contains(&host.id()),
            "host should become routable only after direct verification"
        );
    });
}

pub(crate) fn assert_mesh_requirements_rejected_peer_messages_have_no_mesh_effect() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let host = make_test_node(super::NodeRole::Host { http_port: 9337 })
            .await
            .expect("host node");
        let bridge = make_test_node(super::NodeRole::Worker)
            .await
            .expect("bridge node");
        let rejected = make_test_node(super::NodeRole::Worker)
            .await
            .expect("rejected node");
        let trusted_signer = test_release_signer_key_id(9);
        let policy = requirement_policy(&trusted_signer);

        configure_requirement_node(&host, &policy, Some(&trusted_signer))
            .await
            .expect("configure host policy");
        configure_requirement_node(&bridge, &policy, Some(&trusted_signer))
            .await
            .expect("configure bridge policy");
        configure_requirement_node(&rejected, &policy, None)
            .await
            .expect("configure rejected policy");

        host.start_accepting();
        bridge.start_accepting();
        rejected.start_accepting();

        bridge
            .join(&host.invite_token().await)
            .await
            .expect("bridge joins host");
        wait_for_peer(&host, bridge.id()).await;

        rejected
            .join(&host.invite_token().await)
            .await
            .expect_err("rejected peer should fail admission");
        expect_no_route_table_response(&rejected, &host)
            .await
            .expect("route request should be suppressed");

        let admitted_ids: Vec<_> = host.peers().await.into_iter().map(|peer| peer.id).collect();
        assert_eq!(admitted_ids, vec![bridge.id()]);
        assert!(
            admitted_ids
                .into_iter()
                .all(|peer_id| peer_id != rejected.id()),
            "rejected peer messages must not change mesh membership"
        );
    });
}

pub(crate) fn assert_mesh_requirements_join_rejects_invalid_bootstrap_token() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let host = make_test_node(super::NodeRole::Host { http_port: 9337 })
            .await
            .expect("host node");
        let joiner = make_test_node(super::NodeRole::Worker)
            .await
            .expect("joiner node");
        let owner = crate::crypto::OwnerKeypair::generate();
        let policy = crate::MeshGenesisPolicy::new(
            owner.owner_id(),
            1_717_171_717_000,
            requirement_policy(&test_release_signer_key_id(9)).requirements,
        )
        .expect("policy should validate");
        let signed_policy =
            crate::SignedMeshGenesisPolicy::sign(policy.clone(), &owner).expect("signed policy");
        let addr_bytes = serde_json::to_vec(&host.endpoint_addr_for_advertisement())
            .expect("serializable endpoint addr");

        host.start_accepting();
        joiner.start_accepting();

        let mut token = crate::SignedBootstrapToken::sign(
            vec![addr_bytes],
            &signed_policy,
            Some(current_time_unix_ms() + 60_000),
            &owner,
        )
        .expect("bootstrap token should sign");
        token.signature[0] ^= 0x01;
        let tampered = base64::Engine::encode(
            &base64::engine::general_purpose::URL_SAFE_NO_PAD,
            serde_json::to_vec(&token).expect("serializable token"),
        );

        let err = joiner
            .join(&tampered)
            .await
            .expect_err("tampered bootstrap tokens must be rejected");
        assert!(err.to_string().contains("bootstrap_token_invalid"));
        assert!(joiner.peers().await.is_empty());
        let recent = joiner.recent_mesh_requirement_rejections().await;
        assert_eq!(recent.len(), 1);
        assert_eq!(
            recent[0].reason,
            crate::MeshRequirementRejectReason::BootstrapTokenInvalid
        );
    });
}

pub(crate) fn assert_mesh_requirements_join_accepts_matching_bootstrap_before_policy_state_installed()
 {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let trusted_signer = test_release_signer_key_id(9);
        let policy = requirement_policy(&trusted_signer);
        let policy_hash = policy.canonical_hash_hex().expect("policy hash");
        let host = make_test_node(super::NodeRole::Host { http_port: 9337 })
            .await
            .expect("host node");
        let joiner =
            make_test_node_with_requirements(super::NodeRole::Worker, policy.requirements.clone())
                .await
                .expect("joiner node");

        configure_requirement_node(&host, &policy, Some(&trusted_signer))
            .await
            .expect("configure host policy");
        *joiner.release_attestation.lock().await = Some(test_release_attestation(&trusted_signer));

        assert_eq!(
            *joiner.mesh_policy_hash.lock().await,
            None,
            "fresh constrained joiner must not have active policy state before joining"
        );
        host.start_accepting();
        joiner.start_accepting();

        joiner
            .join(&host.invite_token().await)
            .await
            .expect("matching bootstrap token should install policy and join");

        wait_for_peer(&joiner, host.id()).await;
        wait_for_peer(&host, joiner.id()).await;
        assert_eq!(*joiner.mesh_policy_hash.lock().await, Some(policy_hash));
    });
}

pub(crate) fn assert_mesh_requirements_unrestricted_legacy_mesh_join_stays_compatible() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let host = make_test_node(super::NodeRole::Host { http_port: 9337 })
            .await
            .expect("host node");
        let joiner = make_test_node(super::NodeRole::Worker)
            .await
            .expect("joiner node");

        host.start_accepting();
        joiner.start_accepting();
        joiner
            .join(&host.invite_token().await)
            .await
            .expect("legacy unrestricted meshes should remain join-compatible");

        wait_for_peer(&joiner, host.id()).await;
        wait_for_peer(&host, joiner.id()).await;
    });
}
