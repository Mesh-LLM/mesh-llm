#[tokio::test]
#[serial]
async fn artifact_transfer_peer_eligibility_ignores_public_advertisement_by_default() -> Result<()>
{
    let _transfer_guard = EnvVarGuard::unset("MESH_LLM_ARTIFACT_TRANSFER");
    let node = make_test_node(super::NodeRole::Worker).await?;
    let mut peer = make_test_peer_info(make_test_endpoint_id(0x71));
    peer.artifact_transfer_supported = true;

    assert!(
        !node.artifact_transfer_allowed_for_peer(&peer).await,
        "raw public artifact-transfer advertisement must not make a peer eligible"
    );

    Ok(())
}

#[tokio::test]
#[serial]
async fn artifact_transfer_peer_eligibility_allows_same_or_trusted_owner() -> Result<()> {
    let _transfer_guard = EnvVarGuard::set_str("MESH_LLM_ARTIFACT_TRANSFER", "trusted");
    let node = make_test_node(super::NodeRole::Worker).await?;
    *node.owner_summary.lock().await = verified_owner_summary("owner-a");

    let mut same_owner = make_test_peer_info(make_test_endpoint_id(0x72));
    same_owner.artifact_transfer_supported = true;
    same_owner.owner_summary = verified_owner_summary("owner-a");
    assert!(node.artifact_transfer_allowed_for_peer(&same_owner).await);

    let mut trusted_owner = make_test_peer_info(make_test_endpoint_id(0x73));
    trusted_owner.artifact_transfer_supported = true;
    trusted_owner.owner_summary = verified_owner_summary("owner-b");
    {
        let mut store = node.trust_store.lock().await;
        store.add_trusted_owner("owner-b".to_string(), None);
    }
    assert!(
        node.artifact_transfer_allowed_for_peer(&trusted_owner)
            .await
    );

    let mut untrusted_owner = make_test_peer_info(make_test_endpoint_id(0x74));
    untrusted_owner.artifact_transfer_supported = true;
    untrusted_owner.owner_summary = verified_owner_summary("owner-c");
    assert!(
        !node
            .artifact_transfer_allowed_for_peer(&untrusted_owner)
            .await
    );

    Ok(())
}

#[test]
fn artifact_transfer_authorization_is_limited_to_stage_assignment() {
    let package = tempfile::tempdir().unwrap();
    let (package_ref, manifest_sha256) = write_artifact_authorization_package(package.path());
    let stage0 = make_test_endpoint_id(0x91);
    let stage1 = make_test_endpoint_id(0x92);
    let topology = StageTopologyInstance {
        topology_id: "topology-a".to_string(),
        run_id: "run-a".to_string(),
        model_id: "model-a".to_string(),
        package_ref: package_ref.clone(),
        manifest_sha256: manifest_sha256.clone(),
        stages: vec![
            StageAssignment {
                stage_id: "stage-0".to_string(),
                stage_index: 0,
                node_id: stage0,
                layer_start: 0,
                layer_end: 1,
                endpoint: StageEndpoint {
                    bind_addr: String::new(),
                },
            },
            StageAssignment {
                stage_id: "stage-1".to_string(),
                stage_index: 1,
                node_id: stage1,
                layer_start: 1,
                layer_end: 2,
                endpoint: StageEndpoint {
                    bind_addr: String::new(),
                },
            },
        ],
    };
    let request = |relative_path: &str, expected_size: u64, expected_sha256: String| {
        skippy_stage_proto::StageArtifactTransferRequest {
            r#gen: skippy_protocol::STAGE_PROTOCOL_GENERATION,
            requester_id: stage0.as_bytes().to_vec(),
            topology_id: "topology-a".to_string(),
            run_id: "run-a".to_string(),
            stage_id: "stage-0".to_string(),
            package_ref: package_ref.clone(),
            manifest_sha256: manifest_sha256.clone(),
            relative_path: relative_path.to_string(),
            offset: 0,
            expected_size: Some(expected_size),
            expected_sha256: Some(expected_sha256),
        }
    };

    let layer0 = request("layers/layer-000.gguf", 8, sha256_hex(b"layer000"));
    assert!(
        artifact_transfer_allowed_by_topology(
            std::slice::from_ref(&topology),
            stage0,
            package.path(),
            &layer0,
        )
        .unwrap()
    );

    let mut wrong_topology = layer0.clone();
    wrong_topology.topology_id = "other-topology".to_string();
    assert!(
        !artifact_transfer_allowed_by_topology(
            std::slice::from_ref(&topology),
            stage0,
            package.path(),
            &wrong_topology,
        )
        .unwrap()
    );

    let layer1 = request("layers/layer-001.gguf", 8, sha256_hex(b"layer001"));
    assert!(
        !artifact_transfer_allowed_by_topology(
            std::slice::from_ref(&topology),
            stage0,
            package.path(),
            &layer1,
        )
        .unwrap()
    );

    let projector = request("projectors/mmproj.gguf", 9, sha256_hex(b"projector"));
    assert!(
        artifact_transfer_allowed_by_topology(
            std::slice::from_ref(&topology),
            stage0,
            package.path(),
            &projector,
        )
        .unwrap()
    );

    let manifest = skippy_stage_proto::StageArtifactTransferRequest {
        r#gen: skippy_protocol::STAGE_PROTOCOL_GENERATION,
        requester_id: stage1.as_bytes().to_vec(),
        topology_id: "topology-a".to_string(),
        run_id: "run-a".to_string(),
        stage_id: "stage-1".to_string(),
        package_ref,
        manifest_sha256,
        relative_path: "model-package.json".to_string(),
        offset: 0,
        expected_size: None,
        expected_sha256: None,
    };
    assert!(
        artifact_transfer_allowed_by_topology(&[topology], stage1, package.path(), &manifest)
            .unwrap()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn artifact_transfer_stream_serves_authorized_stage_artifact() -> Result<()> {
    use crate::protocol::{read_len_prefixed, write_len_prefixed};
    use base64::Engine as _;
    use prost::Message as _;

    let cache = tempfile::tempdir().unwrap();
    let _cache_guard = EnvVarGuard::set("HF_HUB_CACHE", cache.path());
    let _transfer_guard = EnvVarGuard::set_str("MESH_LLM_ARTIFACT_TRANSFER", "1");
    let (package_dir, package_ref, manifest_sha256) =
        write_hf_artifact_stream_package(cache.path());
    let server = make_test_node(super::NodeRole::Host { http_port: 9337 }).await?;
    let client = make_test_node(super::NodeRole::Worker).await?;
    server
        .set_mesh_id("artifact-transfer-stream-mesh".to_string())
        .await;
    client
        .set_mesh_id("artifact-transfer-stream-mesh".to_string())
        .await;
    server.start_accepting();
    client.start_accepting();

    let server_id = server.id();
    let client_id = client.id();
    let invite = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(serde_json::to_vec(&server.endpoint.addr())?);
    client.join(&invite).await?;
    wait_for_peer(&client, server_id).await;
    wait_for_peer(&server, client_id).await;
    server
        .record_stage_topology(StageTopologyInstance {
            topology_id: "topology-artifact".to_string(),
            run_id: "run-artifact".to_string(),
            model_id: "model-artifact".to_string(),
            package_ref: package_ref.clone(),
            manifest_sha256: manifest_sha256.clone(),
            stages: vec![StageAssignment {
                stage_id: "stage-0".to_string(),
                stage_index: 0,
                node_id: client_id,
                layer_start: 0,
                layer_end: 1,
                endpoint: StageEndpoint {
                    bind_addr: String::new(),
                },
            }],
        })
        .await;

    let request = skippy_stage_proto::StageArtifactTransferRequest {
        r#gen: skippy_protocol::STAGE_PROTOCOL_GENERATION,
        requester_id: client_id.as_bytes().to_vec(),
        topology_id: "topology-artifact".to_string(),
        run_id: "run-artifact".to_string(),
        stage_id: "stage-0".to_string(),
        package_ref,
        manifest_sha256,
        relative_path: "layers/layer-000.gguf".to_string(),
        offset: 0,
        expected_size: Some(8),
        expected_sha256: Some(sha256_hex(b"layer000")),
    };

    let (mut send, mut recv) = client
        .open_skippy_stage_mesh_stream(server_id, skippy_protocol::STAGE_STREAM_ARTIFACT_TRANSFER)
        .await?;
    write_len_prefixed(&mut send, &request.encode_to_vec()).await?;
    send.finish()?;
    let response_buf = read_len_prefixed(&mut recv).await?;
    let response =
        skippy_stage_proto::StageArtifactTransferResponse::decode(response_buf.as_slice())?;
    assert!(response.accepted, "artifact response: {:?}", response.error);
    assert_eq!(response.total_size, 8);
    let expected_sha = sha256_hex(b"layer000");
    assert_eq!(response.sha256.as_deref(), Some(expected_sha.as_str()));
    let mut bytes = vec![0u8; response.total_size as usize];
    recv.read_exact(&mut bytes).await?;
    assert_eq!(bytes, b"layer000");

    let mut resume_request = request.clone();
    resume_request.offset = 5;
    let (mut resume_send, mut resume_recv) = client
        .open_skippy_stage_mesh_stream(server_id, skippy_protocol::STAGE_STREAM_ARTIFACT_TRANSFER)
        .await?;
    write_len_prefixed(&mut resume_send, &resume_request.encode_to_vec()).await?;
    resume_send.finish()?;
    let resume_response_buf = read_len_prefixed(&mut resume_recv).await?;
    let resume_response =
        skippy_stage_proto::StageArtifactTransferResponse::decode(resume_response_buf.as_slice())?;
    assert!(
        resume_response.accepted,
        "resume artifact response: {:?}",
        resume_response.error
    );
    assert_eq!(resume_response.total_size, 8);
    assert_eq!(
        resume_response.sha256.as_deref(),
        Some(expected_sha.as_str())
    );
    let mut resumed_bytes =
        vec![0u8; (resume_response.total_size - resume_request.offset) as usize];
    resume_recv.read_exact(&mut resumed_bytes).await?;
    assert_eq!(resumed_bytes, b"000");

    let conn = client.stage_connection_to_peer(server_id).await?;
    let (mut legacy_send, mut legacy_recv) = conn.open_bi().await?;
    legacy_send
        .write_all(&[skippy_protocol::STAGE_STREAM_ARTIFACT_TRANSFER])
        .await?;
    write_len_prefixed(&mut legacy_send, &request.encode_to_vec()).await?;
    legacy_send.finish()?;
    let legacy_response_buf = read_len_prefixed(&mut legacy_recv).await?;
    let legacy_response =
        skippy_stage_proto::StageArtifactTransferResponse::decode(legacy_response_buf.as_slice())?;
    assert!(
        legacy_response.accepted,
        "legacy artifact response: {:?}",
        legacy_response.error
    );
    assert_eq!(legacy_response.total_size, 8);
    assert_eq!(
        legacy_response.sha256.as_deref(),
        Some(expected_sha.as_str())
    );
    let mut legacy_bytes = vec![0u8; legacy_response.total_size as usize];
    legacy_recv.read_exact(&mut legacy_bytes).await?;
    assert_eq!(legacy_bytes, b"layer000");
    assert!(package_dir.join("model-package.json").is_file());

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn artifact_transfer_stream_rejects_corrupt_same_size_cached_artifact() -> Result<()> {
    use crate::protocol::{read_len_prefixed, write_len_prefixed};
    use base64::Engine as _;
    use prost::Message as _;

    let cache = tempfile::tempdir().unwrap();
    let _cache_guard = EnvVarGuard::set("HF_HUB_CACHE", cache.path());
    let _transfer_guard = EnvVarGuard::set_str("MESH_LLM_ARTIFACT_TRANSFER", "1");
    let (package_dir, package_ref, manifest_sha256) =
        write_hf_artifact_stream_package(cache.path());
    std::fs::write(package_dir.join("layers/layer-000.gguf"), b"corrupt!").unwrap();
    let server = make_test_node(super::NodeRole::Host { http_port: 9337 }).await?;
    let client = make_test_node(super::NodeRole::Worker).await?;
    server
        .set_mesh_id("artifact-transfer-corrupt-mesh".to_string())
        .await;
    client
        .set_mesh_id("artifact-transfer-corrupt-mesh".to_string())
        .await;
    server.start_accepting();
    client.start_accepting();

    let server_id = server.id();
    let client_id = client.id();
    let invite = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(serde_json::to_vec(&server.endpoint.addr())?);
    client.join(&invite).await?;
    wait_for_peer(&client, server_id).await;
    wait_for_peer(&server, client_id).await;
    server
        .record_stage_topology(StageTopologyInstance {
            topology_id: "topology-artifact-corrupt".to_string(),
            run_id: "run-artifact-corrupt".to_string(),
            model_id: "model-artifact".to_string(),
            package_ref: package_ref.clone(),
            manifest_sha256: manifest_sha256.clone(),
            stages: vec![StageAssignment {
                stage_id: "stage-0".to_string(),
                stage_index: 0,
                node_id: client_id,
                layer_start: 0,
                layer_end: 1,
                endpoint: StageEndpoint {
                    bind_addr: String::new(),
                },
            }],
        })
        .await;

    let request = skippy_stage_proto::StageArtifactTransferRequest {
        r#gen: skippy_protocol::STAGE_PROTOCOL_GENERATION,
        requester_id: client_id.as_bytes().to_vec(),
        topology_id: "topology-artifact-corrupt".to_string(),
        run_id: "run-artifact-corrupt".to_string(),
        stage_id: "stage-0".to_string(),
        package_ref,
        manifest_sha256,
        relative_path: "layers/layer-000.gguf".to_string(),
        offset: 0,
        expected_size: Some(8),
        expected_sha256: Some(sha256_hex(b"layer000")),
    };

    let (mut send, mut recv) = client
        .open_skippy_stage_mesh_stream(server_id, skippy_protocol::STAGE_STREAM_ARTIFACT_TRANSFER)
        .await?;
    write_len_prefixed(&mut send, &request.encode_to_vec()).await?;
    send.finish()?;
    let response_buf = read_len_prefixed(&mut recv).await?;
    let response =
        skippy_stage_proto::StageArtifactTransferResponse::decode(response_buf.as_slice())?;
    assert!(!response.accepted);
    assert_eq!(response.error.as_deref(), Some("artifact unavailable"));

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn artifact_transfer_stream_rejects_public_mesh_without_opt_in() -> Result<()> {
    use crate::protocol::{read_len_prefixed, write_len_prefixed};
    use base64::Engine as _;
    use prost::Message as _;

    let cache = tempfile::tempdir().unwrap();
    let _cache_guard = EnvVarGuard::set("HF_HUB_CACHE", cache.path());
    let _transfer_guard = EnvVarGuard::unset("MESH_LLM_ARTIFACT_TRANSFER");
    let (_package_dir, package_ref, manifest_sha256) =
        write_hf_artifact_stream_package(cache.path());
    let server = make_test_node(super::NodeRole::Host { http_port: 9337 }).await?;
    let client = make_test_node(super::NodeRole::Worker).await?;
    server
        .set_mesh_id("artifact-transfer-disabled-mesh".to_string())
        .await;
    client
        .set_mesh_id("artifact-transfer-disabled-mesh".to_string())
        .await;
    server.start_accepting();
    client.start_accepting();

    let server_id = server.id();
    let client_id = client.id();
    let invite = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(serde_json::to_vec(&server.endpoint.addr())?);
    client.join(&invite).await?;
    wait_for_peer(&client, server_id).await;
    wait_for_peer(&server, client_id).await;
    server
        .record_stage_topology(StageTopologyInstance {
            topology_id: "topology-artifact-disabled".to_string(),
            run_id: "run-artifact-disabled".to_string(),
            model_id: "model-artifact".to_string(),
            package_ref: package_ref.clone(),
            manifest_sha256: manifest_sha256.clone(),
            stages: vec![StageAssignment {
                stage_id: "stage-0".to_string(),
                stage_index: 0,
                node_id: client_id,
                layer_start: 0,
                layer_end: 1,
                endpoint: StageEndpoint {
                    bind_addr: String::new(),
                },
            }],
        })
        .await;

    let request = skippy_stage_proto::StageArtifactTransferRequest {
        r#gen: skippy_protocol::STAGE_PROTOCOL_GENERATION,
        requester_id: client_id.as_bytes().to_vec(),
        topology_id: "topology-artifact-disabled".to_string(),
        run_id: "run-artifact-disabled".to_string(),
        stage_id: "stage-0".to_string(),
        package_ref,
        manifest_sha256,
        relative_path: "layers/layer-000.gguf".to_string(),
        offset: 0,
        expected_size: Some(8),
        expected_sha256: Some(sha256_hex(b"layer000")),
    };

    let (mut send, mut recv) = client
        .open_skippy_stage_mesh_stream(server_id, skippy_protocol::STAGE_STREAM_ARTIFACT_TRANSFER)
        .await?;
    write_len_prefixed(&mut send, &request.encode_to_vec()).await?;
    send.finish()?;
    let response_buf = read_len_prefixed(&mut recv).await?;
    let response =
        skippy_stage_proto::StageArtifactTransferResponse::decode(response_buf.as_slice())?;
    assert!(!response.accepted);
    assert_eq!(
        response.error.as_deref(),
        Some("artifact transfer disabled")
    );

    Ok(())
}

#[tokio::test]
async fn artifact_transfer_body_read_has_idle_timeout() {
    let (_writer, mut reader) = tokio::io::duplex(8);
    let mut buffer = [0u8; 4];

    let error = read_artifact_transfer_chunk(
        &mut reader,
        &mut buffer,
        std::time::Duration::from_millis(10),
    )
    .await
    .expect_err("stalled body read must time out");

    assert!(
        error
            .to_string()
            .contains("artifact transfer body read idle timeout")
    );
}

#[test]
fn artifact_transfer_invalid_resume_offset_removes_preserved_partial() {
    let temp = tempfile::tempdir().unwrap();
    let partial = temp.path().join(".model-package.json.stale.part");
    std::fs::write(&partial, b"stale manifest bytes").unwrap();
    let mut guard = PartialArtifactGuard::preserve_on_error(partial.clone());
    let response = skippy_stage_proto::StageArtifactTransferResponse {
        r#gen: skippy_protocol::STAGE_PROTOCOL_GENERATION,
        accepted: false,
        total_size: 8,
        sha256: Some(sha256_hex(b"manifest")),
        error: Some(ARTIFACT_TRANSFER_INVALID_OFFSET_ERROR.to_string()),
    };

    Node::remove_invalid_resume_partial(&mut guard, 128, &response);

    assert!(!partial.exists());
}

#[test]
fn artifact_transfer_smaller_resume_response_removes_preserved_partial() {
    let temp = tempfile::tempdir().unwrap();
    let partial = temp.path().join(".model-package.json.oversized.part");
    std::fs::write(&partial, b"stale manifest bytes").unwrap();
    let mut guard = PartialArtifactGuard::preserve_on_error(partial.clone());
    let response = skippy_stage_proto::StageArtifactTransferResponse {
        r#gen: skippy_protocol::STAGE_PROTOCOL_GENERATION,
        accepted: true,
        total_size: 8,
        sha256: Some(sha256_hex(b"manifest")),
        error: None,
    };

    Node::remove_invalid_resume_partial(&mut guard, 128, &response);

    assert!(!partial.exists());
}

#[test]
fn partial_artifact_guard_removes_armed_partial_file() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join(".artifact.part");
    std::fs::write(&path, b"partial").unwrap();

    {
        let _guard = PartialArtifactGuard::new(path.clone());
    }

    assert!(!path.exists());
}

#[test]
fn partial_artifact_guard_preserves_disarmed_installed_file() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join(".artifact.part");
    std::fs::write(&path, b"partial").unwrap();

    {
        let mut guard = PartialArtifactGuard::new(path.clone());
        guard.disarm();
    }

    assert!(path.exists());
}
