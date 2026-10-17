//! Real interop test for the `ledger-fetch/1` plugin-mesh-stream carrier
//! (`ledger_fetch_bridge`), over the ACTUAL wire protocol -- same fake-host
//! approach as `tests/mesh_evidence_bridge_interop.rs` (this repo's CI cannot
//! build `mesh-llm-host-runtime`; see `tests/interop.rs`'s header), extended
//! to prove BOTH stream carriers this plugin now declares share its one
//! `on_open_stream` slot correctly: an untagged (evidence-request-shaped)
//! open must still reach the evidence responder (unchanged, proven again
//! here as a regression guard) and a ledger-fetch-tagged open must reach
//! the NEW responder -- never the other way around.

use mesh_llm_plugin::proto::{self, envelope::Payload};
use mesh_llm_plugin::{
    connect_side_stream, read_envelope, write_envelope, LocalListener, LocalStream,
    PROTOCOL_VERSION,
};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixListener;
use tokio::process::Command;
use tokio::time::timeout;

const PLUGIN_BIN: &str = env!("CARGO_BIN_EXE_capsules");
const TEST_TIMEOUT: Duration = Duration::from_secs(10);
const LEDGER_FETCH_CHANNEL: &str = "ledger-fetch/1";
/// Mirrors `ledger_fetch_bridge`'s private `LEDGER_FETCH_METADATA_TAG` --
/// duplicated here (not imported) because this file drives the plugin as an
/// external black box over the real wire, the same way a real host would;
/// it has no access to the plugin's own crate internals.
const LEDGER_FETCH_METADATA_TAG: &str = "ledger-fetch/1";

struct Harness {
    child: tokio::process::Child,
    stream: LocalStream,
    next_request_id: u64,
}

impl Harness {
    async fn spawn(extra_env: &[(&str, &str)]) -> Self {
        let socket_path =
            std::env::temp_dir().join(format!("ledger-fetch-interop-{}.sock", nonce()));
        let _ = std::fs::remove_file(&socket_path);
        let listener = UnixListener::bind(&socket_path).expect("bind fake-host socket");

        let data_dir = std::env::temp_dir().join(format!("ledger-fetch-interop-data-{}", nonce()));
        std::fs::create_dir_all(&data_dir).expect("create the data dir");
        let mut cmd = Command::new(PLUGIN_BIN);
        cmd.env("MESH_LLM_PLUGIN_ENDPOINT", &socket_path)
            .env("MESH_LLM_PLUGIN_TRANSPORT", "unix")
            .env("CAPSULES_BLOCKED_MODELS", "blocked-test-model")
            .env("CAPSULES_LEDGER_FETCH_TIMEOUT_MS", "1500")
            .env("CAPSULES_DATA_DIR", &data_dir)
            // Never the operator's mesh config or sharing switches: a test
            // that wants one sets it.
            .env("MESH_LLM_CONFIG", data_dir.join("no-mesh-config.toml"))
            .env_remove("CAPSULES_SHARE_HISTORY_SEGMENTS")
            .env_remove("CAPSULES_SHARE_STANDARD_HISTORY")
            // Same `/tmp` override `mesh_evidence_bridge_interop.rs` uses --
            // keeps the derived local-listener socket path under
            // `sockaddr_un`'s ~104-byte `sun_path` limit on macOS.
            .env("TMPDIR", "/tmp");
        for (key, value) in extra_env {
            cmd.env(key, value);
        }
        let child = cmd.spawn().expect("spawn the capsules plugin");

        let stream = timeout(
            TEST_TIMEOUT,
            LocalListener::Unix(listener, socket_path).accept(),
        )
        .await
        .expect("plugin connected before timeout")
        .expect("accept plugin connection");

        Self {
            child,
            stream,
            next_request_id: 1,
        }
    }

    fn next_id(&mut self) -> u64 {
        let id = self.next_request_id;
        self.next_request_id += 1;
        id
    }

    async fn send(&mut self, payload: Payload) -> u64 {
        let request_id = self.next_id();
        write_envelope(
            &mut self.stream,
            &proto::Envelope {
                protocol_version: PROTOCOL_VERSION,
                plugin_id: "capsules".to_string(),
                request_id,
                payload: Some(payload),
            },
        )
        .await
        .expect("write envelope to plugin");
        request_id
    }

    async fn recv(&mut self) -> proto::Envelope {
        timeout(TEST_TIMEOUT, read_envelope(&mut self.stream))
            .await
            .expect("plugin responded before timeout")
            .expect("read envelope from plugin")
    }

    async fn initialize(&mut self) -> proto::PluginManifest {
        let request_id = self
            .send(Payload::InitializeRequest(proto::InitializeRequest {
                host_protocol_version: PROTOCOL_VERSION,
                host_capabilities: Vec::new(),
                host_version: "ledger-fetch-interop-test".to_string(),
                host_info_json: "{}".to_string(),
                mesh_visibility: proto::MeshVisibility::Private as i32,
            }))
            .await;
        let envelope = self.recv().await;
        assert_eq!(envelope.request_id, request_id);
        let response = match envelope.payload {
            Some(Payload::InitializeResponse(response)) => response,
            other => panic!("expected InitializeResponse, got {other:?}"),
        };
        let manifest = response.manifest.expect("plugin declares a manifest");
        assert!(
            manifest
                .mesh_channels
                .iter()
                .any(|channel| channel.name == LEDGER_FETCH_CHANNEL),
            "manifest must declare {LEDGER_FETCH_CHANNEL}: {manifest:?}"
        );
        manifest
    }

    /// Seals one real capsule by POSTing an allowed-model chat completion to
    /// the plugin's own declared inference endpoint -- the SAME path
    /// `tests/interop.rs` uses -- and returns its `capsule_id`.
    async fn seal_one_real_capsule(&self, manifest: &proto::PluginManifest) -> String {
        let address = manifest
            .endpoints
            .iter()
            .find(|e| e.kind == proto::EndpointKind::Inference as i32)
            .expect("manifest declares an inference endpoint")
            .address
            .clone()
            .expect("endpoint carries a real HTTP address");

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{address}/v1/chat/completions"))
            .json(&serde_json::json!({
                "model": "an-allowed-model",
                "messages": [{"role": "user", "content": "hi"}],
            }))
            .send()
            .await
            .expect("POST /v1/chat/completions");
        assert_eq!(
            resp.status(),
            reqwest::StatusCode::OK,
            "seal request must be admitted"
        );
        let body: serde_json::Value = resp.json().await.expect("valid JSON body");
        body["admission_policy"]["capsule_id"]
            .as_str()
            .expect("response carries the sealed capsule_id")
            .to_string()
    }

    /// Responder-role driver, tagged as a ledger-fetch open via
    /// `metadata_json` -- see module doc.
    async fn open_ledger_fetch_stream(&mut self) -> LocalStream {
        let request_id = self
            .send(Payload::OpenStreamRequest(proto::OpenStreamRequest {
                stream_id: format!("test-stream-{}", nonce()),
                purpose: proto::StreamPurpose::Generic as i32,
                mode: proto::StreamMode::RawBytes as i32,
                bidirectional: true,
                content_type: Some("application/json".to_string()),
                correlation_id: None,
                metadata_json: Some(LEDGER_FETCH_METADATA_TAG.to_string()),
                expected_bytes: None,
                idle_timeout_ms: None,
            }))
            .await;
        let envelope = self.recv().await;
        assert_eq!(envelope.request_id, request_id);
        let response = match envelope.payload {
            Some(Payload::OpenStreamResponse(response)) => response,
            other => panic!("expected OpenStreamResponse, got {other:?}"),
        };
        assert!(
            response.accepted,
            "plugin must accept the ledger-fetch stream"
        );
        let endpoint = response
            .endpoint
            .expect("accepted response carries an endpoint");
        connect_side_stream(&endpoint, response.transport_kind)
            .await
            .expect("dial the plugin's local listener")
    }

    /// Requester-role driver: send the `mesh_ledger_fetch` tool call, then
    /// service exactly ONE outbound `OpenMeshStreamRequest` the way the real
    /// host's `open_outbound_plugin_mesh_stream` would, before reading the
    /// tool's `RpcResponse`.
    async fn call_mesh_ledger_fetch_tool(
        &mut self,
        peer_id: &str,
        capsule_id: &str,
        respond_open_mesh_stream: impl FnOnce(
            proto::OpenMeshStreamRequest,
        ) -> proto::OpenMeshStreamResponse,
    ) -> proto::Envelope {
        let params_json = serde_json::json!({
            "name": "mesh_ledger_fetch",
            "arguments": {"peer_id": peer_id, "capsule_id": capsule_id},
        })
        .to_string();
        let call_request_id = self
            .send(Payload::RpcRequest(proto::RpcRequest {
                method: "tools/call".to_string(),
                params_json,
            }))
            .await;

        let mesh_stream_envelope = self.recv().await;
        let mesh_stream_request = match mesh_stream_envelope.payload {
            Some(Payload::OpenMeshStreamRequest(request)) => request,
            other => panic!("expected outbound OpenMeshStreamRequest, got {other:?}"),
        };
        assert_eq!(mesh_stream_request.channel, LEDGER_FETCH_CHANNEL);
        assert_eq!(mesh_stream_request.target_peer_id, peer_id);
        assert_eq!(
            mesh_stream_request.metadata_json.as_deref(),
            Some(LEDGER_FETCH_METADATA_TAG),
            "requester must tag its own open so a real host-side responder can dispatch it"
        );
        let mesh_stream_response = respond_open_mesh_stream(mesh_stream_request);
        write_envelope(
            &mut self.stream,
            &proto::Envelope {
                protocol_version: PROTOCOL_VERSION,
                plugin_id: "capsules".to_string(),
                request_id: mesh_stream_envelope.request_id,
                payload: Some(Payload::OpenMeshStreamResponse(mesh_stream_response)),
            },
        )
        .await
        .expect("write OpenMeshStreamResponse to plugin");

        let response_envelope = self.recv().await;
        assert_eq!(response_envelope.request_id, call_request_id);
        response_envelope
    }

    async fn shutdown_process(mut self) {
        let request_id = self
            .send(Payload::ShutdownRequest(proto::ShutdownRequest {
                reason: "ledger fetch bridge interop test complete".to_string(),
            }))
            .await;
        let envelope = self.recv().await;
        assert_eq!(envelope.request_id, request_id);
        assert!(matches!(
            envelope.payload,
            Some(Payload::ShutdownResponse(_))
        ));

        let status = timeout(TEST_TIMEOUT, self.child.wait())
            .await
            .expect("plugin exited before timeout")
            .expect("wait on plugin process");
        assert!(status.success(), "plugin process exited with {status:?}");
    }
}

fn nonce() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    format!(
        "{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

async fn accept_one_remote_peer_connection() -> (UnixListener, std::path::PathBuf) {
    let path = std::env::temp_dir().join(format!("ledger-fetch-fake-peer-{}.sock", nonce()));
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).expect("bind fake peer");
    (listener, path)
}

/// Under the default sharing switch, an unsigned fetch stating an id is
/// declined over the real wire before anything is looked up: a record goes
/// only to the node a signature proves, never to whoever asks.
#[tokio::test]
async fn responder_declines_an_unsigned_fetch_by_default() {
    let mut harness = Harness::spawn(&[]).await;
    let manifest = harness.initialize().await;
    let capsule_id = harness.seal_one_real_capsule(&manifest).await;

    let stream = harness.open_ledger_fetch_stream().await;
    let (mut read_half, mut write_half) = stream.into_split();
    let request_bytes = serde_json::to_vec(
        &serde_json::json!({ "capsule_id": capsule_id, "requester_id": "b0b0b0b0" }),
    )
    .unwrap();
    write_half
        .write_all(&request_bytes)
        .await
        .expect("write ledger-fetch request");
    write_half.shutdown().await.expect("half-close request");

    let mut response_bytes = Vec::new();
    timeout(TEST_TIMEOUT, read_half.read_to_end(&mut response_bytes))
        .await
        .expect("responder answered before timeout")
        .expect("read response bytes");
    let response: serde_json::Value = serde_json::from_slice(&response_bytes).expect("valid JSON");
    assert_eq!(response["status"], "not_authorized");
    assert!(response.get("capsule").is_none());

    harness.shutdown_process().await;
}

/// (A) Responder role, happy path: a mesh-inbound ledger-fetch stream for a
/// capsule_id THIS node really sealed gets answered with the real capsule,
/// the real (base64) signed_statement bytes, and this node's real pubkey --
/// never fabricated, never a second attestation wrapper.
#[tokio::test]
async fn responder_answers_a_real_sealed_capsule_over_the_real_wire() {
    // This capsule names no other side (the fake host sends no requester),
    // so only the `peers` tier serves it; the default declines it (below).
    let mut harness = Harness::spawn(&[("CAPSULES_SHARE_HISTORY_SEGMENTS", "peers")]).await;
    let manifest = harness.initialize().await;
    let capsule_id = harness.seal_one_real_capsule(&manifest).await;

    let stream = harness.open_ledger_fetch_stream().await;
    let (mut read_half, mut write_half) = stream.into_split();
    let request_bytes =
        serde_json::to_vec(&serde_json::json!({ "capsule_id": capsule_id })).unwrap();
    write_half
        .write_all(&request_bytes)
        .await
        .expect("write ledger-fetch request");
    write_half.shutdown().await.expect("half-close request");

    let mut response_bytes = Vec::new();
    timeout(TEST_TIMEOUT, read_half.read_to_end(&mut response_bytes))
        .await
        .expect("responder answered before timeout")
        .expect("read response bytes");
    let response: serde_json::Value = serde_json::from_slice(&response_bytes).expect("valid JSON");

    assert_eq!(response["status"], "found");
    assert_eq!(response["capsule"]["capsule_id"], capsule_id);
    assert!(!response["signed_statement_b64"]
        .as_str()
        .unwrap()
        .is_empty());
    assert!(response["node_pub_key_pem"]
        .as_str()
        .unwrap()
        .contains("PUBLIC KEY"));

    harness.shutdown_process().await;
}

/// (B) Responder role, miss: a capsule_id this node never sealed comes back
/// `not_found`, never a fabricated `found` -- proven over the real wire, not
/// just the pure `answer()` unit tests.
#[tokio::test]
async fn responder_reports_not_found_for_a_real_miss_over_the_real_wire() {
    // Under `peers` an unsigned fetch is looked up; the default refuses it
    // before the ledger is read (above).
    let mut harness = Harness::spawn(&[("CAPSULES_SHARE_HISTORY_SEGMENTS", "peers")]).await;
    harness.initialize().await;

    let stream = harness.open_ledger_fetch_stream().await;
    let (mut read_half, mut write_half) = stream.into_split();
    let request_bytes =
        serde_json::to_vec(&serde_json::json!({ "capsule_id": "never-sealed" })).unwrap();
    write_half
        .write_all(&request_bytes)
        .await
        .expect("write ledger-fetch request");
    write_half.shutdown().await.expect("half-close request");

    let mut response_bytes = Vec::new();
    timeout(TEST_TIMEOUT, read_half.read_to_end(&mut response_bytes))
        .await
        .expect("responder answered before timeout")
        .expect("read response bytes");
    let response: serde_json::Value = serde_json::from_slice(&response_bytes).expect("valid JSON");

    assert_eq!(response["status"], "not_found");
    assert_eq!(response["capsule_id"], "never-sealed");

    harness.shutdown_process().await;
}

/// (C) Both carriers on one `on_open_stream` slot, dispatched correctly: an
/// UNTAGGED open (the evidence-request carrier's own shape,
/// `metadata_json: None`) must still reach the evidence responder, never the
/// ledger-fetch one -- a regression guard for the shared-slot dispatch this
/// change introduces.
#[tokio::test]
async fn untagged_open_still_reaches_the_evidence_responder_not_ledger_fetch() {
    let mut harness = Harness::spawn(&[]).await;
    harness.initialize().await;

    let request_id = harness
        .send(Payload::OpenStreamRequest(proto::OpenStreamRequest {
            stream_id: format!("test-stream-{}", nonce()),
            purpose: proto::StreamPurpose::Generic as i32,
            mode: proto::StreamMode::RawBytes as i32,
            bidirectional: true,
            content_type: Some("application/json".to_string()),
            correlation_id: None,
            metadata_json: None, // the evidence-request carrier's own shape
            expected_bytes: None,
            idle_timeout_ms: None,
        }))
        .await;
    let envelope = harness.recv().await;
    assert_eq!(envelope.request_id, request_id);
    let response = match envelope.payload {
        Some(Payload::OpenStreamResponse(response)) => response,
        other => panic!("expected OpenStreamResponse, got {other:?}"),
    };
    let endpoint = response
        .endpoint
        .expect("accepted response carries an endpoint");
    let stream = connect_side_stream(&endpoint, response.transport_kind)
        .await
        .expect("dial the plugin's local listener");
    let (mut read_half, mut write_half) = stream.into_split();
    // Any request the evidence responder refuses: its signed -00 refusal
    // (`reason` + `sig`) proves which responder the untagged open reached.
    write_half
        .write_all(br#"{"subject":{"kind":"adjudicate"},"halves":[]}"#)
        .await
        .expect("write the request");
    write_half.shutdown().await.expect("half-close request");

    let mut response_bytes = Vec::new();
    timeout(TEST_TIMEOUT, read_half.read_to_end(&mut response_bytes))
        .await
        .expect("responder answered before timeout")
        .expect("read response bytes");
    // A ledger-fetch response would instead be `{"status": ...}` JSON.
    let reply: serde_json::Value = serde_json::from_slice(&response_bytes).expect("a JSON answer");
    assert!(
        reply.get("status").is_none(),
        "not a ledger-fetch answer: {reply}"
    );
    assert!(reply["reason"].is_string(), "{reply}");
    assert!(
        reply["sig"].as_str().is_some_and(|s| s.len() == 128),
        "a signed refusal: {reply}"
    );

    harness.shutdown_process().await;
}

/// (D) Requester role, happy path: the `mesh_ledger_fetch` tool opens a real
/// outbound, correctly-tagged `OpenMeshStreamRequest`, writes the
/// `{"capsule_id": ...}` request, and returns the peer's raw response JSON
/// unchanged -- proven against a REAL dialed connection playing the remote
/// peer, not a stub.
#[tokio::test]
async fn requester_tool_call_round_trips_a_real_dialed_stream() {
    let mut harness = Harness::spawn(&[]).await;
    harness.initialize().await;

    let (peer_listener, peer_path) = accept_one_remote_peer_connection().await;
    let peer_task = tokio::spawn(async move {
        let (mut socket, _) = peer_listener.accept().await.expect("accept from requester");
        let mut request_bytes = Vec::new();
        socket
            .read_to_end(&mut request_bytes)
            .await
            .expect("read ledger-fetch request from requester");
        let response = br#"{"status":"found","capsule":{"capsule_id":"peer-real-id"},"signed_statement_b64":"AAA=","node_pub_key_pem":"-----BEGIN PUBLIC KEY-----\nx\n-----END PUBLIC KEY-----\n"}"#;
        socket.write_all(response).await.expect("write response");
        socket.shutdown().await.expect("half-close response");
        request_bytes
    });

    let response_envelope = harness
        .call_mesh_ledger_fetch_tool("aa".repeat(32).as_str(), "peer-real-id", |_request| {
            proto::OpenMeshStreamResponse {
                stream_id: "test".to_string(),
                accepted: true,
                transport_kind: proto::StreamTransportKind::StreamUnixSocket as i32,
                endpoint: Some(peer_path.to_str().unwrap().to_string()),
                token: None,
                expires_at_unix_ms: None,
                message: None,
            }
        })
        .await;

    let result = match response_envelope.payload {
        Some(Payload::RpcResponse(response)) => response,
        other => panic!("expected a successful RpcResponse, got {other:?}"),
    };
    let call_result: rmcp::model::CallToolResult =
        serde_json::from_str(&result.result_json).expect("decode CallToolResult");
    assert_eq!(call_result.is_error, Some(false));
    let structured = call_result
        .structured_content
        .expect("mesh_ledger_fetch returns structured JSON");
    assert_eq!(structured["status"], "found");
    assert_eq!(structured["capsule"]["capsule_id"], "peer-real-id");

    let request_bytes = timeout(TEST_TIMEOUT, peer_task)
        .await
        .expect("peer task finished")
        .expect("peer task did not panic");
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&request_bytes).unwrap(),
        serde_json::json!({"capsule_id": "peer-real-id", "requester_id": null}),
        "the asker states its own peer id, null until the host has reported it"
    );

    harness.shutdown_process().await;
}

/// (E) Clean failure, never a hang: a peer that never declares the channel
/// has its stream dropped by the real host with no reply -- simulated here
/// by the fake host accepting the mesh-stream open but never dialing/writing
/// anything back. Bounded by `CAPSULES_LEDGER_FETCH_TIMEOUT_MS` (set
/// to 1500ms for this harness), so this test proves the bound, not merely
/// that failure is possible.
#[tokio::test]
async fn requester_reports_a_clean_failure_when_the_peer_never_answers() {
    let mut harness = Harness::spawn(&[]).await;
    harness.initialize().await;

    let (peer_listener, peer_path) = accept_one_remote_peer_connection().await;
    std::mem::forget(peer_listener);

    let started = tokio::time::Instant::now();
    let response_envelope = harness
        .call_mesh_ledger_fetch_tool("bb".repeat(32).as_str(), "whatever", |_request| {
            proto::OpenMeshStreamResponse {
                stream_id: "test".to_string(),
                accepted: true,
                transport_kind: proto::StreamTransportKind::StreamUnixSocket as i32,
                endpoint: Some(peer_path.to_str().unwrap().to_string()),
                token: None,
                expires_at_unix_ms: None,
                message: None,
            }
        })
        .await;
    let elapsed = started.elapsed();

    assert!(
        elapsed < Duration::from_secs(5),
        "requester must fail within its bounded timeout, took {elapsed:?}"
    );
    match response_envelope.payload {
        Some(Payload::ErrorResponse(_)) => {}
        other => panic!("expected a bounded ErrorResponse, got {other:?} after {elapsed:?}"),
    }

    harness.shutdown_process().await;
}
