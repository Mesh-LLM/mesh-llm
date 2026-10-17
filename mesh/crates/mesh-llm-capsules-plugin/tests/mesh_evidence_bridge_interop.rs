//! Real interop test for the `evidence-request/1` plugin-mesh-stream carrier
//! (`mesh_evidence_bridge`), over the ACTUAL wire protocol -- same fake-host
//! approach as `tests/interop.rs` (this repo's CI cannot build
//! `mesh-llm-host-runtime`; see that file's header), extended with the two
//! message types `interop.rs` does not need: `OpenStreamRequest` (drives the
//! RESPONDER role, `on_open_stream`) and `RpcRequest{"tools/call"}` +
//! `OpenMeshStreamRequest` (drives the REQUESTER role, the
//! `mesh_evidence_request` tool).
//!
//! The real mesh-llm host gates BOTH of these on `evidence-request/1` being
//! declared before ever reaching this plugin (`plugin_event_channel_declared`,
//! `mesh/plugin_streams.rs`); this fake host sends them directly, exactly as
//! the real host would after that gate already passed -- proving the
//! plugin's OWN behavior once reached, not the host's gate (untestable here
//! without `mesh-llm-host-runtime`; that gate is upstream, unmodified code).

use mesh_llm_plugin::proto::{self, envelope::Payload};
use mesh_llm_plugin::{
    connect_side_stream, read_envelope, write_envelope, LocalListener, LocalStream,
    PROTOCOL_VERSION,
};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, UnixListener};
use tokio::process::Command;
use tokio::time::timeout;

const PLUGIN_BIN: &str = env!("CARGO_BIN_EXE_capsules");
const TEST_TIMEOUT: Duration = Duration::from_secs(10);
const EVIDENCE_REQUEST_CHANNEL: &str = "evidence-request/1";

struct Harness {
    child: tokio::process::Child,
    stream: LocalStream,
    next_request_id: u64,
    data_dir: std::path::PathBuf,
}

impl Harness {
    async fn spawn(extra_env: &[(&str, &str)]) -> Self {
        let socket_path =
            std::env::temp_dir().join(format!("mesh-evidence-interop-{}.sock", nonce()));
        let _ = std::fs::remove_file(&socket_path);
        let listener = UnixListener::bind(&socket_path).expect("bind fake-host socket");

        let data_dir = std::env::temp_dir().join(format!("mesh-evidence-interop-data-{}", nonce()));
        // The "asked of you" log is opt-in: its directory must exist.
        std::fs::create_dir_all(data_dir.join("received-log"))
            .expect("create the received-log dir");
        let mut cmd = Command::new(PLUGIN_BIN);
        cmd.env("MESH_LLM_PLUGIN_ENDPOINT", &socket_path)
            .env("MESH_LLM_PLUGIN_TRANSPORT", "unix")
            // An isolated data dir per run: never the operator's own.
            .env("CAPSULES_DATA_DIR", &data_dir)
            // Nor the operator's mesh config: no console setting reaches it.
            .env("MESH_LLM_CONFIG", data_dir.join("no-mesh-config.toml"))
            // Nor its sharing switches: a test that wants one sets it.
            .env_remove("CAPSULES_SHARE_HISTORY_SEGMENTS")
            .env_remove("CAPSULES_SHARE_STANDARD_HISTORY")
            .env("CAPSULES_BLOCKED_MODELS", "blocked-test-model")
            .env("CAPSULES_MESH_REQUEST_TIMEOUT_MS", "1500")
            // `bind_side_stream` (mesh-llm-plugin's own `io.rs`) derives the
            // evidence-stream's local socket path from `std::env::temp_dir()`
            // with no override of its own -- on macOS that resolves to a long
            // per-process `/var/folders/.../T/` path which, combined with
            // this plugin's own prefix, sits right at (or over) `sockaddr_un`'s
            // ~104-byte `sun_path` limit (same class of issue `host_runtime_
            // e2e.rs`'s `RealHost` already works around for `$HOME`-derived
            // plugin socket paths). `/tmp` keeps it well under that limit.
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
            data_dir,
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

    async fn initialize(&mut self) {
        let request_id = self
            .send(Payload::InitializeRequest(proto::InitializeRequest {
                host_protocol_version: PROTOCOL_VERSION,
                host_capabilities: Vec::new(),
                host_version: "mesh-evidence-interop-test".to_string(),
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
                .any(|channel| channel.name == EVIDENCE_REQUEST_CHANNEL),
            "manifest must declare {EVIDENCE_REQUEST_CHANNEL}: {manifest:?}"
        );
    }

    /// Responder-role driver: send `OpenStreamRequest` directly (the real
    /// host only ever does this once `evidence-request/1` is already
    /// gate-checked -- see module docs), dial the returned endpoint as the
    /// real host's `bridge_local_stream_bidirectional` would, and return a
    /// connected duplex stream standing in for the remote peer's QUIC bytes.
    async fn open_evidence_stream(&mut self) -> LocalStream {
        let request_id = self
            .send(Payload::OpenStreamRequest(proto::OpenStreamRequest {
                stream_id: format!("test-stream-{}", nonce()),
                purpose: proto::StreamPurpose::Generic as i32,
                mode: proto::StreamMode::RawBytes as i32,
                bidirectional: true,
                content_type: Some("application/json".to_string()),
                correlation_id: None,
                metadata_json: None,
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
        assert!(response.accepted, "plugin must accept the evidence stream");
        let endpoint = response
            .endpoint
            .expect("accepted response carries an endpoint");
        connect_side_stream(&endpoint, response.transport_kind)
            .await
            .expect("dial the plugin's local listener")
    }

    /// Requester-role driver: send the `mesh_evidence_request` tool call,
    /// then service exactly ONE outbound `OpenMeshStreamRequest` the way the
    /// real host's `open_outbound_plugin_mesh_stream` would (accept + reply),
    /// via `respond_open_mesh_stream`, before reading the tool's
    /// `RpcResponse`. Returns the raw envelope so both the success and
    /// error (`ErrorResponse`) shapes are inspectable.
    async fn call_mesh_evidence_tool(
        &mut self,
        peer_id: &str,
        request: serde_json::Value,
        verify: Option<bool>,
        respond_open_mesh_stream: impl FnOnce(
            proto::OpenMeshStreamRequest,
        ) -> proto::OpenMeshStreamResponse,
    ) -> proto::Envelope {
        let mut arguments = serde_json::json!({"peer_id": peer_id, "request": request});
        if let Some(verify) = verify {
            arguments["verify"] = serde_json::json!(verify);
        }
        let params_json = serde_json::json!({
            "name": "mesh_evidence_request",
            "arguments": arguments,
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
        assert_eq!(mesh_stream_request.channel, EVIDENCE_REQUEST_CHANNEL);
        assert_eq!(mesh_stream_request.target_peer_id, peer_id);
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

/// Stands in for the remote peer that the fake host's `respond_open_mesh_
/// stream` reports as reachable: binds a real local listener and accepts
/// exactly one connection, playing "the peer's own responder". A Unix
/// socket, not TCP -- `mesh-llm-plugin`'s own `connect_side_stream` (what
/// `PluginContext::connect_mesh_stream` dials with) only implements
/// `StreamTransportKind::StreamUnixSocket`/`StreamNamedPipe`; `StreamTcp` is
/// a `#[cfg(test)]`-only variant internal to the real host, never dialable
/// from plugin-side code.
async fn accept_one_remote_peer_connection() -> (UnixListener, std::path::PathBuf) {
    let path = std::env::temp_dir().join(format!("mesh-evidence-fake-peer-{}.sock", nonce()));
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).expect("bind fake peer");
    (listener, path)
}

/// A local HTTP listener set as `CAPSULES_EVIDENCE_SERVER_URL`, the
/// evidence-server setting older builds read (this build reads no such
/// setting): it records every request it receives and answers with a canned
/// body. No external evidence server is contacted, so a test asserts it
/// received nothing.
async fn spawn_fake_evidence_server(
    response_body: &'static [u8],
) -> (String, std::sync::Arc<tokio::sync::Mutex<Vec<Vec<u8>>>>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind the listening stub");
    let port = listener.local_addr().unwrap().port();
    let received = std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let received_for_task = received.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let received = received_for_task.clone();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 64 * 1024];
                let read = socket.read(&mut buf).await.unwrap_or(0);
                let request = &buf[..read];
                let header_end = request
                    .windows(4)
                    .position(|w| w == b"\r\n\r\n")
                    .map(|i| i + 4)
                    .unwrap_or(request.len());
                received.lock().await.push(request[header_end..].to_vec());

                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                    response_body.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.write_all(response_body).await;
                let _ = socket.shutdown().await;
            });
        }
    });
    (format!("http://127.0.0.1:{port}"), received)
}

/// Write `request_bytes` as the remote peer, half-close, and read the whole
/// answer.
async fn ask_over_stream(stream: LocalStream, request_bytes: &[u8]) -> Vec<u8> {
    let (mut read_half, mut write_half) = stream.into_split();
    write_half
        .write_all(request_bytes)
        .await
        .expect("write the request");
    write_half.shutdown().await.expect("half-close the request");
    let mut response_bytes = Vec::new();
    timeout(TEST_TIMEOUT, read_half.read_to_end(&mut response_bytes))
        .await
        .expect("responder answered before timeout")
        .expect("read response bytes");
    response_bytes
}

/// (A) Responder role: a mesh-inbound -00 evidence request is answered
/// in-process -- here a signed `coverage_unsatisfiable` refusal (a fresh
/// node has no checkpoint yet), bound to the digest of the bytes the peer
/// sent -- no external evidence server is contacted, and the request is
/// logged to the "asked of you" log. Standard History is shared, so the
/// request gets as far as its coverage.
#[tokio::test]
async fn responder_answers_an_evidence_request_in_process() {
    let (evidence_server_url, received) = spawn_fake_evidence_server(b"{}").await;
    let mut harness = Harness::spawn(&[
        ("CAPSULES_EVIDENCE_SERVER_URL", &evidence_server_url),
        ("CAPSULES_SHARE_STANDARD_HISTORY", "on"),
    ])
    .await;
    harness.initialize().await;

    let stream = harness.open_evidence_stream().await;
    let request_bytes = br#"{"subject":{"checkpoints":null},"coverage":{"min_freshness":1},"requester_id":"node-a"}"#;
    let response: serde_json::Value =
        serde_json::from_slice(&ask_over_stream(stream, request_bytes).await)
            .expect("a JSON answer");

    let check = capsule_emit_evidence_request::refusal::check(&response);
    assert!(check.conformant, "a conforming, signed refusal: {response}");
    assert_eq!(response["reason"], "coverage_unsatisfiable");
    assert_eq!(
        response["request_digest"],
        capsule_emit_evidence_request::digest::request_digest(request_bytes)
    );
    assert!(
        received.lock().await.is_empty(),
        "no external evidence server is contacted"
    );

    let log = std::fs::read_to_string(harness.data_dir.join("received-log/received_log.jsonl"))
        .expect("the request was logged");
    let line: serde_json::Value = serde_json::from_str(log.trim()).expect("one JSON line");
    assert_eq!(line["path"], "evidence-request");
    assert_eq!(line["requester_id"], "node-a");
    assert_eq!(line["subject_kind"], "checkpoints");
    assert_eq!(line["status"], "refused");
    assert_eq!(line["reason"], "coverage_unsatisfiable");

    harness.shutdown_process().await;
}

/// PRD T7, the default half, on the plugin binary: with the default
/// settings, two nodes this one never exchanged with ask for its checkpoints
/// and each gets a signed `policy_declined` (on a fresh node, so it comes
/// before coverage, which would refuse `coverage_unsatisfiable`). Each
/// carries exactly the -00 refusal's members, none naming the asker, and the
/// two agree on all but the request each binds, the time and the signature.
#[tokio::test]
async fn standard_history_is_one_signed_refusal_to_every_non_counterparty_by_default() {
    let mut harness = Harness::spawn(&[]).await;
    harness.initialize().await;

    let mut refusals = Vec::new();
    for asker in ["node-y", "node-z"] {
        let request = serde_json::json!({
            "subject": {"checkpoints": null},
            "coverage": {"min_freshness": 1},
            "requester_id": asker,
        })
        .to_string();
        let stream = harness.open_evidence_stream().await;
        let mut response: serde_json::Value =
            serde_json::from_slice(&ask_over_stream(stream, request.as_bytes()).await)
                .expect("a JSON answer");
        assert!(
            capsule_emit_evidence_request::refusal::check(&response).conformant,
            "a conforming, signed refusal: {response}"
        );
        assert_eq!(response["reason"], "policy_declined", "{response}");
        assert_eq!(
            response["request_digest"],
            capsule_emit_evidence_request::digest::request_digest(request.as_bytes())
        );
        let members = response.as_object_mut().unwrap();
        let mut names: Vec<&str> = members.keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            ["issued_at", "key_id", "reason", "request_digest", "sig"],
            "nothing beside the -00 refusal's members"
        );
        for bound in ["request_digest", "issued_at", "sig"] {
            members.remove(bound).expect("a refusal binds its request");
        }
        refusals.push(response);
    }
    assert_eq!(refusals[0], refusals[1]);

    harness.shutdown_process().await;
}

/// (A2) This node has no referee: a referee's `adjudicate` request is
/// answered in-process with a refusal signed by this node, never a verdict,
/// and never forwarded anywhere. No external evidence server is contacted:
/// the listening stub receives nothing.
#[tokio::test]
async fn responder_refuses_an_adjudicate_request_signed_and_contacts_no_evidence_server() {
    let (door_url, received) = spawn_fake_evidence_server(b"{}").await;
    let mut harness = Harness::spawn(&[("CAPSULES_EVIDENCE_SERVER_URL", &door_url)]).await;
    harness.initialize().await;

    let stream = harness.open_evidence_stream().await;
    let request_bytes = br#"{"subject":{"kind":"adjudicate"},"halves":[]}"#;
    let response_bytes = ask_over_stream(stream, request_bytes).await;

    let reply: serde_json::Value = serde_json::from_slice(&response_bytes).expect("a JSON refusal");
    assert_eq!(reply["reason"], "request_malformed", "{reply}");
    assert!(
        reply.get("adjudication_verdict").is_none(),
        "never a verdict"
    );
    assert!(
        reply["key_id"].as_str().is_some_and(|k| k.len() == 64),
        "{reply}"
    );
    assert!(
        reply["sig"].as_str().is_some_and(|s| s.len() == 128),
        "signed: {reply}"
    );
    assert!(
        received.lock().await.is_empty(),
        "no external evidence server is contacted"
    );

    harness.shutdown_process().await;
}

/// (A3) A request over the side-stream cap is never answered (and never
/// buffered past the cap): the responder drops the stream.
#[tokio::test]
async fn responder_drops_a_request_over_the_side_stream_cap() {
    let mut harness = Harness::spawn(&[]).await;
    harness.initialize().await;

    let stream = harness.open_evidence_stream().await;
    let (mut read_half, mut write_half) = stream.into_split();
    let chunk = vec![b' '; 64 * 1024];
    // Write past the 1 MiB cap; the responder may stop reading first.
    for _ in 0..(1024 * 1024 / chunk.len() + 2) {
        if write_half.write_all(&chunk).await.is_err() {
            break;
        }
    }
    let _ = write_half.shutdown().await;
    let mut response_bytes = Vec::new();
    let _ = timeout(TEST_TIMEOUT, read_half.read_to_end(&mut response_bytes)).await;
    assert!(
        response_bytes.is_empty(),
        "nothing is answered to an over-cap request"
    );

    harness.shutdown_process().await;
}

/// (B) Requester role, happy path: the `mesh_evidence_request` tool opens a
/// real outbound `OpenMeshStreamRequest`, writes the E14 request over the
/// resulting stream, and returns the peer's raw response bytes as the tool's
/// JSON result -- proven against a REAL dialed connection playing the remote
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
            .expect("read E14 request from requester");
        let response = br#"{"subject_kind":"record","bundles":[]}"#;
        socket.write_all(response).await.expect("write response");
        socket.shutdown().await.expect("half-close response");
        request_bytes
    });

    let request = serde_json::json!({"subject": {"kind": "record", "capsule_id": "aa"}});
    let response_envelope = harness
        .call_mesh_evidence_tool(
            "aa".repeat(32).as_str(),
            request.clone(),
            Some(false),
            |_request| proto::OpenMeshStreamResponse {
                stream_id: "test".to_string(),
                accepted: true,
                transport_kind: proto::StreamTransportKind::StreamUnixSocket as i32,
                endpoint: Some(peer_path.to_str().unwrap().to_string()),
                token: None,
                expires_at_unix_ms: None,
                message: None,
            },
        )
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
        .expect("mesh_evidence_request returns structured JSON");
    assert_eq!(structured["subject_kind"], "record");

    let request_bytes = timeout(TEST_TIMEOUT, peer_task)
        .await
        .expect("peer task finished")
        .expect("peer task did not panic");
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&request_bytes).unwrap(),
        request
    );

    harness.shutdown_process().await;
}

/// (B2) Verification is on unless the caller names the opt-out: with no
/// `verify` argument, the peer's answer comes back beside a verification --
/// here `no_announced_key`, because this node's registry names no key for
/// the peer -- never alone and unchecked.
#[tokio::test]
async fn requester_verifies_by_default() {
    let mut harness = Harness::spawn(&[]).await;
    harness.initialize().await;

    let (peer_listener, peer_path) = accept_one_remote_peer_connection().await;
    tokio::spawn(async move {
        let (mut socket, _) = peer_listener.accept().await.expect("accept from requester");
        let mut discard = Vec::new();
        let _ = socket.read_to_end(&mut discard).await;
        let _ = socket.write_all(br#"{"reason":"no_such_subject"}"#).await;
        let _ = socket.shutdown().await;
    });

    let request =
        serde_json::json!({"subject": {"checkpoints": null}, "coverage": {"min_freshness": 1}});
    let response_envelope = harness
        .call_mesh_evidence_tool("dd".repeat(32).as_str(), request, None, |_request| {
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
    let structured = call_result.structured_content.expect("structured JSON");
    assert_eq!(structured["answer"]["reason"], "no_such_subject");
    assert_eq!(structured["verification"]["state"], "no_announced_key");
    assert!(structured["request_digest"]
        .as_str()
        .is_some_and(|d| d.len() == 64));

    harness.shutdown_process().await;
}

/// (C) Clean failure, never a hang: a peer that never declares the channel
/// has its stream dropped by the real host with no reply
/// (`handle_plugin_mesh_stream` returns `Ok(())` without touching `send`/
/// `recv` -- see module docs) -- simulated here by the fake host accepting
/// the mesh-stream open but never dialing/writing anything back. Bounded by
/// `CAPSULES_MESH_REQUEST_TIMEOUT_MS` (set to 1500ms for this test
/// harness, see `Harness::spawn`), so this test proves the bound, not merely
/// that failure is possible.
#[tokio::test]
async fn requester_reports_a_clean_failure_when_the_peer_never_answers() {
    let mut harness = Harness::spawn(&[]).await;
    harness.initialize().await;

    let (peer_listener, peer_path) = accept_one_remote_peer_connection().await;
    // Never accept the connection the plugin dials -- exactly what "the real
    // peer's host silently drops the stream" looks like from here.
    std::mem::forget(peer_listener);

    let started = tokio::time::Instant::now();
    let response_envelope = harness
        .call_mesh_evidence_tool(
            "bb".repeat(32).as_str(),
            serde_json::json!({}),
            None,
            |_request| proto::OpenMeshStreamResponse {
                stream_id: "test".to_string(),
                accepted: true,
                transport_kind: proto::StreamTransportKind::StreamUnixSocket as i32,
                endpoint: Some(peer_path.to_str().unwrap().to_string()),
                token: None,
                expires_at_unix_ms: None,
                message: None,
            },
        )
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

/// (D) Mutant: tamper the bytes the peer writes back, in flight, before the
/// requester ever sees them -- proves the carrier does not itself validate
/// or silently repair anything; that job stays with the requester's own
/// offline verification of the bytes it receives.
#[tokio::test]
async fn tampered_bytes_in_flight_pass_through_unvalidated_and_unrepaired() {
    let mut harness = Harness::spawn(&[]).await;
    harness.initialize().await;

    let (peer_listener, peer_path) = accept_one_remote_peer_connection().await;
    let genuine =
        br#"{"subject_kind":"record","bundles":[{"receipt":{"capsule_id":"real"}}]}"#.to_vec();
    let mut tampered = genuine.clone();
    let flip_at = tampered
        .iter()
        .position(|&b| b == b'r')
        .expect("has a byte to flip");
    tampered[flip_at] = b'X';
    let tampered_for_task = tampered.clone();
    tokio::spawn(async move {
        let (mut socket, _) = peer_listener.accept().await.expect("accept from requester");
        let mut discard = Vec::new();
        let _ = socket.read_to_end(&mut discard).await;
        let _ = socket.write_all(&tampered_for_task).await;
        let _ = socket.shutdown().await;
    });

    let response_envelope = harness
        .call_mesh_evidence_tool(
            "cc".repeat(32).as_str(),
            serde_json::json!({}),
            Some(false),
            |_request| proto::OpenMeshStreamResponse {
                stream_id: "test".to_string(),
                accepted: true,
                transport_kind: proto::StreamTransportKind::StreamUnixSocket as i32,
                endpoint: Some(peer_path.to_str().unwrap().to_string()),
                token: None,
                expires_at_unix_ms: None,
                message: None,
            },
        )
        .await;

    let result = match response_envelope.payload {
        Some(Payload::RpcResponse(response)) => response,
        other => panic!("expected a successful (carrier-level) RpcResponse, got {other:?}"),
    };
    let call_result: rmcp::model::CallToolResult =
        serde_json::from_str(&result.result_json).expect("decode CallToolResult");
    let structured = call_result.structured_content.expect("structured JSON");
    let round_tripped = serde_json::to_vec(&structured).expect("re-encode");

    // The carrier delivered the TAMPERED bytes untouched -- it neither
    // detected nor repaired the flip. (That the requester's own verification
    // rejects a flipped byte is a property of that verification, independent
    // of the carrier, and deliberately not re-proven here.)
    assert_ne!(
        round_tripped, genuine,
        "the tampered byte must reach the caller"
    );

    harness.shutdown_process().await;
}

impl Harness {
    async fn shutdown_process(mut self) {
        let request_id = self
            .send(Payload::ShutdownRequest(proto::ShutdownRequest {
                reason: "mesh evidence bridge interop test complete".to_string(),
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
