//! Real interop test: drives the compiled `capsules` binary
//! over mesh-llm's actual wire protocol (length-prefixed `proto::Envelope`
//! frames over a Unix domain socket — see `mesh-llm-plugin/src/io.rs`), the
//! same way `mesh-llm-host-runtime::plugin::runtime::ExternalPlugin` does,
//! then exercises the plugin's declared inference-provider HTTP endpoint with
//! real HTTP requests.
//!
//! This test acts as a minimal, faithful stand-in for the host side of the
//! protocol rather than depending on `mesh-llm-host-runtime` directly: that
//! crate is not published to crates.io and pulls in mesh-llm's native
//! GPU/runtime toolchain, which would make this public repo's CI depend on
//! build requirements entirely outside this repo's control. Every message
//! type, field, and helper used below (`proto::Envelope`, `InitializeRequest`,
//! `LocalListener`, `read_envelope`/`write_envelope`) comes directly from the
//! published `mesh-llm-plugin` crate — the real wire contract, not an
//! invented one.

use mesh_llm_plugin::proto::{self, envelope::Payload};
use mesh_llm_plugin::{
    read_envelope, write_envelope, LocalListener, LocalStream, PROTOCOL_VERSION,
};
use std::time::Duration;
use tokio::net::UnixListener;
use tokio::process::Command;
use tokio::time::timeout;

const PLUGIN_BIN: &str = env!("CARGO_BIN_EXE_capsules");
const TEST_TIMEOUT: Duration = Duration::from_secs(10);

struct Harness {
    child: tokio::process::Child,
    stream: LocalStream,
    next_request_id: u64,
}

impl Harness {
    async fn spawn(extra_env: &[(&str, &str)]) -> Self {
        Self::spawn_without(extra_env, &[]).await
    }

    /// As `spawn`, with `unset` removed from the plugin's environment.
    async fn spawn_without(extra_env: &[(&str, &str)], unset: &[&str]) -> Self {
        let socket_path = std::env::temp_dir().join(format!("capsules-interop-{}.sock", nonce()));
        let _ = std::fs::remove_file(&socket_path);
        let listener = UnixListener::bind(&socket_path).expect("bind fake-host socket");

        let mut cmd = Command::new(PLUGIN_BIN);
        cmd.env("MESH_LLM_PLUGIN_ENDPOINT", &socket_path)
            .env("MESH_LLM_PLUGIN_TRANSPORT", "unix")
            // An isolated data dir per run: never the operator's own.
            .env(
                "CAPSULES_DATA_DIR",
                std::env::temp_dir().join(format!("capsules-interop-data-{}", nonce())),
            )
            .env("CAPSULES_BLOCKED_MODELS", "blocked-test-model");
        for (key, value) in extra_env {
            cmd.env(key, value);
        }
        for key in unset {
            cmd.env_remove(key);
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

    async fn initialize(&mut self) -> proto::InitializeResponse {
        let request_id = self
            .send(Payload::InitializeRequest(proto::InitializeRequest {
                host_protocol_version: PROTOCOL_VERSION,
                host_capabilities: Vec::new(),
                host_version: "interop-test".to_string(),
                host_info_json: "{}".to_string(),
                mesh_visibility: proto::MeshVisibility::Private as i32,
            }))
            .await;
        let envelope = self.recv().await;
        assert_eq!(envelope.request_id, request_id);
        match envelope.payload {
            Some(Payload::InitializeResponse(response)) => response,
            other => panic!("expected InitializeResponse, got {other:?}"),
        }
    }

    async fn health(&mut self) -> proto::HealthResponse {
        let request_id = self
            .send(Payload::HealthRequest(proto::HealthRequest {}))
            .await;
        let envelope = self.recv().await;
        assert_eq!(envelope.request_id, request_id);
        match envelope.payload {
            Some(Payload::HealthResponse(response)) => response,
            other => panic!("expected HealthResponse, got {other:?}"),
        }
    }

    async fn shutdown(mut self) {
        let request_id = self
            .send(Payload::ShutdownRequest(proto::ShutdownRequest {
                reason: "interop test complete".to_string(),
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

fn inference_endpoint(manifest: &proto::PluginManifest) -> &proto::EndpointManifest {
    manifest
        .endpoints
        .iter()
        .find(|e| e.kind == proto::EndpointKind::Inference as i32)
        .expect("manifest declares an inference endpoint")
}

/// (A) Real handshake: Initialize declares a real HttpBodyMode-less,
/// exact-name-scoped inference-provider endpoint (`managed_by_plugin=true`,
/// `openai_compatible` protocol) — mesh-llm's actual mechanism for a plugin to
/// be dispatched OpenAI-compatible requests, per
/// `mesh-llm-host-runtime::network::openai::ingress::try_route_plugin_model`.
#[tokio::test]
async fn initialize_declares_real_inference_provider_endpoint() {
    let mut harness = Harness::spawn(&[]).await;
    let response = harness.initialize().await;

    assert_eq!(response.plugin_id, "capsules");
    let manifest = response.manifest.expect("plugin declares a manifest");
    let endpoint = inference_endpoint(&manifest);
    assert!(
        endpoint.managed_by_plugin,
        "must register as a provider, not just an observer"
    );
    assert_eq!(endpoint.protocol.as_deref(), Some("openai_compatible"));
    assert!(
        endpoint.address.is_some(),
        "endpoint must declare a real HTTP address"
    );

    harness.shutdown().await;
}

/// (B) Real HTTP probe: the host discovers served models by GETting
/// `/v1/models` on the declared endpoint (mirrors
/// `mesh-llm-host-runtime::plugin::health::probe_endpoint`). Only the
/// declared blocked model is advertised — this IS how "abstain" is realized
/// in the real system: a model this plugin does not advertise is never routed
/// to it at all, so the host's normal routing continues untouched.
#[tokio::test]
async fn v1_models_advertises_only_the_blocked_model() {
    let mut harness = Harness::spawn(&[]).await;
    let response = harness.initialize().await;
    let manifest = response.manifest.unwrap();
    let address = inference_endpoint(&manifest).address.clone().unwrap();

    let body: serde_json::Value = reqwest::get(format!("{address}/v1/models"))
        .await
        .expect("GET /v1/models")
        .json()
        .await
        .expect("valid JSON body");
    let ids: Vec<&str> = body["data"]
        .as_array()
        .expect("data array")
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, vec!["blocked-test-model"]);

    harness.shutdown().await;
}

/// A node whose operator set nothing is an evidence plugin only: no
/// OpenAI-compatible provider endpoint and no admission capability, so it
/// never appears in the host's model list or provider routing. The same for
/// an empty value, `none` and `off`. The admission test endpoint exists only
/// when blocked models are named.
#[tokio::test]
async fn with_no_blocked_models_the_plugin_registers_no_provider_and_no_admission_capability() {
    const KEY: &str = "CAPSULES_BLOCKED_MODELS";
    for value in [None, Some(""), Some("none"), Some("off")] {
        let mut harness = match value {
            None => Harness::spawn_without(&[], &[KEY]).await,
            Some(value) => Harness::spawn(&[(KEY, value)]).await,
        };
        let manifest = harness
            .initialize()
            .await
            .manifest
            .expect("plugin declares a manifest");
        assert!(
            !manifest
                .endpoints
                .iter()
                .any(|e| e.kind == proto::EndpointKind::Inference as i32),
            "{value:?}: no inference endpoint, got {:?}",
            manifest.endpoints
        );
        assert!(
            !manifest
                .capabilities
                .iter()
                .any(|c| c == "admission_policy.v1"),
            "{value:?}: no admission capability, got {:?}",
            manifest.capabilities
        );
        harness.shutdown().await;
    }
}

/// One plugin process per data directory: a second process on the same
/// directory refuses to start with one line naming the directory and the
/// holder's pid, before it touches the ledger. Once the holder exits, cleanly
/// or killed, the next process starts on the same directory.
#[tokio::test]
async fn a_second_process_on_the_same_data_dir_refuses_and_the_dir_frees_on_exit() {
    let dir = std::env::temp_dir().join(format!("capsules-lock-{}", nonce()));
    let dir_env = dir.to_str().expect("utf-8 temp dir").to_string();
    let first = Harness::spawn(&[("CAPSULES_DATA_DIR", &dir_env)]).await;
    let holder = first.child.id().expect("first plugin pid");

    // The second never reaches the host: it must refuse before connecting.
    let second = timeout(
        TEST_TIMEOUT,
        Command::new(PLUGIN_BIN)
            .env("MESH_LLM_PLUGIN_ENDPOINT", dir.join("no-host.sock"))
            .env("MESH_LLM_PLUGIN_TRANSPORT", "unix")
            .env("CAPSULES_DATA_DIR", &dir_env)
            .env("CAPSULES_BLOCKED_MODELS", "blocked-test-model")
            .output(),
    )
    .await
    .expect("second process exited before timeout")
    .expect("run the second process");
    assert!(!second.status.success(), "the second process must refuse");
    let stderr = String::from_utf8_lossy(&second.stderr);
    assert!(
        stderr.contains("is in use by another capsules plugin process"),
        "{stderr}"
    );
    assert!(stderr.contains(&dir_env), "{stderr}");
    assert!(stderr.contains(&format!("pid {holder}")), "{stderr}");

    // Clean exit frees the directory.
    first.shutdown().await;
    let mut killed = Harness::spawn(&[("CAPSULES_DATA_DIR", &dir_env)]).await;
    assert!(killed.initialize().await.manifest.is_some());

    // A crash frees it too: the lock file stays behind, the lock does not.
    killed.child.kill().await.expect("kill the plugin");
    let _ = killed.child.wait().await;
    assert!(dir.join("capsules.lock").exists());
    let mut after_crash = Harness::spawn(&[("CAPSULES_DATA_DIR", &dir_env)]).await;
    assert!(after_crash.initialize().await.manifest.is_some());
    after_crash.shutdown().await;
}

/// (C) Deny: a real HTTP POST for the blocked model is denied with a
/// deny-reason, over the endpoint the host would actually dispatch to.
#[tokio::test]
async fn denies_request_for_blocked_model() {
    let mut harness = Harness::spawn(&[]).await;
    let response = harness.initialize().await;
    let manifest = response.manifest.unwrap();
    let address = inference_endpoint(&manifest).address.clone().unwrap();

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{address}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "blocked-test-model", "messages": []}))
        .send()
        .await
        .expect("POST /v1/chat/completions");
    assert_eq!(resp.status(), reqwest::StatusCode::FORBIDDEN);
    let body: serde_json::Value = resp.json().await.expect("valid JSON body");
    assert_eq!(body["error"]["code"], "blocked_model_prefix");

    harness.shutdown().await;
}

/// (D) Fail-safe: a request the plugin cannot parse is denied, not silently
/// allowed and not a server crash — mirrors the spike's
/// `body_parse_failure` -> deny behavior, now over the real HTTP surface.
#[tokio::test]
async fn denies_malformed_body_fail_safe() {
    let mut harness = Harness::spawn(&[]).await;
    let response = harness.initialize().await;
    let manifest = response.manifest.unwrap();
    let address = inference_endpoint(&manifest).address.clone().unwrap();

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{address}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body("{not valid json")
        .send()
        .await
        .expect("POST /v1/chat/completions");
    assert_eq!(resp.status(), reqwest::StatusCode::FORBIDDEN);

    harness.shutdown().await;
}

/// (E) Full real lifecycle: Initialize -> Health -> Shutdown, over the actual
/// wire protocol, with a clean process exit.
#[tokio::test]
async fn health_and_shutdown_over_real_wire_protocol() {
    let mut harness = Harness::spawn(&[]).await;
    harness.initialize().await;

    let health = harness.health().await;
    assert_eq!(health.status, proto::health_response::Status::Ok as i32);

    harness.shutdown().await;
}
