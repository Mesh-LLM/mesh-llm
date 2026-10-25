//! Explicit live conformance lane: build/package exemplar before running ignored tests.
use super::{affinity, election, handle_api_proxy_connection, mesh};
use crate::plugin::{ExternalPluginSpec, PluginHostMode, PluginManager, ResolvedPlugins};
use mesh_llm_config::{OpenAiExchangeFailurePolicy, OpenAiExchangeGrant};
use mesh_llm_plugin_manager::{PluginInstallOptions, PluginTarget};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, sync::Arc, time::Duration};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    sync::Mutex,
};

const MODEL: &str = "allowed-model";
const NAME: &str = "openai-exchange-observer";
pub(super) struct LiveRecipient {
    pub(super) name: &'static str,
    pub(super) admission: bool,
    pub(super) deny_model: &'static str,
    pub(super) fault: &'static str,
    pub(super) max_queue_bytes: u64,
}
const BUFFERED: &[u8] = br#"{"id":"fixed","object":"chat.completion","created":1,"model":"allowed-model","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}"#;
const SSE: &[u8] = b"data: {\"id\":\"fixed\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"allowed-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"fixed\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"allowed-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";

pub(super) struct LiveHost {
    pub(super) root: tempfile::TempDir,
    pub(super) installed_metadata: mesh_llm_plugin_manager::InstalledPluginMetadata,
    pub(super) manager: PluginManager,
    pub(super) node: mesh::Node,
    pub(super) targets: election::ModelTargets,
    pub(super) backend: tokio::task::JoinHandle<()>,
    pub(super) requests: Arc<Mutex<Vec<Vec<u8>>>>,
}

impl LiveHost {
    pub(super) async fn start(admission: bool, bodies: bool) -> Self {
        Box::pin(Self::start_with_identity(admission, bodies, false, false)).await
    }

    pub(super) async fn start_with_identity(
        admission: bool,
        bodies: bool,
        read_identity: bool,
        delegate_signing: bool,
    ) -> Self {
        Box::pin(Self::start_with_recipients(
            admission,
            bodies,
            read_identity,
            delegate_signing,
            &[],
        ))
        .await
    }
    pub(super) async fn start_with_recipients(
        admission: bool,
        bodies: bool,
        read_identity: bool,
        delegate_signing: bool,
        recipients: &[LiveRecipient],
    ) -> Self {
        let root = tempfile::tempdir().unwrap();
        let archive = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../dist/openai-exchange-observer.tar.gz");
        assert!(
            archive.exists(),
            "run just package-openai-exchange-exemplar first"
        );
        let options = PluginInstallOptions {
            store_root: root.path().join("store"),
            install_root: root.path().join("installed"),
            catalog_url: "unused".into(),
            target: PluginTarget::current().unwrap(),
            bundled_plugins_dir: None,
        };
        let installed = mesh_llm_plugin_manager::install::install_plugin_archive(
            NAME,
            "1.0.0",
            &archive,
            &options,
            &mut |_event| {},
        )
        .unwrap()
        .metadata;
        assert!(installed.executable_path().exists());
        let spec = fixture_spec(
            &root,
            &installed,
            admission,
            bodies,
            read_identity,
            delegate_signing,
        );
        let externals = recipient_specs(spec, &installed, &options, &root, bodies, recipients);
        let specs = ResolvedPlugins {
            externals,
            inactive: Vec::new(),
        };
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        let manager = Box::pin(PluginManager::start(
            &specs,
            PluginHostMode {
                mesh_visibility: mesh_llm_plugin::MeshVisibility::Private,
            },
            tx,
        ))
        .await
        .unwrap();
        assert_eq!(manager.list().await[0].status, "running");
        let mut node = Box::pin(mesh::Node::new_for_tests(mesh::NodeRole::Worker))
            .await
            .unwrap();
        if read_identity || delegate_signing {
            let owner = mesh_llm_identity::OwnerKeypair::generate();
            let now = chrono::Utc::now().timestamp_millis().unsigned_abs();
            let id = node.id();
            let certificate = mesh_llm_identity::sign_node_ownership(
                &owner,
                id.as_bytes(),
                now + 3_600_000,
                None,
                None,
            )
            .unwrap();
            node.owner_keypair = Some(owner);
            *node.owner_attestation.lock().await = Some(certificate);
        }
        node.set_plugin_manager(manager.clone()).await;
        node.start_plugin_channel_forwarder(rx);

        node.set_served_model_descriptors(
            [
                MODEL,
                "blocked-model",
                "backend-error-model",
                "timeout-model",
                "slow-stream-model",
            ]
            .into_iter()
            .map(|name| mesh::ServedModelDescriptor {
                identity: mesh::ServedModelIdentity {
                    model_name: name.into(),
                    ..Default::default()
                },
                ..Default::default()
            })
            .collect(),
        )
        .await;
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = requests.clone();
        let backend = spawn_backend(listener, recorded);
        let mut targets = election::ModelTargets::default();
        for name in [
            MODEL,
            "blocked-model",
            "backend-error-model",
            "timeout-model",
            "slow-stream-model",
        ] {
            targets
                .targets
                .insert(name.into(), vec![election::InferenceTarget::Local(port)]);
        }
        Self {
            root,
            installed_metadata: installed,
            manager,
            node,
            targets,
            backend,
            requests,
        }
    }

    pub(super) async fn connection(&self) -> (tokio::net::TcpStream, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let node = self.node.clone();
        let targets = self.targets.clone();
        let handler = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            crate::network::openai::response::TEST_RESPONSE_FIRST_BYTE_TIMEOUT
                .scope(
                    Duration::from_millis(150),
                    Box::pin(handle_api_proxy_connection(
                        node,
                        stream.into(),
                        targets,
                        affinity::AffinityRouter::new(),
                        crate::runtime::IngressType::LocalOpenAi,
                        None,
                    )),
                )
                .await;
        });
        (
            tokio::net::TcpStream::connect(address).await.unwrap(),
            handler,
        )
    }
    pub(super) async fn request(&self, path: &str, body: &[u8]) -> Vec<u8> {
        let (mut client, handler) = self.connection().await;
        send_request(&mut client, path, body).await;
        let mut response = Vec::new();
        tokio::time::timeout(Duration::from_secs(10), client.read_to_end(&mut response))
            .await
            .unwrap()
            .unwrap();
        handler.await.unwrap();
        response
    }
    async fn unobserved_baseline(&self, path: &str, body: &[u8]) -> Vec<u8> {
        let grants = self
            .manager
            .list()
            .await
            .into_iter()
            .filter_map(|plugin| {
                self.manager
                    .effective_exchange_grant(&plugin.name)
                    .map(|grant| {
                        serde_json::from_value(
                            json!({"name":plugin.name,"openai_exchange_grant":grant}),
                        )
                        .unwrap()
                    })
            })
            .collect();
        let count = self.events().len();
        self.manager
            .apply_exchange_grants(&mesh_llm_config::MeshConfig::default())
            .await;
        let response = self.request(path, body).await;
        assert_eq!(
            self.events().len(),
            count,
            "baseline unexpectedly invoked observer"
        );
        self.manager
            .apply_exchange_grants(&mesh_llm_config::MeshConfig {
                plugins: grants,
                ..Default::default()
            })
            .await;
        response
    }
    pub(super) fn events(&self) -> Vec<Value> {
        self.recipient_events(NAME)
    }
    pub(super) fn recipient_events(&self, name: &str) -> Vec<Value> {
        let file = if name == NAME {
            self.root.path().join("events.jsonl")
        } else {
            self.root.path().join(format!("{name}.jsonl"))
        };
        std::fs::read_to_string(file)
            .unwrap_or_default()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }
    pub(super) async fn stop(self) {
        self.node.take_plugin_manager().await;
        self.manager.shutdown().await;
        self.backend.abort();
        self.node.endpoint.close().await;
    }
}
fn install_recipient_archive(
    original: &mesh_llm_plugin_manager::InstalledPluginMetadata,
    name: &str,
    options: &PluginInstallOptions,
    root: &std::path::Path,
) -> mesh_llm_plugin_manager::InstalledPluginMetadata {
    let staging = root.join(format!("package-{name}"));
    let package = staging.join(name);
    std::fs::create_dir_all(&package).unwrap();
    std::fs::copy(
        original.executable_path(),
        package.join(format!("{name}{}", std::env::consts::EXE_SUFFIX)),
    )
    .unwrap();
    std::fs::copy(
        original.install_path.join("plugin-manifest.json"),
        package.join("plugin-manifest.json"),
    )
    .unwrap();
    std::fs::write(
        package.join("plugin.toml"),
        format!("name = \"{name}\"\nversion = \"1.0.0\"\n"),
    )
    .unwrap();
    let archive = root.join(format!("{name}.tar.gz"));
    assert!(
        std::process::Command::new("tar")
            .arg("-czf")
            .arg(&archive)
            .arg("-C")
            .arg(&staging)
            .arg(name)
            .status()
            .unwrap()
            .success()
    );
    mesh_llm_plugin_manager::install::install_plugin_archive(
        name,
        "1.0.0",
        &archive,
        options,
        &mut |_event| {},
    )
    .unwrap()
    .metadata
}
async fn send_request(client: &mut tokio::net::TcpStream, path: &str, body: &[u8]) {
    client.write_all(format!("POST {path} HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer never-deliver\r\nCookie: session=never-deliver\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",body.len()).as_bytes()).await.unwrap();
    client.write_all(body).await.unwrap();
}
pub(super) async fn read_entity_request(stream: &mut tokio::net::TcpStream) -> Vec<u8> {
    let mut wire = Vec::new();
    let header_end = loop {
        let mut chunk = [0u8; 8192];
        let len = stream.read(&mut chunk).await.unwrap();
        assert!(len > 0);
        wire.extend_from_slice(&chunk[..len]);
        if let Some(pos) = wire.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
    };
    let length: usize = String::from_utf8_lossy(&wire[..header_end])
        .lines()
        .find_map(|line| {
            line.to_ascii_lowercase()
                .strip_prefix("content-length:")
                .map(|v| v.trim().parse().unwrap())
        })
        .unwrap();
    while wire.len() < header_end + length {
        let mut chunk = [0u8; 8192];
        let len = stream.read(&mut chunk).await.unwrap();
        assert!(len > 0);
        wire.extend_from_slice(&chunk[..len]);
    }
    wire[header_end..header_end + length].to_vec()
}
pub(super) fn response_entity(wire: &[u8]) -> Vec<u8> {
    let start = wire.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
    if !String::from_utf8_lossy(&wire[..start])
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked")
    {
        return wire[start..].to_vec();
    }
    let mut body = Vec::new();
    let mut cursor = start;
    loop {
        let end = wire[cursor..]
            .windows(2)
            .position(|w| w == b"\r\n")
            .unwrap()
            + cursor;
        let len = usize::from_str_radix(
            std::str::from_utf8(&wire[cursor..end])
                .unwrap()
                .split(';')
                .next()
                .unwrap(),
            16,
        )
        .unwrap();
        if len == 0 {
            return body;
        }
        cursor = end + 2;
        body.extend_from_slice(&wire[cursor..cursor + len]);
        cursor += len + 2;
    }
}
fn assert_terminal(events: &[Value], outcome: &str, response: &[u8]) {
    let terminals: Vec<_> = events
        .iter()
        .filter(|e| e["phase"] == "exchange_finished")
        .collect();
    assert_eq!(terminals.len(), 1);
    assert_eq!(terminals[0]["execution_outcome"], outcome);
    assert_eq!(
        terminals[0]["response_wire_commitment"]["sha256"],
        hex::encode(Sha256::digest(response_entity(response)))
    );
}
#[tokio::test]
#[ignore = "requires just package-openai-exchange-exemplar"]
async fn installed_lifecycle_denial_never_dispatches_and_body_grants_are_honest() {
    let host = LiveHost::start(true, false).await;
    let body =
        br#"{ "model": "blocked-model", "messages": [{"role":"user","content":"secret prompt"}] }"#;
    let response = host.request("/v1/chat/completions", body).await;
    assert!(response.starts_with(b"HTTP/1.1 403"));
    assert!(host.requests.lock().await.is_empty());
    let events = host.events();
    assert_eq!(events.len(), 2);
    assert_terminal(&events, "policy_denied", &response);
    assert!(
        events
            .iter()
            .all(|e| e.get("body").is_none() && e.get("body_hex").is_none())
    );
    assert!(events.iter().all(
        |e| e["headers"].get("authorization").is_none() && e["headers"].get("cookie").is_none()
    ));
    assert_eq!(
        events[0]["request_wire_digest"]["sha256"],
        hex::encode(Sha256::digest(body))
    );
    host.stop().await;
}
#[tokio::test]
#[ignore = "requires just package-openai-exchange-exemplar"]
async fn installed_lifecycle_allow_preserves_stream_and_buffered_entity_bytes() {
    let host = LiveHost::start(false, true).await;
    for (path, field) in [
        ("/v1/chat/completions", "messages"),
        ("/v1/completions", "prompt"),
        ("/v1/responses", "input"),
    ] {
        for streaming in [false, true] {
            let mut input = json!({"model":MODEL,"stream":streaming,"messages":[{"role":"user","content":"hello"}]});
            if field != "messages" {
                input.as_object_mut().unwrap().remove("messages");
                input[field] = json!("hello");
            }
            let body = serde_json::to_vec(&input).unwrap();
            let baseline = if path == "/v1/chat/completions" {
                Some(host.unobserved_baseline(path, &body).await)
            } else {
                None
            };
            let response = host.request(path, &body).await;
            assert!(
                response.starts_with(b"HTTP/1.1 200"),
                "{}",
                String::from_utf8_lossy(&response)
            );
            if let Some(baseline) = baseline {
                assert!(baseline.starts_with(b"HTTP/1.1 200"));
                assert_eq!(
                    response_entity(&response),
                    response_entity(&baseline),
                    "observer changed the core's final entity bytes"
                );
            }
            let events = host.events();
            let received = events
                .iter()
                .rev()
                .find(|e| e["phase"] == "request_received")
                .unwrap();
            let selected = events
                .iter()
                .rev()
                .find(|e| e["phase"] == "backend_selected")
                .unwrap();
            let terminal = events.last().unwrap();
            assert!(received.get("body_hex").is_none());
            assert_eq!(received["body_stream_kind"], "openai_exchange_original");
            assert_receiver_receipt(&host, received, "openai_exchange_original", &body);
            assert_eq!(
                received["request_wire_digest"]["sha256"],
                hex::encode(Sha256::digest(&body))
            );
            assert_eq!(
                selected["effective_request_wire_digest"]["sha256"],
                hex::encode(Sha256::digest(host.requests.lock().await.last().unwrap()))
            );
            assert_receiver_receipt(
                &host,
                selected,
                "openai_exchange_effective",
                host.requests.lock().await.last().unwrap(),
            );
            assert_eq!(terminal["phase"], "exchange_finished");
            assert_eq!(
                terminal["response_wire_commitment"]["sha256"],
                hex::encode(Sha256::digest(response_entity(&response)))
            );
            assert_eq!(
                terminal["response_wire_commitment"]["side_stream_complete"],
                true
            );
        }
    }
    assert_eq!(host.requests.lock().await.len(), 8);
    let count = host.events().len();
    host.manager
        .apply_exchange_grants(&mesh_llm_config::MeshConfig::default())
        .await;
    let response = host
        .request(
            "/v1/chat/completions",
            br#"{"model":"allowed-model","messages":[{"role":"user","content":"hello"}]}"#,
        )
        .await;
    assert!(response.starts_with(b"HTTP/1.1 200"));
    assert_eq!(host.events().len(), count);
    host.stop().await;
}
#[tokio::test]
#[ignore = "requires just package-openai-exchange-exemplar"]
async fn installed_lifecycle_admission_parses_large_body_from_authenticated_side_stream() {
    let host = LiveHost::start(true, true).await;
    let content = "safe ".repeat(14_000) + "forbidden";
    let body =
        serde_json::to_vec(&json!({"model":MODEL,"messages":[{"role":"user","content":content}]}))
            .unwrap();
    assert!(body.len() > 65536);
    let response = host.request("/v1/chat/completions", &body).await;
    assert!(response.starts_with(b"HTTP/1.1 403"));
    assert!(host.requests.lock().await.is_empty());
    let events = host.events();
    assert!(events[0].get("body").is_none());
    assert_eq!(
        events[0]["request_wire_digest"]["sha256"],
        hex::encode(Sha256::digest(&body))
    );
    host.stop().await;
}
#[tokio::test]
#[ignore = "requires just package-openai-exchange-exemplar"]
async fn installed_lifecycle_accepted_faults_have_one_terminal() {
    for (body,status,outcome,dispatched) in [
        (b"{invalid json".as_slice(),400,"request_invalid",true),
        (br#"{"model":"allowed-model","messages":"invalid"}"#.as_slice(),400,"request_invalid",true),
        (br#"{"model":"backend-error-model","fault":"backend_error","messages":[{"role":"user","content":"hi"}]}"#.as_slice(),500,"backend_error",true),
        (br#"{"model":"timeout-model","fault":"timeout","messages":[{"role":"user","content":"hi"}]}"#.as_slice(),504,"timed_out",true),
    ] {
        let host=LiveHost::start(false,true).await;let response=host.request("/v1/chat/completions",body).await;assert!(response.starts_with(format!("HTTP/1.1 {status}").as_bytes()),"{}",String::from_utf8_lossy(&response));
        assert_eq!(!host.requests.lock().await.is_empty(),dispatched);
        if serde_json::from_slice::<Value>(body).is_err() {assert_eq!(host.requests.lock().await.last().unwrap(),body);}
        let events=host.events();assert_eq!(events.first().unwrap()["phase"],"request_received");assert_eq!(events.first().unwrap()["request_wire_digest"]["sha256"],hex::encode(Sha256::digest(body)));assert_terminal(&events,outcome,&response);host.stop().await;
    }
}
#[tokio::test]
#[ignore = "requires just package-openai-exchange-exemplar"]
async fn installed_lifecycle_client_disconnect_finishes_once() {
    let host = LiveHost::start(false, true).await;
    let (mut client, handler) = host.connection().await;
    send_request(&mut client,"/v1/chat/completions",br#"{"model":"slow-stream-model","fault":"slow_stream","stream":true,"messages":[{"role":"user","content":"hi"}]}"#).await;
    let mut bytes = [0u8; 4096];
    let len = tokio::time::timeout(Duration::from_secs(5), client.read(&mut bytes))
        .await
        .unwrap()
        .unwrap();
    assert!(len > 0);
    drop(client);
    tokio::time::timeout(Duration::from_secs(10), handler)
        .await
        .unwrap()
        .unwrap();
    for _ in 0..100 {
        if host
            .events()
            .iter()
            .any(|e| e["phase"] == "exchange_finished")
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let events = host.events();
    let terminal: Vec<_> = events
        .iter()
        .filter(|e| e["phase"] == "exchange_finished")
        .collect();
    assert_eq!(terminal.len(), 1);
    assert_eq!(terminal[0]["execution_outcome"], "client_cancelled");
    host.stop().await;
}

#[tokio::test]
#[ignore = "requires just package-openai-exchange-exemplar"]
async fn installed_two_admission_plugins_deny_wins_in_both_orders() {
    for other in ["a-admission-observer", "z-admission-observer"] {
        for denier in [NAME, other] {
            let roles = [
                LiveRecipient {
                    name: NAME,
                    admission: true,
                    deny_model: if denier == NAME { MODEL } else { "" },
                    fault: "",
                    max_queue_bytes: 4 * 1024 * 1024,
                },
                LiveRecipient {
                    name: other,
                    admission: true,
                    deny_model: if denier == other { MODEL } else { "" },
                    fault: "",
                    max_queue_bytes: 4 * 1024 * 1024,
                },
            ];
            let host = LiveHost::start_with_recipients(true, true, false, false, &roles).await;
            let response = host
                .request(
                    "/v1/chat/completions",
                    br#"{"model":"allowed-model","messages":[{"role":"user","content":"hello"}]}"#,
                )
                .await;
            assert!(response.starts_with(b"HTTP/1.1 403"));
            assert!(host.requests.lock().await.is_empty());
            for name in [NAME, other] {
                assert_terminal(&host.recipient_events(name), "policy_denied", &response);
            }
            host.stop().await;
        }
    }
}
fn fixture_spec(
    root: &tempfile::TempDir,
    installed: &mesh_llm_plugin_manager::InstalledPluginMetadata,
    admission: bool,
    bodies: bool,
    read_identity: bool,
    delegate_signing: bool,
) -> ExternalPluginSpec {
    let grant = OpenAiExchangeGrant {
        endpoints: vec![
            "chat_completions".into(),
            "completions".into(),
            "responses".into(),
        ],
        phases: vec![
            "request_received".into(),
            "backend_selected".into(),
            "exchange_finished".into(),
        ],
        read_identity_bundle: read_identity,
        delegate_signing_key: delegate_signing,
        signing_scopes: if delegate_signing {
            vec!["mesh.openai.exchange.evidence.sign.v1".into()]
        } else {
            Vec::new()
        },
        max_delegation_ttl_secs: if delegate_signing { 60 } else { 0 },
        request_body: bodies,
        effective_request_body: bodies,
        response_body: bodies,
        metadata: true,
        admission,
        deadline_ms: 2000,
        max_body_bytes: 1024 * 1024,
        max_queue_bytes: 4 * 1024 * 1024,
        max_in_flight: 8,
        failure_policy: OpenAiExchangeFailurePolicy::Required,
        ..Default::default()
    };
    // Installed fixtures explicitly expose setup diagnostics; the packaged
    // exemplar's ordinary manifest has no public operation.
    let mut args = vec!["--identity-probe".into()];
    if admission {
        args.push("--admission".into());
    }
    if bodies {
        args.push("--body".into());
    }
    if read_identity {
        args.push("--identity".into());
    }
    if delegate_signing {
        args.push("--delegate".into());
    }
    ExternalPluginSpec {
        name: NAME.into(),
        command: installed.executable_path().display().to_string(),
        args,
        url: None,
        startup: Default::default(),
        web_ui_enabled: None,
        web_ui_primary_tab: None,
        env: BTreeMap::from([
            (
                "MESH_LLM_EXEMPLAR_RECEIPT_LOG".into(),
                root.path().join("receipts.jsonl").display().to_string(),
            ),
            (
                "MESH_LLM_EXEMPLAR_STREAM_LOG".into(),
                root.path().join("streams.jsonl").display().to_string(),
            ),
            (
                "MESH_LLM_EXEMPLAR_DENY_MODEL".into(),
                "blocked-model".into(),
            ),
            ("MESH_LLM_EXEMPLAR_DENY_TEXT".into(), "forbidden".into()),
            (
                "MESH_LLM_EXEMPLAR_EVENT_LOG".into(),
                root.path().join("events.jsonl").display().to_string(),
            ),
        ]),
        openai_exchange_grant: Some(Box::new(grant)),
        installed_metadata: Some(installed.clone()),
    }
}

fn recipient_specs(
    spec: ExternalPluginSpec,
    installed: &mesh_llm_plugin_manager::InstalledPluginMetadata,
    options: &PluginInstallOptions,
    root: &tempfile::TempDir,
    bodies: bool,
    recipients: &[LiveRecipient],
) -> Vec<ExternalPluginSpec> {
    let mut externals = vec![spec.clone()];
    for role in recipients {
        let mut recipient = spec.clone();
        recipient.name = role.name.into();
        recipient.args = vec!["--plugin-id".into(), role.name.into(), "--optional".into()];
        if bodies {
            recipient.args.push("--body".into());
        }
        if role.admission {
            recipient.args.push("--admission".into());
        }
        recipient.env.insert(
            "MESH_LLM_EXEMPLAR_DENY_MODEL".into(),
            role.deny_model.into(),
        );
        recipient.env.insert(
            "MESH_LLM_EXEMPLAR_STREAM_LOG".into(),
            root.path()
                .join(format!("{}.streams.jsonl", role.name))
                .display()
                .to_string(),
        );
        recipient.env.insert(
            "MESH_LLM_EXEMPLAR_RECEIPT_LOG".into(),
            root.path()
                .join(format!("{}.receipts.jsonl", role.name))
                .display()
                .to_string(),
        );
        recipient
            .env
            .insert("MESH_LLM_EXEMPLAR_FAULT".into(), role.fault.into());
        recipient.env.insert(
            "MESH_LLM_EXEMPLAR_EVENT_LOG".into(),
            root.path()
                .join(if role.name == NAME {
                    "events.jsonl".into()
                } else {
                    format!("{}.jsonl", role.name)
                })
                .display()
                .to_string(),
        );
        let grant = recipient.openai_exchange_grant.as_mut().unwrap();
        grant.admission = role.admission;
        grant.failure_policy = OpenAiExchangeFailurePolicy::BestEffort;
        grant.max_queue_bytes = role.max_queue_bytes;
        if role.name == NAME {
            externals[0] = recipient;
        } else {
            recipient.installed_metadata = Some(install_recipient_archive(
                installed,
                role.name,
                options,
                root.path(),
            ));
            recipient.command = recipient
                .installed_metadata
                .as_ref()
                .unwrap()
                .executable_path()
                .display()
                .to_string();
            externals.push(recipient);
        }
    }

    externals
}

fn spawn_backend(
    listener: TcpListener,
    recorded: Arc<Mutex<Vec<Vec<u8>>>>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let body = read_entity_request(&mut stream).await;
            let parsed = serde_json::from_slice::<Value>(&body);
            let malformed = parsed.is_err();
            let request = parsed.unwrap_or(Value::Null);
            let invalid =
                malformed || (request.get("messages").is_some() && !request["messages"].is_array());
            let streaming = request["stream"] == true;
            let fault = request["fault"].as_str().unwrap_or_else(|| {
                match request["model"].as_str().unwrap_or("") {
                    "backend-error-model" => "backend_error",
                    "timeout-model" => "timeout",
                    "slow-stream-model" => "slow_stream",
                    _ => "none",
                }
            });
            recorded.lock().await.push(body.clone());
            if fault == "timeout" {
                tokio::time::sleep(Duration::from_secs(60)).await;
                continue;
            }
            let bytes: &[u8] = if invalid {
                br#"{"error":{"message":"invalid request body","type":"invalid_request_error"}}"#
            } else if fault == "backend_error" {
                br#"{"error":{"message":"fixture backend failure","type":"server_error"}}"#
            } else if streaming {
                SSE
            } else {
                BUFFERED
            };
            let content_type = if streaming {
                "text/event-stream"
            } else {
                "application/json"
            };
            let status = if invalid {
                "400 Bad Request"
            } else if fault == "backend_error" {
                "500 Internal Server Error"
            } else {
                "200 OK"
            };
            if stream.write_all(format!("HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", bytes.len()).as_bytes()).await.is_err() { continue; }
            for chunk in bytes.chunks(7) {
                if stream.write_all(chunk).await.is_err() {
                    break;
                }
                if fault == "slow_stream" {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
            let _ = stream.shutdown().await;
        }
    })
}

#[tokio::test]
#[ignore = "requires just package-openai-exchange-exemplar"]
async fn installed_required_manifest_live_reduction_fails_closed_before_delivery() {
    for remove_body in [true, false] {
        let host = LiveHost::start(false, true).await;
        let mut grant = host.manager.effective_exchange_grant(NAME).unwrap();
        if remove_body {
            grant.request_body = false;
            grant.effective_request_body = false;
            grant.response_body = false;
        } else {
            grant.metadata = false;
        }
        let config = mesh_llm_config::MeshConfig {
            plugins: vec![
                serde_json::from_value(json!({"name":NAME,"openai_exchange_grant":grant})).unwrap(),
            ],
            ..Default::default()
        };
        host.manager.apply_exchange_grants(&config).await;
        let response=host.request("/v1/chat/completions",br#"{"model":"allowed-model","messages":[{"role":"user","content":"never delivered"}]}"#).await;
        assert!(
            response.starts_with(b"HTTP/1.1 503"),
            "{}",
            String::from_utf8_lossy(&response)
        );
        assert!(host.requests.lock().await.is_empty());
        assert!(host.events().is_empty());
        assert!(
            !host.root.path().join("streams.jsonl").exists(),
            "revoked body was delivered over a side stream"
        );
        host.stop().await;
    }
}

fn assert_receiver_receipt(host: &LiveHost, event: &Value, kind: &str, bytes: &[u8]) {
    let receipts = std::fs::read_to_string(host.root.path().join("receipts.jsonl")).unwrap();
    let receipt = receipts
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .find(|receipt| {
            receipt["metadata"]["exchange_id"] == event["exchange_id"]
                && receipt["metadata"]["kind"] == kind
                && receipt["complete"] == true
        })
        .unwrap();
    assert_eq!(receipt["metadata"]["receipt_protocol"], "sha256-v1");
    assert_eq!(
        receipt["receipt"]["sha256"],
        hex::encode(Sha256::digest(bytes))
    );
    assert_eq!(receipt["receipt"]["byte_count"], bytes.len());
    if kind != "openai_exchange_response" {
        assert_eq!(
            receipt["parsed_body"],
            serde_json::from_slice::<Value>(bytes).unwrap()
        );
    }
}

#[path = "live_streaming_tests.rs"]
mod streaming;

#[path = "live_observer_failure_tests.rs"]
mod observer_failures;

#[path = "chunked_request_live_tests.rs"]
mod chunked_requests;
