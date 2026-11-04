//! The audits of a peer's receipts across two nodes joined over real QUIC:
//! a gateway whose local client asks for the toy model, and a host whose
//! backend serves it with receipts (`receipt_audit::two_node_seam`, which
//! says what the seam does not cover). Each test sends one turn the VRF
//! always draws, and waits for what the audit comes to.
use super::duplicate_alias_repro::duplicate_peer;
use super::*;
use crate::network::receipt_audit::two_node_seam::{
    self as seam, PROMPT, TOY_MODEL, ToyBackend, ToyServing,
};
use crate::network::request_trust::UNSATISFIED_CODE;
use std::sync::Arc;

const RECEIPTS: &str = "x-mesh-trust: receipts\r\n";
/// The longest a test waits for an audit: the challenge window, its grace
/// and the rows wait are 25 s, and a busy machine is slow.
const AUDIT_WAIT: Duration = Duration::from_secs(90);

fn request(path: &str, headers: &str, body: &serde_json::Value) -> String {
    let body = body.to_string();
    format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\n{headers}Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
}

/// A greedy chat completion of the toy prompt.
fn chat(headers: &str) -> String {
    let body = serde_json::json!({
        "model": TOY_MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0,
        "max_tokens": 8,
    });
    request("/v1/chat/completions", headers, &body)
}

/// The same turn, asked as an Anthropic message.
fn message() -> String {
    let body = serde_json::json!({
        "model": TOY_MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0,
        "max_tokens": 8,
    });
    request("/v1/messages", "anthropic-version: 2023-06-01\r\n", &body)
}

fn chat_descriptor() -> mesh::ServedModelDescriptor {
    mesh::ServedModelDescriptor {
        identity: mesh::ServedModelIdentity {
            model_name: TOY_MODEL.to_owned(),
            ..Default::default()
        },
        capabilities_known: true,
        metadata: Some(mesh::ServedModelMetadata {
            workload_class: Some(mesh::ModelWorkloadClass::CausalGeneration),
            ..Default::default()
        }),
        receipts: true,
        ..Default::default()
    }
}

/// A node with its own key and state directory, and `[receipts]` set to
/// `receipts`.
async fn start_node(
    key: iroh::SecretKey,
    state: &tempfile::TempDir,
    receipts: &str,
) -> (mesh::Node, mesh::TunnelChannels) {
    let relay_urls = Vec::new();
    let relay_auths = HashMap::new();
    let (node, channels) = mesh::Node::start_with_spec(mesh::NodeStartupSpec {
        role: mesh::NodeRole::Client,
        relay: mesh::RelayConfig {
            urls: &relay_urls,
            auths: &relay_auths,
            policy: mesh::RelayPolicy::Disabled,
        },
        quic_bind: mesh::QuicBindSelection {
            ip: Some(IpAddr::V4(Ipv4Addr::LOCALHOST)),
            port: Some(0),
        },
        max_vram_gb: Some(0.0),
        enumerate_host: false,
        owner_config: None,
        config_path: None,
        local_mesh_requirements: crate::MeshRequirements::unrestricted(),
        identity: Some(mesh::ScopedNodeIdentity {
            key,
            state_dir: state.path().join("node"),
        }),
    })
    .await
    .expect("start a test node");
    let config = state.path().join("config.toml");
    std::fs::write(&config, format!("[receipts]\n{receipts}\n")).expect("write config");
    *node.config_state.lock().await =
        crate::runtime::config_state::ConfigState::load(&config).expect("load config");
    (node, channels)
}

/// Reads one request, its head and the body its length gives.
async fn read_request(stream: &mut tokio::net::TcpStream) -> Vec<u8> {
    let mut request = Vec::new();
    let mut chunk = [0u8; 8192];
    loop {
        if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&request[..end]).to_ascii_lowercase();
            let length = head
                .lines()
                .find_map(|line| line.strip_prefix("content-length:"))
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(0);
            if request.len() >= end + 4 + length {
                return request;
            }
        }
        match stream.read(&mut chunk).await {
            Ok(0) | Err(_) => return request,
            Ok(read) => request.extend_from_slice(&chunk[..read]),
        }
    }
}

/// The host's backend: every connection is one request `backend` answers.
async fn spawn_backend(backend: Arc<ToyBackend>) -> (u16, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind the toy backend");
    let port = listener.local_addr().expect("backend address").port();
    let task = tokio::spawn(async move {
        while let Ok((mut stream, _)) = listener.accept().await {
            let backend = backend.clone();
            tokio::spawn(async move {
                let request = read_request(&mut stream).await;
                let _ = stream.write_all(&backend.answer(&request)).await;
                let _ = stream.shutdown().await;
            });
        }
    });
    (port, task)
}

/// A gateway that audits every turn, and a host that serves the toy model
/// with receipts under the toy turn's key, joined over QUIC.
struct Mesh {
    gateway: mesh::Node,
    host: mesh::Node,
    gateway_state: tempfile::TempDir,
    _host_state: tempfile::TempDir,
    tasks: Vec<tokio::task::JoinHandle<()>>,
    _tunnels: Vec<TunnelManager>,
    _targets: Vec<watch::Sender<election::ModelTargets>>,
}

impl Mesh {
    async fn start(serving: ToyServing) -> Self {
        let gateway_state = tempfile::tempdir().expect("gateway state");
        let host_state = tempfile::tempdir().expect("host state");
        let (gateway, gateway_channels) = start_node(
            iroh::SecretKey::generate(),
            &gateway_state,
            "audit = true\naudit_one_in = 1",
        )
        .await;
        // the toy turn is signed by this key
        let (host, host_channels) =
            start_node(iroh::SecretKey::from_bytes(&[5; 32]), &host_state, "").await;
        seam::audit_with_the_toy_model(&gateway);
        let (host_port, backend) = spawn_backend(Arc::new(ToyBackend::new(serving))).await;

        host.set_hosted_models(vec![TOY_MODEL.to_owned()]).await;
        host.set_served_model_descriptors(vec![chat_descriptor()])
            .await;
        let mut host_targets = election::ModelTargets::default();
        host_targets.targets.insert(
            TOY_MODEL.to_owned(),
            vec![election::InferenceTarget::Local(host_port)],
        );
        let mut tunnels = Vec::new();
        let mut senders = Vec::new();
        for (node, channels, targets) in [
            (
                &gateway,
                gateway_channels,
                election::ModelTargets::default(),
            ),
            (&host, host_channels, host_targets),
        ] {
            let (tx, rx) = watch::channel(targets);
            let manager =
                TunnelManager::start(node.clone(), channels.rpc, channels.http, channels.stage)
                    .await
                    .expect("start tunnel manager");
            manager.set_http_ingress(rx, AffinityRouter::new());
            tunnels.push(manager);
            senders.push(tx);
            node.start_accepting();
        }
        gateway
            .connect_to_peer(host.endpoint_addr_for_advertisement())
            .await
            .expect("join the host");
        let mut host_peer = duplicate_peer(host.id(), TOY_MODEL);
        host_peer.served_model_descriptors = vec![chat_descriptor()];
        gateway.insert_test_peer(host_peer).await;
        host.insert_test_peer(duplicate_peer(gateway.id(), TOY_MODEL))
            .await;
        Self {
            gateway,
            host,
            gateway_state,
            _host_state: host_state,
            tasks: vec![backend],
            _tunnels: tunnels,
            _targets: senders,
        }
    }

    /// A local client's request to the gateway's API proxy, which knows the
    /// host as the model's elected target.
    async fn ask(&self, raw: &str) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let mut targets = election::ModelTargets::default();
        targets.targets.insert(
            TOY_MODEL.to_owned(),
            vec![election::InferenceTarget::Remote(self.host.id())],
        );
        let (_tx, rx) = watch::channel(targets);
        let proxy = tokio::spawn(crate::network::openai::ingress::api_proxy(
            self.gateway.clone(),
            port,
            rx,
            Some(listener.into()),
            false,
            AffinityRouter::new(),
        ));
        let mut client = tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .unwrap();
        client.write_all(raw.as_bytes()).await.unwrap();
        let mut answer = Vec::new();
        tokio::time::timeout(Duration::from_secs(30), client.read_to_end(&mut answer))
            .await
            .expect("the proxy must answer")
            .unwrap();
        proxy.abort();
        String::from_utf8_lossy(&answer).into_owned()
    }

    /// Waits until `done` holds of the gateway, at most [`AUDIT_WAIT`].
    async fn until(&self, what: &str, done: impl Fn(&mesh::Node) -> bool) {
        let deadline = tokio::time::Instant::now() + AUDIT_WAIT;
        while !done(&self.gateway) {
            assert!(
                tokio::time::Instant::now() < deadline,
                "the gateway never came to: {what}"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    fn blocked(&self) -> bool {
        self.gateway
            .peer_blocks
            .is_blocked(&self.host.id(), crate::network::peer_blocks::now_ms())
    }

    fn strikes(&self) -> usize {
        seam::strikes(&self.gateway, &self.host.id())
    }

    async fn stop(self) {
        self.gateway.receipt_audits.shut_down().await;
        self.gateway.close_endpoint().await;
        self.host.close_endpoint().await;
        for task in self.tasks {
            task.abort();
        }
    }
}

fn assert_ok(answer: &str) {
    assert!(answer.starts_with("HTTP/1.1 200"), "{answer}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_honest_peers_turn_passes_its_audit() {
    let mesh = Mesh::start(ToyServing::Honest).await;
    assert_ok(&mesh.ask(&chat(RECEIPTS)).await);
    let host = mesh.host.id();
    mesh.until("an audit of the host", |gateway| {
        seam::audited(gateway, host)
    })
    .await;
    assert_eq!(mesh.strikes(), 0);
    assert!(!mesh.blocked());
    mesh.stop().await;
}

/// A turn asked as an Anthropic message is audited on the chat completion
/// the peer answered, before it is translated, and is no strike.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_translated_turn_is_audited_on_the_peers_own_answer() {
    let mesh = Mesh::start(ToyServing::Honest).await;
    let answer = mesh.ask(&message()).await;
    assert_ok(&answer);
    assert!(answer.contains("\"type\":\"message\""), "{answer}");
    let host = mesh.host.id();
    mesh.until("an audit of the host", |gateway| {
        seam::audited(gateway, host)
    })
    .await;
    assert_eq!(mesh.strikes(), 0);
    mesh.stop().await;
}

/// A peer that opens another turn than it committed to fails the audit:
/// it is blocked, the evidence is kept, and a request that requires
/// receipts finds no peer.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_cheating_peer_is_blocked_with_evidence() {
    let mesh = Mesh::start(ToyServing::Cheating).await;
    assert_ok(&mesh.ask(&chat(RECEIPTS)).await);
    mesh.until("a block of the host", |_| mesh.blocked()).await;
    assert!(
        !seam::evidence_files(&mesh.gateway_state.path().join("node")).is_empty(),
        "the failed audit kept no evidence"
    );
    let refused = mesh.ask(&chat(RECEIPTS)).await;
    assert!(refused.starts_with("HTTP/1.1 409"), "{refused}");
    assert!(refused.contains(UNSATISFIED_CODE), "{refused}");
    mesh.stop().await;
}

/// A peer that advertises receipts and answers without one is struck.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_peer_that_sends_no_receipt_is_struck() {
    let mesh = Mesh::start(ToyServing::WithoutReceipt).await;
    assert_ok(&mesh.ask(&chat("")).await);
    mesh.until("a strike against the host", |_| mesh.strikes() == 1)
        .await;
    assert!(!mesh.blocked(), "one strike blocks no peer");
    mesh.stop().await;
}
