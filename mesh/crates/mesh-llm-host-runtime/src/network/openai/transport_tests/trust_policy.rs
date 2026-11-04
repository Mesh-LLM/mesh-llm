//! The request trust policy across two nodes joined over real QUIC: which
//! peer a local client's request reaches under the policy, on the API proxy
//! and on the proxy that serves while this node's own model loads, and that
//! a request a peer sent is executed where it lands, under no policy.
use super::duplicate_alias_repro::duplicate_peer;
use super::*;
use crate::network::request_trust::UNSATISFIED_CODE;

const MODEL: &str = "Receipted-Chat-Q4_K_M";
const RECEIPTS: &str = "x-mesh-trust: receipts\r\n";

fn descriptor(workload: mesh::ModelWorkloadClass, receipts: bool) -> mesh::ServedModelDescriptor {
    mesh::ServedModelDescriptor {
        identity: mesh::ServedModelIdentity {
            model_name: MODEL.to_owned(),
            ..Default::default()
        },
        capabilities_known: true,
        metadata: Some(mesh::ServedModelMetadata {
            workload_class: Some(workload),
            ..Default::default()
        }),
        receipts,
        ..Default::default()
    }
}

fn chat_descriptor(receipts: bool) -> mesh::ServedModelDescriptor {
    descriptor(mesh::ModelWorkloadClass::CausalGeneration, receipts)
}

fn chat(headers: &str) -> String {
    let body = serde_json::json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 1,
    })
    .to_string();
    format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n{headers}Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
}

/// A gateway that serves nothing itself, and a host that serves `MODEL` for
/// chat on its own backend, joined over QUIC.
struct Mesh {
    gateway: mesh::Node,
    host: mesh::Node,
    /// The request the host's backend ran, when one reached it.
    host_backend: oneshot::Receiver<Vec<u8>>,
    /// The request the gateway's backend ran: nothing should ever reach it.
    gateway_backend: oneshot::Receiver<Vec<u8>>,
    gateway_config: tempfile::TempDir,
    host_config: tempfile::TempDir,
    tasks: Vec<tokio::task::JoinHandle<()>>,
    _tunnels: Vec<TunnelManager>,
    _targets: Vec<watch::Sender<election::ModelTargets>>,
}

impl Mesh {
    /// `host_receipts`: whether the host advertises receipts for `MODEL`.
    async fn start(host_receipts: bool) -> Self {
        let (gateway, gateway_channels, gateway_config) = start_http_tunnel_test_node().await;
        let (host, host_channels, host_config) = start_http_tunnel_test_node().await;
        let (gateway_port, gateway_backend, gateway_task) = spawn_tunnel_capture().await;
        let (host_port, host_backend, host_task) = spawn_tunnel_capture().await;

        host.set_hosted_models(vec![MODEL.to_owned()]).await;
        host.set_served_model_descriptors(vec![chat_descriptor(host_receipts)])
            .await;
        let mut host_targets = election::ModelTargets::default();
        host_targets.targets.insert(
            MODEL.to_owned(),
            vec![
                election::InferenceTarget::Local(host_port),
                election::InferenceTarget::Remote(gateway.id()),
            ],
        );
        // The gateway's tunnel ingress has a local backend too, so a request
        // forwarded back to it would be seen, not lost.
        let mut gateway_targets = election::ModelTargets::default();
        gateway_targets.targets.insert(
            MODEL.to_owned(),
            vec![election::InferenceTarget::Local(gateway_port)],
        );

        let mut tunnels = Vec::new();
        let mut senders = Vec::new();
        for (node, channels, targets) in [
            (&gateway, gateway_channels, gateway_targets),
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
        let mut host_peer = duplicate_peer(host.id(), MODEL);
        host_peer.served_model_descriptors = vec![chat_descriptor(host_receipts)];
        gateway.insert_test_peer(host_peer).await;
        host.insert_test_peer(duplicate_peer(gateway.id(), MODEL))
            .await;
        Self {
            gateway,
            host,
            host_backend,
            gateway_backend,
            gateway_config,
            host_config,
            tasks: vec![gateway_task, host_task],
            _tunnels: tunnels,
            _targets: senders,
        }
    }

    async fn set_receipts(node: &mesh::Node, directory: &tempfile::TempDir, table: &str) {
        let path = directory.path().join("config.toml");
        std::fs::write(&path, format!("[receipts]\n{table}\n")).expect("write config");
        *node.config_state.lock().await =
            crate::runtime::config_state::ConfigState::load(&path).expect("load config");
    }

    /// A local client's request to the gateway's API proxy, which knows the
    /// host as the model's elected target.
    async fn through_the_api_proxy(&self, raw: &str) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let mut targets = election::ModelTargets::default();
        targets.targets.insert(
            MODEL.to_owned(),
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
        let answer = client_exchange(port, raw).await;
        proxy.abort();
        answer
    }

    /// A local client's request to the proxy the gateway runs while its own
    /// model loads, which tunnels every request to a mesh host.
    async fn through_the_bootstrap_proxy(&self, raw: &str) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let gateway = self.gateway.clone();
        let proxy = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept client");
            handle_mesh_request(gateway, stream.into(), true, AffinityRouter::new()).await;
        });
        let answer = client_exchange(port, raw).await;
        proxy.abort();
        answer
    }

    /// The request that reached the host's backend, if any did.
    fn served_by_host(&mut self) -> Option<String> {
        self.host_backend
            .try_recv()
            .ok()
            .map(|raw| String::from_utf8_lossy(&raw).into_owned())
    }

    async fn stop(mut self) {
        assert!(
            self.gateway_backend.try_recv().is_err(),
            "a request was sent back to the node that routed it"
        );
        self.gateway.close_endpoint().await;
        self.host.close_endpoint().await;
        for task in self.tasks {
            task.abort();
        }
    }
}

async fn client_exchange(port: u16, raw: &str) -> String {
    let mut client = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .unwrap();
    client.write_all(raw.as_bytes()).await.unwrap();
    let mut answer = Vec::new();
    tokio::time::timeout(Duration::from_secs(30), client.read_to_end(&mut answer))
        .await
        .expect("the proxy must answer")
        .unwrap();
    String::from_utf8_lossy(&answer).into_owned()
}

fn assert_unsatisfied(answer: &str) {
    assert!(answer.starts_with("HTTP/1.1 409"), "{answer}");
    assert!(answer.contains(UNSATISFIED_CODE), "{answer}");
}

/// What a receipted host is sent: the turn names this node's receipt key,
/// so it is audited, and the policy header stays with the routing node.
fn assert_audited_turn(forwarded: &str) {
    let forwarded = forwarded.to_ascii_lowercase();
    assert!(forwarded.contains("x-receipt-client-key:"), "{forwarded}");
    // and a seed, since the request fixes none
    assert!(forwarded.contains("x-receipt-seed:"), "{forwarded}");
    assert!(!forwarded.contains("x-mesh-trust"), "{forwarded}");
}

#[tokio::test]
async fn the_api_proxy_routes_a_receipts_request_to_a_peer_that_advertises_them() {
    let mut mesh = Mesh::start(true).await;
    let answer = mesh.through_the_api_proxy(&chat(RECEIPTS)).await;
    assert!(answer.starts_with("HTTP/1.1 200"), "{answer}");
    assert_audited_turn(&mesh.served_by_host().expect("the host serves the turn"));
    mesh.stop().await;
}

#[tokio::test]
async fn the_api_proxy_never_reaches_a_peer_without_receipts_under_the_policy() {
    let mut mesh = Mesh::start(false).await;
    assert_unsatisfied(&mesh.through_the_api_proxy(&chat(RECEIPTS)).await);
    assert!(
        mesh.served_by_host().is_none(),
        "a peer without receipts served"
    );
    // the same peer serves the request under no policy: routing still works
    let answer = mesh.through_the_api_proxy(&chat("")).await;
    assert!(answer.starts_with("HTTP/1.1 200"), "{answer}");
    let forwarded = mesh.served_by_host().expect("the host serves the turn");
    assert!(
        !forwarded
            .to_ascii_lowercase()
            .contains("x-receipt-client-key")
    );
    mesh.stop().await;
}

/// While the node's own model loads, every request goes to a mesh host
/// through the bootstrap proxy. It applies the same policy.
#[tokio::test]
async fn the_bootstrap_proxy_routes_a_receipts_request_to_a_peer_that_advertises_them() {
    let mut mesh = Mesh::start(true).await;
    let answer = mesh.through_the_bootstrap_proxy(&chat(RECEIPTS)).await;
    assert!(answer.starts_with("HTTP/1.1 200"), "{answer}");
    assert_audited_turn(&mesh.served_by_host().expect("the host serves the turn"));
    mesh.stop().await;
}

#[tokio::test]
async fn the_bootstrap_proxy_never_reaches_a_peer_without_receipts_under_the_policy() {
    let mut mesh = Mesh::start(false).await;
    assert_unsatisfied(&mesh.through_the_bootstrap_proxy(&chat(RECEIPTS)).await);
    assert!(
        mesh.served_by_host().is_none(),
        "a peer without receipts served"
    );

    // the node default applies to it too, with no header at all
    Mesh::set_receipts(&mesh.gateway, &mesh.gateway_config, "require = true").await;
    assert_unsatisfied(&mesh.through_the_bootstrap_proxy(&chat("")).await);
    let embeddings = format!(
        "POST /v1/embeddings HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        r#"{"model":"Receipted-Chat-Q4_K_M","input":"x"}"#.len(),
        r#"{"model":"Receipted-Chat-Q4_K_M","input":"x"}"#
    );
    assert_unsatisfied(&mesh.through_the_bootstrap_proxy(&embeddings).await);
    assert!(
        mesh.served_by_host().is_none(),
        "a peer without receipts served"
    );
    mesh.stop().await;
}

/// The scenario of the review, with the chat peer reachable: a peer that
/// advertises receipts serves the model name for embeddings, and the host,
/// without receipts, serves it for chat. A chat request under the policy
/// fails closed on both proxies; it does not fall back to the host.
#[tokio::test]
async fn a_receipted_embedding_peer_does_not_send_a_chat_request_to_a_peer_without_receipts() {
    let mut mesh = Mesh::start(false).await;
    let embedding_id = iroh::SecretKey::from_bytes(&[0x5e; 32]).public();
    let mut embedding_peer = duplicate_peer(embedding_id, MODEL);
    embedding_peer.served_model_descriptors =
        vec![descriptor(mesh::ModelWorkloadClass::Embedding, true)];
    mesh.gateway.insert_test_peer(embedding_peer).await;

    assert_unsatisfied(&mesh.through_the_api_proxy(&chat(RECEIPTS)).await);
    assert_unsatisfied(&mesh.through_the_bootstrap_proxy(&chat(RECEIPTS)).await);
    assert!(
        mesh.served_by_host().is_none(),
        "a peer without receipts served"
    );
    mesh.stop().await;
}

/// `[receipts] offline`: no audit, so no peer's receipts count, and the
/// answer says why.
#[tokio::test]
async fn an_offline_gateway_sends_a_receipts_request_to_no_peer() {
    let mut mesh = Mesh::start(true).await;
    Mesh::set_receipts(&mesh.gateway, &mesh.gateway_config, "offline = true").await;
    for answer in [
        mesh.through_the_api_proxy(&chat(RECEIPTS)).await,
        mesh.through_the_bootstrap_proxy(&chat(RECEIPTS)).await,
    ] {
        assert_unsatisfied(&answer);
        assert!(answer.contains("offline"), "{answer}");
    }
    assert!(mesh.served_by_host().is_none(), "an unaudited peer served");

    // and with audits asked for, an offline node still names no receipt key
    Mesh::set_receipts(
        &mesh.gateway,
        &mesh.gateway_config,
        "offline = true\naudit = true",
    )
    .await;
    let answer = mesh.through_the_api_proxy(&chat("")).await;
    assert!(answer.starts_with("HTTP/1.1 200"), "{answer}");
    let forwarded = mesh.served_by_host().expect("the host serves the turn");
    assert!(
        !forwarded
            .to_ascii_lowercase()
            .contains("x-receipt-client-key")
    );
    mesh.stop().await;
}

/// A request a peer sent over QUIC was assigned to this host. It runs here
/// under no policy, whatever its headers and the host's own settings say,
/// and is never routed on.
#[tokio::test]
async fn a_request_a_peer_sent_runs_on_the_host_under_no_policy() {
    let mut mesh = Mesh::start(false).await;
    Mesh::set_receipts(&mesh.host, &mesh.host_config, "require = true").await;
    let (mut send, mut recv) = mesh
        .gateway
        .open_http_tunnel(mesh.host.id())
        .await
        .expect("open tunnel");
    send.write_all(chat(RECEIPTS).as_bytes()).await.unwrap();
    send.finish().unwrap();
    let answer = tokio::time::timeout(Duration::from_secs(30), recv.read_to_end(16384))
        .await
        .expect("the host must answer")
        .unwrap();
    let answer = String::from_utf8_lossy(&answer);
    assert!(answer.starts_with("HTTP/1.1 200"), "{answer}");
    assert!(
        mesh.served_by_host().is_some(),
        "the host must run a request a peer assigned to it"
    );
    mesh.stop().await;
}

/// A virtual model runs through its plugin, which proves no trust tier. Both
/// proxies refuse it under the policy, with the policy's code, and serve it
/// under none.
#[tokio::test]
async fn a_virtual_model_is_refused_under_the_policy_on_both_proxies() {
    let plugin_manager =
        crate::runtime::proxy::tests::start_standalone_virtual_model_plugin_manager(
            "test-virtual",
            serde_json::json!({
                "id": "chatcmpl-virtual",
                "object": "chat.completion",
                "model": "test-virtual",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "virtual"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
            }),
        )
        .await;
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("test node");
    node.set_plugin_manager(plugin_manager).await;
    let request = |headers: &str| {
        let body = r#"{"model":"test-virtual","messages":[{"role":"user","content":"hi"}]}"#;
        format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n{headers}Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
    };

    for (headers, refused) in [(RECEIPTS, true), ("", false)] {
        let bootstrap = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let bootstrap_port = bootstrap.local_addr().unwrap().port();
        let bootstrap_node = node.clone();
        let bootstrap_proxy = tokio::spawn(async move {
            let (stream, _) = bootstrap.accept().await.expect("accept client");
            handle_mesh_request(bootstrap_node, stream.into(), true, AffinityRouter::new()).await;
        });
        let api = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let api_port = api.local_addr().unwrap().port();
        let (_targets, targets) = watch::channel(election::ModelTargets::default());
        let api_proxy = tokio::spawn(crate::network::openai::ingress::api_proxy(
            node.clone(),
            api_port,
            targets,
            Some(api.into()),
            false,
            AffinityRouter::new(),
        ));
        for port in [bootstrap_port, api_port] {
            let answer = client_exchange(port, &request(headers)).await;
            if refused {
                assert_unsatisfied(&answer);
            } else {
                assert!(answer.starts_with("HTTP/1.1 200"), "{answer}");
            }
        }
        bootstrap_proxy.abort();
        api_proxy.abort();
    }
    node.endpoint.close().await;
}
