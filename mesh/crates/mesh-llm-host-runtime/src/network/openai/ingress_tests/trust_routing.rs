//! The request trust policy through local ingress: every request goes in
//! over a socket, as a client's does, and what answers it is observed.
use super::super::tests::{ingress_client_socket_pair, test_endpoint_id, test_remote_peer};
use super::*;
use crate::network::request_trust::{RequestTrust, UNSATISFIED_CODE};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const MODEL: &str = "acme/chat-model:Q4_K_M";
const CHAT: &str = "/v1/chat/completions";
const RECEIPTS: &str = "x-mesh-trust: receipts\r\n";

/// A loopback backend that answers every request with 200 and counts them.
async fn counting_backend() -> (u16, Arc<AtomicUsize>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind backend");
    let port = listener.local_addr().expect("backend addr").port();
    let served = Arc::new(AtomicUsize::new(0));
    let count = served.clone();
    tokio::spawn(async move {
        while let Ok((mut stream, _)) = listener.accept().await {
            if proxy::read_http_request(&mut stream).await.is_err() {
                continue;
            }
            count.fetch_add(1, Ordering::SeqCst);
            let body = r#"{"id":"c1","object":"chat.completion","choices":[]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes()).await;
        }
    });
    (port, served)
}

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

/// A peer serving `MODEL` for `workload`, with or without receipts. Its
/// address leads nowhere: a request routed to it fails, it is not served.
fn peer(seed: u32, workload: mesh::ModelWorkloadClass, receipts: bool) -> mesh::PeerInfo {
    let mut peer = test_remote_peer(seed, MODEL);
    peer.served_model_descriptors = vec![descriptor(workload, receipts)];
    peer
}

fn chat_peer(seed: u32, receipts: bool) -> mesh::PeerInfo {
    peer(seed, mesh::ModelWorkloadClass::CausalGeneration, receipts)
}

struct Ingress {
    node: mesh::Node,
    targets: election::ModelTargets,
    /// How many requests this node's own runtime served.
    served_here: Arc<AtomicUsize>,
    config: tempfile::TempDir,
}

impl Ingress {
    /// A node serving `MODEL` for chat on its own runtime, which issues
    /// receipts or not.
    async fn serving(issues_receipts: bool) -> Self {
        let mut ingress = Self::without_a_runtime().await;
        let (port, served_here) = counting_backend().await;
        ingress.served_here = served_here;
        let node = &ingress.node;
        node.set_hosted_models(vec![MODEL.to_owned()]).await;
        node.set_served_model_descriptors(vec![descriptor(
            mesh::ModelWorkloadClass::CausalGeneration,
            false,
        )])
        .await;
        node.set_issues_receipts(MODEL, issues_receipts);
        ingress.targets.targets.insert(
            MODEL.to_owned(),
            vec![election::InferenceTarget::Local(port)],
        );
        ingress
    }

    /// A node that serves nothing itself.
    async fn without_a_runtime() -> Self {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .expect("test node should start");
        Self {
            node,
            targets: election::ModelTargets::default(),
            served_here: Arc::default(),
            config: tempfile::tempdir().expect("config directory"),
        }
    }

    /// Sets the node's `[receipts]` table.
    async fn receipts_config(&self, table: &str) {
        let path = self.config.path().join("config.toml");
        std::fs::write(&path, format!("[receipts]\n{table}\n")).expect("write config");
        *self.node.config_state.lock().await =
            crate::runtime::config_state::ConfigState::load(&path).expect("load config");
    }

    async fn add_peer(&mut self, peer: mesh::PeerInfo) -> iroh::EndpointId {
        let id = peer.id;
        self.node.insert_test_peer(peer).await;
        self.targets
            .targets
            .entry(MODEL.to_owned())
            .or_default()
            .push(election::InferenceTarget::Remote(id));
        id
    }

    fn ctx<'a>(&'a self, affinity: &'a affinity::AffinityRouter) -> IngressRouteContext<'a> {
        IngressRouteContext {
            node: &self.node,
            targets: &self.targets,
            affinity,
            plugin_manager: None,
            requested_by_node_id: None,
            automatic_retry: None,
            exchange_channel: None,
        }
    }

    /// Sends `raw` to this node's API as a local client and returns the
    /// whole answer.
    async fn send(&self, raw: Vec<u8>) -> String {
        self.send_from(raw, crate::runtime::IngressType::LocalOpenAi, None)
            .await
    }

    async fn send_from(
        &self,
        raw: Vec<u8>,
        ingress_type: crate::runtime::IngressType,
        requested_by: Option<iroh::EndpointId>,
    ) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind ingress");
        let address = listener.local_addr().expect("ingress addr");
        let (node, targets) = (self.node.clone(), self.targets.clone());
        let ingress = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept client");
            handle_api_proxy_connection(
                node,
                stream.into(),
                targets,
                affinity::AffinityRouter::new(),
                ingress_type,
                requested_by,
            )
            .await;
        });
        let mut client = tokio::net::TcpStream::connect(address)
            .await
            .expect("connect client");
        client.write_all(&raw).await.expect("write request");
        let mut answer = Vec::new();
        tokio::time::timeout(
            std::time::Duration::from_secs(30),
            client.read_to_end(&mut answer),
        )
        .await
        .expect("ingress must answer")
        .expect("read answer");
        ingress.await.expect("ingress task");
        String::from_utf8_lossy(&answer).into_owned()
    }

    fn served_here(&self) -> usize {
        self.served_here.load(Ordering::SeqCst)
    }
}

/// A JSON `POST` to `path` with `headers` (each `Name: value\r\n`).
fn post(path: &str, headers: &str, body: &str) -> Vec<u8> {
    format!(
        "POST {path} HTTP/1.1\r\nHost: t\r\n{headers}Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .into_bytes()
}

fn chat(headers: &str) -> Vec<u8> {
    let body = format!(r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hi"}}]}}"#);
    post(CHAT, headers, &body)
}

/// `raw` as an ingress reads it off a socket: no trust policy decided yet.
async fn parsed(raw: Vec<u8>) -> proxy::BufferedHttpRequest {
    let (mut writer, mut reader) = tokio::io::duplex(64 * 1024);
    writer.write_all(&raw).await.expect("write request");
    proxy::read_http_request(&mut reader)
        .await
        .expect("parse request")
}

fn assert_unsatisfied(answer: &str) {
    assert!(answer.starts_with("HTTP/1.1 409"), "{answer}");
    assert!(answer.contains(UNSATISFIED_CODE), "{answer}");
}

fn assert_served(answer: &str) {
    assert!(answer.starts_with("HTTP/1.1 200"), "{answer}");
}

#[tokio::test]
async fn a_receipts_request_is_served_only_by_a_runtime_that_issues_them() {
    let plain = Ingress::serving(false).await;
    assert_unsatisfied(&plain.send(chat(RECEIPTS)).await);
    assert_eq!(plain.served_here(), 0, "a runtime without receipts served");
    // the same node serves a request under no policy, as before
    assert_served(&plain.send(chat("")).await);
    assert_served(&plain.send(chat("x-mesh-trust: any\r\n")).await);
    assert_eq!(plain.served_here(), 2);

    let receipted = Ingress::serving(true).await;
    assert_served(&receipted.send(chat(RECEIPTS)).await);
    assert_eq!(receipted.served_here(), 1);
}

#[tokio::test]
async fn the_node_default_requires_receipts_without_a_header() {
    let plain = Ingress::serving(false).await;
    plain.receipts_config("require = true").await;
    assert_unsatisfied(&plain.send(chat("")).await);
    // a header cannot weaken the node default
    let weaker = plain.send(chat("x-mesh-trust: any\r\n")).await;
    assert!(weaker.starts_with("HTTP/1.1 400"), "{weaker}");
    assert_eq!(plain.served_here(), 0);
}

/// The scenario of the review: a receipted peer serves the model name for
/// embeddings, a peer without receipts serves it for chat. Neither is a
/// chat candidate under the policy, and the workload fallback must not turn
/// the second one into one.
#[tokio::test]
async fn a_peer_with_receipts_for_another_workload_does_not_qualify_a_chat_peer_without() {
    let mut ingress = Ingress::without_a_runtime().await;
    let embedding = ingress
        .add_peer(peer(0x41, mesh::ModelWorkloadClass::Embedding, true))
        .await;
    let chat_only = ingress.add_peer(chat_peer(0x42, false)).await;
    assert_unsatisfied(&ingress.send(chat(RECEIPTS)).await);

    // the pools routing reads hold neither peer for this request
    let trust = RequestTrust::Receipts { audit: true };
    let pool =
        workload_routing::ingress_candidates(&ingress.node, MODEL, CHAT, &ingress.targets, trust)
            .await;
    assert!(pool.is_empty(), "{pool:?}");
    let hosts = vec![embedding, chat_only];
    let permitted =
        crate::network::request_trust::permitted_hosts(&ingress.node, trust, MODEL, CHAT, hosts)
            .await;
    assert!(permitted.is_empty(), "{permitted:?}");
    // without the policy the chat peer is the fallback, as before
    let any = RequestTrust::UNRESTRICTED;
    let pool =
        workload_routing::ingress_candidates(&ingress.node, MODEL, CHAT, &ingress.targets, any)
            .await;
    assert_eq!(pool, vec![election::InferenceTarget::Remote(chat_only)]);
}

/// A peer that advertises receipts is the only remote candidate, wherever
/// routing looks for one: the elected pool, the fallback read from mesh
/// state, and the peer route resolved when this node serves nothing.
#[tokio::test]
async fn every_pool_holds_only_the_peers_that_qualify_when_it_is_read() {
    let mut ingress = Ingress::without_a_runtime().await;
    let receipted = ingress.add_peer(chat_peer(0x51, true)).await;
    let plain = ingress.add_peer(chat_peer(0x52, false)).await;
    let trust = RequestTrust::Receipts { audit: true };
    let only_receipted = vec![election::InferenceTarget::Remote(receipted)];
    let node = &ingress.node;

    let elected =
        workload_routing::ingress_candidates(node, MODEL, CHAT, &ingress.targets, trust).await;
    assert_eq!(elected, only_receipted);
    // no elected target at all: the fallback reads every host of the model
    let nothing_elected = election::ModelTargets::default();
    let fallback =
        workload_routing::ingress_candidates(node, MODEL, CHAT, &nothing_elected, trust).await;
    assert_eq!(fallback, only_receipted);

    let affinity = affinity::AffinityRouter::new();
    let ctx = IngressRouteContext {
        targets: &nothing_elected,
        ..ingress.ctx(&affinity)
    };
    let RemoteMeshRoute::Targets(route) =
        resolve_remote_mesh_route(&ctx, MODEL, None, &[], trust, CHAT).await
    else {
        panic!("a receipted peer serves the model");
    };
    assert_eq!(route.candidates(MODEL), only_receipted);
    // a pinned peer that does not qualify is no candidate either
    let pinned = resolve_remote_mesh_route(&ctx, MODEL, Some(plain), &[], trust, CHAT).await;
    assert!(matches!(pinned, RemoteMeshRoute::TargetUnavailable { .. }));

    // a peer that stops advertising receipts stops qualifying at once
    let mut downgraded = chat_peer(0x51, false);
    downgraded.id = receipted;
    node.insert_test_peer(downgraded).await;
    let after =
        workload_routing::ingress_candidates(node, MODEL, CHAT, &ingress.targets, trust).await;
    assert!(after.is_empty(), "{after:?}");
    let stale = election::InferenceTarget::Remote(receipted);
    assert!(!crate::network::request_trust::permits(node, trust, MODEL, CHAT, &stale).await);
}

/// A pool handed straight to the attempt loop, as a path that skipped the
/// early filter would hand it: the loop asks the policy itself, and
/// contacts nothing that does not qualify.
#[tokio::test]
async fn the_attempt_loop_refuses_a_candidate_the_policy_does_not_permit() {
    let ingress = Ingress::serving(false).await;
    let (server, mut client) = ingress_client_socket_pair().await;
    let mut request = parsed(chat(RECEIPTS)).await;
    trust_policy::decide(&ingress.node, &mut request, false)
        .await
        .expect("the header is well formed");
    let affinity = affinity::AffinityRouter::new();
    let outcome = proxy::route_model_request(
        ingress.node.clone(),
        server.into(),
        &ingress.targets,
        MODEL,
        &request,
        proxy::RouteModelRequestContext {
            local_execution_only: false,
            automatic_retry: None,
            required_tokens: None,
            affinity: &affinity,
            route_observer: OpenAiRouteObserver::default(),
            served_by_header: None,
            peer_capsule_id: None,
            served_by_node_id: None,
        },
    )
    .await;
    assert!(matches!(
        outcome,
        proxy::RouteDispatchOutcome::Responded(409)
    ));
    let mut answer = String::new();
    client.read_to_string(&mut answer).await.expect("read");
    assert_unsatisfied(&answer);
    assert_eq!(ingress.served_here(), 0);
}

/// A request read off a socket carries no decision. A path that routes it
/// without asking for one serves nothing.
#[tokio::test]
async fn a_request_nothing_decided_for_is_served_by_nothing() {
    let ingress = Ingress::serving(true).await;
    let mut undecided = parsed(chat("")).await;
    assert_eq!(undecided.trust, RequestTrust::Undecided);
    let affinity = affinity::AffinityRouter::new();
    let (server, mut client) = ingress_client_socket_pair().await;
    route_request(
        false,
        server.into(),
        &mut undecided,
        &ingress.ctx(&affinity),
        Some(MODEL),
        None,
        OpenAiRouteObserver::default(),
    )
    .await;
    let mut answer = String::new();
    client.read_to_string(&mut answer).await.expect("read");
    assert_unsatisfied(&answer);
    assert_eq!(ingress.served_here(), 0);
}

#[tokio::test]
async fn the_policy_refuses_the_endpoints_receipts_do_not_cover() {
    let ingress = Ingress::serving(true).await;
    ingress.receipts_config("require = true").await;
    let body = format!(r#"{{"model":"{MODEL}","input":"hi"}}"#);
    assert_unsatisfied(&ingress.send(post("/v1/embeddings", "", &body)).await);
    assert_eq!(ingress.served_here(), 0);

    // listing models is no inference: the policy leaves it alone
    let models = b"GET /v1/models HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n";
    assert_served(&ingress.send(models.to_vec()).await);
}

/// A client audits a turn by opening its receipt on the node that served
/// it. Under `require` that request is not refused as "not chat": it
/// reaches the runtime that issues the receipts, and only such a runtime.
#[tokio::test]
async fn opening_a_receipt_reaches_the_runtime_under_the_policy() {
    let body = format!(r#"{{"model":"{MODEL}","receipt_id":"r1"}}"#);
    let receipted = Ingress::serving(true).await;
    receipted.receipts_config("require = true").await;
    assert_served(&receipted.send(post("/v1/receipts/open", "", &body)).await);
    assert_eq!(receipted.served_here(), 1);

    let plain = Ingress::serving(false).await;
    plain.receipts_config("require = true").await;
    assert_unsatisfied(&plain.send(post("/v1/receipts/open", "", &body)).await);
    assert_eq!(plain.served_here(), 0);
}

#[tokio::test]
async fn a_realtime_session_is_refused_under_the_policy() {
    let ingress = Ingress::serving(true).await;
    let upgrade = |headers: &str| {
        format!(
            "GET /v1/realtime?model={MODEL} HTTP/1.1\r\nHost: t\r\n{headers}Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n"
        )
        .into_bytes()
    };
    assert_unsatisfied(&ingress.send(upgrade(RECEIPTS)).await);
    ingress.receipts_config("require = true").await;
    assert_unsatisfied(&ingress.send(upgrade("")).await);
    assert_eq!(ingress.served_here(), 0);
}

/// A plugin endpoint serves the model name, and a plugin cannot prove the
/// tier: under the policy it is no way to serve the request, whichever
/// fallback would have reached it.
#[tokio::test]
async fn a_plugin_endpoint_never_serves_a_receipts_request() {
    let (plugin_port, served_by_plugin) = counting_backend().await;
    let manager = crate::plugin::PluginManager::for_test_summaries(Vec::new());
    manager
        .set_test_inference_endpoints(vec![crate::plugin::InferenceEndpointRoute {
            plugin_name: "test-provider".into(),
            endpoint_id: "chat".into(),
            protocol: None,
            address: format!("http://127.0.0.1:{plugin_port}"),
            models: vec![MODEL.into()],
        }])
        .await;
    // no runtime here, and a peer that serves the name for another workload:
    // the shape in which the plugin fallback fires
    let mut ingress = Ingress::without_a_runtime().await;
    ingress.node.set_plugin_manager(manager).await;
    ingress
        .add_peer(peer(0x61, mesh::ModelWorkloadClass::Embedding, true))
        .await;

    assert_served(&ingress.send(chat("")).await);
    assert_eq!(served_by_plugin.load(Ordering::SeqCst), 1);
    assert_unsatisfied(&ingress.send(chat(RECEIPTS)).await);
    assert_eq!(
        served_by_plugin.load(Ordering::SeqCst),
        1,
        "a plugin served a request that requires receipts"
    );
}

#[tokio::test]
async fn a_policy_this_node_cannot_read_is_a_400_without_the_code() {
    let ingress = Ingress::serving(true).await;
    for header in ["x-mesh-trust: fast\r\n", "x-mesh-trust: receipts,\r\n"] {
        let answer = ingress.send(chat(header)).await;
        assert!(answer.starts_with("HTTP/1.1 400"), "{answer}");
        assert!(!answer.contains(UNSATISFIED_CODE), "{answer}");
    }
    let unavailable = ingress.send(chat("x-mesh-trust: tee\r\n")).await;
    assert!(unavailable.starts_with("HTTP/1.1 400"), "{unavailable}");
    assert!(unavailable.contains("not available"), "{unavailable}");
    assert_eq!(ingress.served_here(), 0);
}

/// `[receipts] offline` turns every audit off, so a peer's receipts go
/// unchecked and no peer qualifies; this node's own runtime still does.
#[tokio::test]
async fn an_offline_node_serves_a_receipts_request_itself_or_not_at_all() {
    let mut plain = Ingress::serving(false).await;
    plain.receipts_config("offline = true").await;
    plain.add_peer(chat_peer(0x71, true)).await;
    let answer = plain.send(chat(RECEIPTS)).await;
    assert_unsatisfied(&answer);
    assert!(answer.contains("offline"), "{answer}");
    assert_eq!(plain.served_here(), 0);

    let mut receipted = Ingress::serving(true).await;
    receipted
        .receipts_config("offline = true\nrequire = true")
        .await;
    receipted.add_peer(chat_peer(0x72, true)).await;
    assert_served(&receipted.send(chat("")).await);
    assert_eq!(receipted.served_here(), 1);
}

/// A request a peer assigned to this node is executed here, whatever
/// policy its headers or this node's settings name: the policy was the
/// sender's to apply, and nothing is routed.
#[tokio::test]
async fn a_request_a_peer_assigned_is_executed_here_under_no_policy() {
    let ingress = Ingress::serving(false).await;
    ingress.receipts_config("require = true").await;
    let sender = test_endpoint_id(0x09);
    let answer = ingress
        .send_from(
            chat(RECEIPTS),
            crate::runtime::IngressType::RemoteQuicHttp,
            Some(sender),
        )
        .await;
    assert_served(&answer);
    assert_eq!(ingress.served_here(), 1);
}

/// `x-mesh-target` narrows where a request may run; under the policy a
/// pinned candidate that does not qualify is rejected, not served.
#[tokio::test]
async fn a_pinned_candidate_that_does_not_qualify_is_rejected() {
    let mut ingress = Ingress::serving(false).await;
    let plain_peer = ingress.add_peer(chat_peer(0x81, false)).await;
    for pinned in [ingress.node.id(), plain_peer] {
        let target = hex::encode(pinned.as_bytes());
        let headers = format!("{RECEIPTS}x-mesh-target: {target}\r\n");
        assert_unsatisfied(&ingress.send(chat(&headers)).await);
    }
    assert_eq!(ingress.served_here(), 0);
}

/// The 409 of a dispatch that cannot honor the policy carries the policy's
/// code, so a client can tell it from a routing-header conflict.
#[tokio::test]
async fn a_dispatch_that_cannot_honor_the_policy_answers_with_its_code() {
    let ingress = Ingress::serving(true).await;
    let affinity = affinity::AffinityRouter::new();
    let ctx = ingress.ctx(&affinity);
    let no_model = AutoRouteDecision {
        effective_model: None,
        required_tokens: None,
    };
    for (headers, coded) in [(RECEIPTS, true), ("", false)] {
        let target = hex::encode(test_endpoint_id(0x0a).as_bytes());
        let headers = format!("{headers}x-mesh-target: {target}\r\n");
        let mut request = parsed(chat(&headers)).await;
        trust_policy::decide(ctx.node, &mut request, false)
            .await
            .expect("well formed");
        let (server, mut client) = ingress_client_socket_pair().await;
        let refused = enforce_mesh_routing_headers_before_dispatch(
            server.into(),
            &request,
            &no_model,
            None,
            OpenAiRouteObserver::default(),
        )
        .await;
        assert!(matches!(
            refused,
            Err(proxy::RouteDispatchOutcome::Responded(409))
        ));
        let mut answer = String::new();
        client.read_to_string(&mut answer).await.expect("read");
        assert!(answer.starts_with("HTTP/1.1 409"), "{answer}");
        assert_eq!(answer.contains(UNSATISFIED_CODE), coded, "{answer}");
    }
}

/// A request that excludes the only runtime that qualifies cannot be served
/// under its policy, and its 409 says so with the policy's code. The same
/// exclusion under no policy stays the routing-header 409, without it.
#[tokio::test]
async fn excluding_the_only_runtime_that_qualifies_fails_with_the_policys_code() {
    let ingress = Ingress::serving(true).await;
    let exclude = format!(
        "x-mesh-exclude: {}\r\n",
        hex::encode(ingress.node.id().as_bytes())
    );
    assert_unsatisfied(&ingress.send(chat(&format!("{RECEIPTS}{exclude}"))).await);

    let answer = ingress.send(chat(&exclude)).await;
    assert!(answer.starts_with("HTTP/1.1 409"), "{answer}");
    assert!(!answer.contains(UNSATISFIED_CODE), "{answer}");
    assert_eq!(ingress.served_here(), 0);
}

/// A peer that advertises receipts for a model it serves under a name this
/// node cannot tie to a Hugging Face file does not qualify: every audit of
/// its turns would end as no audit, so its receipts would prove nothing.
#[tokio::test]
async fn a_peer_with_receipts_under_a_name_this_node_cannot_tie_does_not_qualify() {
    const LOCAL: &str = "local-gguf/sha256-ab12cd34ef56";
    let mut ingress = Ingress::without_a_runtime().await;
    let mut peer = test_remote_peer(0x61, LOCAL);
    let mut served = descriptor(mesh::ModelWorkloadClass::CausalGeneration, true);
    served.identity.model_name = LOCAL.to_owned();
    peer.served_model_descriptors = vec![served];
    let host = peer.id;
    ingress.node.insert_test_peer(peer).await;
    ingress
        .targets
        .targets
        .entry(LOCAL.to_owned())
        .or_default()
        .push(election::InferenceTarget::Remote(host));

    let trust = RequestTrust::Receipts { audit: true };
    let permitted = crate::network::request_trust::permitted_hosts(
        &ingress.node,
        trust,
        LOCAL,
        CHAT,
        vec![host],
    )
    .await;
    assert!(permitted.is_empty(), "{permitted:?}");
    let target = election::InferenceTarget::Remote(host);
    assert!(
        !crate::network::request_trust::permits(&ingress.node, trust, LOCAL, CHAT, &target).await
    );
    // under no policy the peer serves the model, as before
    let any = crate::network::request_trust::permitted_hosts(
        &ingress.node,
        RequestTrust::UNRESTRICTED,
        LOCAL,
        CHAT,
        vec![host],
    )
    .await;
    assert_eq!(any, vec![host]);

    let body = format!(r#"{{"model":"{LOCAL}","messages":[{{"role":"user","content":"hi"}}]}}"#);
    let answer = ingress.send(post(CHAT, RECEIPTS, &body)).await;
    assert_unsatisfied(&answer);
    assert!(answer.contains("cannot tie that name"), "{answer}");
}
