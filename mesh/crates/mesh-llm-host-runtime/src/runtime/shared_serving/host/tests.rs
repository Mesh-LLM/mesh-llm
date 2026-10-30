use super::*;
use crate::mesh::{Node, NodeRole};
use async_trait::async_trait;
use skippy_inference_api::{
    CompletionRequest, CompletionResponse, InferenceExecutionScope, InferenceRequestContext,
    InferenceResult, ModelObject, Usage,
};

mod ingress;
mod native;
mod reconnect;
mod release_attestation;
mod released_peer;
mod startup;
mod trust;

#[test]
/// An abandoned join must publish failure and release only that membership's mesh claim.
fn interrupted_join_releases_its_mesh_claim_without_touching_other_memberships() {
    let (status, _) = watch::channel(MembershipStatus {
        name: "alpha".into(),
        phase: MembershipPhase::Joining,
        endpoint_id: None,
        mesh_id: None,
        peers: 0,
        admission: Default::default(),
        last_error: None,
    });
    let owners = std::sync::Mutex::new(BTreeMap::from([
        ("mesh-a".into(), "alpha".into()),
        ("mesh-b".into(), "beta".into()),
    ]));
    drop(JoiningStatus {
        status: &status,
        owners: &owners,
        name: "alpha",
    });
    assert!(matches!(status.borrow().phase, MembershipPhase::Failed));
    assert_eq!(
        *owners.lock().unwrap(),
        BTreeMap::from([("mesh-b".into(), "beta".into())])
    );
}

#[derive(Default)]
struct RecordingBackend {
    scopes: std::sync::Mutex<Vec<InferenceExecutionScope>>,
}

#[async_trait]
impl InferenceBackend for RecordingBackend {
    /// Reject chat on the completion-only recording fixture.
    async fn chat_completion(
        &self,
        _: skippy_inference_api::ChatCompletionRequest,
    ) -> InferenceResult<skippy_inference_api::ChatCompletionResponse> {
        Err(skippy_inference_api::InferenceError::unsupported(
            "completion-only test backend",
        ))
    }

    /// Reject chat streams on the completion-only recording fixture.
    async fn chat_completion_stream(
        &self,
        _: skippy_inference_api::ChatCompletionRequest,
        _: InferenceRequestContext,
    ) -> InferenceResult<skippy_inference_api::ChatCompletionStream> {
        Err(skippy_inference_api::InferenceError::unsupported(
            "completion-only test backend",
        ))
    }

    /// Expose the single model advertised by the isolated test host.
    async fn models(&self) -> InferenceResult<Vec<ModelObject>> {
        Ok(vec![ModelObject::new("shared-test")])
    }

    /// Record the trusted membership scope before returning a deterministic completion.
    async fn completion_with_context(
        &self,
        request: CompletionRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<CompletionResponse> {
        self.scopes
            .lock()
            .unwrap()
            .push(context.execution_scope().expect("trusted membership scope"));
        Ok(CompletionResponse::new(
            request.model,
            "shared weights",
            Usage::default(),
        ))
    }
}

struct Fixture {
    host: Arc<SharedServingHost>,
    backend: Arc<RecordingBackend>,
    seeds: Vec<Node>,
    directory: tempfile::TempDir,
}

impl Fixture {
    /// Start two independent loopback seed meshes sharing one recording backend.
    async fn new() -> Self {
        let fixture = Self::detached().await;
        let (alpha, beta) = tokio::join!(
            fixture.host.reconnect("alpha"),
            fixture.host.reconnect("beta")
        );
        alpha.expect("alpha fixture must join its independent seed");
        beta.expect("beta fixture must join its independent seed");
        fixture
    }

    /// Construct a loaded host and independent seeds before attempting any memberships.
    async fn detached() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let backend = Arc::new(RecordingBackend::default());
        let mut seeds = Vec::new();
        let mut memberships = Vec::new();
        for name in ["alpha", "beta"] {
            let seed = Node::new_for_tests(NodeRole::Client).await.unwrap();
            seed.set_mesh_id_force(format!("shared-test-{name}")).await;
            seed.start_accepting();
            let accepting_seed = seed.clone();
            tokio::spawn(async move { accepting_seed.accept_loop().await });
            let token_file = directory.path().join(format!("{name}.token"));
            std::fs::write(&token_file, seed.invite_token().await).unwrap();
            memberships.push(ServingMembershipConfig {
                name: name.into(),
                token_file,
                trust_store: None,
                owner_key: None,
            });
            seeds.push(seed);
        }
        let model_path = directory.path().join("model.gguf");
        std::fs::write(&model_path, b"fixture").unwrap();
        let config = SharedServingConfig {
            model_path,
            model_id: "shared-test".into(),
            state_dir: directory.path().join("host"),
            context_size: 512,
            gpu_layers: 0,
            concurrency: 1,
            queue_capacity: 4,
            membership_queue_capacity: 2,
            lan_only: true,
            bind_ip: Some(std::net::Ipv4Addr::LOCALHOST.into()),
            memberships,
        };
        config.validate().unwrap();
        let lock = lock_state(&config).unwrap();
        let descriptor = ServedModelDescriptor {
            identity: ServedModelIdentity {
                model_name: config.model_id.clone(),
                source_kind: ModelSourceKind::LocalGguf,
                is_primary: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let host =
            SharedServingHost::assemble(config, descriptor, backend.clone(), None, lock).unwrap();
        Self {
            host,
            backend,
            seeds,
            directory,
        }
    }

    /// Send identical client/cache metadata through a selected membership's QUIC ingress.
    async fn request(&self, index: usize, name: &str) -> String {
        let body = r#"{"model":"shared-test","prompt":"same prompt","max_tokens":1,"prompt_cache_key":"same-client-key","user":"same-user"}"#;
        let raw = format!(
            "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        self.raw_request(index, name, &raw).await
    }

    /// Send raw HTTP over a seed-to-attachment QUIC tunnel with a bounded response wait.
    async fn raw_request(&self, index: usize, name: &str, raw: &str) -> String {
        let status = self.host.status().await;
        let member = status
            .memberships
            .iter()
            .find(|member| member.name == name)
            .unwrap();
        let endpoint = member.endpoint_id.as_ref().unwrap().parse().unwrap();
        let (mut send, mut recv) = self.seeds[index].open_http_tunnel(endpoint).await.unwrap();
        send.write_all(raw.as_bytes()).await.unwrap();
        send.finish().unwrap();
        let response = tokio::time::timeout(Duration::from_secs(10), recv.read_to_end(32768))
            .await
            .unwrap()
            .unwrap();
        String::from_utf8(response).unwrap()
    }

    /// Drain the shared test host and close both isolated seed endpoints.
    async fn close(self) {
        self.host.shutdown().await.unwrap();
        for seed in self.seeds {
            seed.close_endpoint().await;
        }
    }
}

#[tokio::test]
/// Exercise isolated identities, peers and execution scopes across pause, leave and reconnect.
async fn two_independent_quic_meshes_share_backend_without_sharing_membership_or_execution_scope() {
    let fixture = Fixture::new().await;
    let before = fixture.host.status().await;
    assert_eq!(before.loaded_models, 1);
    assert!(
        before
            .memberships
            .iter()
            .all(|member| matches!(member.phase, MembershipPhase::Connected)),
        "{:#?}",
        before.memberships
    );
    assert_ne!(
        before.memberships[0].endpoint_id,
        before.memberships[1].endpoint_id
    );
    assert_ne!(before.memberships[0].mesh_id, before.memberships[1].mesh_id);
    for (index, name) in ["alpha", "beta"].iter().enumerate() {
        let response = fixture.request(index, name).await;
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        assert!(response.contains("shared weights"));
        let record = fixture.host.record(name).unwrap();
        let attachment = record.attachment.lock().await;
        let node = &attachment.as_ref().unwrap().node;
        let peers = node.peers().await;
        assert!(
            peers
                .iter()
                .any(|peer| peer.id == fixture.seeds[index].id())
        );
        assert!(
            !peers
                .iter()
                .any(|peer| peer.id == fixture.seeds[1 - index].id())
        );
        assert!(node.full_model_only);
        assert_eq!(node.vram_bytes, 0);
        assert_eq!(node.local_runtime_capacity_bytes, 0);
        assert_eq!(node.host_ram_capacity_bytes, 0);
    }
    let scopes = fixture.backend.scopes.lock().unwrap().clone();
    assert_eq!(scopes.len(), 2);
    assert_ne!(scopes[0], scopes[1]);
    fixture.host.pause("alpha").await.unwrap();
    let paused = fixture.request(0, "alpha").await;
    assert!(!paused.starts_with("HTTP/1.1 200"), "{paused}");
    assert!(fixture.request(1, "beta").await.starts_with("HTTP/1.1 200"));
    fixture.host.resume("alpha").await.unwrap();
    assert!(
        fixture
            .request(0, "alpha")
            .await
            .starts_with("HTTP/1.1 200")
    );
    fixture.host.leave("alpha").await.unwrap();
    assert!(fixture.request(1, "beta").await.starts_with("HTTP/1.1 200"));
    fixture.host.reconnect("alpha").await.unwrap();
    assert_eq!(
        fixture.host.status().await.memberships[0].endpoint_id,
        before.memberships[0].endpoint_id
    );
    assert!(
        fixture
            .request(0, "alpha")
            .await
            .starts_with("HTTP/1.1 200")
    );
    assert_ne!(
        *fixture.backend.scopes.lock().unwrap().last().unwrap(),
        scopes[0]
    );
    fixture.close().await;
}

#[tokio::test]
/// Reject uppercase labels (as in the SDK) and duplicate lowercase labels
/// before either can alias persisted state on a case-insensitive filesystem.
async fn membership_labels_cannot_alias_state_on_case_insensitive_filesystems() {
    let fixture = Fixture::new().await;
    let mut config = fixture.host.config.clone();
    config.memberships[0].name = "ALPHA".into();
    assert!(
        config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("invalid membership name")
    );
    config.memberships[0].name = "alpha".into();
    config.validate().unwrap();
    config.memberships[1].name = "alpha".into();
    assert!(
        config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("ignoring ASCII case")
    );
    fixture.close().await;
}

#[tokio::test]
/// A corrupted token must fail its reconnect while the other mesh keeps serving.
async fn bad_reconnect_token_fails_closed_without_stopping_other_membership() {
    let fixture = Fixture::new().await;
    std::fs::write(
        fixture.directory.path().join("alpha.token"),
        b"invalid token",
    )
    .unwrap();
    assert!(fixture.host.reconnect("alpha").await.is_err());
    assert!(matches!(
        fixture.host.status().await.memberships[0].phase,
        MembershipPhase::Failed
    ));
    assert!(fixture.request(1, "beta").await.starts_with("HTTP/1.1 200"));
    fixture.close().await;
}

#[tokio::test]
/// Interrupted advertisement must not leave status contradicting effective admission.
async fn abandoned_pause_and_resume_report_effective_admission_state() {
    let fixture = Fixture::new().await;
    let node = fixture
        .host
        .record("alpha")
        .unwrap()
        .attachment
        .lock()
        .await
        .as_ref()
        .unwrap()
        .node
        .clone();
    // Suspend advertisement after the synchronous admission change. Dropping
    // the operator future must not leave the phase describing the old policy.
    let models = node.models.lock().await;
    let mut pause = Box::pin(fixture.host.pause("alpha"));
    assert!(futures_util::poll!(pause.as_mut()).is_pending());
    drop(pause);
    let paused = fixture.host.status().await.memberships.remove(0);
    assert!(matches!(paused.phase, MembershipPhase::Paused));
    assert!(!paused.admission.accepting);
    drop(models);
    fixture.host.pause("alpha").await.unwrap();
    assert!(fixture.request(1, "beta").await.starts_with("HTTP/1.1 200"));

    let models = node.models.lock().await;
    let mut resume = Box::pin(fixture.host.resume("alpha"));
    assert!(futures_util::poll!(resume.as_mut()).is_pending());
    drop(resume);
    let resumed = fixture.host.status().await.memberships.remove(0);
    assert!(matches!(resumed.phase, MembershipPhase::Connected));
    assert!(resumed.admission.accepting);
    drop(models);
    fixture.host.resume("alpha").await.unwrap();
    assert!(
        fixture
            .request(0, "alpha")
            .await
            .starts_with("HTTP/1.1 200")
    );
    fixture.close().await;
}

#[tokio::test]
/// Reject a second attachment to one mesh and refuse new work after shutdown.
async fn duplicate_mesh_attachment_is_rejected_and_shutdown_cannot_resume() {
    let fixture = Fixture::new().await;
    std::fs::write(
        fixture.directory.path().join("alpha.token"),
        fixture.seeds[1].invite_token().await,
    )
    .unwrap();
    assert!(fixture.host.reconnect("alpha").await.is_err());
    assert!(fixture.request(1, "beta").await.starts_with("HTTP/1.1 200"));
    fixture.host.shutdown().await.unwrap();
    assert!(fixture.host.resume("beta").await.is_err());
    assert!(fixture.host.reconnect("beta").await.is_err());
    fixture.close().await;
}

#[tokio::test]
/// A live shared host must exclude a second owner of its state directory.
async fn shared_state_directory_has_one_lifetime_owner() {
    let fixture = Fixture::new().await;
    assert!(lock_state(&fixture.host.config).is_err());
    fixture.close().await;
}

#[tokio::test]
/// Require operator authentication for status and lifecycle changes without exposing credentials.
async fn management_http_requires_operator_authentication_for_reads_and_writes() {
    let fixture = Fixture::new().await;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let app = super::super::control::router(fixture.host.clone(), "private-operator-token");
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let client = reqwest::Client::new();
    let status = format!("{base}/api/shared-serving");
    let pause = format!("{status}/memberships/alpha/pause");
    assert_eq!(client.get(&status).send().await.unwrap().status(), 401);
    assert_eq!(client.post(&pause).send().await.unwrap().status(), 401);
    assert_eq!(
        client
            .post(&pause)
            .bearer_auth("wrong")
            .send()
            .await
            .unwrap()
            .status(),
        401
    );
    let response = client
        .get(&status)
        .bearer_auth("private-operator-token")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body = response.text().await.unwrap();
    assert!(body.contains("shared-test"));
    assert!(!body.contains("private-operator-token"));
    assert!(!body.contains(".token"));
    assert_eq!(
        client
            .post(&pause)
            .bearer_auth("private-operator-token")
            .send()
            .await
            .unwrap()
            .status(),
        204
    );
    assert!(matches!(
        fixture.host.status().await.memberships[0].phase,
        MembershipPhase::Paused
    ));
    assert!(fixture.request(1, "beta").await.starts_with("HTTP/1.1 200"));
    server.abort();
    fixture.close().await;
}

#[tokio::test]
/// Even admitted peers must not acquire model-control, plugin or unknown stream access.
async fn admitted_mesh_cannot_open_model_control_or_plugin_streams() {
    let fixture = Fixture::new().await;
    let record = fixture.host.record("alpha").unwrap();
    let attachment = record.attachment.lock().await;
    let node = &attachment.as_ref().unwrap().node;
    let connection = fixture.seeds[0]
        .endpoint
        .connect(
            Node::decode_invite_token(&node.invite_token().await).unwrap(),
            crate::protocol::ALPN,
        )
        .await
        .unwrap();
    for stream_type in [
        crate::protocol::STREAM_TUNNEL,
        crate::protocol::STREAM_CONFIG_PUSH,
        crate::protocol::STREAM_PLUGIN_CHANNEL,
        0xff,
    ] {
        let (mut send, mut recv) = connection.open_bi().await.unwrap();
        send.write_all(&[stream_type]).await.unwrap();
        send.finish().unwrap();
        let received = tokio::time::timeout(Duration::from_secs(5), recv.read_to_end(1024))
            .await
            .unwrap();
        assert!(
            received.is_err() || received.unwrap().is_empty(),
            "forbidden stream produced a response"
        );
    }
    drop(attachment);
    assert!(fixture.request(1, "beta").await.starts_with("HTTP/1.1 200"));
    fixture.close().await;
}
