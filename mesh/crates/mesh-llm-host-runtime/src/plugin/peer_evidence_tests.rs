use super::*;
use crate::plugin::{BridgeFuture, PluginRpcBridge, RpcResult};
use mesh_llm_plugin::{PluginRuntime, plugin_server_info};
use std::sync::{Arc, atomic::AtomicUsize};

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn declaration(namespace: &str) -> proto::PeerEvidenceVerifierManifest {
    peer_evidence::peer_evidence_verifier(namespace, "verify")
}

/// Evidence published under `namespace`, as a provider's record carries it.
fn published(namespace: &str) -> BTreeMap<String, Vec<u8>> {
    BTreeMap::from([(namespace.into(), br#"{"bond":"reference"}"#.to_vec())])
}

impl EvidenceRegistry {
    /// Commits a verifier result and reports whether it accepts.
    fn store(
        &self,
        key: VerdictKey,
        epoch: u64,
        result: Result<PeerEvidenceVerificationResult>,
    ) -> bool {
        self.commit(key, epoch, result).accepted
    }
}

fn accepted_for(seconds: u64) -> Result<PeerEvidenceVerificationResult> {
    Ok(
        serde_json::from_value(json!({"accepted": true, "valid_until": unix_now() + seconds}))
            .unwrap(),
    )
}

#[test]
fn only_one_verifier_is_supported() {
    let mut registry = EvidenceRegistry::default();
    assert!(!registry.has_verifier());
    registry.register("a", &declaration("proof")).unwrap();
    assert!(registry.register("b", &declaration("other")).is_err());
    assert!(registry.has_verifier());
}

#[test]
fn verdicts_are_bound_to_mesh_node_and_evidence_and_expire() {
    let mut registry = EvidenceRegistry::default();
    registry.register("v", &declaration("bond")).unwrap();
    let evidence = BTreeMap::from([("bond".into(), b"reference".to_vec())]);
    let key = VerdictKey::new("mesh", "host", b"reference");
    assert!(registry.store(key.clone(), 0, accepted_for(300)));
    assert!(registry.provider_cached("mesh", "host", &evidence));
    assert!(!registry.provider_cached("other", "host", &evidence));
    assert!(!registry.provider_cached("mesh", "other", &evidence));
    assert!(!registry.provider_cached("mesh", "host", &BTreeMap::new()));
    assert!(!registry.provider_cached(
        "mesh",
        "host",
        &BTreeMap::from([("bond".into(), b"another".to_vec())])
    ));

    let now = Instant::now();
    let verdict = Verdict {
        expires: now,
        renew: now,
        ..lock(&registry.verdicts)[&key]
    };
    lock(&registry.verdicts).insert(key.clone(), verdict);
    assert_eq!(registry.provider_decision("mesh", "host", &evidence), None);
    // An expired verdict is due at once.
    assert_eq!(registry.next_due(|_| false), Some(now));

    // An already expired result is a short rejection.
    assert!(!registry.store(key.clone(), 0, accepted_for(0)));
    assert_eq!(
        registry.provider_decision("mesh", "host", &evidence),
        Some(false)
    );
    assert!(registry.next_due(|_| false).unwrap() <= Instant::now() + FAILURE_RETRY);
}

#[test]
fn acceptances_are_renewed_before_they_expire() {
    let mut registry = EvidenceRegistry::default();
    registry.register("v", &declaration("bond")).unwrap();
    let evidence = BTreeMap::from([("bond".into(), b"reference".to_vec())]);
    let key = VerdictKey::new("mesh", "host", b"reference");
    assert!(registry.store(key.clone(), 0, accepted_for(60)));
    assert!(!registry.needs_verification("mesh", "host", &evidence));
    let verdict = lock(&registry.verdicts)[&key];
    assert_eq!(verdict.expires - verdict.renew, RENEWAL_LEAD);
    assert_eq!(registry.next_due(|_| false), Some(verdict.renew));

    // Once renewal is due, the provider stays eligible until it expires, and
    // stays due until a renewal is underway.
    let due = Instant::now();
    lock(&registry.verdicts).get_mut(&key).unwrap().renew = due;
    assert!(registry.needs_verification("mesh", "host", &evidence));
    assert!(registry.provider_cached("mesh", "host", &evidence));
    assert_eq!(registry.next_due(|_| false), Some(due));
    assert_eq!(registry.next_due(|node| node == "host"), None);

    // A failed renewal keeps the acceptance and tries again shortly.
    assert!(registry.store(key.clone(), 0, Err(anyhow::anyhow!("unreachable RPC"))));
    let retried = lock(&registry.verdicts)[&key];
    assert_eq!(retried.expires, verdict.expires);
    assert!(retried.renew <= Instant::now() + FAILURE_RETRY);
    assert!(registry.provider_cached("mesh", "host", &evidence));

    // An acceptance too short to renew early, such as the rest of an exit
    // countdown, lasts until it expires.
    assert!(registry.store(key.clone(), 0, accepted_for(15)));
    let short = lock(&registry.verdicts)[&key];
    assert_eq!(short.renew, short.expires);
}

#[test]
fn rejections_replace_an_acceptance_at_once() {
    let mut registry = EvidenceRegistry::default();
    registry.register("v", &declaration("bond")).unwrap();
    let evidence = BTreeMap::from([("bond".into(), b"reference".to_vec())]);
    let key = VerdictKey::new("mesh", "host", b"reference");
    assert!(registry.store(key.clone(), 0, accepted_for(60)));
    let rejected =
        serde_json::from_value(json!({"accepted": false, "valid_until": unix_now() + 60}));
    assert!(!registry.store(key.clone(), 0, Ok(rejected.unwrap())));
    assert_eq!(
        registry.provider_decision("mesh", "host", &evidence),
        Some(false)
    );

    // So does one without a deadline, which is checked again shortly.
    assert!(registry.store(key.clone(), 0, accepted_for(60)));
    let rejected = serde_json::from_value(json!({"accepted": false}));
    assert!(!registry.store(key.clone(), 0, Ok(rejected.unwrap())));
    assert_eq!(
        registry.provider_decision("mesh", "host", &evidence),
        Some(false)
    );
    assert!(lock(&registry.verdicts)[&key].renew <= Instant::now() + FAILURE_RETRY);
}

#[test]
fn verdicts_end_at_their_deadline() {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    // Two seconds ahead, so the deadline is still ahead when the second
    // ticks over during the test.
    let deadline = now.as_secs() + 2;
    let result = json!({"accepted": true, "valid_until": deadline});
    let verdict = verdict_from(serde_json::from_value(result).unwrap()).unwrap();
    // A deadline in the middle of a second is not extended to its end.
    assert!(verdict.expires <= Instant::now() + (Duration::from_secs(deadline) - now));
}

#[test]
fn far_verdicts_last_until_invalidated() {
    let mut registry = EvidenceRegistry::default();
    registry.register("v", &declaration("bond")).unwrap();
    let evidence = published("bond");
    let key = VerdictKey::new("mesh", "host", &evidence["bond"]);
    let far = 2 * MAX_VERDICT_AGE.as_secs();
    assert!(registry.store(key.clone(), 0, accepted_for(far)));
    assert!(registry.provider_cached("mesh", "host", &evidence));
    // The lifetime is capped, which keeps `Instant` arithmetic in range.
    let verdict = lock(&registry.verdicts)[&key];
    assert!(verdict.expires <= Instant::now() + MAX_VERDICT_AGE);

    // A result that was in flight across an invalidation is discarded.
    registry.invalidate("v");
    assert!(!registry.provider_cached("mesh", "host", &evidence));
    assert!(!registry.store(key.clone(), 0, accepted_for(300)));
    assert_eq!(registry.provider_decision("mesh", "host", &evidence), None);
    assert!(registry.store(key, 1, accepted_for(300)));
}

#[test]
fn only_current_evidence_of_current_providers_stays_cached() {
    let mut registry = EvidenceRegistry::default();
    registry.register("v", &declaration("bond")).unwrap();
    let evidence = |bytes: &[u8]| BTreeMap::from([("bond".to_string(), bytes.to_vec())]);
    for (node, bytes) in [("kept", b"a"), ("replaced", b"a"), ("departed", b"a")] {
        let key = VerdictKey::new("mesh", node, bytes);
        assert!(registry.store(key, 0, accepted_for(300)));
    }
    registry.retain_current(
        "mesh",
        &[
            ("kept".into(), evidence(b"a")),
            ("replaced".into(), evidence(b"b")),
        ],
    );
    assert_eq!(
        registry.provider_decision("mesh", "kept", &evidence(b"a")),
        Some(true)
    );
    assert_eq!(
        registry.provider_decision("mesh", "replaced", &evidence(b"a")),
        None
    );
    assert_eq!(
        registry.provider_decision("mesh", "departed", &evidence(b"a")),
        None
    );
    assert_eq!(lock(&registry.verdicts).len(), 1);
}

#[test]
fn clearing_the_verifier_withdraws_its_publication() {
    let mut registry = EvidenceRegistry::default();
    registry.register("a", &declaration("proof")).unwrap();
    registry.replace_publication(1, Some(b"first".to_vec()));
    registry.clear_plugin("b");
    assert_eq!(
        registry.publications(),
        BTreeMap::from([("proof".into(), b"first".to_vec())])
    );
    registry.clear_plugin("a");
    assert!(registry.publications().is_empty());
}

struct CountingVerifier {
    calls: Arc<AtomicUsize>,
    verdict: Value,
}

impl PluginRpcBridge for CountingVerifier {
    fn handle_request(
        &self,
        _: String,
        _: String,
        _: String,
    ) -> BridgeFuture<Result<RpcResult, proto::ErrorResponse>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let result_json = self.verdict.to_string();
        Box::pin(async move { Ok(RpcResult { result_json }) })
    }
    fn handle_notification(&self, _: String, _: String, _: String) -> BridgeFuture<()> {
        Box::pin(async {})
    }
}

#[tokio::test]
async fn verification_calls_the_verifier_once_per_current_verdict() {
    for (verdict, accepted) in [
        (
            json!({"accepted": true, "valid_until": unix_now() + 300}),
            true,
        ),
        (
            json!({"accepted": false, "valid_until": unix_now() + 300, "reason": "no proof"}),
            false,
        ),
        (json!({"unexpected": true}), false),
    ] {
        let calls = Arc::new(AtomicUsize::new(0));
        let manager = PluginManager::for_test_peer_evidence(
            Arc::new(CountingVerifier {
                calls: calls.clone(),
                verdict,
            }),
            None,
        );
        for _ in 0..2 {
            assert_eq!(
                manager
                    .verify_provider_evidence("mesh", "peer", &published("test.bond.v1"))
                    .await
                    .accepted,
                accepted
            );
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn providers_without_evidence_are_rejected_without_asking() {
    let calls = Arc::new(AtomicUsize::new(0));
    let verdict = json!({"accepted": true, "valid_until": unix_now() + 300});
    let manager = PluginManager::for_test_peer_evidence(
        Arc::new(CountingVerifier {
            calls: calls.clone(),
            verdict,
        }),
        None,
    );
    let registry = manager.peer_evidence();
    let none = BTreeMap::new();
    assert!(
        !manager
            .verify_provider_evidence("mesh", "peer", &none)
            .await
            .accepted
    );
    assert!(!registry.needs_verification("mesh", "peer", &none));
    assert_eq!(
        registry.provider_decision("mesh", "peer", &none),
        Some(false)
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn due_acceptances_are_verified_again() {
    let calls = Arc::new(AtomicUsize::new(0));
    let verdict = json!({"accepted": true, "valid_until": unix_now() + 300});
    let manager = PluginManager::for_test_peer_evidence(
        Arc::new(CountingVerifier {
            calls: calls.clone(),
            verdict,
        }),
        None,
    );
    let evidence = published("test.bond.v1");
    let verify = async || {
        manager
            .verify_provider_evidence("mesh", "peer", &evidence)
            .await
            .accepted
    };
    assert!(verify().await);
    for verdict in lock(&manager.peer_evidence().verdicts).values_mut() {
        verdict.renew = Instant::now();
    }
    assert!(verify().await);
    assert!(verify().await);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

async fn verifier_spec(
    plugin: impl mesh_llm_plugin::Plugin + Clone + Sync + 'static,
) -> ExternalPluginSpec {
    let name = plugin.plugin_id().to_owned();
    let metadata: mesh_llm_plugin_manager::InstalledPluginMetadata =
        serde_json::from_value(json!({
            "name": name,
            "source_repository": "local:fixture",
            "installed_version": "0.0.0",
            "target_triple": "fixture",
            "downloaded_asset_name": "fixture.tar.gz",
            "install_path": "unused-test-package",
            "enabled": true,
        }))
        .unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mut spec = crate::plugin::config::in_process_builtin_spec(&name);
    spec.command = metadata.executable_path().to_str().unwrap().into();
    spec.url = Some(format!("test+tcp://{}", listener.local_addr().unwrap()));
    spec.startup.optional = false;
    spec.installed_metadata = Some(metadata);
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let (mut proxy, sdk) = tokio::io::duplex(64 * 1024);
        tokio::select! {
            _ = tokio::io::copy_bidirectional(&mut socket, &mut proxy) => {},
            _ = PluginRuntime::run_with_stream(plugin, mesh_llm_plugin::LocalStream::Memory(sdk)) => {},
        }
    });
    spec
}

fn verifier_manifest(namespace: &str) -> proto::PluginManifest {
    mesh_llm_plugin::plugin_manifest![
        declaration(namespace),
        mesh_llm_plugin::operation::<PeerEvidenceVerificationRequest>("verify", "Verify")
    ]
}

async fn start(spec: ExternalPluginSpec) -> Result<PluginManager> {
    let (tx, _rx) = tokio::sync::mpsc::channel(32);
    PluginManager::start(
        &crate::plugin::config::ResolvedPlugins {
            externals: vec![spec],
            inactive: vec![],
        },
        crate::plugin::PluginHostMode {
            mesh_visibility: mesh_llm_plugin::MeshVisibility::Private,
        },
        tx,
    )
    .await
}

#[test]
fn configured_verifier_cannot_start_lazily() {
    let manifest = verifier_manifest("proof");
    let mut spec = crate::plugin::config::in_process_builtin_spec("verifier");
    assert!(validate_startup(&spec, Some(&manifest)).unwrap().is_some());
    spec.startup.lazy_start = true;
    assert!(validate_startup(&spec, Some(&manifest)).is_err());
    assert!(validate_startup(&spec, None).unwrap().is_none());
}

/// Answers every verification with an error-marked acceptance.
#[derive(Clone)]
struct ErrorVerdict;

#[mesh_llm_plugin::async_trait]
impl mesh_llm_plugin::Plugin for ErrorVerdict {
    fn plugin_id(&self) -> &str {
        "verifier"
    }
    fn plugin_version(&self) -> String {
        "0.0.0".into()
    }
    fn server_info(&self) -> rmcp::model::ServerConfig {
        plugin_server_info("verifier", "0.0.0", "Verifier", "Fixture", None::<String>)
    }
    fn manifest(&self) -> Option<proto::PluginManifest> {
        Some(verifier_manifest("proof"))
    }
    async fn invoke_service(
        &mut self,
        _: proto::InvokeServiceRequest,
        _: &mut mesh_llm_plugin::PluginContext<'_>,
    ) -> mesh_llm_plugin::PluginResult<Option<proto::InvokeServiceResponse>> {
        Ok(Some(proto::InvokeServiceResponse {
            output_json: json!({"accepted": true, "valid_until": u64::MAX}).to_string(),
            is_error: true,
        }))
    }
}

#[tokio::test]
async fn verifier_errors_are_rejections() {
    let manager = start(verifier_spec(ErrorVerdict).await).await.unwrap();
    assert!(manager.peer_evidence().has_verifier());
    let evidence = published("proof");
    assert!(
        !manager
            .verify_provider_evidence("mesh", "peer", &evidence)
            .await
            .accepted
    );
    assert_eq!(
        manager
            .peer_evidence()
            .provider_decision("mesh", "peer", &evidence),
        Some(false)
    );
    manager.shutdown().await;
    assert!(
        manager
            .peer_evidence()
            .provider_decision("mesh", "peer", &evidence)
            .is_none()
    );
}

#[test]
fn verifier_declarations_need_authentication_and_survive_restarts() {
    let proof = declaration("proof");
    let first = std::sync::OnceLock::new();
    assert!(validate_generation(Some(&proof), false, &first).is_err());
    assert!(validate_generation(Some(&proof), true, &first).is_ok());
    assert!(validate_generation(Some(&proof), true, &first).is_ok());
    assert!(validate_generation(Some(&declaration("other")), true, &first).is_err());
    assert!(validate_generation(None, true, &first).is_err());

    // A plugin that started without a verifier cannot add one later.
    let first = std::sync::OnceLock::new();
    assert!(validate_generation(None, false, &first).is_ok());
    assert!(validate_generation(Some(&proof), true, &first).is_err());
}

#[test]
fn a_closed_connection_withdraws_only_its_own_publication() {
    let mut registry = EvidenceRegistry::default();
    registry
        .register("verifier", &declaration("proof"))
        .unwrap();
    assert!(registry.replace_publication(2, Some(b"{}".to_vec())));
    // Republishing the same evidence changes nothing peers see.
    assert!(!registry.replace_publication(3, Some(b"{}".to_vec())));

    // A notice from an older connection arriving late leaves it in place.
    registry.withdraw_generation("verifier", 2);
    assert!(!registry.publications().is_empty());
    registry.withdraw_generation("other", 3);
    assert!(!registry.publications().is_empty());
    registry.withdraw_generation("verifier", 3);
    assert!(registry.publications().is_empty());
}

#[test]
fn only_an_acceptance_after_none_makes_a_provider_newly_eligible() {
    let mut registry = EvidenceRegistry::default();
    registry.register("v", &declaration("bond")).unwrap();
    let key = VerdictKey::new("mesh", "host", b"reference");
    let commit = |result| registry.commit(key.clone(), 0, result);
    assert_eq!(
        commit(accepted_for(300)),
        Verification {
            accepted: true,
            newly_eligible: true
        }
    );
    // A renewal of a current acceptance changes nothing.
    assert!(!commit(accepted_for(300)).newly_eligible);
    // An acceptance that lapsed during the call is new again.
    lock(&registry.verdicts).get_mut(&key).unwrap().expires = Instant::now();
    assert!(commit(accepted_for(300)).newly_eligible);
    let rejected = serde_json::from_value(json!({"accepted": false}));
    assert_eq!(commit(Ok(rejected.unwrap())), Verification::default());
    assert!(commit(accepted_for(300)).newly_eligible);
}
