use super::*;
use crate::plugin::{BridgeFuture, PluginRpcBridge, RpcResult};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};

struct Verifier {
    calls: Arc<AtomicUsize>,
    expiry: Arc<AtomicU64>,
    accepted: Arc<AtomicBool>,
}

struct UnresponsiveVerifier(Arc<AtomicUsize>);
impl PluginRpcBridge for UnresponsiveVerifier {
    fn handle_request(
        &self,
        _: String,
        _: String,
        _: String,
    ) -> BridgeFuture<Result<RpcResult, proto::ErrorResponse>> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Box::pin(std::future::pending())
    }
    fn handle_notification(&self, _: String, _: String, _: String) -> BridgeFuture<()> {
        Box::pin(async {})
    }
}

#[tokio::test]
async fn peer_evidence_shared_refresh_deadline_caches_a_finite_timeout_rejection() {
    let calls = Arc::new(AtomicUsize::new(0));
    let manager =
        PluginManager::for_test_peer_evidence(Arc::new(UnresponsiveVerifier(calls.clone())), None);
    let evidence = BTreeMap::new();
    assert!(
        !manager
            .verify_provider_evidence_until(
                "mesh",
                "peer",
                &evidence,
                tokio::time::Instant::now() + Duration::from_millis(20)
            )
            .await
    );
    assert_eq!(
        manager
            .peer_evidence()
            .provider_decision("mesh", "peer", &evidence),
        Some(false)
    );
    assert!(
        !manager
            .verify_provider_evidence("mesh", "peer", &evidence)
            .await
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let entry = manager
        .peer_evidence()
        .cached(&EvidenceRegistry::lease_key(
            "evidence", "mesh", "peer", None,
        ))
        .unwrap();
    assert!(!entry.cache_until_restart);
    assert!(entry.valid_until > unix_seconds());
}

struct ControlledVerifier {
    calls: Arc<AtomicUsize>,
    entered: Arc<tokio::sync::Notify>,
    release: Arc<tokio::sync::Notify>,
}
impl PluginRpcBridge for ControlledVerifier {
    fn handle_request(
        &self,
        _: String,
        _: String,
        _: String,
    ) -> BridgeFuture<Result<RpcResult, proto::ErrorResponse>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let entered = self.entered.clone();
        let release = self.release.clone();
        Box::pin(async move {
            entered.notify_one();
            release.notified().await;
            Ok(RpcResult {
                result_json: json!({"accepted":true,"cache_until_restart":true}).to_string(),
            })
        })
    }
    fn handle_notification(&self, _: String, _: String, _: String) -> BridgeFuture<()> {
        Box::pin(async {})
    }
}

#[tokio::test]
async fn peer_evidence_timed_out_or_cancelled_callers_keep_the_actual_invocation_serialized() {
    for cancel in [false, true] {
        let calls = Arc::new(AtomicUsize::new(0));
        let entered = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let manager = PluginManager::for_test_peer_evidence(
            Arc::new(ControlledVerifier {
                calls: calls.clone(),
                entered: entered.clone(),
                release: release.clone(),
            }),
            None,
        );
        let caller = {
            let manager = manager.clone();
            tokio::spawn(async move {
                manager
                    .verify_provider_evidence_until(
                        "mesh",
                        "peer",
                        &BTreeMap::new(),
                        tokio::time::Instant::now() + Duration::from_millis(20),
                    )
                    .await
            })
        };
        entered.notified().await;
        if cancel {
            caller.abort();
            assert!(caller.await.unwrap_err().is_cancelled());
        } else {
            assert!(!caller.await.unwrap());
        }
        manager.expire_peer_evidence_for_test();
        assert!(
            !manager
                .verify_provider_evidence_until(
                    "mesh",
                    "peer",
                    &BTreeMap::new(),
                    tokio::time::Instant::now() + Duration::from_millis(20),
                )
                .await
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        manager.expire_peer_evidence_for_test();
        let waiter = {
            let manager = manager.clone();
            tokio::spawn(async move {
                manager
                    .verify_provider_evidence("mesh", "peer", &BTreeMap::new())
                    .await
            })
        };
        release.notify_one();
        assert!(waiter.await.unwrap());
        assert!(
            manager
                .peer_evidence()
                .provider_cached("mesh", "peer", &BTreeMap::new())
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}

impl PluginRpcBridge for Verifier {
    fn handle_request(
        &self,
        _: String,
        _: String,
        _: String,
    ) -> BridgeFuture<Result<RpcResult, proto::ErrorResponse>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let until = self.expiry.load(Ordering::SeqCst);
        let accepted = self.accepted.load(Ordering::SeqCst);
        Box::pin(async move {
            tokio::task::yield_now().await;
            Ok(RpcResult {
                result_json: json!({"accepted":accepted,"valid_until":until}).to_string(),
            })
        })
    }
    fn handle_notification(&self, _: String, _: String, _: String) -> BridgeFuture<()> {
        Box::pin(async {})
    }
}

#[tokio::test]
async fn missing_evidence_uses_the_verifier_and_shares_the_existing_cache() {
    let calls = Arc::new(AtomicUsize::new(0));
    let accepted = Arc::new(AtomicBool::new(true));
    let manager = PluginManager::for_test_peer_evidence(
        Arc::new(Verifier {
            calls: calls.clone(),
            expiry: Arc::new(AtomicU64::new(unix_seconds() + 86400)),
            accepted: accepted.clone(),
        }),
        None,
    );
    let missing = BTreeMap::new();
    for _ in 0..3 {
        assert!(
            manager
                .verify_provider_evidence("mesh", "peer", &missing)
                .await
        );
        assert!(
            manager
                .peer_evidence()
                .provider_cached("mesh", "peer", &missing)
        );
    }
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let published = BTreeMap::from([("test.bond.v1".into(), b"{}".to_vec())]);
    assert!(
        !manager
            .peer_evidence()
            .provider_cached("mesh", "peer", &published)
    );
    assert!(
        manager
            .verify_provider_evidence("mesh", "peer", &published)
            .await
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    accepted.store(false, Ordering::SeqCst);
    manager.expire_peer_evidence_for_test();
    for _ in 0..3 {
        assert!(
            !manager
                .verify_provider_evidence("mesh", "peer", &missing)
                .await
        );
    }
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn verifier_restarts_invalidate_only_its_own_cached_decisions() {
    let calls = Arc::new(AtomicUsize::new(0));
    let manager = PluginManager::for_test_peer_evidence(
        Arc::new(Verifier {
            calls: calls.clone(),
            expiry: Arc::new(AtomicU64::new(unix_seconds() + 86400)),
            accepted: Arc::new(AtomicBool::new(true)),
        }),
        None,
    );
    let missing = BTreeMap::new();
    assert!(
        manager
            .verify_provider_evidence("mesh", "peer", &missing)
            .await
    );
    manager
        .refresh_peer_evidence_verifier("ordinary", 0)
        .await
        .unwrap();
    assert!(
        manager
            .verify_provider_evidence("mesh", "peer", &missing)
            .await
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    manager
        .refresh_peer_evidence_verifier("evidence", 0)
        .await
        .unwrap();
    assert!(
        manager
            .verify_provider_evidence("mesh", "peer", &missing)
            .await
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn disconnection_invalidates_only_the_current_verifier_generation() {
    let mut manager = PluginManager::for_test_peer_evidence(
        Arc::new(Verifier {
            calls: Arc::default(),
            expiry: Arc::new(AtomicU64::new(unix_seconds() + 86400)),
            accepted: Arc::new(AtomicBool::new(true)),
        }),
        None,
    );
    let missing = BTreeMap::new();
    assert!(
        manager
            .verify_provider_evidence("mesh", "peer", &missing)
            .await
    );
    let plugin = super::discovery_tests::plugin("evidence", Some("test.bond.v1"));
    plugin.manifest().await.unwrap();
    let generation = plugin.bound_generation().await.unwrap();
    Arc::get_mut(&mut manager.inner)
        .unwrap()
        .plugins
        .insert("evidence".into(), plugin);
    manager
        .peer_evidence_disconnected("evidence", generation + 1)
        .await;
    assert!(
        manager
            .peer_evidence()
            .provider_cached("mesh", "peer", &missing)
    );
    manager
        .peer_evidence_disconnected("evidence", generation)
        .await;
    assert!(
        !manager
            .peer_evidence()
            .provider_cached("mesh", "peer", &missing)
    );
    manager.shutdown().await;
}

#[tokio::test]
async fn concurrent_misses_and_repeated_rejections_share_one_verification() {
    let calls = Arc::new(AtomicUsize::new(0));
    let accepted = Arc::new(AtomicBool::new(false));
    let manager = PluginManager::for_test_peer_evidence(
        Arc::new(Verifier {
            calls: calls.clone(),
            expiry: Arc::new(AtomicU64::new(unix_seconds() + 86400)),
            accepted: accepted.clone(),
        }),
        None,
    );
    let evidence = BTreeMap::from([("test.bond.v1".into(), b"{}".to_vec())]);
    let (first, concurrent) = tokio::join!(
        manager.verify_provider_evidence("mesh", "peer", &evidence),
        manager.verify_provider_evidence("mesh", "peer", &evidence)
    );
    assert!(!first && !concurrent);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    accepted.store(true, Ordering::SeqCst);
    assert!(
        !manager
            .verify_provider_evidence("mesh", "peer", &evidence)
            .await
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "a cached rejection must not poll again early"
    );
    manager.expire_peer_evidence_for_test();
    assert!(
        manager
            .verify_provider_evidence("mesh", "peer", &evidence)
            .await
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn stopping_a_plugin_preserves_other_verifiers_cached_decisions() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut manager = PluginManager::for_test_bridge(
        &["first", "second"],
        Arc::new(Verifier {
            calls: calls.clone(),
            expiry: Arc::new(AtomicU64::new(unix_seconds() + 86400)),
            accepted: Arc::new(AtomicBool::new(true)),
        }),
    );
    for name in ["first", "second"] {
        let manifest = mesh_llm_plugin::plugin_manifest![
            peer_evidence::peer_evidence_verifier(format!("test.{name}"), "verify"),
            mesh_llm_plugin::operation::<PeerEvidenceVerificationRequest>(
                "verify",
                "Verify evidence"
            )
        ];
        Arc::get_mut(&mut manager.inner)
            .unwrap()
            .peer_evidence
            .register(name, &manifest)
            .unwrap();
        manager
            .inner
            .test_manifests
            .lock()
            .await
            .insert(name.into(), manifest);
    }
    let evidence = BTreeMap::from([
        ("test.first".into(), b"{}".to_vec()),
        ("test.second".into(), b"{}".to_vec()),
    ]);
    assert!(
        manager
            .verify_provider_evidence("mesh", "peer", &evidence)
            .await
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    manager.peer_evidence().clear_plugin("unrelated");
    assert!(
        manager
            .peer_evidence()
            .provider_cached("mesh", "peer", &evidence)
    );
    assert!(
        manager
            .verify_provider_evidence("mesh", "peer", &evidence)
            .await
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    for (name, count) in [("first", 3), ("second", 4)] {
        manager.peer_evidence().clear_plugin(name);
        assert!(
            !manager
                .peer_evidence()
                .provider_cached("mesh", "peer", &evidence)
        );
        assert!(
            manager
                .verify_provider_evidence("mesh", "peer", &evidence)
                .await
        );
        assert_eq!(calls.load(Ordering::SeqCst), count);
    }
}

#[tokio::test]
async fn a_restarted_verifier_must_keep_its_declared_contract() {
    let calls = Arc::new(AtomicUsize::new(0));
    let manager = PluginManager::for_test_peer_evidence(
        Arc::new(Verifier {
            calls: calls.clone(),
            expiry: Arc::new(AtomicU64::new(unix_seconds() + 86400)),
            accepted: Arc::new(AtomicBool::new(true)),
        }),
        None,
    );
    let evidence = BTreeMap::from([("test.bond.v1".into(), b"{}".to_vec())]);
    assert!(
        manager
            .verify_provider_evidence("mesh", "peer", &evidence)
            .await
    );
    let original = manager.inner.test_manifests.lock().await["evidence"].clone();
    manager.peer_evidence().clear_plugin("evidence");
    for mutation in 0..4 {
        let mut changed = original.clone();
        match mutation {
            0 => changed.peer_evidence_verifier = None,
            1 => {
                changed
                    .peer_evidence_verifier
                    .as_mut()
                    .unwrap()
                    .contract_version = 2
            }
            2 => changed.peer_evidence_verifier.as_mut().unwrap().namespace = "other".into(),
            _ => changed.peer_evidence_verifier.as_mut().unwrap().handler = "other".into(),
        }
        manager
            .inner
            .test_manifests
            .lock()
            .await
            .insert("evidence".into(), changed);
        assert!(
            !manager
                .verify_provider_evidence("mesh", "peer", &evidence)
                .await
        );
    }
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    manager
        .inner
        .test_manifests
        .lock()
        .await
        .insert("evidence".into(), original);
    assert!(
        manager
            .verify_provider_evidence("mesh", "peer", &evidence)
            .await
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn verifier_expiry_is_honored_without_a_core_cap_and_cached() {
    let calls = Arc::new(AtomicUsize::new(0));
    let expiry = Arc::new(AtomicU64::new(unix_seconds() + 86400));
    let manager = PluginManager::for_test_peer_evidence(
        Arc::new(Verifier {
            calls: calls.clone(),
            expiry: expiry.clone(),
            accepted: Arc::new(AtomicBool::new(true)),
        }),
        None,
    );
    let bytes = serde_json::to_vec(&json!({"bond":"ref"})).unwrap();
    let evidence = BTreeMap::from([("test.bond.v1".into(), bytes.clone())]);
    let key = EvidenceRegistry::lease_key("evidence", "mesh", "host", Some(&bytes));
    assert!(
        manager
            .verify_provider_evidence("mesh", "host", &evidence)
            .await
    );
    let fresh = manager.peer_evidence().leases.lock().unwrap()[&key];
    assert_eq!(fresh.valid_until, expiry.load(Ordering::SeqCst));
    assert!(
        manager
            .verify_provider_evidence("mesh", "host", &evidence)
            .await
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    manager.expire_peer_evidence_for_test();
    let due = manager.peer_evidence().leases.lock().unwrap()[&key].deadline;
    EvidenceRegistry::insert(
        &mut manager.peer_evidence().leases.lock().unwrap(),
        EvidenceRegistry::lease_key("evidence", "mesh", "another-host", Some(&bytes)),
        fresh,
    );
    let providers = vec![
        ("mesh".into(), "host".into(), evidence.clone()),
        ("mesh".into(), "another-host".into(), evidence.clone()),
    ];
    assert!(!manager.peer_evidence().providers_cached(&providers));
    let before = due - Duration::from_secs(1);
    for _ in 0..2 {
        manager.peer_evidence().cache_changed.notify_one();
        manager.peer_evidence().cache_changed.notified().await;
        assert_eq!(
            manager.peer_evidence().next_refresh_deadline(before),
            Some(due),
            "notifications must not hide a deadline before its refresh is queued"
        );
    }
    assert_eq!(
        manager.peer_evidence().next_refresh_deadline(due),
        Some(fresh.deadline)
    );
    expiry.store(unix_seconds() + 45, Ordering::SeqCst);
    assert!(
        manager
            .verify_provider_evidence("mesh", "host", &evidence)
            .await
    );
    let until = manager.peer_evidence().leases.lock().unwrap()[&key].valid_until;
    assert_eq!(until, expiry.load(Ordering::SeqCst));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(manager.peer_evidence().providers_cached(&providers));
    for (invalid_deadline, expected_calls) in [(0, 3), (u64::MAX, 4)] {
        manager.expire_peer_evidence_for_test();
        expiry.store(invalid_deadline, Ordering::SeqCst);
        for _ in 0..2 {
            assert!(
                !manager
                    .verify_provider_evidence("mesh", "host", &evidence)
                    .await
            );
        }
        let denial = manager.peer_evidence().cached(&key).unwrap();
        assert!(!denial.accepted);
        assert!(!denial.cache_until_restart);
        assert_eq!(calls.load(Ordering::SeqCst), expected_calls);
    }
}

#[tokio::test]
async fn policy_only_verdicts_survive_time_expiry_but_not_evidence_changes_or_restart() {
    struct Policy(Arc<AtomicUsize>, bool);
    impl PluginRpcBridge for Policy {
        fn handle_request(
            &self,
            _: String,
            _: String,
            _: String,
        ) -> BridgeFuture<Result<RpcResult, proto::ErrorResponse>> {
            self.0.fetch_add(1, Ordering::SeqCst);
            let accepted = self.1;
            Box::pin(async move {
                Ok(RpcResult {
                    result_json: json!({"accepted":accepted,"cache_until_restart":true})
                        .to_string(),
                })
            })
        }
        fn handle_notification(&self, _: String, _: String, _: String) -> BridgeFuture<()> {
            Box::pin(async {})
        }
    }
    for accepted in [true, false] {
        let calls = Arc::new(AtomicUsize::new(0));
        let manager =
            PluginManager::for_test_peer_evidence(Arc::new(Policy(calls.clone(), accepted)), None);
        let empty = BTreeMap::new();
        assert_eq!(
            manager
                .verify_provider_evidence("mesh", "peer", &empty)
                .await,
            accepted
        );
        manager.expire_peer_evidence_for_test();
        assert_eq!(
            manager
                .verify_provider_evidence("mesh", "peer", &empty)
                .await,
            accepted
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let changed = BTreeMap::from([("test.bond.v1".into(), b"{}".to_vec())]);
        assert_eq!(
            manager
                .verify_provider_evidence("mesh", "peer", &changed)
                .await,
            accepted
        );
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        manager.peer_evidence().clear_plugin("evidence");
        assert_eq!(
            manager
                .verify_provider_evidence("mesh", "peer", &empty)
                .await,
            accepted
        );
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }
}

#[test]
fn full_cache_evicts_rejections_and_never_rejects_an_accepted_provider() {
    let mut entries = HashMap::new();
    let entry = Eligibility {
        accepted: false,
        valid_until: 0,
        deadline: Instant::now(),
        cache_until_restart: true,
        last_used: Instant::now(),
        runtime_generation: 0,
    };
    for n in 0..MAX_LEASES {
        EvidenceRegistry::insert(
            &mut entries,
            EvidenceRegistry::lease_key("p", "mesh", &n.to_string(), None),
            entry,
        );
    }
    let key = EvidenceRegistry::lease_key("p", "mesh", "accepted", None);
    EvidenceRegistry::insert(
        &mut entries,
        key.clone(),
        Eligibility {
            accepted: true,
            ..entry
        },
    );
    assert_eq!(entries.len(), MAX_LEASES);
    assert!(entries[&key].accepted);
    for n in 0..MAX_LEASES {
        EvidenceRegistry::insert(
            &mut entries,
            EvidenceRegistry::lease_key("p", "mesh", &format!("new-{n}"), None),
            entry,
        );
    }
    assert_eq!(entries.len(), MAX_LEASES);
    assert!(entries[&key].accepted);
    let expired = entries.get_mut(&key).unwrap();
    expired.cache_until_restart = false;
    expired.deadline = Instant::now() - Duration::from_secs(1);
    let rejected = EvidenceRegistry::lease_key("p", "mesh", "after-expiry", None);
    EvidenceRegistry::insert(&mut entries, rejected.clone(), entry);
    assert_eq!(entries.len(), MAX_LEASES);
    assert!(entries.contains_key(&rejected));
    assert!(!entries.contains_key(&key));
}

#[tokio::test]
async fn transient_verifier_failures_retry_after_a_finite_deadline() {
    struct Failing(Arc<AtomicUsize>);
    impl PluginRpcBridge for Failing {
        fn handle_request(
            &self,
            _: String,
            _: String,
            _: String,
        ) -> BridgeFuture<Result<RpcResult, proto::ErrorResponse>> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Box::pin(async {
                Err(proto::ErrorResponse {
                    code: -1,
                    message: "temporarily unavailable".into(),
                    data_json: String::new(),
                })
            })
        }
        fn handle_notification(&self, _: String, _: String, _: String) -> BridgeFuture<()> {
            Box::pin(async {})
        }
    }
    let calls = Arc::new(AtomicUsize::new(0));
    let manager = PluginManager::for_test_peer_evidence(Arc::new(Failing(calls.clone())), None);
    for _ in 0..2 {
        assert!(
            !manager
                .verify_provider_evidence("mesh", "peer", &BTreeMap::new())
                .await
        );
    }
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(
        !manager
            .peer_evidence()
            .leases
            .lock()
            .unwrap()
            .values()
            .next()
            .unwrap()
            .cache_until_restart
    );
    manager.expire_peer_evidence_for_test();
    assert!(
        !manager
            .verify_provider_evidence("mesh", "peer", &BTreeMap::new())
            .await
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn malformed_evidence_caches_a_finite_rejection_without_invoking_the_verifier() {
    let calls = Arc::new(AtomicUsize::new(0));
    let manager = PluginManager::for_test_peer_evidence(
        Arc::new(Verifier {
            calls: calls.clone(),
            expiry: Arc::new(AtomicU64::new(unix_seconds() + 60)),
            accepted: Arc::new(AtomicBool::new(true)),
        }),
        None,
    );
    let bytes = b"not JSON";
    let evidence = BTreeMap::from([("test.bond.v1".into(), bytes.to_vec())]);
    for _ in 0..2 {
        assert!(
            !manager
                .verify_provider_evidence("mesh", "host", &evidence)
                .await
        );
    }
    let key = EvidenceRegistry::lease_key("evidence", "mesh", "host", Some(bytes));
    let denial = manager.peer_evidence().cached(&key).unwrap();
    assert!(!denial.accepted);
    assert!(!denial.cache_until_restart);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        manager
            .peer_evidence()
            .provider_decision("mesh", "host", &Default::default()),
        None,
    );
}

#[test]
fn publication_generations_handle_replacement_before_delayed_disconnect() {
    let mut registry = EvidenceRegistry::default();
    registry
        .register(
            "p",
            &mesh_llm_plugin::plugin_manifest![
                peer_evidence::peer_evidence_verifier("proof", "verify"),
                mesh_llm_plugin::operation::<Value>("verify", "Verify")
            ],
        )
        .unwrap();
    registry
        .publish("p", 1, Some(1), Some(b"old".to_vec()), usize::MAX)
        .unwrap();
    registry.retain_publication_generation("p", Some(2));
    assert!(registry.publications().is_empty());
    registry
        .publish("p", 2, Some(2), Some(b"new".to_vec()), usize::MAX)
        .unwrap();
    registry.withdraw_publication_generation("p", 1);
    assert_eq!(registry.publications()["proof"], b"new".to_vec());
    registry.withdraw_publication_generation("p", 2);
    assert!(registry.publications().is_empty());
}

#[tokio::test]
async fn verifier_runtime_removal_immediately_hides_cached_verdicts_and_publications() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut manager = PluginManager::for_test_peer_evidence(
        Arc::new(Verifier {
            calls: calls.clone(),
            expiry: Arc::new(AtomicU64::new(unix_seconds() + 86400)),
            accepted: Arc::new(AtomicBool::new(true)),
        }),
        None,
    );
    let active = Arc::new(AtomicU64::new(1));
    Arc::get_mut(&mut manager.inner)
        .unwrap()
        .peer_evidence
        .verifiers
        .get_mut("evidence")
        .unwrap()
        .active_generation = Some(active.clone());
    let empty = BTreeMap::new();
    assert!(
        manager
            .verify_provider_evidence("mesh", "peer", &empty)
            .await
    );
    manager
        .peer_evidence()
        .publish("evidence", 1, Some(1), Some(b"{}".to_vec()), usize::MAX)
        .unwrap();
    assert!(
        manager
            .peer_evidence()
            .provider_cached("mesh", "peer", &empty)
    );
    assert_eq!(manager.peer_evidence().publications().len(), 1);
    active.store(0, Ordering::Release);
    assert!(
        !manager
            .peer_evidence()
            .provider_cached("mesh", "peer", &empty)
    );
    assert!(manager.peer_evidence().publications().is_empty());
    assert!(
        manager
            .peer_evidence()
            .publish("evidence", 2, Some(2), Some(b"{}".to_vec()), usize::MAX)
            .is_err()
    );
    active.store(2, Ordering::Release);
    assert!(
        !manager
            .peer_evidence()
            .provider_cached("mesh", "peer", &empty)
    );
    assert!(manager.peer_evidence().publications().is_empty());
    assert!(
        manager
            .verify_provider_evidence("mesh", "peer", &empty)
            .await
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn a_new_verifier_declaration_after_startup_requires_a_mesh_restart() {
    let mut manager = PluginManager::for_test_peer_evidence(
        Arc::new(Verifier {
            calls: Arc::default(),
            expiry: Arc::new(AtomicU64::new(unix_seconds() + 86400)),
            accepted: Arc::new(AtomicBool::new(true)),
        }),
        None,
    );
    Arc::get_mut(&mut manager.inner)
        .unwrap()
        .bridged_plugins
        .insert("ordinary".into());
    manager
        .refresh_peer_evidence_verifier("ordinary", 1)
        .await
        .unwrap();
    manager.inner.test_manifests.lock().await.insert(
        "ordinary".into(),
        mesh_llm_plugin::plugin_manifest![
            peer_evidence::peer_evidence_verifier("new.proof", "verify"),
            mesh_llm_plugin::operation::<PeerEvidenceVerificationRequest>("verify", "Verify"),
        ],
    );
    assert!(
        manager
            .refresh_peer_evidence_verifier("ordinary", 2)
            .await
            .unwrap_err()
            .to_string()
            .contains("restart Mesh")
    );
}
