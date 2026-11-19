use super::*;
use crate::plugin::{ExternalPlugin, InProcessPluginRunner};
use mesh_llm_plugin::{PluginMetadata, PluginRuntime, SimplePlugin, plugin_server_info};
use std::sync::Arc;

#[derive(Clone)]
struct ErrorVerdict(Arc<std::sync::atomic::AtomicUsize>);
#[mesh_llm_plugin::async_trait]
impl mesh_llm_plugin::Plugin for ErrorVerdict {
    fn plugin_id(&self) -> &str {
        "error-verifier"
    }
    fn plugin_version(&self) -> String {
        "0.0.0".into()
    }
    fn server_info(&self) -> rmcp::model::ServerConfig {
        plugin_server_info(
            "error-verifier",
            "0.0.0",
            "Verifier",
            "Fixture",
            None::<String>,
        )
    }
    fn manifest(&self) -> Option<proto::PluginManifest> {
        Some(mesh_llm_plugin::plugin_manifest![
            peer_evidence::peer_evidence_verifier("test.error", "verify"),
            mesh_llm_plugin::operation::<PeerEvidenceVerificationRequest>("verify", "Verify")
        ])
    }
    async fn invoke_service(
        &mut self,
        _: proto::InvokeServiceRequest,
        _: &mut mesh_llm_plugin::PluginContext<'_>,
    ) -> mesh_llm_plugin::PluginResult<Option<proto::InvokeServiceResponse>> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(Some(proto::InvokeServiceResponse {
            output_json: json!({"accepted":true,"cache_until_restart":true}).to_string(),
            is_error: true,
        }))
    }
}

#[tokio::test]
async fn peer_evidence_error_marked_output_is_a_finite_cached_rejection() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let fixture = ErrorVerdict(calls.clone());
    let runner: InProcessPluginRunner =
        Arc::new(move |stream| Box::pin(PluginRuntime::run_with_stream(fixture.clone(), stream)));
    let (tx, _rx) = tokio::sync::mpsc::channel(32);
    let mut spec = crate::plugin::config::in_process_builtin_spec("error-verifier");
    spec.startup.optional = false;
    let manager = PluginManager::start_with_in_process(
        &crate::plugin::config::ResolvedPlugins {
            externals: vec![spec],
            inactive: vec![],
        },
        crate::plugin::PluginHostMode {
            mesh_visibility: mesh_llm_plugin::MeshVisibility::Private,
        },
        tx,
        crate::plugin::InProcessPlugins::default().with("error-verifier", runner),
    )
    .await
    .unwrap();
    let evidence = BTreeMap::new();
    for _ in 0..2 {
        assert!(
            !manager
                .verify_provider_evidence("mesh", "peer", &evidence)
                .await
        );
    }
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let entry = manager
        .peer_evidence()
        .cached(&EvidenceRegistry::lease_key(
            "error-verifier",
            "mesh",
            "peer",
            None,
        ))
        .unwrap();
    assert!(!entry.accepted && !entry.cache_until_restart);
    assert!(entry.valid_until > unix_seconds());
    manager.expire_peer_evidence_for_test();
    assert!(
        !manager
            .verify_provider_evidence("mesh", "peer", &evidence)
            .await
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    manager.shutdown().await;
}

pub(super) fn plugin(name: &str, namespace: Option<&str>) -> ExternalPlugin {
    let name = name.to_owned();
    let manifest = namespace.map_or_else(proto::PluginManifest::default, |namespace| {
        mesh_llm_plugin::plugin_manifest![
            peer_evidence::peer_evidence_verifier(namespace, "verify"),
            mesh_llm_plugin::operation::<PeerEvidenceVerificationRequest>("verify", "Verify")
        ]
    });
    let runner_name = name.clone();
    let runner: InProcessPluginRunner = Arc::new(move |stream| {
        let metadata = PluginMetadata::new(
            runner_name.clone(),
            "0.0.0",
            plugin_server_info("verifier", "0.0.0", "Verifier", "Fixture", None::<String>),
        )
        .with_manifest(manifest.clone());
        Box::pin(PluginRuntime::run_with_stream(
            SimplePlugin::new(metadata),
            stream,
        ))
    });
    crate::plugin::runtime::tests::in_process_plugin(&name, runner)
}

#[tokio::test]
async fn peer_evidence_is_discovered_from_manifests_without_permissions() {
    let mut ordinary = plugin("ordinary", None);
    ordinary.spec.startup.lazy_start = true;
    let mut plugins = BTreeMap::from([
        ("ordinary".into(), ordinary),
        ("verifier".into(), plugin("verifier", Some("proof"))),
    ]);
    plugins["verifier"].manifest().await.unwrap();
    let registry = EvidenceRegistry::from_plugins(&plugins).await.unwrap();
    assert!(registry.has_verifiers());
    assert_eq!(registry.verifiers.len(), 1);
    assert_eq!(
        registry.verifier("verifier").unwrap().declaration.namespace,
        "proof"
    );
    assert!(registry.verifier("ordinary").is_err());
    assert!(plugins["ordinary"].manifest_snapshot().await.is_none());

    plugins.insert("duplicate".into(), plugin("duplicate", Some("proof")));
    plugins["duplicate"].manifest().await.unwrap();
    assert!(EvidenceRegistry::from_plugins(&plugins).await.is_err());
    for plugin in plugins.values() {
        plugin.shutdown().await;
    }
}

#[tokio::test]
async fn peer_evidence_startup_allows_recovery_only_for_a_known_unrelated_manifest() {
    for namespace in [None, Some("proof")] {
        let plugin = plugin("stopped", namespace);
        plugin.manifest().await.unwrap();
        plugin.active_generation.store(0, Ordering::Release);
        let plugins = BTreeMap::from([("stopped".into(), plugin)]);
        assert_eq!(
            EvidenceRegistry::from_plugins(&plugins).await.is_ok(),
            namespace.is_none(),
        );
        plugins["stopped"].shutdown().await;
    }
    let plugins = BTreeMap::from([("unknown".into(), plugin("unknown", None))]);
    assert!(EvidenceRegistry::from_plugins(&plugins).await.is_err());
    plugins["unknown"].shutdown().await;
}
