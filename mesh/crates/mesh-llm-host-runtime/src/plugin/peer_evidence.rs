//! Authenticated publication under manifest-owned, versioned verifier declarations.
use super::{PluginManager, proto};
use anyhow::{Context, Result, ensure};
use mesh_llm_plugin::peer_evidence;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashMap},
    sync::Mutex,
};
pub(crate) const MAX_EVIDENCE_BYTES: usize = peer_evidence::MAX_EVIDENCE_BYTES;

#[derive(Clone)]
struct Verifier {
    declaration: proto::PeerEvidenceVerifierManifest,
}
#[derive(Default)]
pub(crate) struct EvidenceRegistry {
    verifiers: BTreeMap<String, Verifier>,
    publications: Mutex<HashMap<String, Vec<u8>>>,
}

pub(super) fn require_verifier(
    manifest: Option<&proto::PluginManifest>,
) -> Result<&proto::PeerEvidenceVerifierManifest> {
    peer_evidence::validate_manifest(manifest.context("peer evidence requires a plugin manifest")?)?
        .context("plugin has no peer evidence verifier declaration")
}
impl EvidenceRegistry {
    pub(super) async fn from_plugins(
        plugins: &BTreeMap<String, super::ExternalPlugin>,
    ) -> Result<Self> {
        let mut registry = Self::default();
        for (name, plugin) in plugins {
            if let Some(manifest) = plugin.manifest().await?
                && manifest.peer_evidence_verifier.is_some()
            {
                registry.register(name, &manifest)?;
            }
        }
        ensure!(
            registry.verifiers.is_empty(),
            "peer evidence verification is not supported by this host revision"
        );
        Ok(registry)
    }
    fn register(&mut self, plugin: &str, manifest: &proto::PluginManifest) -> Result<()> {
        let declaration = require_verifier(Some(manifest))?;
        ensure!(self.verifiers.len() < 8, "too many peer evidence verifiers");
        ensure!(
            !self.verifiers.contains_key(plugin),
            "duplicate peer evidence verifier"
        );
        ensure!(
            !self
                .verifiers
                .values()
                .any(|g| g.declaration.namespace == declaration.namespace),
            "duplicate peer evidence namespace"
        );
        self.verifiers.insert(
            plugin.into(),
            Verifier {
                declaration: declaration.clone(),
            },
        );
        Ok(())
    }
    pub(crate) fn publications(&self) -> BTreeMap<String, Vec<u8>> {
        let publications = self.publications.lock().unwrap_or_else(|e| e.into_inner());
        self.verifiers
            .iter()
            .filter_map(|(name, verifier)| {
                publications
                    .get(name)
                    .map(|bytes| (verifier.declaration.namespace.clone(), bytes.clone()))
            })
            .collect()
    }
    pub(crate) fn clear_plugin(&self, name: &str) {
        self.publications
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(name);
    }
    fn verifier(&self, plugin: &str) -> Result<&Verifier> {
        self.verifiers
            .get(plugin)
            .context("plugin has no peer evidence verifier")
    }
}
impl PluginManager {
    pub(crate) fn peer_evidence(&self) -> &EvidenceRegistry {
        &self.inner.peer_evidence
    }
    async fn current_evidence_verifier(&self, plugin: &str) -> Result<Verifier> {
        let verifier = self.peer_evidence().verifier(plugin)?.clone();
        let manifest = self.manifest_snapshot(plugin).await;
        let declaration = require_verifier(manifest.as_ref())?;
        ensure!(
            *declaration == verifier.declaration,
            "peer evidence verifier declaration changed; restart Mesh"
        );
        Ok(verifier)
    }
}

pub(crate) fn is_method(method: &str) -> bool {
    method == "PublishPeerEvidence"
}

pub(crate) async fn handle_request(
    node: &crate::mesh::Node,
    plugin: &str,
    request: proto::RpcRequest,
) -> Result<proto::RpcResponse, proto::ErrorResponse> {
    handle(node, plugin, request)
        .await
        .map_err(|error| proto::ErrorResponse {
            code: -32602,
            message: error.to_string(),
            data_json: String::new(),
        })
}

async fn handle(
    node: &crate::mesh::Node,
    plugin: &str,
    request: proto::RpcRequest,
) -> Result<proto::RpcResponse> {
    ensure!(is_method(&request.method), "unknown peer evidence method");
    ensure!(
        request.params_json.len() <= MAX_EVIDENCE_BYTES + 256,
        "oversized peer evidence request"
    );
    let manager = node
        .plugin_manager()
        .await
        .context("plugin manager unavailable")?;
    let registry = manager.peer_evidence();
    manager.current_evidence_verifier(plugin).await?;
    let params: Value = serde_json::from_str(&request.params_json)?;
    let evidence = params.get("evidence").context("evidence missing")?;
    let bytes = serde_json::to_vec(evidence)?;
    ensure!(bytes.len() <= MAX_EVIDENCE_BYTES, "oversized evidence");
    let mut publications = registry
        .publications
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if evidence.is_null() {
        publications.remove(plugin);
    } else {
        publications.insert(plugin.into(), bytes);
    }
    let result = json!({"published":!evidence.is_null()});
    Ok(proto::RpcResponse {
        result_json: serde_json::to_string(&result)?,
    })
}

#[cfg(test)]
mod publication_tests {
    use super::*;
    fn manifest(namespace: &str) -> proto::PluginManifest {
        mesh_llm_plugin::plugin_manifest![
            peer_evidence::peer_evidence_verifier(namespace, "verify"),
            mesh_llm_plugin::operation::<Value>("verify", "Verify evidence")
        ]
    }
    #[test]
    fn only_publication_is_an_evidence_host_rpc() {
        assert!(is_method("PublishPeerEvidence"));
        assert!(!is_method("ReadPeerEvidenceContext"));
        assert!(!is_method("GetPeerEvidence"));
    }

    #[test]
    fn verifier_declarations_are_unique_and_bounded() {
        assert!(require_verifier(None).is_err());
        assert!(require_verifier(Some(&proto::PluginManifest::default())).is_err());
        let mut registry = EvidenceRegistry::default();
        assert!(registry.verifier("ungranted").is_err());
        registry.register("a", &manifest("proof")).unwrap();
        assert!(registry.register("b", &manifest("proof")).is_err());
        let mut unsupported = manifest("other");
        unsupported
            .peer_evidence_verifier
            .as_mut()
            .unwrap()
            .contract_version = 2;
        assert!(registry.register("b", &unsupported).is_err());
        for namespace in [String::new(), "n".repeat(65)] {
            assert!(
                EvidenceRegistry::default()
                    .register("a", &manifest(&namespace))
                    .is_err()
            );
        }
        for index in 1..8 {
            registry
                .register(&index.to_string(), &manifest(&format!("proof.{index}")))
                .unwrap();
        }
        assert!(
            registry
                .register("ninth", &manifest("proof.ninth"))
                .is_err()
        );
    }
    #[test]
    fn publications_use_declared_namespaces_and_stop_clears_only_the_owner() {
        let mut registry = EvidenceRegistry::default();
        registry.register("a", &manifest("proof.a")).unwrap();
        registry.register("b", &manifest("proof.b")).unwrap();
        {
            let mut publications = registry.publications.lock().unwrap();
            publications.insert("a".into(), b"first".to_vec());
            publications.insert("b".into(), b"second".to_vec());
            publications.insert("ungranted".into(), b"hidden".to_vec());
        }
        assert_eq!(registry.publications().len(), 2);
        registry.clear_plugin("a");
        assert_eq!(
            registry.publications(),
            BTreeMap::from([("proof.b".into(), b"second".to_vec())])
        );
    }
}
