//! Manifest-discovered verification and a single provider eligibility cache.
use super::{PluginManager, config::ExternalPluginSpec, proto};
use anyhow::{Context, Result, ensure};
use mesh_llm_plugin::peer_evidence::{
    self, PeerEvidenceVerificationRequest, PeerEvidenceVerificationResult,
};
use prost::Message;
use serde_json::{Value, json};
use sha2::Digest;
use std::{
    collections::{BTreeMap, HashMap},
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::Semaphore;

pub(crate) const MAX_EVIDENCE_BYTES: usize = peer_evidence::MAX_EVIDENCE_BYTES;
pub(crate) const MANIFEST_ENFORCEMENT_CAPABILITY: &str = "peer_evidence.manifest_enforcement.v1";
const MAX_LEASES: usize = 4096;
pub(crate) type ProviderEvidence = (String, String, BTreeMap<String, Vec<u8>>);

#[cfg(test)]
#[path = "peer_evidence/discovery_tests.rs"]
mod discovery_tests;

mod verification;

#[derive(Clone, Copy)]
struct Eligibility {
    accepted: bool,
    valid_until: u64,
    deadline: Instant,
    cache_until_restart: bool,
    last_used: Instant,
    runtime_generation: u64,
}
impl Eligibility {
    fn current(&self) -> bool {
        self.current_at(unix_seconds(), Instant::now())
    }
    fn current_at(&self, wall: u64, monotonic: Instant) -> bool {
        self.cache_until_restart || (self.valid_until > wall && self.deadline > monotonic)
    }
}

struct Verifier {
    declaration: proto::PeerEvidenceVerifierManifest,
    generation: AtomicU64,
    active_generation: Option<std::sync::Arc<AtomicU64>>,
}
#[derive(Clone, PartialEq, Eq, Hash)]
struct EvidenceKey {
    plugin: String,
    mesh: String,
    node: String,
    digest: Option<[u8; 32]>,
}
pub(crate) struct EvidenceRegistry {
    verifiers: BTreeMap<String, Verifier>,
    publications: Mutex<HashMap<String, (u64, Vec<u8>)>>,
    leases: Mutex<HashMap<EvidenceKey, Eligibility>>,
    slots: std::sync::Arc<Semaphore>,
    pub(crate) refresh: std::sync::Arc<tokio::sync::Mutex<()>>,
    pub(crate) refresh_requested: AtomicBool,
    pub(crate) cache_changed: std::sync::Arc<tokio::sync::Notify>,
}
impl Default for EvidenceRegistry {
    fn default() -> Self {
        Self {
            verifiers: Default::default(),
            publications: Default::default(),
            leases: Default::default(),
            slots: std::sync::Arc::new(Semaphore::new(1)),
            refresh: Default::default(),
            refresh_requested: AtomicBool::new(false),
            cache_changed: Default::default(),
        }
    }
}
#[derive(Debug, thiserror::Error)]
#[error("declared peer evidence verifier failed startup validation")]
pub(super) struct VerifierStartupFailure;

pub(super) fn require_verifier(
    manifest: Option<&proto::PluginManifest>,
) -> Result<&proto::PeerEvidenceVerifierManifest> {
    peer_evidence::validate_manifest(manifest.context("peer evidence requires a plugin manifest")?)?
        .context("plugin has no peer evidence verifier declaration")
}
pub(super) fn validate_startup(
    spec: &ExternalPluginSpec,
    manifest: Option<&proto::PluginManifest>,
) -> Result<()> {
    let declaration = manifest
        .map(peer_evidence::validate_manifest)
        .transpose()?
        .flatten();
    if let Some(expected) = spec
        .installed_metadata
        .as_ref()
        .and_then(|metadata| metadata.manifest.as_ref())
        .and_then(|manifest| manifest.peer_evidence_verifier.as_ref())
    {
        let actual =
            declaration.context("packaged peer evidence verifier missing from live manifest")?;
        ensure!(
            actual.contract_version == expected.contract_version
                && actual.namespace == expected.namespace
                && actual.handler == expected.handler,
            "live peer evidence verifier differs from installed package"
        );
    }
    if declaration.is_some() {
        ensure!(
            !spec.startup.optional && !spec.startup.lazy_start,
            "peer evidence verifiers require eager, non-optional startup"
        );
    }
    Ok(())
}
impl EvidenceRegistry {
    pub(super) async fn from_plugins(
        plugins: &BTreeMap<String, super::ExternalPlugin>,
    ) -> Result<Self> {
        let mut registry = Self::default();
        for (name, plugin) in plugins {
            let manifest = plugin.manifest_snapshot().await;
            validate_startup(&plugin.spec, manifest.as_ref())?;
            if !plugin.spec.startup.optional
                && !plugin.spec.startup.lazy_start
                && plugin.active_generation.load(Ordering::Acquire) == 0
                && manifest
                    .as_ref()
                    .is_none_or(|manifest| manifest.peer_evidence_verifier.is_some())
            {
                ensure!(
                    plugin.summary().await.status == "disabled",
                    "required plugin '{name}' stopped during verifier discovery"
                );
            }
            if let Some(manifest) = manifest
                && manifest.peer_evidence_verifier.is_some()
            {
                registry.register(name, &manifest)?;
                registry.verifiers.get_mut(name).unwrap().active_generation =
                    Some(plugin.active_generation.clone());
            }
        }
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
                .any(|verifier| verifier.declaration.namespace == declaration.namespace),
            "duplicate peer evidence namespace"
        );
        self.verifiers.insert(
            plugin.into(),
            Verifier {
                declaration: declaration.clone(),
                generation: AtomicU64::new(0),
                active_generation: None,
            },
        );
        Ok(())
    }
    pub(crate) fn has_verifiers(&self) -> bool {
        !self.verifiers.is_empty()
    }
    pub(crate) fn next_refresh_deadline(&self, refreshed_through: Instant) -> Option<Instant> {
        self.leases
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|(key, entry)| {
                !entry.cache_until_restart
                    && entry.deadline > refreshed_through
                    && self.verifiers[&key.plugin]
                        .active_generation
                        .as_ref()
                        .is_none_or(|active| {
                            entry.runtime_generation != 0
                                && active.load(Ordering::Acquire) == entry.runtime_generation
                        })
            })
            .map(|(_, entry)| entry.deadline)
            .min()
    }
    pub(crate) fn publications(&self) -> BTreeMap<String, Vec<u8>> {
        let publications = self.publications.lock().unwrap_or_else(|e| e.into_inner());
        self.publication_snapshot(&publications)
    }
    fn publication_snapshot(
        &self,
        publications: &HashMap<String, (u64, Vec<u8>)>,
    ) -> BTreeMap<String, Vec<u8>> {
        self.verifiers
            .iter()
            .filter_map(|(name, verifier)| {
                publications
                    .get(name)
                    .filter(|(generation, _)| {
                        verifier.active_generation.as_ref().is_none_or(|active| {
                            *generation != 0 && active.load(Ordering::Acquire) == *generation
                        })
                    })
                    .map(|(_, bytes)| (verifier.declaration.namespace.clone(), bytes.clone()))
            })
            .collect()
    }
    pub(crate) fn clear_plugin(&self, name: &str) {
        self.publications
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(name);
        self.invalidate_verifier(name);
    }
    fn retain_publication_generation(&self, name: &str, generation: Option<u64>) {
        self.publications
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|plugin, (published, _)| plugin != name || Some(*published) == generation);
    }
    fn withdraw_publication_generation(&self, name: &str, generation: u64) {
        self.publications
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|plugin, (published, _)| plugin != name || *published != generation);
    }
    fn invalidate_verifier(&self, name: &str) {
        let Ok(verifier) = self.verifier(name) else {
            return;
        };
        let mut leases = self.leases.lock().unwrap_or_else(|e| e.into_inner());
        verifier.generation.fetch_add(1, Ordering::AcqRel);
        leases.retain(|key, _| key.plugin != name);
        self.cache_changed.notify_one();
    }
    fn verifier(&self, plugin: &str) -> Result<&Verifier> {
        self.verifiers
            .get(plugin)
            .context("plugin has no peer evidence verifier")
    }
    fn lease_key(plugin: &str, mesh: &str, node: &str, bytes: Option<&[u8]>) -> EvidenceKey {
        EvidenceKey {
            plugin: plugin.into(),
            mesh: mesh.into(),
            node: node.into(),
            digest: bytes.map(|bytes| sha2::Sha256::digest(bytes).into()),
        }
    }
    fn cached(&self, key: &EvidenceKey) -> Option<Eligibility> {
        let active = self.verifier(&key.plugin).ok()?.active_generation.as_ref();
        self.leases
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_mut(key)
            .filter(|entry| {
                entry.current()
                    && active.is_none_or(|generation| {
                        let generation = generation.load(Ordering::Acquire);
                        generation != 0 && generation == entry.runtime_generation
                    })
            })
            .map(|entry| {
                entry.last_used = Instant::now();
                *entry
            })
    }
    fn insert(
        leases: &mut HashMap<EvidenceKey, Eligibility>,
        key: EvidenceKey,
        entry: Eligibility,
    ) {
        if !leases.contains_key(&key) && leases.len() >= MAX_LEASES {
            let oldest = leases
                .iter()
                .min_by_key(|(_, entry)| (entry.current(), entry.accepted, entry.last_used))
                .map(|(key, _)| key.clone());
            if let Some(oldest) = oldest {
                // Rejections cannot displace current accepted providers.
                if !entry.accepted && leases[&oldest].accepted && leases[&oldest].current() {
                    return;
                }
                leases.remove(&oldest);
            }
        }
        leases.insert(key, entry);
    }
    fn publish(
        &self,
        plugin: &str,
        generation: u64,
        current: Option<u64>,
        evidence: Option<Vec<u8>>,
        evidence_budget: usize,
    ) -> Result<()> {
        ensure!(
            current == Some(generation),
            "publication from a stale plugin generation"
        );
        ensure!(
            self.verifier(plugin)?
                .active_generation
                .as_ref()
                .is_none_or(|active| active.load(Ordering::Acquire) == generation),
            "peer evidence generation has not completed initialization"
        );
        let mut publications = self.publications.lock().unwrap_or_else(|e| e.into_inner());
        match evidence {
            Some(bytes) => {
                let mut candidate = self.publication_snapshot(&publications);
                candidate.insert(
                    self.verifier(plugin)?.declaration.namespace.clone(),
                    bytes.clone(),
                );
                ensure!(
                    crate::proto::node::NodeRecord {
                        peer_evidence: candidate,
                        ..Default::default()
                    }
                    .encoded_len()
                        <= evidence_budget,
                    "peer evidence exceeds the remaining signed node record budget; reduce advertised metadata"
                );
                publications.insert(plugin.into(), (generation, bytes));
            }
            None => {
                publications.remove(plugin);
            }
        }
        Ok(())
    }
    pub(crate) fn provider_cached(
        &self,
        mesh: &str,
        node: &str,
        evidence: &BTreeMap<String, Vec<u8>>,
    ) -> bool {
        self.provider_decision(mesh, node, evidence) == Some(true)
    }
    pub(crate) fn providers_cached(&self, providers: &[ProviderEvidence]) -> bool {
        let verifiers: Vec<_> = self.verifiers.iter().collect();
        let keys: Vec<_> = verifiers
            .iter()
            .enumerate()
            .flat_map(|(index, (plugin, verifier))| {
                providers.iter().map(move |(mesh, node, evidence)| {
                    (
                        index,
                        Self::lease_key(
                            plugin,
                            mesh,
                            node,
                            evidence
                                .get(&verifier.declaration.namespace)
                                .map(Vec::as_slice),
                        ),
                    )
                })
            })
            .collect();
        let mut leases = self.leases.lock().unwrap_or_else(|e| e.into_inner());
        let generations: Vec<_> = verifiers
            .iter()
            .map(|(_, verifier)| {
                verifier
                    .active_generation
                    .as_ref()
                    .map(|active| active.load(Ordering::Acquire))
            })
            .collect();
        let wall = unix_seconds();
        let monotonic = Instant::now();
        for (index, key) in keys {
            let Some(entry) = leases.get_mut(&key) else {
                return false;
            };
            if !entry.accepted
                || !entry.current_at(wall, monotonic)
                || generations[index].is_some_and(|generation| {
                    generation == 0 || generation != entry.runtime_generation
                })
            {
                return false;
            }
            entry.last_used = monotonic;
        }
        verifiers
            .iter()
            .zip(generations)
            .all(|((_, verifier), expected)| {
                verifier
                    .active_generation
                    .as_ref()
                    .map(|active| active.load(Ordering::Acquire))
                    == expected
            })
    }

    pub(crate) fn provider_decision(
        &self,
        mesh: &str,
        node: &str,
        evidence: &BTreeMap<String, Vec<u8>>,
    ) -> Option<bool> {
        let mut decision = Some(true);
        for (plugin, verifier) in &self.verifiers {
            match self.cached(&Self::lease_key(
                plugin,
                mesh,
                node,
                evidence
                    .get(&verifier.declaration.namespace)
                    .map(Vec::as_slice),
            )) {
                Some(entry) if !entry.accepted => return Some(false),
                None => decision = None,
                _ => {}
            }
        }
        decision
    }
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
impl PluginManager {
    #[cfg(test)]
    pub(crate) fn evict_peer_evidence_for_test(&self) {
        let registry = &self.inner.peer_evidence;
        registry.leases.lock().unwrap().clear();
        registry.cache_changed.notify_one();
    }
    #[cfg(test)]
    pub(crate) fn expire_peer_evidence_for_test(&self) {
        for entry in self.inner.peer_evidence.leases.lock().unwrap().values_mut() {
            entry.deadline = Instant::now() - Duration::from_secs(1);
        }
    }
    #[cfg(test)]
    pub(crate) fn for_test_peer_evidence(
        bridge: std::sync::Arc<dyn super::PluginRpcBridge>,
        evidence: Option<Value>,
    ) -> Self {
        let mut manager = Self::for_test_bridge(&["evidence"], bridge);
        let manifest = mesh_llm_plugin::plugin_manifest![
            peer_evidence::peer_evidence_verifier("test.bond.v1", "verify"),
            mesh_llm_plugin::operation::<PeerEvidenceVerificationRequest>(
                "verify",
                "Verify evidence"
            )
        ];
        manager
            .inner
            .test_manifests
            .try_lock()
            .unwrap()
            .insert("evidence".into(), manifest.clone());
        let registry = &mut std::sync::Arc::get_mut(&mut manager.inner)
            .unwrap()
            .peer_evidence;
        registry.register("evidence", &manifest).unwrap();
        if let Some(evidence) = evidence {
            registry.publications.lock().unwrap().insert(
                "evidence".into(),
                (0, serde_json::to_vec(&evidence).unwrap()),
            );
        }
        manager
    }
    pub(crate) fn peer_evidence(&self) -> &EvidenceRegistry {
        &self.inner.peer_evidence
    }
    pub(crate) async fn refresh_peer_evidence_verifier(
        &self,
        plugin: &str,
        generation: u64,
    ) -> Result<bool> {
        let manifest = self.manifest_snapshot(plugin).await;
        let declared = manifest
            .as_ref()
            .map(peer_evidence::validate_manifest)
            .transpose()?
            .flatten();
        let registered = self
            .peer_evidence()
            .verifiers
            .get(plugin)
            .map(|verifier| &verifier.declaration);
        ensure!(
            registered == declared,
            "peer evidence verifier declaration changed; restart Mesh"
        );
        if registered.is_some() {
            if let Some(external) = self.inner.plugins.get(plugin) {
                let runtime = external.runtime.lock().await;
                if runtime.as_ref().map(|runtime| runtime.generation) != Some(generation) {
                    return Ok(false);
                }
                self.peer_evidence().retain_publication_generation(
                    plugin,
                    runtime.as_ref().map(|runtime| runtime.generation),
                );
            }
            self.peer_evidence().invalidate_verifier(plugin);
        }
        Ok(registered.is_some())
    }
    pub(crate) async fn peer_evidence_disconnected(&self, plugin: &str, generation: u64) {
        if self.peer_evidence().verifier(plugin).is_err() {
            return;
        }
        let Some(external) = self.inner.plugins.get(plugin) else {
            self.peer_evidence().clear_plugin(plugin);
            return;
        };
        let runtime = external.runtime.lock().await;
        self.peer_evidence()
            .withdraw_publication_generation(plugin, generation);
        if runtime
            .as_ref()
            .is_none_or(|current| current.generation == generation)
        {
            self.peer_evidence().clear_plugin(plugin);
        }
    }
    async fn current_evidence_verifier(&self, plugin: &str) -> Result<&Verifier> {
        let verifier = self.peer_evidence().verifier(plugin)?;
        let manifest = self.manifest_snapshot(plugin).await;
        ensure!(
            *require_verifier(manifest.as_ref())? == verifier.declaration,
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
    generation: u64,
    request: proto::RpcRequest,
) -> Result<proto::RpcResponse, proto::ErrorResponse> {
    handle(node, plugin, generation, request)
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
    generation: u64,
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
    let local = node.snapshot_local_announcement_data().await;
    let announcement = node.build_local_announcement(local);
    let record = crate::protocol::node_record::node_record_from_wire_ann(
        &crate::protocol::local_ann_to_proto_ann(&announcement),
        &announcement.addr,
    );
    let evidence_budget = crate::protocol::node_record::MAX_NODE_RECORD_BODY_BYTES
        .saturating_sub(record.encoded_len());
    let external = manager
        .inner
        .plugins
        .get(plugin)
        .context("plugin is not running")?;
    let runtime = external.runtime.lock().await;
    // Commit under the runtime lock so disconnect cannot overtake this write.
    registry.publish(
        plugin,
        generation,
        runtime.as_ref().map(|runtime| runtime.generation),
        (!evidence.is_null()).then_some(bytes),
        evidence_budget,
    )?;
    let result = json!({"published":!evidence.is_null()});
    Ok(proto::RpcResponse {
        result_json: serde_json::to_string(&result)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publications_preserve_the_signed_record_budget_and_allow_withdrawal() {
        let names: Vec<_> = (0..256)
            .map(|n| format!("{n:03}{}", "m".repeat(505)))
            .collect();
        let record = crate::proto::node::NodeRecord {
            serving_models: names.clone(),
            requested_models: names,
            ..Default::default()
        };
        let remaining =
            crate::protocol::node_record::MAX_NODE_RECORD_BODY_BYTES - record.encoded_len();
        assert!(remaining < MAX_EVIDENCE_BYTES);
        let mut registry = EvidenceRegistry::default();
        let manifest = mesh_llm_plugin::plugin_manifest![
            peer_evidence::peer_evidence_verifier("proof", "verify"),
            mesh_llm_plugin::operation::<PeerEvidenceVerificationRequest>("verify", "Verify")
        ];
        registry.register("p", &manifest).unwrap();
        registry
            .publish("p", 1, Some(1), Some(b"{}".to_vec()), remaining)
            .unwrap();
        let full_reference = serde_json::to_vec(&"e".repeat(MAX_EVIDENCE_BYTES - 2)).unwrap();
        assert_eq!(full_reference.len(), MAX_EVIDENCE_BYTES);
        assert!(
            registry
                .publish("p", 1, Some(1), Some(full_reference), remaining)
                .is_err()
        );
        assert_eq!(registry.publications()["proof"], b"{}");
        registry.publish("p", 1, Some(1), None, 0).unwrap();
        assert!(registry.publications().is_empty());
    }

    #[test]
    fn leases_expire_and_are_bound_to_namespace_mesh_host_and_reference() {
        let mut registry = EvidenceRegistry::default();
        registry.verifiers.insert(
            "verifier".into(),
            Verifier {
                declaration: peer_evidence::peer_evidence_verifier("example.bond", "verify"),
                generation: AtomicU64::new(0),
                active_generation: None,
            },
        );
        let evidence = BTreeMap::from([("example.bond".into(), b"reference".to_vec())]);
        let key = EvidenceRegistry::lease_key("verifier", "mesh", "host", Some(b"reference"));
        registry.leases.lock().unwrap().insert(
            key.clone(),
            Eligibility {
                accepted: true,
                valid_until: unix_seconds() + 300,
                deadline: Instant::now() + Duration::from_secs(300),
                cache_until_restart: false,
                last_used: Instant::now(),
                runtime_generation: 0,
            },
        );
        assert!(registry.provider_cached("mesh", "host", &evidence));
        assert!(!registry.provider_cached("other-mesh", "host", &evidence));
        assert!(!registry.provider_cached("mesh", "other-host", &evidence));
        assert!(!registry.provider_cached(
            "mesh",
            "host",
            &BTreeMap::from([("example.bond".into(), b"another".to_vec())])
        ));
        assert!(!registry.provider_cached("mesh", "host", &BTreeMap::new()));
        registry.leases.lock().unwrap().insert(
            key.clone(),
            Eligibility {
                accepted: true,
                valid_until: unix_seconds() + 300,
                deadline: Instant::now() - Duration::from_secs(1),
                cache_until_restart: false,
                last_used: Instant::now(),
                runtime_generation: 0,
            },
        );
        assert!(!registry.provider_cached("mesh", "host", &evidence));
        registry.leases.lock().unwrap().insert(
            key,
            Eligibility {
                accepted: true,
                valid_until: unix_seconds() - 1,
                deadline: Instant::now() + Duration::from_secs(300),
                cache_until_restart: false,
                last_used: Instant::now(),
                runtime_generation: 0,
            },
        );
        assert!(!registry.provider_cached("mesh", "host", &evidence));
    }
}

#[cfg(test)]
mod publication_tests {
    use super::*;
    #[test]
    fn stale_publications_cannot_restore_or_overwrite_replacement_evidence() {
        let mut registry = EvidenceRegistry::default();
        registry.register("p", &manifest("proof")).unwrap();
        registry
            .publish("p", 1, Some(1), Some(b"old".to_vec()), usize::MAX)
            .unwrap();
        registry.clear_plugin("p");
        assert!(
            registry
                .publish("p", 1, None, Some(b"stale".to_vec()), usize::MAX)
                .is_err()
        );
        registry.publish("p", 2, Some(2), None, usize::MAX).unwrap();
        assert!(
            registry
                .publish("p", 1, Some(2), Some(b"stale".to_vec()), usize::MAX)
                .is_err()
        );
        assert!(registry.publications.lock().unwrap().is_empty());
    }
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
        assert!(registry.verifier("undeclared").is_err());
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
            publications.insert("a".into(), (0, b"first".to_vec()));
            publications.insert("b".into(), (0, b"second".to_vec()));
            publications.insert("undeclared".into(), (0, b"hidden".to_vec()));
        }
        assert_eq!(registry.publications().len(), 2);
        registry.clear_plugin("a");
        assert_eq!(
            registry.publications(),
            BTreeMap::from([("proof.b".into(), b"second".to_vec())])
        );
    }
}

#[cfg(test)]
#[path = "peer_evidence/validity_tests.rs"]
mod validity_tests;

#[cfg(test)]
mod invalidation_tests {
    use super::*;
    use crate::plugin::{BridgeFuture, PluginRpcBridge, RpcResult};
    use std::sync::Arc;

    struct DelayedVerifier {
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    }
    impl PluginRpcBridge for DelayedVerifier {
        fn handle_request(
            &self,
            _: String,
            _: String,
            _: String,
        ) -> BridgeFuture<Result<RpcResult, proto::ErrorResponse>> {
            let (started, release) = (self.started.clone(), self.release.clone());
            Box::pin(async move {
                started.notify_one();
                release.notified().await;
                Ok(RpcResult {
                    result_json: json!({"accepted":true,"valid_until":unix_seconds()+86400})
                        .to_string(),
                })
            })
        }
        fn handle_notification(&self, _: String, _: String, _: String) -> BridgeFuture<()> {
            Box::pin(async {})
        }
    }
    #[tokio::test]
    async fn stopping_a_plugin_invalidates_only_its_own_inflight_verification() {
        for stopped in ["evidence", "unrelated", "other_verifier"] {
            let started = Arc::new(tokio::sync::Notify::new());
            let release = Arc::new(tokio::sync::Notify::new());
            let mut manager = PluginManager::for_test_peer_evidence(
                Arc::new(DelayedVerifier {
                    started: started.clone(),
                    release: release.clone(),
                }),
                None,
            );
            let other = mesh_llm_plugin::plugin_manifest![
                peer_evidence::peer_evidence_verifier("other.namespace", "verify"),
                mesh_llm_plugin::operation::<PeerEvidenceVerificationRequest>(
                    "verify",
                    "Verify evidence"
                )
            ];
            Arc::get_mut(&mut manager.inner)
                .unwrap()
                .peer_evidence
                .register("other_verifier", &other)
                .unwrap();
            let key = EvidenceRegistry::lease_key("evidence", "mesh", "peer", Some(b"{}"));
            let task = {
                let manager = manager.clone();
                tokio::spawn(async move {
                    manager
                        .verify_evidence(
                            "evidence",
                            "mesh",
                            "peer",
                            Some(b"{}"),
                            tokio::time::Instant::now() + Duration::from_secs(12),
                        )
                        .await
                        .is_ok_and(|entry| entry.accepted)
                })
            };
            started.notified().await;
            manager.peer_evidence().clear_plugin(stopped);
            release.notify_one();
            let expected = stopped != "evidence";
            assert_eq!(task.await.unwrap(), expected, "stopped {stopped}");
            assert_eq!(
                manager
                    .peer_evidence()
                    .cached(&key)
                    .is_some_and(|entry| entry.accepted),
                expected,
                "stopped {stopped}"
            );
        }
    }
}
