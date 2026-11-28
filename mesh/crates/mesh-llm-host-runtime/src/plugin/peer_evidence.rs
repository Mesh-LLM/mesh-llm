//! The manifest-discovered peer evidence verifier, this node's publication
//! and cached provider verdicts.
use super::{PluginManager, config::ExternalPluginSpec, proto};
use anyhow::{Context, Result, bail, ensure};
use mesh_llm_plugin::peer_evidence::{
    self, MAX_EVIDENCE_BYTES, PeerEvidenceVerificationRequest, PeerEvidenceVerificationResult,
};
use serde_json::{Value, json};
use sha2::Digest;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

/// Keeps `Instant` arithmetic in range; a verdict with a far deadline
/// effectively lasts until its verifier restarts.
const MAX_VERDICT_AGE: Duration = Duration::from_secs(365 * 24 * 60 * 60);
const FAILURE_RETRY: Duration = Duration::from_secs(5);
/// An acceptance is verified again this long before it expires, so its
/// provider stays eligible while the verifier answers.
const RENEWAL_LEAD: Duration = Duration::from_secs(10);

#[derive(Clone, Copy)]
struct Verdict {
    accepted: bool,
    expires: Instant,
    /// When to verify again. An acceptance that lasts at least twice the
    /// lead is renewed that long before it expires; a shorter one, such as
    /// the rest of an exiting bond's countdown, is not, so renewals near a
    /// fixed deadline cannot repeat.
    renew: Instant,
}

impl Verdict {
    fn new(accepted: bool, lifetime: Duration) -> Self {
        let expires = Instant::now() + lifetime;
        let renew = if accepted && lifetime >= 2 * RENEWAL_LEAD {
            expires - RENEWAL_LEAD
        } else {
            expires
        };
        Self {
            accepted,
            expires,
            renew,
        }
    }
}

struct Verifier {
    plugin: String,
    declaration: proto::PeerEvidenceVerifierManifest,
    /// Bumped whenever the verdicts become invalid, so a call that started
    /// before the bump cannot store its result.
    epoch: AtomicU64,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct VerdictKey {
    mesh: String,
    node: String,
    digest: [u8; 32],
}

impl VerdictKey {
    fn new(mesh: &str, node: &str, evidence: &[u8]) -> Self {
        Self {
            mesh: mesh.into(),
            node: node.into(),
            digest: sha2::Sha256::digest(evidence).into(),
        }
    }
}

/// This node's evidence, with the plugin connection that published it.
struct Publication {
    generation: u64,
    bytes: Vec<u8>,
}

/// What a verification found: whether the verifier accepts the evidence, and
/// whether that made the provider newly eligible.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Verification {
    pub(crate) accepted: bool,
    pub(crate) newly_eligible: bool,
}

/// The node's one peer evidence verifier, its own publication and the
/// verifier's cached verdicts.
#[derive(Default)]
pub(crate) struct EvidenceRegistry {
    verifier: Option<Verifier>,
    /// This node's own evidence under the verifier's namespace.
    publication: Mutex<Option<Publication>>,
    verdicts: Mutex<HashMap<VerdictKey, Verdict>>,
    /// Wakes the provider verification task.
    pub(crate) refresh: tokio::sync::Notify,
}

/// Validates the verifier a starting plugin declares in its live manifest.
pub(super) fn validate_startup<'a>(
    spec: &ExternalPluginSpec,
    manifest: Option<&'a proto::PluginManifest>,
) -> Result<Option<&'a proto::PeerEvidenceVerifierManifest>> {
    let declaration = manifest
        .map(peer_evidence::validate_manifest)
        .transpose()?
        .flatten();
    // Verifiers are registered when the plugins start, so a deferred one
    // could never enforce.
    ensure!(
        declaration.is_none() || !spec.startup.lazy_start,
        "peer evidence verifiers cannot start lazily"
    );
    Ok(declaration)
}

/// A verifier decides which providers the host routes to, so it needs an
/// OS-authenticated process, and a restart must keep the declaration the
/// first generation made, which is the one the host registered.
pub(super) fn validate_generation(
    declaration: Option<&proto::PeerEvidenceVerifierManifest>,
    authenticated: bool,
    first: &std::sync::OnceLock<Option<proto::PeerEvidenceVerifierManifest>>,
) -> Result<()> {
    ensure!(
        declaration.is_none() || authenticated,
        "peer evidence verifiers require an OS-authenticated launched plugin process"
    );
    ensure!(
        first.get_or_init(|| declaration.cloned()).as_ref() == declaration,
        "a restarted plugin cannot change its peer evidence verifier"
    );
    Ok(())
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

impl EvidenceRegistry {
    /// Registers the verifier each plugin's first generation declared, which
    /// stays fixed when a later connection drops and clears its manifest.
    pub(super) fn from_plugins(plugins: &BTreeMap<String, super::ExternalPlugin>) -> Result<Self> {
        let mut registry = Self::default();
        for (name, plugin) in plugins {
            if let Some(declaration) = plugin.declared_peer_evidence_verifier() {
                registry.register(name, declaration)?;
            }
        }
        Ok(registry)
    }

    fn register(
        &mut self,
        plugin: &str,
        declaration: &proto::PeerEvidenceVerifierManifest,
    ) -> Result<()> {
        ensure!(
            self.verifier.is_none(),
            "only one peer evidence verifier is supported"
        );
        self.verifier = Some(Verifier {
            plugin: plugin.into(),
            declaration: declaration.clone(),
            epoch: AtomicU64::new(0),
        });
        Ok(())
    }

    pub(crate) fn has_verifier(&self) -> bool {
        self.verifier.is_some()
    }

    /// The verifier when `plugin` runs it.
    fn verifier_of(&self, plugin: &str) -> Option<&Verifier> {
        self.verifier
            .as_ref()
            .filter(|verifier| verifier.plugin == plugin)
    }

    /// This node's evidence, keyed by namespace as in its node record.
    pub(crate) fn publications(&self) -> BTreeMap<String, Vec<u8>> {
        let publication = lock(&self.publication);
        match (&self.verifier, publication.as_ref()) {
            (Some(verifier), Some(publication)) => BTreeMap::from([(
                verifier.declaration.namespace.clone(),
                publication.bytes.clone(),
            )]),
            _ => BTreeMap::new(),
        }
    }

    /// Stores or withdraws this node's evidence for connection `generation`,
    /// and reports whether the evidence changed.
    fn replace_publication(&self, generation: u64, bytes: Option<Vec<u8>>) -> bool {
        let mut publication = lock(&self.publication);
        let changed = publication.as_ref().map(|current| &current.bytes) != bytes.as_ref();
        *publication = bytes.map(|bytes| Publication { generation, bytes });
        changed
    }

    /// Withdraws the publication connection `generation` made, after that
    /// connection closed; a newer connection's publication stays.
    pub(crate) fn withdraw_generation(&self, plugin: &str, generation: u64) {
        if self.verifier_of(plugin).is_some() {
            let mut publication = lock(&self.publication);
            if publication
                .as_ref()
                .is_some_and(|current| current.generation == generation)
            {
                *publication = None;
            }
        }
    }

    /// Drops the plugin's publication and verdicts after it stopped.
    pub(crate) fn clear_plugin(&self, plugin: &str) {
        if self.verifier_of(plugin).is_some() {
            *lock(&self.publication) = None;
        }
        self.invalidate(plugin);
    }

    /// Discards the plugin's verdicts and asks for fresh verification.
    pub(crate) fn invalidate(&self, plugin: &str) {
        let Some(verifier) = self.verifier_of(plugin) else {
            return;
        };
        let mut verdicts = lock(&self.verdicts);
        verifier.epoch.fetch_add(1, Ordering::AcqRel);
        verdicts.clear();
        drop(verdicts);
        self.refresh.notify_one();
    }

    /// The provider's evidence under the verifier's namespace.
    fn published<'a>(&self, evidence: &'a BTreeMap<String, Vec<u8>>) -> Option<&'a [u8]> {
        let verifier = self.verifier.as_ref()?;
        evidence
            .get(&verifier.declaration.namespace)
            .map(Vec::as_slice)
    }

    /// The verdict key for the provider's evidence, or `None` without a
    /// verifier or without evidence, when there is nothing to ask.
    fn key(
        &self,
        mesh: &str,
        node: &str,
        evidence: &BTreeMap<String, Vec<u8>>,
    ) -> Option<VerdictKey> {
        let bytes = self.published(evidence)?;
        Some(VerdictKey::new(mesh, node, bytes))
    }

    /// `Some(true)` when the verifier currently accepts the evidence, or there
    /// is no verifier, `Some(false)` when it currently rejects it or the
    /// provider published none, and `None` when undecided.
    pub(crate) fn provider_decision(
        &self,
        mesh: &str,
        node: &str,
        evidence: &BTreeMap<String, Vec<u8>>,
    ) -> Option<bool> {
        if self.verifier.is_none() {
            return Some(true);
        }
        let Some(key) = self.key(mesh, node, evidence) else {
            return Some(false);
        };
        lock(&self.verdicts)
            .get(&key)
            .filter(|verdict| verdict.expires > Instant::now())
            .map(|verdict| verdict.accepted)
    }

    pub(crate) fn provider_cached(
        &self,
        mesh: &str,
        node: &str,
        evidence: &BTreeMap<String, Vec<u8>>,
    ) -> bool {
        self.provider_decision(mesh, node, evidence) == Some(true)
    }

    /// When the next verdict is due for renewal or verification again, which
    /// may already have passed, ignoring the providers `skip` names, such as
    /// those being verified. A verdict is due no later than it expires.
    pub(crate) fn next_due(&self, skip: impl Fn(&str) -> bool) -> Option<Instant> {
        lock(&self.verdicts)
            .iter()
            .filter(|(key, _)| !skip(&key.node))
            .map(|(_, verdict)| verdict.renew)
            .min()
    }

    /// When the provider's current acceptance expires, if it has one.
    pub(crate) fn acceptance_expiry(
        &self,
        mesh: &str,
        node: &str,
        evidence: &BTreeMap<String, Vec<u8>>,
    ) -> Option<Instant> {
        let key = self.key(mesh, node, evidence)?;
        let now = Instant::now();
        lock(&self.verdicts)
            .get(&key)
            .filter(|verdict| verdict.accepted && verdict.expires > now)
            .map(|verdict| verdict.expires)
    }

    /// The cached verdict, unless it is due for renewal.
    fn settled_verdict(&self, key: &VerdictKey) -> Option<bool> {
        lock(&self.verdicts)
            .get(key)
            .filter(|verdict| verdict.renew > Instant::now())
            .map(|verdict| verdict.accepted)
    }

    /// Whether the verifier lacks a verdict on the provider's evidence, or
    /// has one due for renewal.
    pub(crate) fn needs_verification(
        &self,
        mesh: &str,
        node: &str,
        evidence: &BTreeMap<String, Vec<u8>>,
    ) -> bool {
        self.key(mesh, node, evidence)
            .is_some_and(|key| self.settled_verdict(&key).is_none())
    }

    fn commit(
        &self,
        key: VerdictKey,
        epoch: u64,
        result: Result<PeerEvidenceVerificationResult>,
    ) -> Verification {
        if let Ok(result) = &result
            && !result.accepted
        {
            let reason = result.reason.as_deref().unwrap_or("none given");
            tracing::debug!(node = %key.node, reason, "peer evidence rejected");
        }
        let verdict = result.and_then(verdict_from);
        let mut verdicts = lock(&self.verdicts);
        let current = self
            .verifier
            .as_ref()
            .map(|verifier| verifier.epoch.load(Ordering::Acquire));
        if current != Some(epoch) {
            return Verification::default();
        }
        let now = Instant::now();
        let previous = verdicts.get(&key).copied();
        let verdict = verdict.unwrap_or_else(|error| {
            tracing::debug!(node = %key.node, %error, "peer evidence verification failed");
            match previous {
                // A failed renewal keeps the acceptance until it expires
                // and tries again shortly.
                Some(old) if old.accepted && old.expires > now => Verdict {
                    renew: (now + FAILURE_RETRY).min(old.expires),
                    ..old
                },
                _ => Verdict::new(false, FAILURE_RETRY),
            }
        });
        verdicts.insert(key, verdict);
        let was_eligible = previous.is_some_and(|old| old.accepted && old.expires > now);
        Verification {
            accepted: verdict.accepted,
            newly_eligible: verdict.accepted && !was_eligible,
        }
    }

    /// Keeps only the verdicts on the current evidence of current providers,
    /// so departed providers and replaced evidence cannot fill the cache.
    pub(crate) fn retain_current(
        &self,
        mesh: &str,
        providers: &[(String, BTreeMap<String, Vec<u8>>)],
    ) {
        let current: HashSet<_> = providers
            .iter()
            .filter_map(|(node, evidence)| self.key(mesh, node, evidence))
            .collect();
        lock(&self.verdicts).retain(|key, _| current.contains(key));
    }
}

fn verdict_from(result: PeerEvidenceVerificationResult) -> Result<Verdict> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let lifetime = Duration::from_secs(result.valid_until)
        .checked_sub(now)
        .filter(|lifetime| !lifetime.is_zero());
    match (result.accepted, lifetime) {
        (accepted, Some(lifetime)) => Ok(Verdict::new(accepted, lifetime.min(MAX_VERDICT_AGE))),
        // A rejection applies even without a deadline, and is checked again
        // shortly.
        (false, None) => Ok(Verdict::new(false, FAILURE_RETRY)),
        (true, None) => bail!("peer evidence verdict expired"),
    }
}

impl PluginManager {
    pub(crate) fn peer_evidence(&self) -> &EvidenceRegistry {
        &self.inner.peer_evidence
    }

    /// Asks the verifier about the provider's evidence when it has no current
    /// verdict, or one due for renewal, and reports what it found. A provider
    /// that published no evidence is rejected without asking.
    pub(crate) async fn verify_provider_evidence(
        &self,
        mesh: &str,
        node: &str,
        evidence: &BTreeMap<String, Vec<u8>>,
    ) -> Verification {
        let registry = self.peer_evidence();
        let Some(verifier) = &registry.verifier else {
            return Verification {
                accepted: true,
                newly_eligible: false,
            };
        };
        let Some(bytes) = registry.published(evidence) else {
            return Verification::default();
        };
        let key = VerdictKey::new(mesh, node, bytes);
        if let Some(accepted) = registry.settled_verdict(&key) {
            return Verification {
                accepted,
                newly_eligible: false,
            };
        }
        let epoch = verifier.epoch.load(Ordering::Acquire);
        let result = self
            .call_verifier(
                &verifier.plugin,
                &verifier.declaration.handler,
                mesh,
                node,
                bytes,
            )
            .await;
        registry.commit(key, epoch, result)
    }

    async fn call_verifier(
        &self,
        plugin: &str,
        handler: &str,
        mesh: &str,
        node: &str,
        evidence: &[u8],
    ) -> Result<PeerEvidenceVerificationResult> {
        let evidence = serde_json::from_slice(evidence).context("invalid peer evidence JSON")?;
        let request = PeerEvidenceVerificationRequest {
            mesh_id: mesh.into(),
            node_id: node.into(),
            evidence,
        };
        #[cfg(test)]
        if self.is_test_bridge_enabled(plugin) {
            return self
                .invoke_service_json(plugin, proto::ServiceKind::Operation, handler, &request)
                .await;
        }
        let response = self
            .inner
            .plugins
            .get(plugin)
            .context("peer evidence verifier is not loaded")?
            .invoke_service(
                proto::ServiceKind::Operation,
                handler,
                &serde_json::to_string(&request)?,
                Some(Duration::from_secs(super::REQUEST_TIMEOUT_SECS)),
            )
            .await?;
        ensure!(
            !response.is_error,
            "peer evidence verifier returned an error"
        );
        Ok(serde_json::from_str(&response.output_json)?)
    }

    #[cfg(test)]
    pub(crate) fn for_test_peer_evidence(
        bridge: std::sync::Arc<dyn super::PluginRpcBridge>,
        evidence: Option<Value>,
    ) -> Self {
        let mut manager = Self::for_test_bridge(&["evidence"], bridge);
        let registry = &mut std::sync::Arc::get_mut(&mut manager.inner)
            .unwrap()
            .peer_evidence;
        registry
            .register(
                "evidence",
                &peer_evidence::peer_evidence_verifier("test.bond.v1", "verify"),
            )
            .unwrap();
        registry.replace_publication(
            0,
            evidence.map(|evidence| serde_json::to_vec(&evidence).unwrap()),
        );
        manager
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
    publish(node, plugin, generation, request)
        .await
        .map_err(|error| proto::ErrorResponse {
            code: -32602,
            message: error.to_string(),
            data_json: String::new(),
        })
}

/// Stores or withdraws this node's evidence and announces the signed node
/// record carrying the change.
async fn publish(
    node: &crate::mesh::Node,
    plugin: &str,
    generation: u64,
    request: proto::RpcRequest,
) -> Result<proto::RpcResponse> {
    let params: Value = serde_json::from_str(&request.params_json)?;
    let evidence = params.get("evidence").context("evidence missing")?;
    let bytes = (!evidence.is_null())
        .then(|| serde_json::to_vec(evidence))
        .transpose()?;
    ensure!(
        bytes
            .as_ref()
            .is_none_or(|bytes| bytes.len() <= MAX_EVIDENCE_BYTES),
        "oversized peer evidence"
    );
    let manager = node
        .plugin_manager()
        .await
        .context("plugin manager unavailable")?;
    let registry = manager.peer_evidence();
    ensure!(
        registry.verifier_of(plugin).is_some(),
        "plugin has no peer evidence verifier"
    );
    let published = bytes.is_some();
    // Checked and written while the plugin's connection cannot be replaced
    // or stopped, so a request from an older connection cannot overwrite a
    // newer one's publication or outlive a stop.
    let changed = manager
        .inner
        .plugins
        .get(plugin)
        .context("plugin is not loaded")?
        .while_running(generation, || {
            registry.replace_publication(generation, bytes)
        })
        .await
        .context("the publishing plugin connection is no longer running")?;
    // Peers see the evidence only in this node's next signed record, so a
    // change is announced now rather than at the next periodic gossip.
    if changed {
        node.regossip().await;
    }
    Ok(proto::RpcResponse {
        result_json: json!({ "published": published }).to_string(),
    })
}

#[cfg(test)]
#[path = "peer_evidence_tests.rs"]
mod tests;
