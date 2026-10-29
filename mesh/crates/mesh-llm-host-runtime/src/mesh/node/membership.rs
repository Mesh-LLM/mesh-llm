//! Mesh memberships: serving this node's models into further meshes.
//!
//! A membership is a second, lighter [`Node`] with its own endpoint key, peer
//! table, owner allowlist, demand and gossip. It shares the serving state of
//! the node it belongs to: the models, their descriptors and readiness, the
//! hardware it advertises, and the local inference targets behind them. So a
//! model loaded once is served into every mesh the node is a member of.
//!
//! Isolation comes from the separate identity. Each membership node accepts
//! only its own connections, admits only owners on its own allowlist, and
//! gossips only its own peer table, so peers in one mesh never learn about the
//! peers of another. That holds only while the meshes stay apart, so the node
//! and its memberships never peer with one another ([`LocalEndpoints`]), and a
//! membership refuses peers that announce a mesh other than its own. Peer
//! requests that arrive through a membership execute on the local model only,
//! as every peer request does.
//!
//! What a membership node does not share, it does not do: it runs no plugins,
//! no owner control listener, and no split-stage transport, and it does not
//! advertise them.

use super::*;
use std::collections::{BTreeMap, HashSet};

/// How long shutdown waits for every membership to leave its mesh, together.
const MEMBERSHIPS_CLOSE_BUDGET: std::time::Duration = std::time::Duration::from_secs(10);

/// What a membership node is started with.
pub(crate) struct MembershipNodeConfig<'a> {
    /// The mesh's name, from which every member derives the same mesh ID.
    pub(crate) mesh_name: &'a str,
    pub(crate) secret_key: SecretKey,
    pub(crate) relay: RelayConfig<'a>,
    /// The UDP port to bind, so the invite token, which carries the address,
    /// survives a restart. Falls back to any port when it is taken.
    pub(crate) quic_port: Option<u16>,
    /// The owner IDs whose nodes this membership admits; nobody else.
    pub(crate) trusted_owners: &'a [String],
    pub(crate) enumerate_host: bool,
}

/// The endpoint IDs of one serving node and its memberships. Each serves a
/// different mesh, so none of them may ever peer with another: that would
/// merge the meshes.
#[derive(Clone, Default)]
pub(crate) struct LocalEndpoints(Arc<std::sync::RwLock<HashSet<EndpointId>>>);

impl LocalEndpoints {
    fn insert(&self, id: EndpointId) {
        if let Ok(mut ids) = self.0.write() {
            ids.insert(id);
        }
    }

    pub(crate) fn remove(&self, id: EndpointId) {
        if let Ok(mut ids) = self.0.write() {
            ids.remove(&id);
        }
    }

    pub(crate) fn contains(&self, id: EndpointId) -> bool {
        self.0.read().is_ok_and(|ids| ids.contains(&id))
    }
}

/// A running membership: its node, the tasks that serve it, and what it was
/// started with, so a failed replacement can start it again.
pub(crate) struct MembershipEntry {
    pub(crate) node: Node,
    pub(crate) api_port: u16,
    pub(crate) key_path: std::path::PathBuf,
    pub(crate) config: crate::runtime::MeshMembershipConfig,
    /// The last join failure, if the latest round of joins had one.
    pub(crate) join_error: Arc<std::sync::Mutex<Option<String>>>,
    pub(crate) tasks: Vec<tokio::task::JoinHandle<()>>,
}

impl MembershipEntry {
    /// The UDP port the membership's endpoint is bound to.
    pub(crate) fn quic_port(&self) -> Option<u16> {
        self.node
            .endpoint
            .bound_sockets()
            .first()
            .map(std::net::SocketAddr::port)
    }

    /// Stops the membership's tasks, and with them the streams it is
    /// serving, tells its peers it is leaving, and closes its endpoint. The
    /// node it belongs to keeps serving. Returns once the tasks are gone, so
    /// the membership's API port is free again.
    pub(crate) async fn close(self) {
        for task in &self.tasks {
            task.abort();
        }
        for task in self.tasks {
            let _ = task.await;
        }
        self.node.broadcast_leaving().await;
        self.node.close_endpoint().await;
        self.node.local_endpoints.remove(self.node.id());
    }
}

/// The memberships of one node, by name.
#[derive(Clone, Default)]
pub(crate) struct MeshMemberships {
    entries: Arc<Mutex<BTreeMap<String, MembershipEntry>>>,
    /// Held across a whole add or remove, so changes apply one at a time.
    changes: Arc<Mutex<()>>,
    /// Set once shutdown has taken the memberships; nothing joins after.
    closed: Arc<std::sync::atomic::AtomicBool>,
}

impl MeshMemberships {
    pub(crate) async fn lock_changes(&self) -> tokio::sync::OwnedMutexGuard<()> {
        self.changes.clone().lock_owned().await
    }

    /// Adds a membership. After shutdown it is handed back, to be closed.
    pub(crate) async fn insert(
        &self,
        name: String,
        entry: MembershipEntry,
    ) -> std::result::Result<(), Box<MembershipEntry>> {
        let mut entries = self.entries.lock().await;
        if self.closed.load(std::sync::atomic::Ordering::Acquire) {
            return Err(Box::new(entry));
        }
        entries.insert(name, entry);
        Ok(())
    }

    pub(crate) async fn remove(&self, name: &str) -> Option<MembershipEntry> {
        self.entries.lock().await.remove(name)
    }

    async fn take_all(&self) -> Vec<MembershipEntry> {
        let mut entries = self.entries.lock().await;
        self.closed
            .store(true, std::sync::atomic::Ordering::Release);
        std::mem::take(&mut *entries).into_values().collect()
    }

    /// Calls `f` with each membership, by name.
    pub(crate) async fn with_each<T>(&self, f: impl Fn(&str, &MembershipEntry) -> T) -> Vec<T> {
        self.entries
            .lock()
            .await
            .iter()
            .map(|(name, entry)| f(name, entry))
            .collect()
    }

    async fn nodes(&self) -> Vec<Node> {
        self.with_each(|_, entry| entry.node.clone()).await
    }
}

/// The endpoint an unsigned invite token names. Signed tokens carry mesh
/// requirements, which a membership's mesh does not have, so they are refused.
pub(crate) fn membership_token_endpoint(invite_token: &str) -> Result<EndpointId> {
    match parse_invite_token(invite_token)
        .map_err(|reason| anyhow::anyhow!("join rejected: {}", reason.code()))?
    {
        InviteTokenMaterial::Legacy(addr) => Ok(addr.id),
        InviteTokenMaterial::Signed(_) => anyhow::bail!(
            "join rejected: a membership joins by unsigned invite tokens; signed tokens carry mesh requirements"
        ),
    }
}

impl Node {
    /// Re-gossips this node's state, and that of its memberships, which
    /// advertise the same serving state.
    pub(super) async fn regossip_with_memberships(&self) {
        self.regossip_connections().await;
        for node in self.memberships.nodes().await {
            node.regossip_connections().await;
        }
    }

    /// Joins a membership's mesh by another member's invite token. Returns
    /// `false`, without dialling, for this node's own token, which a list of
    /// the mesh's members naturally includes. Tokens naming this node's other
    /// meshes, and signed tokens, are refused.
    pub(crate) async fn join_membership(&self, invite_token: &str) -> Result<bool> {
        let target = membership_token_endpoint(invite_token)?;
        if target == self.endpoint.id() {
            return Ok(false);
        }
        anyhow::ensure!(
            !self.local_endpoints.contains(target),
            "join rejected: the invite token names another mesh this node serves"
        );
        self.join(invite_token).await.map(|()| true)
    }

    /// Whether a peer announcement may join this node's mesh: never one of
    /// this node's own other endpoints, and, for a membership, never a node
    /// that announces a different mesh.
    pub(crate) async fn check_mesh_peer(
        &self,
        peer_id: EndpointId,
        announced_mesh_id: Option<&str>,
    ) -> Result<()> {
        anyhow::ensure!(
            peer_id == self.endpoint.id() || !self.local_endpoints.contains(peer_id),
            "peer {} is another mesh this node serves",
            peer_id.fmt_short()
        );
        if self.is_membership
            && let (Some(ours), Some(theirs)) = (self.mesh_id().await, announced_mesh_id)
        {
            anyhow::ensure!(
                ours == theirs,
                "peer {} announces mesh {theirs}, not this membership's mesh {ours}",
                peer_id.fmt_short()
            );
        }
        Ok(())
    }

    /// Applies [`Self::check_mesh_peer`] to an announced peer: a direct peer
    /// that fails it is an error, and a transitive one is skipped.
    pub(crate) async fn admits_mesh_peer(
        &self,
        peer_id: EndpointId,
        remote: EndpointId,
        announced_mesh_id: Option<&str>,
    ) -> Result<bool> {
        match self.check_mesh_peer(peer_id, announced_mesh_id).await {
            Ok(()) => Ok(true),
            Err(error) if peer_id == remote => Err(error),
            Err(error) => {
                tracing::debug!("ignoring transitive peer: {error:#}");
                Ok(false)
            }
        }
    }

    /// Closes every membership, together under one budget. Part of runtime
    /// shutdown, before the node's own endpoint closes; a membership still
    /// starting closes itself instead of joining.
    pub(crate) async fn close_memberships(&self) {
        let entries = self.memberships.take_all().await;
        let closes =
            futures_util::future::join_all(entries.into_iter().map(MembershipEntry::close));
        if tokio::time::timeout(MEMBERSHIPS_CLOSE_BUDGET, closes)
            .await
            .is_err()
        {
            tracing::warn!(
                budget_secs = MEMBERSHIPS_CLOSE_BUDGET.as_secs(),
                "mesh memberships did not all close within the shutdown budget"
            );
        }
    }

    /// Starts a membership node: a new endpoint and peer table, admitting only
    /// `trusted_owners`, that advertises and serves this node's models. The
    /// returned node accepts connections at once; the caller wires its tunnel
    /// channels to local execution.
    pub(crate) async fn start_membership_node(
        &self,
        config: MembershipNodeConfig<'_>,
    ) -> Result<(Node, TunnelChannels)> {
        let owner_keypair = self
            .owner_keypair
            .clone()
            .context("a mesh membership needs an owner key: its peers admit it by owner")?;
        // Everything that can fail without a socket comes first, so a failure
        // leaves no endpoint to close.
        let endpoint_id = config.secret_key.public();
        let owner_attestation = self
            .sign_membership_ownership(&owner_keypair, endpoint_id, config.enumerate_host)
            .await?;
        let trust_store = self.membership_trust_store(config.trusted_owners).await;
        let owner_summary = verify_node_ownership(
            Some(&owner_attestation),
            endpoint_id.as_bytes(),
            &trust_store,
            TrustPolicy::Off,
            current_time_unix_ms(),
        );
        let (endpoint, quic_bind) = self.bind_membership_endpoint(&config).await?;
        if config.relay.policy.uses_relay() {
            wait_for_endpoint_online(
                &endpoint,
                "Membership relay connected",
                "Membership relay connection timed out (5s) — proceeding without relay",
            )
            .await;
        }
        let public_addr = if config.relay.policy.uses_raw_stun() {
            stun_public_addr(&endpoint, config.relay.policy.uses_relay()).await
        } else {
            None
        };
        // Benchmarks are host detail, kept back as the GPU name is.
        let host_figure = |figure: &Arc<tokio::sync::Mutex<Option<Vec<f64>>>>| {
            if config.enumerate_host {
                figure.clone()
            } else {
                Arc::new(tokio::sync::Mutex::new(None))
            }
        };
        self.local_endpoints.insert(self.endpoint.id());
        self.local_endpoints.insert(endpoint_id);

        let (peer_change_tx, peer_change_rx) = watch::channel(0usize);
        let (tunnel_tx, tunnel_rx) = tokio::sync::mpsc::channel(256);
        let (tunnel_http_tx, tunnel_http_rx) = tokio::sync::mpsc::channel(256);
        let (stage_transport_tx, stage_transport_rx) = tokio::sync::mpsc::channel(256);
        let node = Node {
            // This membership's identity and transport.
            endpoint,
            endpoint_secret_key: config.secret_key,
            public_addr,
            quic_bind,
            relay_policy: config.relay.policy,
            owner_keypair: Some(owner_keypair),
            local_mesh_requirements: crate::MeshRequirements::unrestricted(),
            local_endpoints: self.local_endpoints.clone(),
            is_membership: true,
            // This membership's mesh: peers, demand, and mesh identity.
            state: Arc::new(Mutex::new(MembershipState::default())),
            plugin_message_history: Default::default(),
            direct_rescue_endpoints: super::super::direct_rescue::DirectRescueEndpoints::default(),
            host_role_claims: Arc::new(Mutex::new(HostRoleClaims::default())),
            model_demand: Arc::new(std::sync::Mutex::new(HashMap::new())),
            // What it advertises is what it serves: not the files on disk, nor
            // what the node's own mesh wants.
            available_models: Arc::new(Mutex::new(Vec::new())),
            requested_models: Arc::new(Mutex::new(Vec::new())),
            explicit_model_interests: Arc::new(Mutex::new(Vec::new())),
            requirement_mesh_state: Arc::new(Mutex::new(None)),
            #[cfg(test)]
            adopted_membership_file: None,
            mesh_id: Arc::new(Mutex::new(None)),
            mesh_policy_hash: Arc::new(Mutex::new(None)),
            signed_genesis_policy: Arc::new(Mutex::new(None)),
            bootstrap_token: Arc::new(Mutex::new(None)),
            join_targets: Arc::new(Mutex::new(Vec::new())),
            first_joined_mesh_ts: Arc::new(Mutex::new(None)),
            accepting: Arc::new((
                tokio::sync::Notify::new(),
                std::sync::atomic::AtomicBool::new(false),
            )),
            peer_change_tx,
            peer_change_rx,
            cache_affinity_inventory: Arc::new(std::sync::Mutex::new(
                mesh_llm_routing::cache_inventory::CacheInventory::default(),
            )),
            swarm_capture: Arc::new(std::sync::Mutex::new(None)),
            plugin_frame_telemetry: Arc::new(PluginFrameTelemetry::default()),
            tunnel_tx,
            tunnel_http_tx,
            stage_transport_tx,
            // No split stages, plugins, or owner control through a membership.
            native_inference: self.native_inference.clone(),
            stage_control_tx: Arc::new(Mutex::new(None)),
            stage_control_lifecycle: Arc::new(Mutex::new(None)),
            stage_transport_bridges: Arc::new(Mutex::new(HashMap::new())),
            stage_transport_aliases: Arc::new(Mutex::new(HashMap::new())),
            stage_topologies: Arc::new(Mutex::new(StageTopologyState::default())),
            plugin_manager: Arc::new(Mutex::new(None)),
            plugin_keys: crate::mesh::plugin_keys::PluginKeys::default(),
            control_listener: Arc::new(Mutex::new(None)),
            owner_lifecycle_response_cache: OwnerLifecycleResponseCache::default(),
            model_intent_tx: Arc::new(tokio::sync::Mutex::new(None)),
            #[cfg(feature = "payments")]
            payment_recovery: Arc::new(Mutex::new(None)),
            memberships: MeshMemberships::default(),
            // This membership's admission: its own allowlist, always enforced.
            owner_attestation: Arc::new(Mutex::new(Some(owner_attestation))),
            // Renewed in memory, never saved: the ownership file is the main
            // node's.
            owner_attestation_path: None,
            unsaved_owner_attestation: Arc::default(),
            owner_summary: Arc::new(Mutex::new(owner_summary)),
            trust_store: Arc::new(Mutex::new(trust_store)),
            trust_policy: TrustPolicy::Allowlist,
            enumerate_host: config.enumerate_host,
            public_mesh: false,
            // The serving state it shares with this node.
            role: self.role.clone(),
            models: self.models.clone(),
            model_source: self.model_source.clone(),
            serving_models: self.serving_models.clone(),
            served_model_descriptors: self.served_model_descriptors.clone(),
            served_model_generations: self.served_model_generations.clone(),
            model_runtime_descriptors: self.model_runtime_descriptors.clone(),
            hosted_models: self.hosted_models.clone(),
            llama_ready: self.llama_ready.clone(),
            vram_bytes: self.vram_bytes,
            local_runtime_capacity_bytes: self.local_runtime_capacity_bytes,
            host_ram_offload_gain_bytes: self.host_ram_offload_gain_bytes,
            inflight_requests: self.inflight_requests.clone(),
            inflight_change_tx: self.inflight_change_tx.clone(),
            routing_metrics: self.routing_metrics.clone(),
            routing_telemetry: self.routing_telemetry.clone(),
            local_request_metrics: self.local_request_metrics.clone(),
            runtime_data_producer: self.runtime_data_producer.clone(),
            display_name: self.display_name.clone(),
            release_attestation: self.release_attestation.clone(),
            release_attestation_summary: self.release_attestation_summary.clone(),
            gpu_name: self.gpu_name.clone(),
            hostname: self.hostname.clone(),
            is_soc: self.is_soc,
            gpu_vram: self.gpu_vram.clone(),
            gpu_reserved_bytes: self.gpu_reserved_bytes.clone(),
            advertised_memory: self.advertised_memory,
            gpu_mem_bandwidth_gbps: host_figure(&self.gpu_mem_bandwidth_gbps),
            gpu_compute_tflops_fp32: host_figure(&self.gpu_compute_tflops_fp32),
            gpu_compute_tflops_fp16: host_figure(&self.gpu_compute_tflops_fp16),
            config_state: self.config_state.clone(),
            peer_blocks: self.peer_blocks.clone(),
            config_revision_tx: self.config_revision_tx.clone(),
            #[cfg(feature = "payments")]
            payments: self.payments.clone(),
            activity_policy_guard: self.activity_policy_guard.clone(),
            drain_timeout_secs: self.drain_timeout_secs,
            drain_timeout_max_secs: self.drain_timeout_max_secs,
            runtime_intents: self.runtime_intents.clone(),
            runtime_instance_lifecycles: self.runtime_instance_lifecycles.clone(),
        };
        // The mesh ID is set before the first peer can offer its own.
        if let Err(error) = node
            .initialize_mesh_identity_as_originator(Some(config.mesh_name), None)
            .await
        {
            node.close_endpoint().await;
            self.local_endpoints.remove(endpoint_id);
            return Err(error);
        }
        node.set_first_joined_mesh_ts_if_absent(current_time_unix_ms())
            .await;
        node.spawn_accept_loop();
        node.start_accepting();
        Ok((
            node,
            TunnelChannels {
                rpc: tunnel_rx,
                http: tunnel_http_rx,
                stage: stage_transport_rx,
            },
        ))
    }

    /// An ownership certificate for a membership endpoint, signed and never
    /// saved, here or on renewal: the node-ownership file is the main node's.
    async fn sign_membership_ownership(
        &self,
        owner_keypair: &crate::crypto::OwnerKeypair,
        endpoint_id: EndpointId,
        enumerate_host: bool,
    ) -> Result<SignedNodeOwnership> {
        let node_label = self
            .owner_attestation
            .lock()
            .await
            .as_ref()
            .and_then(|ownership| ownership.claim.node_label.clone());
        Ok(sign_node_ownership(
            owner_keypair,
            endpoint_id.as_bytes(),
            current_time_unix_ms() + DEFAULT_NODE_CERT_LIFETIME_SECS * 1000,
            node_label,
            enumerate_host.then(|| self.hostname.clone()).flatten(),
        )?)
    }

    /// A membership's trust store: its own allowlist, with the node's
    /// revocations, so revoking an owner or a node certificate reaches every
    /// mesh the node serves.
    async fn membership_trust_store(&self, trusted_owners: &[String]) -> TrustStore {
        let node_store = self.trust_store.lock().await;
        TrustStore {
            revoked_owners: node_store.revoked_owners.clone(),
            revoked_node_certs: node_store.revoked_node_certs.clone(),
            revoked_node_ids: node_store.revoked_node_ids.clone(),
            ..TrustStore::default()
        }
        .merged_with_trusted_owners(trusted_owners)
    }

    /// Binds the membership's endpoint on its previous UDP port when it can.
    async fn bind_membership_endpoint(
        &self,
        config: &MembershipNodeConfig<'_>,
    ) -> Result<(Endpoint, QuicBindSelection)> {
        let bind = |port| QuicBindSelection {
            ip: self.quic_bind.ip,
            port,
        };
        if let Some(port) = config.quic_port {
            match bind_mesh_endpoint(config.secret_key.clone(), config.relay, bind(Some(port)))
                .await
            {
                Ok(endpoint) => return Ok((endpoint, bind(Some(port)))),
                Err(error) => tracing::info!(
                    port,
                    "membership's previous UDP port is unavailable, so its invite token changes: {error:#}"
                ),
            }
        }
        let endpoint =
            bind_mesh_endpoint(config.secret_key.clone(), config.relay, bind(None)).await?;
        Ok((endpoint, bind(None)))
    }
}
