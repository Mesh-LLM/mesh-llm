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
//! peers of another. Peer requests that arrive through a membership execute on
//! the local model only, as every peer request does.
//!
//! What a membership node does not share, it does not do: it runs no plugins,
//! no owner control listener, and no split-stage transport.

use super::*;
use std::collections::BTreeMap;

/// What a membership node is started with.
pub(crate) struct MembershipNodeConfig<'a> {
    pub(crate) secret_key: SecretKey,
    pub(crate) relay: RelayConfig<'a>,
    /// The owner IDs whose nodes this membership admits; nobody else.
    pub(crate) trusted_owners: &'a [String],
    pub(crate) enumerate_host: bool,
}

/// A running membership: its node and the tasks that serve it.
pub(crate) struct MembershipEntry {
    pub(crate) node: Node,
    pub(crate) api_port: u16,
    pub(crate) tasks: Vec<tokio::task::JoinHandle<()>>,
}

impl MembershipEntry {
    /// Stops the membership's tasks, tells its peers it is leaving, and
    /// closes its endpoint. The node it belongs to keeps serving. Returns once
    /// the tasks are gone, so the membership's API port is free again.
    pub(crate) async fn close(self) {
        for task in &self.tasks {
            task.abort();
        }
        for task in self.tasks {
            let _ = task.await;
        }
        self.node.broadcast_leaving().await;
        self.node.close_endpoint().await;
    }
}

/// The memberships of one node, by name.
#[derive(Clone, Default)]
pub(crate) struct MeshMemberships {
    entries: Arc<Mutex<BTreeMap<String, MembershipEntry>>>,
}

impl MeshMemberships {
    /// Adds a membership, returning the one it replaces.
    pub(crate) async fn insert(
        &self,
        name: String,
        entry: MembershipEntry,
    ) -> Option<MembershipEntry> {
        self.entries.lock().await.insert(name, entry)
    }

    pub(crate) async fn remove(&self, name: &str) -> Option<MembershipEntry> {
        self.entries.lock().await.remove(name)
    }

    pub(crate) async fn take_all(&self) -> Vec<MembershipEntry> {
        std::mem::take(&mut *self.entries.lock().await)
            .into_values()
            .collect()
    }

    /// Each membership's name, node and local API port.
    pub(crate) async fn list(&self) -> Vec<(String, Node, u16)> {
        self.entries
            .lock()
            .await
            .iter()
            .map(|(name, entry)| (name.clone(), entry.node.clone(), entry.api_port))
            .collect()
    }

    async fn nodes(&self) -> Vec<Node> {
        self.entries
            .lock()
            .await
            .values()
            .map(|entry| entry.node.clone())
            .collect()
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
    /// the mesh's members naturally includes. Signed tokens carry mesh
    /// requirements, which a membership's mesh does not have, so they are
    /// refused.
    pub(crate) async fn join_membership(&self, invite_token: &str) -> Result<bool> {
        match parse_invite_token(invite_token)
            .map_err(|reason| anyhow::anyhow!("join rejected: {}", reason.code()))?
        {
            InviteTokenMaterial::Legacy(addr) if addr.id == self.endpoint.id() => Ok(false),
            InviteTokenMaterial::Legacy(_) => self.join(invite_token).await.map(|()| true),
            InviteTokenMaterial::Signed(_) => anyhow::bail!(
                "join rejected: a membership joins by unsigned invite tokens; signed tokens carry mesh requirements"
            ),
        }
    }

    /// Closes every membership. Part of runtime shutdown, before the node's
    /// own endpoint closes.
    pub(crate) async fn close_memberships(&self) {
        for entry in self.memberships.take_all().await {
            entry.close().await;
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
        let quic_bind = QuicBindSelection {
            ip: self.quic_bind.ip,
            port: None,
        };
        let endpoint =
            bind_mesh_endpoint(config.secret_key.clone(), config.relay, quic_bind).await?;
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
        let node_label = self
            .owner_attestation
            .lock()
            .await
            .as_ref()
            .and_then(|ownership| ownership.claim.node_label.clone());
        // Signed for this endpoint, and not saved: the node-ownership file is
        // the main node's.
        let owner_attestation = sign_node_ownership(
            &owner_keypair,
            endpoint.id().as_bytes(),
            current_time_unix_ms() + DEFAULT_NODE_CERT_LIFETIME_SECS * 1000,
            node_label,
            config
                .enumerate_host
                .then(|| self.hostname.clone())
                .flatten(),
        )?;
        let trust_store = TrustStore::default().merged_with_trusted_owners(config.trusted_owners);
        let owner_summary = verify_node_ownership(
            Some(&owner_attestation),
            endpoint.id().as_bytes(),
            &trust_store,
            TrustPolicy::Off,
            current_time_unix_ms(),
        );

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
            // This membership's mesh: peers, demand, and mesh identity.
            state: Arc::new(Mutex::new(MembershipState::default())),
            plugin_message_history: Default::default(),
            direct_rescue_endpoints: super::super::direct_rescue::DirectRescueEndpoints::default(),
            host_role_claims: Arc::new(Mutex::new(HostRoleClaims::default())),
            model_demand: Arc::new(std::sync::Mutex::new(HashMap::new())),
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
            available_models: self.available_models.clone(),
            requested_models: self.requested_models.clone(),
            explicit_model_interests: self.explicit_model_interests.clone(),
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
            gpu_mem_bandwidth_gbps: self.gpu_mem_bandwidth_gbps.clone(),
            gpu_compute_tflops_fp32: self.gpu_compute_tflops_fp32.clone(),
            gpu_compute_tflops_fp16: self.gpu_compute_tflops_fp16.clone(),
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
}
