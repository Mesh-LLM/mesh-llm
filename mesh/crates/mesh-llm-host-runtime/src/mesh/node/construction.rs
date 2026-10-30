//! Node construction with explicit identity/storage ownership.

use super::startup::*;
use super::*;

pub(crate) struct ScopedNodeIdentity {
    pub key: SecretKey,
    pub state_dir: std::path::PathBuf,
}

pub(crate) struct NodeStartupSpec<'a> {
    pub role: NodeRole,
    pub relay: RelayConfig<'a>,
    pub quic_bind: QuicBindSelection,
    pub max_vram_gb: Option<f64>,
    pub enumerate_host: bool,
    pub owner_config: Option<OwnerRuntimeConfig>,
    pub config_path: Option<&'a std::path::Path>,
    pub local_mesh_requirements: crate::MeshRequirements,
    pub identity: Option<ScopedNodeIdentity>,
}

impl Node {
    #[expect(
        clippy::too_many_arguments,
        reason = "preserve the existing public constructor while scoped startup uses NodeStartupSpec"
    )]
    /// Start an ordinary node with the process-default identity and storage.
    /// Returns transport receivers for the caller to wire into runtime services.
    pub async fn start(
        role: NodeRole,
        relay: RelayConfig<'_>,
        quic_bind: QuicBindSelection,
        max_vram_gb: Option<f64>,
        enumerate_host: bool,
        owner_config: Option<OwnerRuntimeConfig>,
        config_path: Option<&std::path::Path>,
        local_mesh_requirements: crate::MeshRequirements,
    ) -> Result<(Self, TunnelChannels)> {
        Self::start_with_spec(NodeStartupSpec {
            role,
            relay,
            quic_bind,
            max_vram_gb,
            enumerate_host,
            owner_config,
            config_path,
            local_mesh_requirements,
            identity: None,
        })
        .await
    }

    /// Construct a node with explicit transport, ownership and optional scoped identity.
    ///
    /// A scoped identity uses its own persisted membership and advertises no split
    /// capacity. The caller must attach any release proof before accepting or joining peers.
    pub(crate) async fn start_with_spec(
        spec: NodeStartupSpec<'_>,
    ) -> Result<(Self, TunnelChannels)> {
        let NodeStartupSpec {
            role,
            relay,
            quic_bind,
            max_vram_gb,
            enumerate_host,
            owner_config,
            config_path,
            local_mesh_requirements,
            identity,
        } = spec;
        let secret_key = match identity.as_ref() {
            Some(identity) => identity.key.clone(),
            None => startup_secret_key(&role).await?,
        };
        let (endpoint, public_addr) =
            bind_discoverable_endpoint(secret_key.clone(), relay, quic_bind).await?;

        let (peer_change_tx, peer_change_rx) = watch::channel(0usize);
        let (inflight_change_tx, _inflight_change_rx) = watch::channel(0u64);
        let (tunnel_tx, tunnel_rx) = tokio::sync::mpsc::channel(256);
        let (tunnel_http_tx, tunnel_http_rx) = tokio::sync::mpsc::channel(256);
        let (stage_transport_tx, stage_transport_rx) = tokio::sync::mpsc::channel(256);

        let config_state_init = {
            let path = crate::plugin::config_path(config_path)
                .unwrap_or_else(|_| std::path::PathBuf::from("config.toml"));
            crate::runtime::config_state::ConfigState::load(&path)?
        };
        let mut hardware =
            advertised_hardware_for_start(config_state_init.config(), &role, max_vram_gb);
        if identity.is_some() {
            hardware.vram_bytes = 0;
            hardware.local_runtime_capacity_bytes = 0;
            hardware.host_ram_capacity_bytes = 0;
            hardware.host_ram_offload_gain_bytes = 0;
        }
        let owner_runtime = init_owner_runtime(
            owner_config.as_ref(),
            endpoint.id(),
            hardware.hostname.clone(),
            identity
                .as_ref()
                .map(|identity| identity.state_dir.as_path()),
        )?;
        let owner_summary = verify_node_ownership(
            owner_runtime.owner_attestation.as_ref(),
            endpoint.id().as_bytes(),
            &owner_runtime.trust_store,
            TrustPolicy::Off,
            current_time_unix_ms(),
        );
        let config_revision_init = config_state_init.revision();
        let runtime_data_collector = crate::runtime_data::RuntimeDataCollector::new();
        let runtime_data_producer =
            runtime_data_collector.producer(crate::runtime_data::RuntimeDataSource {
                scope: "routing",
                plugin_data_key: None,
                plugin_endpoint_key: None,
            });

        let owner_keypair = owner_config
            .as_ref()
            .and_then(|config| config.keypair.clone());
        let policy = StartupPolicies::new(owner_config.as_ref());

        let node = Node {
            endpoint,
            endpoint_secret_key: secret_key.clone(),
            public_addr,
            quic_bind,
            relay_policy: relay.policy,
            owner_keypair,
            local_mesh_requirements,
            state: Arc::new(Mutex::new(MembershipState::default())),
            plugin_message_history: Default::default(),
            direct_rescue_endpoints: super::direct_rescue::DirectRescueEndpoints::default(),
            role: Arc::new(Mutex::new(role)),
            host_role_claims: Arc::new(Mutex::new(HostRoleClaims::default())),
            models: Arc::new(Mutex::new(Vec::new())),
            model_source: Arc::new(Mutex::new(None)),
            serving_models: Arc::new(Mutex::new(Vec::new())),
            served_model_descriptors: Arc::new(Mutex::new(Vec::new())),
            served_model_generations: Arc::new(Mutex::new(HashMap::new())),
            model_runtime_descriptors: Arc::new(Mutex::new(Vec::new())),
            hosted_models: Arc::new(Mutex::new(Vec::new())),
            llama_ready: Arc::new(Mutex::new(false)),
            available_models: Arc::new(Mutex::new(Vec::new())),
            requested_models: Arc::new(Mutex::new(Vec::new())),
            explicit_model_interests: Arc::new(Mutex::new(Vec::new())),
            model_demand: Arc::new(std::sync::Mutex::new(HashMap::new())),
            requirement_mesh_state: Arc::new(Mutex::new(None)),
            adopted_membership_file: identity
                .as_ref()
                .map(|identity| identity.state_dir.join("adopted_membership.json")),
            identity_state_dir: identity.as_ref().map(|identity| identity.state_dir.clone()),
            full_model_only: identity.is_some(),
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
            vram_bytes: hardware.vram_bytes,
            local_runtime_capacity_bytes: hardware.local_runtime_capacity_bytes,
            host_ram_capacity_bytes: hardware.host_ram_capacity_bytes,
            host_ram_offload_gain_bytes: hardware.host_ram_offload_gain_bytes,
            peer_change_tx,
            peer_change_rx,
            inflight_requests: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            inflight_change_tx,
            routing_metrics: crate::network::metrics::RoutingMetrics::default(),
            routing_telemetry: Arc::new(std::sync::Mutex::new(None)),
            cache_affinity_inventory: Arc::new(std::sync::Mutex::new(
                mesh_llm_routing::cache_inventory::CacheInventory::default(),
            )),
            swarm_capture: Arc::new(std::sync::Mutex::new(None)),
            local_request_metrics: Arc::new(LocalRequestMetricsSampler::default()),
            plugin_frame_telemetry: Arc::new(PluginFrameTelemetry::default()),
            runtime_data_producer,
            tunnel_tx,
            tunnel_http_tx,
            stage_transport_tx,
            stage_control_tx: Arc::new(Mutex::new(None)),
            stage_control_lifecycle: Arc::new(Mutex::new(None)),
            stage_transport_bridges: Arc::new(Mutex::new(HashMap::new())),
            stage_transport_aliases: Arc::new(Mutex::new(HashMap::new())),
            stage_topologies: Arc::new(Mutex::new(StageTopologyState::default())),
            plugin_manager: Arc::new(Mutex::new(None)),
            display_name: Arc::new(Mutex::new(None)),
            owner_attestation: Arc::new(Mutex::new(owner_runtime.owner_attestation)),
            plugin_keys: crate::mesh::plugin_keys::PluginKeys::default(),
            release_attestation: Arc::new(Mutex::new(None)),
            release_attestation_summary: Arc::new(Mutex::new(
                crate::ReleaseAttestationSummary::default(),
            )),
            owner_summary: Arc::new(Mutex::new(owner_summary)),
            control_listener: Arc::new(Mutex::new(None)),
            trust_store: Arc::new(Mutex::new(owner_runtime.trust_store)),
            trust_policy: owner_runtime.trust_policy,
            enumerate_host,
            gpu_name: hardware.gpu_name,
            hostname: hardware.hostname,
            is_soc: hardware.is_soc,
            gpu_vram: hardware.gpu_vram,
            gpu_reserved_bytes: hardware.gpu_reserved_bytes,
            advertised_memory: hardware.memory,
            gpu_mem_bandwidth_gbps: Arc::new(tokio::sync::Mutex::new(None)),
            gpu_compute_tflops_fp32: Arc::new(tokio::sync::Mutex::new(None)),
            gpu_compute_tflops_fp16: Arc::new(tokio::sync::Mutex::new(None)),
            config_state: Arc::new(tokio::sync::Mutex::new(config_state_init)),
            peer_blocks: identity.as_ref().map_or_else(
                crate::network::peer_blocks::PeerBlocks::for_this_node,
                |identity| crate::network::peer_blocks::PeerBlocks::load(&identity.state_dir),
            ),
            config_revision_tx: {
                let (tx, _rx) = tokio::sync::watch::channel(config_revision_init);
                Arc::new(tx)
            },
            #[cfg(feature = "payments")]
            payments: Arc::new(tokio::sync::OnceCell::new()),
            #[cfg(feature = "payments")]
            payment_recovery: Arc::new(Mutex::new(None)),
            activity_policy_guard: policy.activity,
            public_mesh: policy.public_mesh,
            drain_timeout_secs: policy.drain_timeout_secs,
            drain_timeout_max_secs: policy.drain_timeout_max_secs,
            runtime_intents: Arc::new(std::sync::Mutex::new(Vec::new())),
            runtime_instance_lifecycles: Arc::new(std::sync::Mutex::new(HashMap::new())),
            owner_lifecycle_response_cache: OwnerLifecycleResponseCache::default(),
            model_intent_tx: Arc::new(tokio::sync::Mutex::new(None)),
            memberships: MeshMemberships::default(),
            owner_attestation_path: saved_owner_attestation_path(
                identity
                    .as_ref()
                    .map(|identity| identity.state_dir.as_path()),
            ),
            unsaved_owner_attestation: Arc::default(),
            local_endpoints: LocalEndpoints::default(),
            is_membership: false,
        };

        node.maybe_start_control_listener(
            secret_key,
            owner_config.as_ref().and_then(|config| config.control_bind),
            owner_config
                .as_ref()
                .and_then(|config| config.control_advertise_addr),
        )
        .await?;

        // Accept loop starts but waits for start_accepting() before processing connections.
        // This lets a node exist before it is ready to accept mesh traffic.
        node.spawn_accept_loop();

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

struct StartupPolicies {
    activity: crate::runtime::activity_policy::ActivityPolicyGuard,
    public_mesh: bool,
    drain_timeout_secs: u64,
    drain_timeout_max_secs: u64,
}

impl StartupPolicies {
    /// Resolve activity, public-mesh and drain policies from optional owner settings.
    fn new(owner: Option<&OwnerRuntimeConfig>) -> Self {
        let activity = owner
            .map(|owner| owner.activity.clone())
            .unwrap_or_default();
        Self {
            activity: crate::runtime::activity_policy::ActivityPolicyGuard::new(&activity),
            public_mesh: owner.is_some_and(|owner| owner.public_mesh),
            drain_timeout_secs: owner
                .map_or(mesh_llm_config::DEFAULT_DRAIN_TIMEOUT_SECS, |owner| {
                    owner.drain_timeout_secs
                }),
            drain_timeout_max_secs: owner
                .map_or(mesh_llm_config::DEFAULT_DRAIN_TIMEOUT_MAX_SECS, |owner| {
                    owner.drain_timeout_max_secs
                }),
        }
    }
}

/// Bind the endpoint and obtain discovery addresses permitted by the relay policy.
/// Relay readiness is bounded; LAN-only policy skips public STUN.
async fn bind_discoverable_endpoint(
    secret_key: SecretKey,
    relay: RelayConfig<'_>,
    quic_bind: QuicBindSelection,
) -> Result<(Endpoint, Option<crate::mesh::stun::PublicAddr>)> {
    let endpoint = bind_mesh_endpoint(secret_key, relay, quic_bind).await?;
    if relay.policy.uses_relay() {
        // Ensure invite tokens can include the relay; sinkholed networks still
        // proceed with direct UDP after the bounded wait.
        wait_for_endpoint_online(
            &endpoint,
            "Relay connected",
            "Relay connection timed out (5s) — proceeding without relay",
        )
        .await;
    }
    // Probe the endpoint's actual sockets. A configured bind port is not proof
    // of the externally visible NAT-mapped address.
    let public_addr = if relay.policy.uses_raw_stun() {
        stun_public_addr(&endpoint, relay.policy.uses_relay()).await
    } else {
        tracing::info!("Raw STUN: disabled by LAN-only discovery mode");
        None
    };
    Ok((endpoint, public_addr))
}
