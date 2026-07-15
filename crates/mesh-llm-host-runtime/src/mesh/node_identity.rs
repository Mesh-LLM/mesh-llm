use super::*;

pub(crate) fn direct_admission_attestation_hash(
    release_attestation: Option<&crate::ReleaseBuildAttestation>,
) -> String {
    release_attestation
        .map(|attestation| {
            attestation
                .canonical_hash_hex()
                .unwrap_or_else(|_| "invalid-release-attestation".to_string())
        })
        .unwrap_or_else(|| "missing-release-attestation".to_string())
}

pub(crate) fn signed_policy_matches_owner(
    signed_policy: &crate::SignedMeshGenesisPolicy,
    policy: &crate::MeshGenesisPolicy,
    owner: &crate::crypto::OwnerKeypair,
) -> bool {
    signed_policy.policy == *policy
        && signed_policy.origin_sign_public_key.as_slice() == owner.verifying_key().as_bytes()
}

pub(crate) fn sign_requirement_bootstrap_token(
    addr: &EndpointAddr,
    policy: &crate::MeshGenesisPolicy,
    signed_policy: Option<&crate::SignedMeshGenesisPolicy>,
    owner: &crate::crypto::OwnerKeypair,
) -> Result<(crate::SignedMeshGenesisPolicy, crate::SignedBootstrapToken)> {
    let signed_policy = if let Some(signed) =
        signed_policy.filter(|signed| signed_policy_matches_owner(signed, policy, owner))
    {
        signed.clone()
    } else {
        crate::SignedMeshGenesisPolicy::sign(policy.clone(), owner)
            .map_err(|reason| anyhow::anyhow!("failed to sign genesis policy: {reason:?}"))?
    };
    let token = crate::SignedBootstrapToken::sign(
        vec![serde_json::to_vec(addr).expect("serializable endpoint addr")],
        &signed_policy,
        Some(current_time_unix_ms() + SIGNED_BOOTSTRAP_TOKEN_LIFETIME_MS),
        owner,
    )
    .map_err(|reason| anyhow::anyhow!("failed to sign bootstrap token: {reason:?}"))?;
    Ok((signed_policy, token))
}

pub(crate) fn encode_signed_bootstrap_token(token: &crate::SignedBootstrapToken) -> String {
    let json = serde_json::to_vec(token).expect("serializable bootstrap token");
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(json)
}

pub(crate) fn signed_bootstrap_token_matches_invite_context(
    token: &crate::SignedBootstrapToken,
    addr: &EndpointAddr,
    mesh_id: &str,
    policy_hash: &str,
    policy: &crate::MeshGenesisPolicy,
) -> bool {
    if token.mesh_id != mesh_id
        || token.policy_hash != policy_hash
        || token.genesis_policy != *policy
    {
        return false;
    }
    match decode_signed_bootstrap_addrs(token) {
        Ok(addrs) => addrs.iter().any(|cached_addr| cached_addr == addr),
        Err(_) => false,
    }
}

impl Node {
    pub async fn initialize_mesh_identity_as_originator(
        &self,
        name: Option<&str>,
        nostr_pubkey: Option<&str>,
    ) -> Result<String> {
        if self.local_mesh_requirements.is_unrestricted() {
            let mesh_id = generate_mesh_id(name, nostr_pubkey);
            self.set_mesh_id_force(mesh_id.clone()).await;
            return Ok(mesh_id);
        }

        let signed_policy = self.load_or_create_signed_genesis_policy()?;
        let policy_hash = signed_policy
            .policy
            .canonical_hash_hex()
            .map_err(|reason| anyhow::anyhow!("invalid local mesh policy hash: {reason:?}"))?;
        let mesh_id = signed_policy
            .policy
            .policy_derived_mesh_id()
            .map_err(|reason| anyhow::anyhow!("invalid policy-derived mesh ID: {reason:?}"))?;
        self.install_requirement_aware_mesh_state(
            mesh_id.clone(),
            policy_hash,
            signed_policy.policy.clone(),
            Some(signed_policy),
            None,
        )
        .await?;
        Ok(mesh_id)
    }

    pub async fn invite_token(&self) -> String {
        let mut addr = self.endpoint_addr_for_advertisement();
        // Inject STUN-discovered public address if relay STUN didn't provide one.
        if let Some(pub_addr) = self.public_addr
            && !endpoint_addr_has_public_ipv4(&addr)
        {
            addr.addrs.insert(TransportAddr::Ip(pub_addr));
        }
        addr = filter_endpoint_addr_for_bind_ip(
            addr,
            self.quic_bind.ip,
            self.relay_policy.uses_raw_stun(),
        );
        let mesh_id = self.mesh_id.lock().await.clone();
        let policy_hash = self.mesh_policy_hash.lock().await.clone();
        let policy = self.genesis_policy.lock().await.clone();
        let signed_policy_guard = self.signed_genesis_policy.lock().await.clone();
        let cached_token = self.bootstrap_token.lock().await.clone();

        if let (Some(mesh_id), Some(policy_hash), Some(policy)) = (mesh_id, policy_hash, policy) {
            return self
                .requirement_aware_invite_token(
                    &addr,
                    mesh_id,
                    policy_hash,
                    policy,
                    signed_policy_guard,
                    cached_token,
                )
                .await;
        }

        if let Some(token) = self.valid_cached_bootstrap_token(cached_token).await {
            return encode_signed_bootstrap_token(&token);
        }
        encode_endpoint_addr_token(&addr)
    }

    pub(crate) async fn requirement_aware_invite_token(
        &self,
        addr: &EndpointAddr,
        mesh_id: String,
        policy_hash: String,
        policy: crate::MeshGenesisPolicy,
        signed_policy: Option<crate::SignedMeshGenesisPolicy>,
        cached_token: Option<crate::SignedBootstrapToken>,
    ) -> String {
        if let Some(token) = self
            .matching_cached_invite_token(
                cached_token.clone(),
                addr,
                &mesh_id,
                &policy_hash,
                &policy,
            )
            .await
        {
            return encode_signed_bootstrap_token(&token);
        }

        if let Some(invite_token) = self
            .sign_requirement_invite_token(
                addr,
                &mesh_id,
                &policy_hash,
                &policy,
                signed_policy.as_ref(),
            )
            .await
        {
            return invite_token;
        }

        if let Some(token) = self.valid_cached_bootstrap_token(cached_token).await {
            return encode_signed_bootstrap_token(&token);
        }

        tracing::warn!(
            "requirement-aware mesh has no valid signed bootstrap token; refusing to emit legacy invite token"
        );
        String::new()
    }

    pub(crate) async fn matching_cached_invite_token(
        &self,
        cached_token: Option<crate::SignedBootstrapToken>,
        addr: &EndpointAddr,
        mesh_id: &str,
        policy_hash: &str,
        policy: &crate::MeshGenesisPolicy,
    ) -> Option<crate::SignedBootstrapToken> {
        let token = self.valid_cached_bootstrap_token(cached_token).await?;
        signed_bootstrap_token_matches_invite_context(&token, addr, mesh_id, policy_hash, policy)
            .then_some(token)
    }

    pub(crate) async fn sign_requirement_invite_token(
        &self,
        addr: &EndpointAddr,
        mesh_id: &str,
        policy_hash: &str,
        policy: &crate::MeshGenesisPolicy,
        signed_policy: Option<&crate::SignedMeshGenesisPolicy>,
    ) -> Option<String> {
        let owner = self.requirement_origin_owner(policy, signed_policy)?;
        match sign_requirement_bootstrap_token(addr, policy, signed_policy, owner) {
            Ok((signed_policy, token)) => {
                *self.signed_genesis_policy.lock().await = Some(signed_policy);
                *self.bootstrap_token.lock().await = Some(token.clone());
                debug_assert_eq!(mesh_id, token.mesh_id);
                debug_assert_eq!(policy_hash, token.policy_hash);
                Some(encode_signed_bootstrap_token(&token))
            }
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "failed to sign requirement-aware bootstrap token; refusing to emit legacy invite token"
                );
                Some(String::new())
            }
        }
    }

    pub(crate) async fn valid_cached_bootstrap_token(
        &self,
        cached_token: Option<crate::SignedBootstrapToken>,
    ) -> Option<crate::SignedBootstrapToken> {
        if let Some(token) = cached_token {
            if token.verify_at(current_time_unix_ms()).is_ok() {
                return Some(token);
            }
            *self.bootstrap_token.lock().await = None;
        }
        None
    }

    pub(crate) fn requirement_origin_owner(
        &self,
        policy: &crate::MeshGenesisPolicy,
        signed_policy: Option<&crate::SignedMeshGenesisPolicy>,
    ) -> Option<&crate::crypto::OwnerKeypair> {
        self.owner_keypair.as_ref().filter(|owner| {
            signed_policy.is_some_and(|signed| signed_policy_matches_owner(signed, policy, owner))
                || policy.origin_owner_id == owner.owner_id()
        })
    }

    pub(crate) fn endpoint_addr_for_advertisement(&self) -> EndpointAddr {
        let mut addr = self.endpoint.addr();
        if self.quic_bind.ip.is_some() {
            addr = filter_endpoint_addr_for_bind_ip(
                addr,
                self.quic_bind.ip,
                self.relay_policy.uses_raw_stun(),
            );
        }
        addr
    }

    /// The local node's reachable [`EndpointAddr`], filtered to the bound LAN
    /// interface in the same way the invite token is. Used by mDNS reverse-dial
    /// so a host can advertise (and peers can learn) a direct address to dial
    /// back on the working direction.
    pub fn advertised_endpoint_addr(&self) -> EndpointAddr {
        self.endpoint_addr_for_advertisement()
    }

    /// Dial a peer by its [`EndpointAddr`] directly (no token decode).
    ///
    /// Used by mDNS reverse-dial: when a relay-less direct connection cannot be
    /// established in one direction (multi-homed initiator), the other side
    /// dials back on the direction that works.
    pub async fn dial_peer_addr(&self, addr: EndpointAddr) -> Result<()> {
        self.state.lock().await.dead_peers.remove(&addr.id);
        self.connect_to_peer(addr).await
    }

    /// The set of peer endpoint IDs we currently hold a connection to.
    ///
    /// Used by mDNS reverse-dial to avoid redialing already-connected peers.
    pub async fn connected_peer_ids(&self) -> std::collections::HashSet<EndpointId> {
        self.state
            .lock()
            .await
            .connections
            .keys()
            .copied()
            .collect()
    }

    /// LAN IPv4 socket addresses of all known peers (from gossip/tokens),
    /// regardless of connection state. Used by the LAN beacon to unicast a
    /// dial-back hint directly to peers when multicast is unavailable.
    pub async fn known_peer_lan_ipv4(&self) -> Vec<std::net::SocketAddrV4> {
        let state = self.state.lock().await;
        let mut out = Vec::new();
        for peer in state.peers.values() {
            out.extend(lan_bootstrap::lan_ipv4_candidates(&peer.addr));
        }
        out
    }

    /// Decode an invite token into an [`EndpointAddr`] without connecting.
    /// Returns `Err` if the token is not valid base64 or not valid JSON.
    pub fn decode_invite_token(invite_token: &str) -> Result<EndpointAddr> {
        match parse_invite_token(invite_token)
            .map_err(|reason| anyhow::anyhow!("invite token rejected: {}", reason.code()))?
        {
            InviteTokenMaterial::Legacy(addr) => Ok(addr),
            InviteTokenMaterial::Signed(token) => {
                token.verify().map_err(|reason| {
                    anyhow::anyhow!("invite token rejected: {}", reason.code())
                })?;
                decode_signed_bootstrap_addrs(&token)?
                    .into_iter()
                    .next()
                    .ok_or_else(|| {
                        anyhow::anyhow!("bootstrap token does not contain any endpoint addresses")
                    })
            }
        }
    }

    #[cfg(test)]
    pub async fn sync_from_peer_for_tests(&self, remote: &Self) {
        let remote_id = remote.endpoint.id();
        let their_announcements = remote.collect_announcements().await;
        for ann in &their_announcements {
            if ann.addr.id == self.endpoint.id() {
                continue;
            }
            if ann.addr.id == remote_id {
                if let Some(ref their_id) = ann.mesh_id {
                    self.set_mesh_id(their_id.clone()).await;
                }
                self.merge_remote_demand(&ann.model_demand);
                self.add_peer(
                    remote_id,
                    ann.addr.clone(),
                    ann,
                    Some(NODE_PROTOCOL_GENERATION),
                )
                .await;
            } else {
                self.update_transitive_peer(ann.addr.id, &ann.addr, ann, remote_id)
                    .await;
            }
        }
    }

    pub(crate) async fn build_mesh_event(
        &self,
        kind: crate::plugin::proto::mesh_event::Kind,
        peer: Option<crate::plugin::proto::MeshPeer>,
        detail_json: String,
    ) -> crate::plugin::proto::MeshEvent {
        crate::plugin::proto::MeshEvent {
            kind: kind as i32,
            peer,
            local_peer_id: endpoint_id_hex(self.endpoint.id()),
            mesh_id: self.mesh_id.lock().await.clone().unwrap_or_default(),
            detail_json,
        }
    }

    /// Enable accepting inbound connections. Call before join() or when ready to participate.
    /// Until this is called, the accept loop blocks waiting.
    pub fn start_accepting(&self) {
        self.accepting
            .1
            .store(true, std::sync::atomic::Ordering::Release);
        self.accepting.0.notify_waiters();
        let node = self.clone();
        tokio::spawn(async move {
            let plugin_manager = node.plugin_manager.lock().await.clone();
            if let Some(plugin_manager) = plugin_manager {
                let _ = plugin_manager
                    .broadcast_mesh_event(
                        node.build_mesh_event(
                            crate::plugin::proto::mesh_event::Kind::LocalAccepting,
                            None,
                            String::new(),
                        )
                        .await,
                    )
                    .await;
            }
        });
    }

    pub async fn join(&self, invite_token: &str) -> Result<()> {
        let addr = match parse_invite_token(invite_token)
            .map_err(|reason| anyhow::anyhow!("join rejected: {}", reason.code()))?
        {
            InviteTokenMaterial::Legacy(addr) => addr,
            InviteTokenMaterial::Signed(token) => {
                let addrs = match self.validate_bootstrap_token(&token).await {
                    Ok(addrs) => addrs,
                    Err(reason) => {
                        self.record_mesh_requirement_rejection(
                            MeshRequirementRejectionSource::Join,
                            None,
                            reason.clone(),
                        )
                        .await;
                        return Err(anyhow::anyhow!("join rejected: {}", reason.code()));
                    }
                };
                self.install_requirement_aware_mesh_state(
                    token.mesh_id.clone(),
                    token.policy_hash.clone(),
                    token.genesis_policy.clone(),
                    None,
                    Some(*token),
                )
                .await?;
                addrs.into_iter().next().ok_or_else(|| {
                    anyhow::anyhow!("bootstrap token does not contain any endpoint addresses")
                })?
            }
        };
        // Clear dead status — explicit join should always attempt connection
        self.state.lock().await.dead_peers.remove(&addr.id);
        self.remember_join_target(addr.clone()).await;
        self.connect_to_peer(addr).await
    }

    /// Record a join target address so the LAN beacon can unicast a dial-back
    /// hint to it even before a direct connection forms.
    ///
    /// If a target with the same endpoint id is already recorded, its address
    /// is replaced with the newer one. A peer that restarts or rebinds to a new
    /// QUIC port advertises a fresh `EndpointAddr` under the same id, and the
    /// beacon must dial that rather than keep unicasting to the stale socket.
    pub(crate) async fn remember_join_target(&self, addr: EndpointAddr) {
        let mut targets = self.join_targets.lock().await;
        if let Some(existing) = targets.iter_mut().find(|t| t.id == addr.id) {
            *existing = addr;
        } else {
            targets.push(addr);
        }
    }

    /// LAN IPv4 socket addresses of recorded join targets (from invite tokens),
    /// used by the LAN beacon for dial-back unicast before peers are connected.
    pub async fn join_target_lan_ipv4(&self) -> Vec<std::net::SocketAddrV4> {
        let targets = self.join_targets.lock().await;
        let mut out = Vec::new();
        for addr in targets.iter() {
            out.extend(lan_bootstrap::lan_ipv4_candidates(addr));
        }
        out
    }

    /// Like [`join`], but retries once after a delay on transient (connect/timeout)
    /// errors.  Decode errors (invalid base64/JSON) fail immediately.
    pub async fn join_with_retry(&self, invite_token: &str) -> Result<()> {
        let addr = match parse_invite_token(invite_token)
            .map_err(|reason| anyhow::anyhow!("join rejected: {}", reason.code()))?
        {
            InviteTokenMaterial::Legacy(addr) => addr,
            InviteTokenMaterial::Signed(token) => {
                let addrs = match self.validate_bootstrap_token(&token).await {
                    Ok(addrs) => addrs,
                    Err(reason) => {
                        self.record_mesh_requirement_rejection(
                            MeshRequirementRejectionSource::Join,
                            None,
                            reason.clone(),
                        )
                        .await;
                        return Err(anyhow::anyhow!("join rejected: {}", reason.code()));
                    }
                };
                self.install_requirement_aware_mesh_state(
                    token.mesh_id.clone(),
                    token.policy_hash.clone(),
                    token.genesis_policy.clone(),
                    None,
                    Some(*token),
                )
                .await?;
                addrs.into_iter().next().ok_or_else(|| {
                    anyhow::anyhow!("bootstrap token does not contain any endpoint addresses")
                })?
            }
        };

        // Three attempts with increasing backoff.  Relay-only joins need
        // WebSocket setup + QUIC handshake at high RTT — two attempts at
        // 15s were not enough.  Three at 30s with 5s/10s gaps give ~105s
        // total budget which covers all but the worst relay conditions.
        let backoffs = [5, 10];
        self.state.lock().await.dead_peers.remove(&addr.id);
        self.remember_join_target(addr.clone()).await;
        let mut last_err = match self.connect_to_peer(addr.clone()).await {
            Ok(()) => return Ok(()),
            Err(e) => e,
        };
        for (attempt, delay_secs) in backoffs.iter().enumerate() {
            tracing::info!(
                "Join attempt {} failed ({last_err:#}), retrying in {delay_secs}s...",
                attempt + 1
            );
            tokio::time::sleep(std::time::Duration::from_secs(*delay_secs)).await;
            self.state.lock().await.dead_peers.remove(&addr.id);
            match self.connect_to_peer(addr.clone()).await {
                Ok(()) => return Ok(()),
                Err(e) => last_err = e,
            }
        }
        Err(last_err)
    }
}
