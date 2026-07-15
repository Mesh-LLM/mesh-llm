use super::*;
use crate::mesh::identity_persistence::mesh_genesis_policy_path;

pub(crate) fn preflight_pushed_config_for_current_node(
    config: &crate::plugin::MeshConfig,
) -> Result<()> {
    let survey = crate::system::hardware::query(&[
        crate::system::hardware::Metric::GpuName,
        crate::system::hardware::Metric::GpuFacts,
    ]);
    preflight_pushed_config_for_current_node_with_gpus(config, &survey.gpus)
}

pub(crate) fn preflight_pushed_config_for_current_node_with_gpus(
    config: &crate::plugin::MeshConfig,
    gpus: &[crate::system::hardware::GpuFacts],
) -> Result<()> {
    if config.gpu.assignment != crate::plugin::GpuAssignment::Pinned {
        return Ok(());
    }

    for model in &config.models {
        let gpu = crate::system::hardware::resolve_pinned_gpu_strict(model.gpu_id.as_deref(), gpus)
            .map_err(anyhow::Error::new)
            .with_context(|| {
                format!(
                    "pushed config model '{}' failed pinned GPU preflight",
                    model.model
                )
            })?;

        let stable_id = gpu
            .stable_id
            .as_deref()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "pushed config model '{}' resolved pinned GPU at index {} without a stable_id",
                    model.model,
                    gpu.index
                )
            })
            .with_context(|| {
                format!(
                    "pushed config model '{}' failed pinned GPU preflight",
                    model.model
                )
            })?;

        if gpu.backend_device.is_none() {
            return Err(anyhow::anyhow!(
                "pushed config model '{}' resolved pinned GPU '{}' at index {} without a backend_device",
                model.model,
                stable_id,
                gpu.index
            ))
            .with_context(|| {
                format!(
                    "pushed config model '{}' failed pinned GPU preflight",
                    model.model
                )
            });
        }
    }

    Ok(())
}

impl Node {
    pub(crate) fn load_or_create_signed_genesis_policy(
        &self,
    ) -> Result<crate::SignedMeshGenesisPolicy> {
        let owner = self.owner_keypair.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "requirement-aware meshes require an owner identity so the genesis policy and bootstrap token can be signed"
            )
        })?;
        if let Ok(serialized) = std::fs::read(mesh_genesis_policy_path())
            && let Ok(existing) =
                serde_json::from_slice::<crate::SignedMeshGenesisPolicy>(&serialized)
            && existing.verify().is_ok()
            && existing.policy.origin_owner_id == owner.owner_id()
            && existing.policy.requirements == self.local_mesh_requirements
            && existing.origin_sign_public_key == owner.verifying_key().as_bytes().to_vec()
        {
            return Ok(existing);
        }

        let signed = crate::SignedMeshGenesisPolicy::sign(
            crate::MeshGenesisPolicy::new(
                owner.owner_id(),
                current_time_unix_ms(),
                self.local_mesh_requirements.clone(),
            )
            .map_err(|reason| anyhow::anyhow!("invalid local mesh genesis policy: {reason:?}"))?,
            owner,
        )
        .map_err(|reason| anyhow::anyhow!("failed to sign mesh genesis policy: {reason:?}"))?;
        let bytes = serde_json::to_vec_pretty(&signed).context("serialize mesh genesis policy")?;
        let path = mesh_genesis_policy_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
        crate::crypto::write_keystore_bytes_atomically(&path, &bytes)?;
        Ok(signed)
    }

    pub(crate) async fn active_mesh_policy_state(&self) -> Option<ActiveMeshPolicyState> {
        let mesh_id = self.mesh_id.lock().await.clone()?;
        let policy_hash = self.mesh_policy_hash.lock().await.clone()?;
        let policy = self.genesis_policy.lock().await.clone()?;
        Some(ActiveMeshPolicyState {
            mesh_id,
            policy_hash,
            policy,
        })
    }

    pub(crate) fn mesh_requirement_rejection_event(
        &self,
        source: MeshRequirementRejectionSource,
        peer_id: Option<EndpointId>,
        reason: MeshRequirementRejectReason,
    ) -> MeshRequirementRejectionEvent {
        MeshRequirementRejectionEvent {
            observed_at_unix_ms: current_time_unix_ms(),
            source,
            message: reason.message().to_string(),
            reason,
            peer_id: peer_id.map(|id| id.fmt_short().to_string()),
        }
    }

    pub(crate) async fn record_mesh_requirement_rejection(
        &self,
        source: MeshRequirementRejectionSource,
        peer_id: Option<EndpointId>,
        reason: MeshRequirementRejectReason,
    ) {
        let event = self.mesh_requirement_rejection_event(source.clone(), peer_id, reason.clone());
        let source_label = match source {
            MeshRequirementRejectionSource::Join => "join",
            MeshRequirementRejectionSource::Gossip => "gossip",
            MeshRequirementRejectionSource::TopologyDisclosure => "topology disclosure",
        };
        if let Some(peer_id) = event.peer_id.as_deref() {
            emit_mesh_warning(format!(
                "mesh {source_label} rejected for peer {peer_id} [{}]: {}",
                reason.code(),
                event.message
            ));
        } else {
            emit_mesh_warning(format!(
                "mesh {source_label} rejected [{}]: {}",
                reason.code(),
                event.message
            ));
        }
        tracing::warn!(
            source = source_label,
            reason = reason.code(),
            peer_id = event.peer_id.as_deref().unwrap_or(""),
            message = %event.message,
            "mesh requirement rejection"
        );
        let mut state = self.state.lock().await;
        state.recent_mesh_rejections.push_front(event);
        while state.recent_mesh_rejections.len() > RECENT_MESH_REJECTION_LIMIT {
            state.recent_mesh_rejections.pop_back();
        }
        drop(state);
        self.runtime_data_producer.mark_status_dirty();
    }

    pub(crate) async fn mesh_requirement_policy_summary(
        &self,
    ) -> Option<MeshRequirementPolicySummary> {
        self.active_mesh_policy_state()
            .await
            .map(|state| MeshRequirementPolicySummary {
                policy_hash: state.policy_hash,
                requirements: state.policy.requirements,
            })
    }

    pub(crate) async fn recent_mesh_requirement_rejections(
        &self,
    ) -> Vec<MeshRequirementRejectionEvent> {
        self.state
            .lock()
            .await
            .recent_mesh_rejections
            .iter()
            .cloned()
            .collect()
    }

    #[cfg(test)]
    pub(crate) async fn set_active_mesh_policy_for_tests(
        &self,
        policy: crate::MeshGenesisPolicy,
    ) -> MeshRequirementPolicySummary {
        let policy_hash = policy
            .canonical_hash_hex()
            .expect("policy hash should serialize");
        let mesh_id = policy
            .policy_derived_mesh_id()
            .expect("policy-derived mesh id should serialize");
        *self.mesh_id.lock().await = Some(mesh_id);
        *self.mesh_policy_hash.lock().await = Some(policy_hash.clone());
        *self.genesis_policy.lock().await = Some(policy.clone());
        MeshRequirementPolicySummary {
            policy_hash,
            requirements: policy.requirements,
        }
    }

    pub(crate) async fn install_requirement_aware_mesh_state(
        &self,
        mesh_id: String,
        policy_hash: String,
        policy: crate::MeshGenesisPolicy,
        signed_policy: Option<crate::SignedMeshGenesisPolicy>,
        bootstrap_token: Option<crate::SignedBootstrapToken>,
    ) -> Result<()> {
        let current_mesh_id = self.mesh_id().await;
        if current_mesh_id
            .as_deref()
            .is_some_and(|current| current != mesh_id.as_str())
        {
            anyhow::bail!(
                "mesh ID conflict: local mesh is '{}' but bootstrap token requires '{}'",
                current_mesh_id.unwrap_or_default(),
                mesh_id
            );
        }
        *self.mesh_policy_hash.lock().await = Some(policy_hash);
        *self.genesis_policy.lock().await = Some(policy);
        *self.signed_genesis_policy.lock().await = signed_policy;
        *self.bootstrap_token.lock().await = bootstrap_token;
        self.set_mesh_id_force(mesh_id).await;
        Ok(())
    }

    pub(crate) async fn validate_bootstrap_token(
        &self,
        token: &crate::SignedBootstrapToken,
    ) -> std::result::Result<Vec<EndpointAddr>, MeshRequirementRejectReason> {
        token.verify()?;
        if !self.local_mesh_requirements.is_unrestricted() {
            if let Some(active_policy) = self.active_mesh_policy_state().await {
                if token.policy_hash.as_str() != active_policy.policy_hash
                    || token.genesis_policy != active_policy.policy
                {
                    return Err(MeshRequirementRejectReason::MeshPolicyMismatch);
                }
            } else {
                self.local_mesh_requirements.validate()?;
                if token.genesis_policy.requirements != self.local_mesh_requirements {
                    return Err(MeshRequirementRejectReason::MeshPolicyMismatch);
                }
            }
        }
        decode_signed_bootstrap_addrs(token)
            .map_err(|_| MeshRequirementRejectReason::BootstrapTokenInvalid)
    }

    pub(crate) async fn validate_peer_announcement_against_active_policy(
        &self,
        _peer_id: EndpointId,
        ann: &PeerAnnouncement,
    ) -> std::result::Result<(), MeshRequirementRejectReason> {
        let Some(active_policy) = self.active_mesh_policy_state().await else {
            return Ok(());
        };
        if ann.mesh_id.as_deref() != Some(active_policy.mesh_id.as_str()) {
            return Err(MeshRequirementRejectReason::MeshPolicyMismatch);
        }
        if ann.mesh_policy_hash.as_deref() != Some(active_policy.policy_hash.as_str()) {
            return Err(MeshRequirementRejectReason::MeshPolicyMismatch);
        }
        if let Some(signed_policy) = ann.genesis_policy.as_ref() {
            signed_policy.verify()?;
            if signed_policy.policy != active_policy.policy {
                return Err(MeshRequirementRejectReason::MeshPolicyMismatch);
            }
            if signed_policy.policy.canonical_hash_hex()? != active_policy.policy_hash {
                return Err(MeshRequirementRejectReason::MeshPolicyMismatch);
            }
            *self.signed_genesis_policy.lock().await = Some(signed_policy.clone());
        }
        Ok(())
    }

    pub(crate) async fn validate_direct_peer_requirements(
        &self,
        peer_id: EndpointId,
        ann: &PeerAnnouncement,
        negotiated_protocol_generation: Option<u32>,
    ) -> std::result::Result<(), MeshRequirementRejectReason> {
        self.validate_peer_announcement_against_active_policy(peer_id, ann)
            .await?;

        let active_policy = self.active_mesh_policy_state().await;
        let release_attestation = peer_release_attestation_status(ann.release_attestation.as_ref());
        let direct_proof = match &active_policy {
            None => DirectPeerProofStatus::NotChecked,
            Some(active_policy) => match ann.direct_admission_proof.as_ref() {
                None => DirectPeerProofStatus::Missing,
                Some(proof) => match self.verify_direct_peer_admission_proof(
                    peer_id,
                    ann,
                    active_policy,
                    proof,
                ) {
                    Ok(()) => DirectPeerProofStatus::Verified,
                    Err(
                        err @ (MeshRequirementRejectReason::DirectProofStale
                        | MeshRequirementRejectReason::DirectProofSenderIdMismatch),
                    ) => return Err(err),
                    Err(_) => DirectPeerProofStatus::Invalid,
                },
            },
        };
        let input = crate::MeshRequirementEvaluationInput {
            advertised_node_version: ann.version.clone(),
            negotiated_protocol_generation,
            policy_hash: ann.mesh_policy_hash.clone(),
            release_attestation,
            direct_proof,
            bootstrap: crate::BootstrapStatus::NotChecked,
        };

        if let Some(active_policy) = active_policy.as_ref()
            && active_policy
                .policy
                .requirements
                .release_attestation
                .required
            && let MeshRequirementDecision::Rejected(
                reason @ (MeshRequirementRejectReason::CertifiedBinaryRequired
                | MeshRequirementRejectReason::BuildProofInvalid
                | MeshRequirementRejectReason::ReleaseSignerUntrusted
                | MeshRequirementRejectReason::BuildProofMissing),
            ) = active_policy.policy.evaluate(&input)
        {
            return Err(reason);
        }

        match evaluate_direct_peer_admission(
            active_policy.as_ref().map(|state| &state.policy),
            &input,
        ) {
            MeshRequirementDecision::Accepted => Ok(()),
            MeshRequirementDecision::Rejected(reason) => Err(reason),
        }
    }

    pub(crate) fn verify_direct_peer_admission_proof(
        &self,
        peer_id: EndpointId,
        ann: &PeerAnnouncement,
        active_policy: &ActiveMeshPolicyState,
        proof: &crate::DirectNodeAdmissionProof,
    ) -> std::result::Result<(), MeshRequirementRejectReason> {
        proof.verify_for_live_sender(peer_id.as_bytes(), current_time_unix_ms())?;
        if proof.mesh_id.trim() != active_policy.mesh_id
            || proof.policy_hash.trim() != active_policy.policy_hash
        {
            return Err(MeshRequirementRejectReason::BuildProofInvalid);
        }
        if ann.mesh_id.as_deref() != Some(proof.mesh_id.as_str())
            || ann.mesh_policy_hash.as_deref() != Some(proof.policy_hash.as_str())
        {
            return Err(MeshRequirementRejectReason::BuildProofInvalid);
        }
        let expected_attestation_hash =
            direct_admission_attestation_hash(ann.release_attestation.as_ref());
        if proof.attestation_hash.trim() != expected_attestation_hash {
            return Err(MeshRequirementRejectReason::BuildProofInvalid);
        }
        Ok(())
    }

    pub(crate) fn build_self_direct_admission_proof(
        &self,
        mesh_id: &str,
        policy_hash: &str,
        release_attestation: Option<&crate::ReleaseBuildAttestation>,
    ) -> Option<crate::DirectNodeAdmissionProof> {
        let attestation_hash = direct_admission_attestation_hash(release_attestation);
        let signing_key =
            ed25519_dalek::SigningKey::from_bytes(&self.endpoint_secret_key.to_bytes());
        let mut proof = crate::DirectNodeAdmissionProof {
            version: 1,
            sender_id: self.endpoint.id().as_bytes().to_vec(),
            mesh_id: mesh_id.to_string(),
            policy_hash: policy_hash.to_string(),
            attestation_hash,
            timestamp_unix_ms: current_time_unix_ms(),
            signature_algorithm: "ed25519".to_string(),
            signature: Vec::new(),
        };
        proof.signature = ed25519_dalek::Signer::sign(&signing_key, &proof.canonical_bytes().ok()?)
            .to_bytes()
            .to_vec();
        Some(proof)
    }
}
