//! Persisted requirement-aware membership and conflict-checked state transitions.
//!
//! Paths are supplied by the host; signatures and policy identity are verified
//! before saved membership is returned or replaced.

use crate::requirements::{
    MeshGenesisPolicy, MeshRequirementRejectReason, SignedBootstrapToken, SignedMeshGenesisPolicy,
};
use anyhow::{Context, Result};
use iroh::EndpointAddr;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequirementAwareMeshState {
    pub mesh_id: String,
    pub policy_hash: String,
    pub policy: MeshGenesisPolicy,
    pub signed_policy: Option<SignedMeshGenesisPolicy>,
    pub bootstrap_token: Option<SignedBootstrapToken>,
}

fn same_requirement_mesh(
    left: &RequirementAwareMeshState,
    right: &RequirementAwareMeshState,
) -> bool {
    left.mesh_id == right.mesh_id
        && left.policy_hash == right.policy_hash
        && left.policy == right.policy
}

pub fn install_requirement_mesh_state_transition(
    current: &mut Option<RequirementAwareMeshState>,
    requested: RequirementAwareMeshState,
) -> Result<bool> {
    if let Some(installed) = current.as_ref()
        && !same_requirement_mesh(installed, &requested)
    {
        anyhow::bail!(
            "mesh ID conflict: local mesh is '{}' but bootstrap token requires '{}'",
            installed.mesh_id,
            requested.mesh_id
        );
    }
    let was_empty = current.is_none();
    *current = Some(requested);
    Ok(was_empty)
}

const MAX_ADOPTED_PEER_ADDRS: usize = 16;

#[derive(Debug, Serialize, Deserialize)]
pub struct AdoptedMeshMembership {
    pub mesh_id: String,
    pub policy_hash: String,
    pub signed_policy: SignedMeshGenesisPolicy,
    #[serde(default)]
    pub peer_addrs: Vec<EndpointAddr>,
}

impl AdoptedMeshMembership {
    fn verify(&self) -> std::result::Result<(), MeshRequirementRejectReason> {
        self.signed_policy.verify()?;
        if self.signed_policy.policy.policy_derived_mesh_id()? != self.mesh_id
            || self.signed_policy.policy.canonical_hash_hex()? != self.policy_hash
        {
            return Err(MeshRequirementRejectReason::MeshPolicyMismatch);
        }
        Ok(())
    }

    pub fn matches_token(&self, token: &SignedBootstrapToken) -> bool {
        self.mesh_id == token.mesh_id
            && self.policy_hash == token.policy_hash
            && self.signed_policy.policy == token.genesis_policy
            && self.signed_policy.origin_sign_public_key == token.origin_sign_public_key
    }
}

pub fn load_adopted_mesh_membership(
    path: &std::path::Path,
) -> Result<Option<AdoptedMeshMembership>> {
    let serialized = match std::fs::read(path) {
        Ok(serialized) => serialized,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).with_context(|| format!("read {}", path.display())),
    };
    let membership = serde_json::from_slice::<AdoptedMeshMembership>(&serialized)
        .with_context(|| format!("parse {}", path.display()))?;
    membership
        .verify()
        .map_err(|reason| anyhow::anyhow!("verify adopted mesh membership: {reason:?}"))?;
    Ok(Some(membership))
}

fn preferred_adopted_peer_addrs(mut peer_addrs: Vec<EndpointAddr>) -> Vec<EndpointAddr> {
    // Callers put current addresses first; keep their priority and freshest value.
    let mut seen = std::collections::HashSet::new();
    peer_addrs.retain(|addr| seen.insert(addr.id));
    peer_addrs.truncate(MAX_ADOPTED_PEER_ADDRS);
    peer_addrs
}

pub fn persist_adopted_mesh_membership(
    path: &std::path::Path,
    state: &RequirementAwareMeshState,
    peer_addrs: Vec<EndpointAddr>,
) -> Result<()> {
    let Some(signed_policy) = state.signed_policy.clone() else {
        return Ok(());
    };
    let peer_addrs = preferred_adopted_peer_addrs(peer_addrs);
    let membership = AdoptedMeshMembership {
        mesh_id: state.mesh_id.clone(),
        policy_hash: state.policy_hash.clone(),
        signed_policy,
        peer_addrs,
    };
    membership
        .verify()
        .map_err(|reason| anyhow::anyhow!("verify adopted mesh membership: {reason:?}"))?;
    let bytes = serde_json::to_vec_pretty(&membership).context("serialize adopted membership")?;
    if std::fs::read(path).is_ok_and(|existing| existing == bytes) {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    mesh_llm_identity::keystore::write_keystore_bytes_atomically(path, &bytes)?;
    Ok(())
}

pub fn enrich_requirement_mesh_state(
    current: &mut Option<RequirementAwareMeshState>,
    verified: &RequirementAwareMeshState,
    signed_policy: SignedMeshGenesisPolicy,
) -> std::result::Result<(), MeshRequirementRejectReason> {
    let Some(installed) = current.as_mut() else {
        return Err(MeshRequirementRejectReason::MeshPolicyMismatch);
    };
    if !same_requirement_mesh(installed, verified) {
        return Err(MeshRequirementRejectReason::MeshPolicyMismatch);
    }
    installed.signed_policy = Some(signed_policy);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use iroh::{SecretKey, TransportAddr};

    fn requirement_state(
        mesh_id: &str,
        owner: &mesh_llm_identity::OwnerKeypair,
    ) -> RequirementAwareMeshState {
        let policy = MeshGenesisPolicy::new(
            owner.owner_id(),
            1_717_171_717_000,
            crate::requirements::MeshRequirements::default(),
        )
        .expect("test policy");
        RequirementAwareMeshState {
            mesh_id: mesh_id.to_string(),
            policy_hash: policy.canonical_hash_hex().expect("policy hash"),
            policy,
            signed_policy: None,
            bootstrap_token: None,
        }
    }

    #[test]
    fn current_peer_precedes_history_and_keeps_latest_address() {
        let mut addresses: Vec<_> = (0..=MAX_ADOPTED_PEER_ADDRS)
            .map(|_| EndpointAddr::new(SecretKey::generate().public()))
            .collect();
        addresses.sort_by_key(|addr| addr.id);
        let stale = addresses.pop().unwrap();
        let mut current = stale.clone();
        current
            .addrs
            .insert(TransportAddr::Ip("127.0.0.1:12345".parse().unwrap()));
        let mut input = vec![current.clone(), stale];
        input.extend(addresses);
        let selected = preferred_adopted_peer_addrs(input);
        assert_eq!(selected.len(), MAX_ADOPTED_PEER_ADDRS);
        assert_eq!(selected[0], current);
        assert_eq!(
            selected.iter().filter(|addr| addr.id == current.id).count(),
            1
        );
    }

    #[test]
    fn persisted_membership_round_trips_and_rejects_tampered_identity() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("membership.json");
        assert!(load_adopted_mesh_membership(&path).unwrap().is_none());
        let owner = mesh_llm_identity::OwnerKeypair::generate();
        let mut state = requirement_state("", &owner);
        state.mesh_id = state.policy.policy_derived_mesh_id().unwrap();
        state.signed_policy =
            Some(SignedMeshGenesisPolicy::sign(state.policy.clone(), &owner).unwrap());
        let peer = EndpointAddr::new(SecretKey::generate().public());
        persist_adopted_mesh_membership(&path, &state, vec![peer.clone()]).unwrap();

        let restored = load_adopted_mesh_membership(&path).unwrap().unwrap();
        assert_eq!(restored.mesh_id, state.mesh_id);
        assert_eq!(restored.policy_hash, state.policy_hash);
        assert_eq!(restored.peer_addrs, vec![peer]);
        let mut serialized: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        serialized["mesh_id"] = serde_json::Value::String("different-mesh".into());
        std::fs::write(&path, serde_json::to_vec(&serialized).unwrap()).unwrap();
        assert!(load_adopted_mesh_membership(&path).is_err());
    }

    #[test]
    fn invalid_persisted_identity_cannot_replace_existing_membership() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("membership.json");
        let owner = mesh_llm_identity::OwnerKeypair::generate();
        let mut state = requirement_state("", &owner);
        state.mesh_id = state.policy.policy_derived_mesh_id().unwrap();
        state.signed_policy =
            Some(SignedMeshGenesisPolicy::sign(state.policy.clone(), &owner).unwrap());
        persist_adopted_mesh_membership(&path, &state, vec![]).unwrap();
        let original = std::fs::read(&path).unwrap();

        state.policy_hash = "different-policy".into();
        assert!(persist_adopted_mesh_membership(&path, &state, vec![]).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), original);
    }

    #[test]
    fn install_transition_rejects_conflicting_mesh() {
        // Given an installed requirement-aware mesh state.
        let installed = requirement_state(
            "installed-mesh",
            &mesh_llm_identity::OwnerKeypair::generate(),
        );
        let requested = requirement_state(
            "requested-mesh",
            &mesh_llm_identity::OwnerKeypair::generate(),
        );
        let mut current = Some(installed.clone());

        // When a conflicting state attempts to install.
        let result = install_requirement_mesh_state_transition(&mut current, requested);

        // Then the installed state remains unchanged.
        assert!(result.is_err());
        assert_eq!(current, Some(installed));
    }

    #[test]
    fn signed_policy_enrichment_rejects_stale_snapshot() {
        // Given verification against a snapshot that has since been replaced.
        let verified_owner = mesh_llm_identity::OwnerKeypair::generate();
        let verified_snapshot = requirement_state("verified-mesh", &verified_owner);
        let replacement = requirement_state(
            "replacement-mesh",
            &mesh_llm_identity::OwnerKeypair::generate(),
        );
        let mut current = Some(replacement.clone());
        let signed_policy =
            SignedMeshGenesisPolicy::sign(verified_snapshot.policy.clone(), &verified_owner)
                .expect("signed policy");

        // When enrichment attempts to publish the verified signed policy.
        let result = enrich_requirement_mesh_state(&mut current, &verified_snapshot, signed_policy);

        // Then the newer state remains unchanged.
        assert_eq!(result, Err(MeshRequirementRejectReason::MeshPolicyMismatch));
        assert_eq!(current, Some(replacement));
    }
}
