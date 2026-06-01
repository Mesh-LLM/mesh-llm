use std::{error::Error, fmt};

use mesh_llm_identity::{
    NodeOwnershipClaim, OwnershipStatus, OwnershipSummary, SignedNodeOwnership, TrustPolicy,
    TrustStore, verify_node_ownership,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlPlaneAuthError {
    MissingLocalOwnerIdentity {
        local_status: OwnershipStatus,
    },
    MissingRemoteOwnerAttestation,
    RemoteOwnerMismatch {
        local_owner_id: String,
        remote_owner_id: String,
    },
    RemoteOwnershipInvalid {
        status: OwnershipStatus,
        owner_id: Option<String>,
        cert_id: Option<String>,
    },
    TargetNodeMismatch {
        expected_node_id: String,
        actual_node_id: String,
    },
    UnsupportedTrustPolicy {
        policy: TrustPolicy,
    },
}

impl fmt::Display for ControlPlaneAuthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingLocalOwnerIdentity { local_status } => {
                write!(f, "missing local owner identity ({local_status:?})")
            }
            Self::MissingRemoteOwnerAttestation => write!(f, "missing remote owner attestation"),
            Self::RemoteOwnerMismatch {
                local_owner_id,
                remote_owner_id,
            } => write!(
                f,
                "remote owner mismatch (local {local_owner_id}, remote {remote_owner_id})"
            ),
            Self::RemoteOwnershipInvalid {
                status,
                owner_id,
                cert_id,
            } => write!(
                f,
                "remote ownership invalid ({status:?}, owner_id={}, cert_id={})",
                owner_id.as_deref().unwrap_or("unknown"),
                cert_id.as_deref().unwrap_or("unknown")
            ),
            Self::TargetNodeMismatch {
                expected_node_id,
                actual_node_id,
            } => write!(
                f,
                "target node mismatch (expected {expected_node_id}, got {actual_node_id})"
            ),
            Self::UnsupportedTrustPolicy { policy } => {
                write!(f, "unsupported control-plane trust policy {policy:?}")
            }
        }
    }
}

impl Error for ControlPlaneAuthError {}

pub fn verify_control_plane_target_node(
    target_node_id: &[u8],
    actual_local_endpoint_id: &[u8; 32],
) -> Result<(), ControlPlaneAuthError> {
    if target_node_id == actual_local_endpoint_id {
        return Ok(());
    }
    Err(ControlPlaneAuthError::TargetNodeMismatch {
        expected_node_id: hex::encode(actual_local_endpoint_id),
        actual_node_id: hex::encode(target_node_id),
    })
}

pub fn verify_control_plane_peer_ownership(
    local_owner: &OwnershipSummary,
    remote_ownership: Option<&crate::proto::node::SignedNodeOwnership>,
    actual_remote_endpoint_id: &[u8; 32],
    trust_store: &TrustStore,
    trust_policy: TrustPolicy,
    now_unix_ms: u64,
) -> Result<OwnershipSummary, ControlPlaneAuthError> {
    let Some(local_owner_id) = local_owner
        .owner_id
        .as_ref()
        .filter(|_| local_owner.verified)
    else {
        return Err(ControlPlaneAuthError::MissingLocalOwnerIdentity {
            local_status: local_owner.status.clone(),
        });
    };

    match trust_policy {
        TrustPolicy::Off | TrustPolicy::PreferOwned | TrustPolicy::RequireOwned => {}
        TrustPolicy::Allowlist => {
            return Err(ControlPlaneAuthError::UnsupportedTrustPolicy {
                policy: trust_policy,
            });
        }
    }

    let remote_ownership = remote_ownership
        .map(proto_signed_node_ownership_to_local)
        .ok_or(ControlPlaneAuthError::MissingRemoteOwnerAttestation)?;
    let remote_summary = verify_node_ownership(
        Some(&remote_ownership),
        actual_remote_endpoint_id,
        trust_store,
        TrustPolicy::Off,
        now_unix_ms,
    );

    match remote_summary.status {
        OwnershipStatus::Verified => {
            let Some(remote_owner_id) = remote_summary.owner_id.as_ref() else {
                return Err(ControlPlaneAuthError::RemoteOwnershipInvalid {
                    status: remote_summary.status.clone(),
                    owner_id: None,
                    cert_id: remote_summary.cert_id.clone(),
                });
            };
            if remote_owner_id != local_owner_id {
                return Err(ControlPlaneAuthError::RemoteOwnerMismatch {
                    local_owner_id: local_owner_id.clone(),
                    remote_owner_id: remote_owner_id.clone(),
                });
            }
            Ok(remote_summary)
        }
        OwnershipStatus::MismatchedNodeId => Err(ControlPlaneAuthError::TargetNodeMismatch {
            expected_node_id: hex::encode(actual_remote_endpoint_id),
            actual_node_id: remote_ownership.claim.node_endpoint_id,
        }),
        _ => Err(ControlPlaneAuthError::RemoteOwnershipInvalid {
            status: remote_summary.status.clone(),
            owner_id: remote_summary.owner_id.clone(),
            cert_id: remote_summary.cert_id.clone(),
        }),
    }
}

fn proto_signed_node_ownership_to_local(
    attestation: &crate::proto::node::SignedNodeOwnership,
) -> SignedNodeOwnership {
    SignedNodeOwnership {
        claim: NodeOwnershipClaim {
            version: attestation.version,
            cert_id: attestation.cert_id.clone(),
            owner_id: attestation.owner_id.clone(),
            owner_sign_public_key: hex::encode(&attestation.owner_sign_public_key),
            node_endpoint_id: hex::encode(&attestation.node_endpoint_id),
            issued_at_unix_ms: attestation.issued_at_unix_ms,
            expires_at_unix_ms: attestation.expires_at_unix_ms,
            node_label: attestation.node_label.clone(),
            hostname_hint: attestation.hostname_hint.clone(),
        },
        signature: hex::encode(&attestation.signature),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_llm_identity::{OwnerKeypair, sign_node_ownership};

    fn current_time_unix_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    fn proto_signed_node_ownership(
        ownership: &SignedNodeOwnership,
    ) -> crate::proto::node::SignedNodeOwnership {
        crate::proto::node::SignedNodeOwnership {
            version: ownership.claim.version,
            cert_id: ownership.claim.cert_id.clone(),
            owner_id: ownership.claim.owner_id.clone(),
            owner_sign_public_key: hex::decode(&ownership.claim.owner_sign_public_key)
                .expect("test owner_sign_public_key must decode"),
            node_endpoint_id: hex::decode(&ownership.claim.node_endpoint_id)
                .expect("test node_endpoint_id must decode"),
            issued_at_unix_ms: ownership.claim.issued_at_unix_ms,
            expires_at_unix_ms: ownership.claim.expires_at_unix_ms,
            node_label: ownership.claim.node_label.clone(),
            hostname_hint: ownership.claim.hostname_hint.clone(),
            signature: hex::decode(&ownership.signature).expect("test signature must decode"),
        }
    }

    fn verified_local_owner_summary(owner: &OwnerKeypair) -> OwnershipSummary {
        OwnershipSummary {
            owner_id: Some(owner.owner_id()),
            status: OwnershipStatus::Verified,
            verified: true,
            ..OwnershipSummary::default()
        }
    }

    #[test]
    fn control_plane_auth_same_owner_without_gossip() {
        let owner = OwnerKeypair::generate();
        let local_owner = verified_local_owner_summary(&owner);
        let remote_node_endpoint_id = [0x52; 32];
        let remote_ownership = sign_node_ownership(
            &owner,
            &remote_node_endpoint_id,
            current_time_unix_ms() + 60_000,
            Some("remote-worker".into()),
            None,
        )
        .unwrap();

        let summary = verify_control_plane_peer_ownership(
            &local_owner,
            Some(&proto_signed_node_ownership(&remote_ownership)),
            &remote_node_endpoint_id,
            &TrustStore::default(),
            TrustPolicy::Off,
            current_time_unix_ms(),
        )
        .expect("same-owner direct control attestation must succeed without gossip state");

        assert!(summary.verified);
        assert_eq!(summary.owner_id.as_deref(), Some(owner.owner_id().as_str()));
        assert_eq!(summary.node_label.as_deref(), Some("remote-worker"));
    }

    #[test]
    fn control_plane_auth_rejects_wrong_owner() {
        let local_owner = OwnerKeypair::generate();
        let remote_owner = OwnerKeypair::generate();
        let remote_node_endpoint_id = [0x62; 32];
        let remote_ownership = sign_node_ownership(
            &remote_owner,
            &remote_node_endpoint_id,
            current_time_unix_ms() + 60_000,
            None,
            None,
        )
        .unwrap();

        let err = verify_control_plane_peer_ownership(
            &verified_local_owner_summary(&local_owner),
            Some(&proto_signed_node_ownership(&remote_ownership)),
            &remote_node_endpoint_id,
            &TrustStore::default(),
            TrustPolicy::Off,
            current_time_unix_ms(),
        )
        .expect_err("different-owner control attestation must fail closed");

        assert!(matches!(
            err,
            ControlPlaneAuthError::RemoteOwnerMismatch { .. }
        ));
    }

    #[test]
    fn control_plane_auth_rejects_wrong_node_id() {
        let owner = OwnerKeypair::generate();
        let local_owner = verified_local_owner_summary(&owner);
        let claimed_node_endpoint_id = [0x71; 32];
        let actual_remote_endpoint_id = [0x72; 32];
        let remote_ownership = sign_node_ownership(
            &owner,
            &claimed_node_endpoint_id,
            current_time_unix_ms() + 60_000,
            None,
            None,
        )
        .unwrap();

        let err = verify_control_plane_peer_ownership(
            &local_owner,
            Some(&proto_signed_node_ownership(&remote_ownership)),
            &actual_remote_endpoint_id,
            &TrustStore::default(),
            TrustPolicy::Off,
            current_time_unix_ms(),
        )
        .expect_err("wrong peer node id must fail closed");

        assert!(matches!(
            err,
            ControlPlaneAuthError::TargetNodeMismatch { .. }
        ));
    }

    #[test]
    fn control_plane_auth_rejects_bad_signature() {
        let owner = OwnerKeypair::generate();
        let local_owner = verified_local_owner_summary(&owner);
        let remote_node_endpoint_id = [0x81; 32];
        let mut remote_ownership = proto_signed_node_ownership(
            &sign_node_ownership(
                &owner,
                &remote_node_endpoint_id,
                current_time_unix_ms() + 60_000,
                None,
                None,
            )
            .unwrap(),
        );
        remote_ownership.signature[0] ^= 0xFF;

        let err = verify_control_plane_peer_ownership(
            &local_owner,
            Some(&remote_ownership),
            &remote_node_endpoint_id,
            &TrustStore::default(),
            TrustPolicy::Off,
            current_time_unix_ms(),
        )
        .expect_err("bad-signature control attestation must fail closed");

        assert!(matches!(
            err,
            ControlPlaneAuthError::RemoteOwnershipInvalid {
                status: OwnershipStatus::InvalidSignature,
                ..
            }
        ));
    }

    #[test]
    fn control_plane_auth_rejects_missing_local_owner_identity() {
        let owner = OwnerKeypair::generate();
        let remote_node_endpoint_id = [0x91; 32];
        let remote_ownership = sign_node_ownership(
            &owner,
            &remote_node_endpoint_id,
            current_time_unix_ms() + 60_000,
            None,
            None,
        )
        .unwrap();

        let err = verify_control_plane_peer_ownership(
            &OwnershipSummary::default(),
            Some(&proto_signed_node_ownership(&remote_ownership)),
            &remote_node_endpoint_id,
            &TrustStore::default(),
            TrustPolicy::Off,
            current_time_unix_ms(),
        )
        .expect_err("missing local owner identity must fail closed");

        assert!(matches!(
            err,
            ControlPlaneAuthError::MissingLocalOwnerIdentity {
                local_status: OwnershipStatus::Unsigned,
            }
        ));
    }

    #[test]
    fn control_plane_auth_rejects_unsupported_trust_policy() {
        let owner = OwnerKeypair::generate();
        let local_owner = verified_local_owner_summary(&owner);
        let remote_node_endpoint_id = [0xA1; 32];
        let remote_ownership = sign_node_ownership(
            &owner,
            &remote_node_endpoint_id,
            current_time_unix_ms() + 60_000,
            None,
            None,
        )
        .unwrap();

        let err = verify_control_plane_peer_ownership(
            &local_owner,
            Some(&proto_signed_node_ownership(&remote_ownership)),
            &remote_node_endpoint_id,
            &TrustStore::default(),
            TrustPolicy::Allowlist,
            current_time_unix_ms(),
        )
        .expect_err("allowlist trust policy must fail closed for owner-control auth");

        assert!(matches!(
            err,
            ControlPlaneAuthError::UnsupportedTrustPolicy {
                policy: TrustPolicy::Allowlist,
            }
        ));
    }

    #[test]
    fn control_plane_auth_rejects_target_node_mismatch() {
        let err = verify_control_plane_target_node(&[0xCD; 32], &[0xAB; 32])
            .expect_err("wrong target node id must fail closed");

        assert!(matches!(
            err,
            ControlPlaneAuthError::TargetNodeMismatch { .. }
        ));
    }
}
