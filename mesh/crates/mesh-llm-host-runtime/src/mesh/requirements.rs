//! Mesh genesis/requirements admission policy — now owned by
//! `mesh-llm-membership`.
//!
//! The full dependency-neutral admission policy (mesh genesis policy,
//! signed genesis policies, bootstrap tokens, direct-node admission proofs,
//! version/protocol/attestation requirement evaluation, and rejection
//! taxonomy) lives in `mesh_llm_membership::requirements`. This host module is
//! a compatibility re-export so existing callers and the crate-root
//! re-exports keep resolving unchanged.

pub use mesh_llm_membership::requirements::{
    BootstrapStatus, DIRECT_NODE_ADMISSION_PROOF_MAX_CLOCK_SKEW_MS, DirectNodeAdmissionProof,
    DirectPeerProofStatus, MeshGenesisPolicy, MeshRequirementDecision,
    MeshRequirementEvaluationInput, MeshRequirementPolicySummary, MeshRequirementRejectReason,
    MeshRequirementRejectionEvent, MeshRequirementRejectionSource, MeshRequirements,
    NodeVersionBounds, PeerReleaseAttestationStatus, ProtocolGenerationBounds,
    ReleaseAttestationRequirement, SignedBootstrapToken, SignedMeshGenesisPolicy,
    evaluate_direct_peer_admission, peer_release_attestation_status,
};

pub(crate) use mesh_llm_membership::requirements::current_time_unix_ms;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use mesh_llm_protocol::protocol::NODE_PROTOCOL_GENERATION;

    pub(crate) fn restricted_requirements() -> MeshRequirements {
        MeshRequirements {
            node_version: NodeVersionBounds {
                min: Some("0.65.0".into()),
                max: Some("0.66.0".into()),
            },
            protocol_generation: ProtocolGenerationBounds {
                min: Some(1),
                max: Some(2),
            },
            release_attestation: ReleaseAttestationRequirement {
                required: true,
                allowed_signer_keys: vec!["signer-b".into(), "signer-a".into()],
            },
        }
    }

    pub(crate) fn assert_mesh_requirements_policy_canonical_hash_is_stable() {
        let (policy, mut local_input) = MeshGenesisPolicy::for_local_node(
            "owner-123",
            1_717_171_717_000,
            restricted_requirements(),
        )
        .expect("policy should validate");
        local_input.advertised_node_version = Some("0.66.0".into());

        let first = policy.canonical_hash_hex().expect("hash should compute");
        let second = policy
            .canonical_hash_hex()
            .expect("hash should compute twice");

        assert_eq!(first, second);
        assert_eq!(
            policy.evaluate(&local_input),
            MeshRequirementDecision::Rejected(MeshRequirementRejectReason::CertifiedBinaryRequired),
            "the local-input helper should still reflect unsigned default attestation state"
        );
        assert_eq!(
            first, "fb6461159de3e62f1debc8f490b7d10f77fb9c11519ce967a1d24e8cea19ade2",
            "keep this hash stable unless the canonical encoding intentionally changes"
        );
    }

    pub(crate) fn assert_mesh_requirements_policy_change_changes_mesh_id() {
        let baseline =
            MeshGenesisPolicy::new("owner-123", 1_717_171_717_000, restricted_requirements())
                .expect("policy should validate");
        let changed = MeshGenesisPolicy::new(
            "owner-123",
            1_717_171_717_000,
            MeshRequirements {
                node_version: NodeVersionBounds {
                    min: Some("0.65.1".into()),
                    max: Some("0.66.0".into()),
                },
                ..restricted_requirements()
            },
        )
        .expect("changed policy should validate");

        assert_ne!(
            baseline
                .policy_derived_mesh_id()
                .expect("baseline mesh id should compute"),
            changed
                .policy_derived_mesh_id()
                .expect("changed mesh id should compute")
        );
    }

    pub(crate) fn assert_mesh_requirements_bootstrap_token_validates_origin_signature() {
        let owner = mesh_llm_identity::OwnerKeypair::generate();
        let signed_policy = SignedMeshGenesisPolicy::sign(
            MeshGenesisPolicy::new(
                owner.owner_id(),
                1_717_171_717_000,
                restricted_requirements(),
            )
            .expect("policy should validate"),
            &owner,
        )
        .expect("signed policy should validate");
        let mut token = SignedBootstrapToken::sign(
            vec![
                serde_json::to_vec(&serde_json::json!({
                    "id": hex::encode([1u8; 32]),
                    "addrs": []
                }))
                .expect("json should serialize"),
            ],
            &signed_policy,
            Some(1_717_171_717_000 + 60_000),
            &owner,
        )
        .expect("signed token should validate");

        assert!(token.verify_at(1_717_171_717_000).is_ok());
        token.signature[0] ^= 0x55;
        assert_eq!(
            token.verify_at(1_717_171_717_000),
            Err(MeshRequirementRejectReason::BootstrapTokenInvalid)
        );
    }

    pub(crate) fn assert_mesh_requirements_bootstrap_rejects_expired_token() {
        let owner = mesh_llm_identity::OwnerKeypair::generate();
        let signed_policy = SignedMeshGenesisPolicy::sign(
            MeshGenesisPolicy::new(
                owner.owner_id(),
                1_717_171_717_000,
                restricted_requirements(),
            )
            .expect("policy should validate"),
            &owner,
        )
        .expect("signed policy should validate");
        let token = SignedBootstrapToken::sign(
            vec![
                serde_json::to_vec(&serde_json::json!({
                    "id": hex::encode([2u8; 32]),
                    "addrs": []
                }))
                .expect("json should serialize"),
            ],
            &signed_policy,
            Some(1_717_171_717_000 + 5),
            &owner,
        )
        .expect("signed token should validate");

        assert_eq!(
            token.verify_at(1_717_171_717_000 + 6),
            Err(MeshRequirementRejectReason::BootstrapTokenExpired)
        );
    }

    pub(crate) fn assert_mesh_requirements_bootstrap_rejects_policy_hash_mismatch() {
        let owner = mesh_llm_identity::OwnerKeypair::generate();
        let signed_policy = SignedMeshGenesisPolicy::sign(
            MeshGenesisPolicy::new(
                owner.owner_id(),
                1_717_171_717_000,
                restricted_requirements(),
            )
            .expect("policy should validate"),
            &owner,
        )
        .expect("signed policy should validate");
        let mut token = SignedBootstrapToken::sign(
            vec![
                serde_json::to_vec(&serde_json::json!({
                    "id": hex::encode([3u8; 32]),
                    "addrs": []
                }))
                .expect("json should serialize"),
            ],
            &signed_policy,
            Some(1_717_171_717_000 + 60_000),
            &owner,
        )
        .expect("signed token should validate");
        token.policy_hash = "deadbeef".to_string();

        assert_eq!(
            token.verify_at(1_717_171_717_000),
            Err(MeshRequirementRejectReason::MeshPolicyMismatch)
        );
    }

    pub(crate) fn assert_mesh_requirements_policy_hash_derives_mesh_id() {
        let policy =
            MeshGenesisPolicy::new("owner-123", 1_717_171_717_000, restricted_requirements())
                .expect("policy should validate");
        assert_eq!(
            policy
                .policy_derived_mesh_id()
                .expect("mesh id should compute"),
            policy.canonical_hash_hex().expect("hash should compute")
        );
    }

    pub(crate) fn assert_mesh_requirements_version_bounds_unset_min_only_max_only_and_exact() {
        let unrestricted = MeshRequirements::unrestricted();
        let stable_input = MeshRequirementEvaluationInput {
            advertised_node_version: Some("0.65.1".into()),
            negotiated_protocol_generation: Some(NODE_PROTOCOL_GENERATION),
            direct_proof: DirectPeerProofStatus::Verified,
            ..Default::default()
        };
        assert_eq!(
            unrestricted.evaluate(&stable_input),
            MeshRequirementDecision::Accepted
        );

        let min_only = MeshRequirements {
            node_version: NodeVersionBounds {
                min: Some("0.65.1".into()),
                max: None,
            },
            ..MeshRequirements::unrestricted()
        };
        assert_eq!(
            min_only.evaluate(&stable_input),
            MeshRequirementDecision::Accepted
        );
        assert_eq!(
            min_only.evaluate(&MeshRequirementEvaluationInput {
                advertised_node_version: Some("0.65.0".into()),
                ..stable_input.clone()
            }),
            MeshRequirementDecision::Rejected(MeshRequirementRejectReason::NodeVersionBelowMinimum)
        );

        let max_only = MeshRequirements {
            node_version: NodeVersionBounds {
                min: None,
                max: Some("0.65.1".into()),
            },
            ..MeshRequirements::unrestricted()
        };
        assert_eq!(
            max_only.evaluate(&stable_input),
            MeshRequirementDecision::Accepted
        );
        assert_eq!(
            max_only.evaluate(&MeshRequirementEvaluationInput {
                advertised_node_version: Some("0.65.2".into()),
                ..stable_input.clone()
            }),
            MeshRequirementDecision::Rejected(MeshRequirementRejectReason::NodeVersionAboveMaximum)
        );

        let exact = MeshRequirements {
            node_version: NodeVersionBounds {
                min: Some("0.65.1".into()),
                max: Some("0.65.1".into()),
            },
            ..MeshRequirements::unrestricted()
        };
        assert_eq!(
            exact.evaluate(&stable_input),
            MeshRequirementDecision::Accepted
        );
        assert_eq!(
            exact.evaluate(&MeshRequirementEvaluationInput {
                advertised_node_version: Some("0.65.1-alpha.1".into()),
                direct_proof: DirectPeerProofStatus::Missing,
                ..stable_input.clone()
            }),
            MeshRequirementDecision::Rejected(MeshRequirementRejectReason::NodeVersionBelowMinimum)
        );
        assert_eq!(
            exact.evaluate(&MeshRequirementEvaluationInput {
                advertised_node_version: Some("0.65.1+build.99".into()),
                direct_proof: DirectPeerProofStatus::Invalid,
                ..stable_input
            }),
            MeshRequirementDecision::Accepted,
            "exact precedence checks should still accept build metadata variants"
        );
    }

    pub(crate) fn assert_mesh_requirements_protocol_bounds_reject_unknown_only_when_constrained() {
        let unconstrained = MeshRequirements::unrestricted();
        let unrestricted_policy =
            MeshGenesisPolicy::new("owner-123", 1_717_171_717_000, unconstrained)
                .expect("unrestricted policy should validate");
        assert_eq!(
            unrestricted_policy.evaluate(&MeshRequirementEvaluationInput {
                bootstrap: BootstrapStatus::Valid,
                ..Default::default()
            }),
            MeshRequirementDecision::Accepted
        );

        let constrained = MeshRequirements {
            protocol_generation: ProtocolGenerationBounds {
                min: Some(1),
                max: Some(2),
            },
            ..MeshRequirements::unrestricted()
        };
        let constrained_policy =
            MeshGenesisPolicy::new("owner-123", 1_717_171_717_000, constrained)
                .expect("constrained policy should validate");
        assert_eq!(
            constrained_policy.evaluate(&MeshRequirementEvaluationInput::default()),
            MeshRequirementDecision::Rejected(
                MeshRequirementRejectReason::ProtocolGenerationUnknown
            )
        );
        assert_eq!(
            constrained_policy.evaluate(&MeshRequirementEvaluationInput {
                bootstrap: BootstrapStatus::Invalid,
                negotiated_protocol_generation: Some(0),
                ..Default::default()
            }),
            MeshRequirementDecision::Rejected(MeshRequirementRejectReason::BootstrapTokenInvalid)
        );
        assert_eq!(
            constrained_policy.evaluate(&MeshRequirementEvaluationInput {
                bootstrap: BootstrapStatus::Expired,
                negotiated_protocol_generation: Some(1),
                ..Default::default()
            }),
            MeshRequirementDecision::Rejected(MeshRequirementRejectReason::BootstrapTokenExpired)
        );
        assert_eq!(
            constrained_policy.evaluate(&MeshRequirementEvaluationInput {
                negotiated_protocol_generation: Some(0),
                ..Default::default()
            }),
            MeshRequirementDecision::Rejected(
                MeshRequirementRejectReason::ProtocolGenerationBelowMinimum
            )
        );
        assert_eq!(
            constrained_policy.evaluate(&MeshRequirementEvaluationInput {
                negotiated_protocol_generation: Some(3),
                ..Default::default()
            }),
            MeshRequirementDecision::Rejected(
                MeshRequirementRejectReason::ProtocolGenerationAboveMaximum
            )
        );
        assert_eq!(
            constrained_policy.evaluate(&MeshRequirementEvaluationInput {
                negotiated_protocol_generation: Some(1),
                ..Default::default()
            }),
            MeshRequirementDecision::Accepted
        );
    }

    pub(crate) fn assert_mesh_requirements_rejects_unsigned_when_attestation_required() {
        let constrained = MeshRequirements {
            release_attestation: ReleaseAttestationRequirement {
                required: true,
                allowed_signer_keys: vec!["trusted-signer".into()],
            },
            ..MeshRequirements::unrestricted()
        };

        assert_eq!(
            constrained.evaluate(&MeshRequirementEvaluationInput::default()),
            MeshRequirementDecision::Rejected(MeshRequirementRejectReason::CertifiedBinaryRequired)
        );
        assert_eq!(
            constrained.evaluate(&MeshRequirementEvaluationInput {
                release_attestation: PeerReleaseAttestationStatus::Invalid,
                ..Default::default()
            }),
            MeshRequirementDecision::Rejected(MeshRequirementRejectReason::BuildProofInvalid)
        );
        assert_eq!(
            constrained.evaluate(&MeshRequirementEvaluationInput {
                release_attestation: PeerReleaseAttestationStatus::Present {
                    signer_key: Some("untrusted-signer".into()),
                    attested_version: Some(env!("CARGO_PKG_VERSION").to_string()),
                },
                ..Default::default()
            }),
            MeshRequirementDecision::Rejected(MeshRequirementRejectReason::ReleaseSignerUntrusted)
        );
        assert_eq!(
            constrained.evaluate(&MeshRequirementEvaluationInput {
                release_attestation: PeerReleaseAttestationStatus::Present {
                    signer_key: Some("trusted-signer".into()),
                    attested_version: Some(env!("CARGO_PKG_VERSION").to_string()),
                },
                ..Default::default()
            }),
            MeshRequirementDecision::Accepted
        );
    }

    pub(crate) fn assert_mesh_requirements_accept_trusted_signer_with_compatible_peer_version() {
        let constrained = MeshRequirements {
            node_version: NodeVersionBounds {
                min: Some("0.65.0".into()),
                max: Some("0.65.9".into()),
            },
            protocol_generation: ProtocolGenerationBounds {
                min: Some(1),
                max: Some(1),
            },
            release_attestation: ReleaseAttestationRequirement {
                required: true,
                allowed_signer_keys: vec!["trusted-signer".into()],
            },
        };

        assert_eq!(
            constrained.evaluate(&MeshRequirementEvaluationInput {
                advertised_node_version: Some("0.65.4".into()),
                negotiated_protocol_generation: Some(1),
                release_attestation: PeerReleaseAttestationStatus::Present {
                    signer_key: Some("trusted-signer".into()),
                    attested_version: Some("0.65.4".into()),
                },
                ..Default::default()
            }),
            MeshRequirementDecision::Accepted
        );
    }

    pub(crate) fn assert_mesh_requirements_rejection_reasons_are_stable() {
        let stable = [
            (
                MeshRequirementRejectReason::CertifiedBinaryRequired,
                "certified_binary_required",
                "this mesh requires a certified mesh-llm binary; use a certified compiled binary to join.",
            ),
            (
                MeshRequirementRejectReason::BuildProofMissing,
                "build_proof_missing",
                "the peer's certified build proof is missing required signer metadata.",
            ),
            (
                MeshRequirementRejectReason::BuildProofInvalid,
                "build_proof_invalid",
                "the peer's certified build proof could not be verified.",
            ),
            (
                MeshRequirementRejectReason::ReleaseSignerUntrusted,
                "release_signer_untrusted",
                "the peer's certified build proof was signed by an untrusted release signer.",
            ),
            (
                MeshRequirementRejectReason::AttestationPolicyMismatch,
                "attestation_policy_mismatch",
                "the certified build or policy attestation does not match this mesh's requirements.",
            ),
            (
                MeshRequirementRejectReason::MeshPolicyMismatch,
                "mesh_policy_mismatch",
                "the peer or bootstrap token advertised a different mesh policy than this mesh requires.",
            ),
            (
                MeshRequirementRejectReason::BootstrapTokenInvalid,
                "bootstrap_token_invalid",
                "the bootstrap token is invalid for this mesh.",
            ),
            (
                MeshRequirementRejectReason::BootstrapTokenExpired,
                "bootstrap_token_expired",
                "the bootstrap token has expired for this mesh.",
            ),
            (
                MeshRequirementRejectReason::NodeVersionBelowMinimum,
                "node_version_below_minimum",
                "the peer mesh-llm version is below this mesh's minimum allowed version.",
            ),
            (
                MeshRequirementRejectReason::NodeVersionAboveMaximum,
                "node_version_above_maximum",
                "the peer mesh-llm version is above this mesh's maximum allowed version.",
            ),
            (
                MeshRequirementRejectReason::NodeVersionMalformed,
                "node_version_malformed",
                "the peer advertised a malformed mesh-llm node version.",
            ),
            (
                MeshRequirementRejectReason::ProtocolGenerationBelowMinimum,
                "protocol_generation_below_minimum",
                "the peer protocol generation is below this mesh's minimum allowed generation.",
            ),
            (
                MeshRequirementRejectReason::ProtocolGenerationAboveMaximum,
                "protocol_generation_above_maximum",
                "the peer protocol generation is above this mesh's maximum allowed generation.",
            ),
            (
                MeshRequirementRejectReason::ProtocolGenerationUnknown,
                "protocol_generation_unknown",
                "the peer did not advertise a protocol generation required by this mesh.",
            ),
            (
                MeshRequirementRejectReason::TopologyDisclosureDenied,
                "topology_disclosure_denied",
                "topology disclosure was denied until the peer completes mesh admission.",
            ),
        ];

        for (reason, expected_code, expected_message) in stable {
            assert_eq!(reason.code(), expected_code);
            assert_eq!(reason.message(), expected_message);
            assert_eq!(serde_json::to_value(&reason).unwrap(), expected_code);
        }
    }
}
