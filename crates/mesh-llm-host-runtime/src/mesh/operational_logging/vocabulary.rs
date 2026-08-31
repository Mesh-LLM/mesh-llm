use crate::crypto::OwnershipStatus;
use crate::logging::OperationalAuditContext;

pub(super) const OPERATIONAL_AUDIT_INFO: &str = "info";
pub(super) const OPERATIONAL_AUDIT_WARNING: &str = "warning";

/// Static outcomes that are safe to publish through the local operational log.
///
/// Variants and codes follow the reviewed mesh audit vocabulary. `QuicInboundAccepted`
/// covers authenticated mesh and Skippy-stage QUIC connections; its `protocol_gen`
/// summary is the generation of the negotiated protocol family. There is no
/// `quic_alpn_accepted` code because no explicit mesh-ALPN validation branch exists.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MeshOperationalEvent {
    QuicHandlerFailed(MeshHandlerFailureClass),
    QuicInboundAccepted(MeshQuicInboundOutcome),
    ControlHandlerFailed(MeshHandlerFailureClass),
    ControlAlpnRejected,
    ControlConnectionAccepted,
    GossipPolicyRejected(MeshPolicyRejectionReason),
    GossipDirectPeerPromoted,
    GossipIncompatibleVersionRejected,
    GossipPeerRemoved(MeshPeerRemovalReason),
    AutoJoinSucceeded,
    AutoJoinFailed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MeshPeerRemovalReason {
    StaleDirectAndTransitive,
    HeartbeatUnreachable,
    PeerDownProbeFailed,
    ClosedConnectionNoAddress,
    ReconnectFailed,
    RecoveredGossipFailed,
    CleanShutdown,
    TunnelOpenFailed,
}

impl MeshPeerRemovalReason {
    pub(crate) const fn reason_code(self) -> &'static str {
        match self {
            Self::StaleDirectAndTransitive => "stale_direct_and_transitive",
            Self::HeartbeatUnreachable => "heartbeat_unreachable",
            Self::PeerDownProbeFailed => "peer_down_probe_failed",
            Self::ClosedConnectionNoAddress => "closed_connection_no_address",
            Self::ReconnectFailed => "reconnect_failed",
            Self::RecoveredGossipFailed => "recovered_gossip_failed",
            Self::CleanShutdown => "clean_shutdown",
            Self::TunnelOpenFailed => "tunnel_open_failed",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MeshQuicInboundOutcome {
    Accepted,
    Readmitted,
}

impl MeshQuicInboundOutcome {
    const fn outcome(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Readmitted => "readmitted",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MeshHandlerFailureClass {
    Capacity,
    Internal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MeshPolicyRejectionReason {
    AttestationRequired,
    AttestationExpired,
    AttestationInvalid,
    AttestationNodeMismatch,
    Revoked,
    AttestationRevoked,
    NodeRevoked,
    AttestationProtocolUnsupported,
    Untrusted,
}

impl MeshPolicyRejectionReason {
    pub(crate) const fn from_ownership_status(status: &OwnershipStatus) -> Option<Self> {
        match status {
            OwnershipStatus::Verified => None,
            OwnershipStatus::Unsigned => Some(Self::AttestationRequired),
            OwnershipStatus::Expired => Some(Self::AttestationExpired),
            OwnershipStatus::InvalidSignature => Some(Self::AttestationInvalid),
            OwnershipStatus::MismatchedNodeId => Some(Self::AttestationNodeMismatch),
            OwnershipStatus::RevokedOwner => Some(Self::Revoked),
            OwnershipStatus::RevokedCert => Some(Self::AttestationRevoked),
            OwnershipStatus::RevokedNodeId => Some(Self::NodeRevoked),
            OwnershipStatus::UnsupportedProtocol => Some(Self::AttestationProtocolUnsupported),
            OwnershipStatus::UntrustedOwner => Some(Self::Untrusted),
        }
    }

    pub(super) const fn reason_code(self) -> &'static str {
        match self {
            Self::AttestationRequired => "owner_attestation_required",
            Self::AttestationExpired => "owner_attestation_expired",
            Self::AttestationInvalid => "owner_attestation_invalid",
            Self::AttestationNodeMismatch => "owner_attestation_node_mismatch",
            Self::Revoked => "owner_revoked",
            Self::AttestationRevoked => "owner_attestation_revoked",
            Self::NodeRevoked => "owner_node_revoked",
            Self::AttestationProtocolUnsupported => "owner_attestation_protocol_unsupported",
            Self::Untrusted => "owner_untrusted",
        }
    }
}

impl MeshHandlerFailureClass {
    pub(super) const fn reason_code(self) -> &'static str {
        match self {
            Self::Capacity => "capacity",
            Self::Internal => "internal",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MeshHandlerFailureBoundary {
    AcceptSetup,
    AlpnRead,
    Handshake,
    CapacityPermit,
    StreamAccept,
    ControlDispatch,
}

impl MeshHandlerFailureBoundary {
    pub(crate) const fn failure_class(self) -> MeshHandlerFailureClass {
        match self {
            Self::AcceptSetup
            | Self::AlpnRead
            | Self::Handshake
            | Self::StreamAccept
            | Self::ControlDispatch => MeshHandlerFailureClass::Internal,
            Self::CapacityPermit => MeshHandlerFailureClass::Capacity,
        }
    }
}

impl MeshOperationalEvent {
    pub(super) const fn level(self) -> &'static str {
        match self {
            Self::QuicInboundAccepted(_)
            | Self::ControlConnectionAccepted
            | Self::GossipDirectPeerPromoted
            | Self::GossipPeerRemoved(_)
            | Self::AutoJoinSucceeded => OPERATIONAL_AUDIT_INFO,
            Self::QuicHandlerFailed(_)
            | Self::ControlHandlerFailed(_)
            | Self::ControlAlpnRejected
            | Self::GossipPolicyRejected(_)
            | Self::GossipIncompatibleVersionRejected
            | Self::AutoJoinFailed => OPERATIONAL_AUDIT_WARNING,
        }
    }

    pub(super) const fn code(self) -> &'static str {
        match self {
            Self::QuicHandlerFailed(_) => "mesh_quic_handler_failed",
            Self::QuicInboundAccepted(_) => "mesh_quic_inbound_accepted",
            Self::ControlHandlerFailed(_) => "mesh_control_handler_failed",
            Self::ControlAlpnRejected => "mesh_control_alpn_rejected",
            Self::ControlConnectionAccepted => "mesh_control_connection_accepted",
            Self::GossipPolicyRejected(_) => "gossip_policy_rejected",
            Self::GossipDirectPeerPromoted => "gossip_direct_peer_promoted",
            Self::GossipIncompatibleVersionRejected => "gossip_incompatible_version_rejected",
            Self::GossipPeerRemoved(_) => "gossip_peer_removed",
            Self::AutoJoinSucceeded => "mesh_auto_join_succeeded",
            Self::AutoJoinFailed => "mesh_auto_join_failed",
        }
    }

    pub(super) const fn outcome(self) -> &'static str {
        match self {
            Self::QuicInboundAccepted(outcome) => outcome.outcome(),
            Self::ControlConnectionAccepted => "accepted",
            Self::GossipDirectPeerPromoted => "promoted",
            Self::GossipPeerRemoved(_) => "removed",
            Self::AutoJoinSucceeded => "succeeded",
            Self::QuicHandlerFailed(_) | Self::ControlHandlerFailed(_) | Self::AutoJoinFailed => {
                "failed"
            }
            Self::ControlAlpnRejected
            | Self::GossipPolicyRejected(_)
            | Self::GossipIncompatibleVersionRejected => "rejected",
        }
    }

    pub(super) const fn reason_code(self) -> Option<&'static str> {
        match self {
            Self::QuicHandlerFailed(class) | Self::ControlHandlerFailed(class) => {
                Some(class.reason_code())
            }
            Self::ControlAlpnRejected => Some("alpn_unsupported"),
            Self::GossipPolicyRejected(reason) => Some(reason.reason_code()),
            Self::GossipIncompatibleVersionRejected => Some("protocol_version_unsupported"),
            Self::GossipPeerRemoved(reason) => Some(reason.reason_code()),
            Self::AutoJoinFailed => Some("candidate_failed"),
            Self::QuicInboundAccepted(_)
            | Self::ControlConnectionAccepted
            | Self::GossipDirectPeerPromoted
            | Self::AutoJoinSucceeded => None,
        }
    }

    pub(super) const fn decorate_context(
        self,
        mut context: OperationalAuditContext,
    ) -> OperationalAuditContext {
        context = context.outcome(self.outcome());
        match self.reason_code() {
            Some(reason) => context.reason_code(reason),
            None => context,
        }
    }
}
