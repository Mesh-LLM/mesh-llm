//! Bounded, metadata-only operational audit vocabulary for mesh boundaries.
//!
//! Dynamic context is restricted to authenticated peer identity and selected
//! connection-path observations. Tokens, ALPN bytes, hostnames, and raw errors
//! never enter this adapter.
//! Pre-authentication failures remain identity-free; they carry only static
//! outcome/reason codes and an elapsed duration when that boundary owns a timer.

#[cfg(test)]
use crate::logging::LoggingService;
use crate::logging::{
    OperationalAuditContext, OperationalAuditPathType, OperationalAuditRecord,
    OperationalAuditSeverity,
};
use crate::mesh::SelectedPathObservation;
use iroh::EndpointId;

mod vocabulary;

#[cfg(test)]
use vocabulary::MeshHandlerFailureClass;
pub(crate) use vocabulary::{
    MeshHandlerFailureBoundary, MeshOperationalEvent, MeshPeerRemovalReason,
    MeshPolicyRejectionReason, MeshQuicInboundOutcome,
};
use vocabulary::{OPERATIONAL_AUDIT_INFO, OPERATIONAL_AUDIT_WARNING};

const OPERATIONAL_AUDIT_SOURCE: &str = "mesh";

fn operational_audit_record(
    event: MeshOperationalEvent,
    context: Option<OperationalAuditContext>,
) -> OperationalAuditRecord {
    let level = event.level();
    let severity = match level {
        OPERATIONAL_AUDIT_INFO => OperationalAuditSeverity::Info,
        OPERATIONAL_AUDIT_WARNING => OperationalAuditSeverity::Warning,
        _ => OperationalAuditSeverity::Error,
    };
    let record = OperationalAuditRecord::builder(OPERATIONAL_AUDIT_SOURCE, event.code())
        .severity(severity)
        .build();
    if matches!(
        event,
        MeshOperationalEvent::AutoJoinSucceeded | MeshOperationalEvent::AutoJoinFailed
    ) {
        return record;
    }
    let Some(context) = context else {
        return record;
    };
    record.with_context(event.decorate_context(context))
}

pub(crate) fn mesh_peer_operational_context(
    peer: EndpointId,
    path: Option<SelectedPathObservation>,
) -> OperationalAuditContext {
    let context = OperationalAuditContext::new().mesh_peer_subject(&hex::encode(peer.as_bytes()));
    let Some(path) = path else {
        return context;
    };
    match path.path_type {
        "direct" => context.network_path(
            OperationalAuditPathType::Direct,
            path.observed_direct_remote_addr,
        ),
        "relay" => context.network_path(OperationalAuditPathType::Relay, None),
        _ => context,
    }
}

/// Record one mesh boundary result through the process-local logging service.
/// Logging is optional and this intentionally never affects mesh serving.
pub(crate) fn record_mesh_operational_event(event: MeshOperationalEvent) {
    let record = operational_audit_record(event, Some(OperationalAuditContext::new()));
    #[cfg(test)]
    capture_mesh_operational_audit_for_test(&record);
    let Some(state) = crate::logging_runtime_state() else {
        return;
    };
    let _ = state.write_operational_audit(record);
}

pub(crate) fn record_mesh_operational_event_with_context(
    event: MeshOperationalEvent,
    context: OperationalAuditContext,
) {
    let record = operational_audit_record(event, Some(context));
    #[cfg(test)]
    capture_mesh_operational_audit_for_test(&record);
    let Some(state) = crate::logging_runtime_state() else {
        return;
    };
    let _ = state.write_operational_audit(record);
}

#[cfg(test)]
static MESH_OPERATIONAL_AUDIT_CAPTURE: std::sync::Mutex<
    Option<tokio::sync::mpsc::UnboundedSender<OperationalAuditRecord>>,
> = std::sync::Mutex::new(None);

#[cfg(test)]
pub(crate) struct MeshOperationalAuditCaptureGuard {
    sender: tokio::sync::mpsc::UnboundedSender<OperationalAuditRecord>,
}

#[cfg(test)]
impl Drop for MeshOperationalAuditCaptureGuard {
    fn drop(&mut self) {
        let mut capture = MESH_OPERATIONAL_AUDIT_CAPTURE
            .lock()
            .expect("mesh operational audit capture lock");
        if capture
            .as_ref()
            .is_some_and(|sender| sender.same_channel(&self.sender))
        {
            *capture = None;
        }
    }
}

#[cfg(test)]
pub(crate) fn capture_mesh_operational_audits() -> (
    tokio::sync::mpsc::UnboundedReceiver<OperationalAuditRecord>,
    MeshOperationalAuditCaptureGuard,
) {
    let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
    *MESH_OPERATIONAL_AUDIT_CAPTURE
        .lock()
        .expect("mesh operational audit capture lock") = Some(sender.clone());
    (receiver, MeshOperationalAuditCaptureGuard { sender })
}

#[cfg(test)]
fn capture_mesh_operational_audit_for_test(record: &OperationalAuditRecord) {
    if let Some(sender) = MESH_OPERATIONAL_AUDIT_CAPTURE
        .lock()
        .expect("mesh operational audit capture lock")
        .as_ref()
    {
        let _ = sender.send(record.clone());
    }
}

#[cfg(test)]
fn record_mesh_operational_event_with_context_and_service(
    service: &LoggingService,
    event: MeshOperationalEvent,
    context: OperationalAuditContext,
) {
    let _ = service.write_operational_audit(operational_audit_record(event, Some(context)));
}

#[cfg(test)]
mod tests {
    use super::{
        MeshHandlerFailureBoundary, MeshHandlerFailureClass, MeshOperationalEvent,
        MeshPeerRemovalReason, MeshPolicyRejectionReason, MeshQuicInboundOutcome,
        capture_mesh_operational_audits, mesh_peer_operational_context,
        record_mesh_operational_event, record_mesh_operational_event_with_context_and_service,
    };
    use crate::crypto::OwnershipStatus;
    use crate::logging::{LoggingService, OperationalAuditContext, ServiceConfig};
    use crate::mesh::SelectedPathObservation;
    use iroh::{EndpointId, SecretKey};
    use serial_test::serial;
    use tokio::sync::mpsc::error::TryRecvError;

    #[test]
    #[serial]
    fn stale_capture_guard_does_not_clear_replacement() {
        let (_receiver_a, guard_a) = capture_mesh_operational_audits();
        let (mut receiver_b, guard_b) = capture_mesh_operational_audits();

        drop(guard_a);
        record_mesh_operational_event(MeshOperationalEvent::ControlConnectionAccepted);

        let captured = loop {
            match receiver_b.try_recv() {
                Ok(record) if record.code() == "mesh_control_connection_accepted" => break record,
                Ok(_) => {}
                Err(error) => panic!("replacement capture must receive the audit: {error}"),
            }
        };
        assert_eq!(captured.code(), "mesh_control_connection_accepted");

        drop(guard_b);
        record_mesh_operational_event(MeshOperationalEvent::ControlAlpnRejected);

        loop {
            match receiver_b.try_recv() {
                Ok(record) => assert_ne!(record.code(), "mesh_control_alpn_rejected"),
                Err(TryRecvError::Disconnected) => break,
                Err(TryRecvError::Empty) => panic!("active capture registration remains"),
            }
        }
    }

    #[test]
    fn mesh_boundary_outcomes_emit_authenticated_peer_context_without_raw_secrets() {
        let service = LoggingService::new_disabled(ServiceConfig::default());
        let peer = EndpointId::from(SecretKey::from_bytes(&[0x42; 32]).public());
        let peer_hex = hex::encode(peer.as_bytes());
        let direct_addr = "192.0.2.42:11204".parse().expect("direct address");
        let context = mesh_peer_operational_context(
            peer,
            Some(SelectedPathObservation {
                path_type: "direct",
                rtt_ms: Some(17),
                observed_direct_remote_addr: Some(direct_addr),
            }),
        )
        .numeric_summary("direct_peers", 3);
        record_mesh_operational_event_with_context_and_service(
            &service,
            MeshOperationalEvent::GossipDirectPeerPromoted,
            context,
        );

        let audit: serde_json::Value = serde_json::from_str(
            &service
                .bus_ref()
                .drain()
                .into_iter()
                .next()
                .expect("peer audit")
                .payload,
        )
        .expect("audit payload");
        assert_eq!(audit["kind"], "audit");
        assert_eq!(audit["severity"], "info");
        assert_eq!(audit["code"], "gossip_direct_peer_promoted");
        assert_eq!(audit["subject_kind"], "mesh_peer");
        assert_eq!(audit["subject_id"], peer_hex);
        assert_eq!(audit["path_type"], "direct");
        assert_eq!(audit["remote_addr"], direct_addr.to_string());
        assert_eq!(audit["outcome"], "promoted");
        assert!(audit.get("duration_ms").is_none());
        assert_eq!(
            audit["numeric_summaries"],
            serde_json::json!({"direct_peers": 3})
        );

        let serialized = serde_json::to_string(&audit).expect("serialized audit payload");
        for raw_value in [
            "node=untrusted-lab-host",
            "token=mesh-secret-bootstrap-token",
            "mesh-llm/1-private-alpn",
            "connection refused at secret.example.test",
            "untrusted-lab-host.example.test",
        ] {
            assert!(
                !serialized.contains(raw_value),
                "raw secret, ALPN, error, and hostname data must not enter the audit payload"
            );
        }
    }

    #[test]
    fn relay_peer_context_never_records_an_address() {
        let peer = EndpointId::from(SecretKey::from_bytes(&[0x24; 32]).public());
        let context = mesh_peer_operational_context(
            peer,
            Some(SelectedPathObservation {
                path_type: "relay",
                rtt_ms: Some(31),
                observed_direct_remote_addr: Some(
                    "203.0.113.10:443".parse().expect("relay-shaped address"),
                ),
            }),
        );
        let service = LoggingService::new_disabled(ServiceConfig::default());
        record_mesh_operational_event_with_context_and_service(
            &service,
            MeshOperationalEvent::QuicInboundAccepted(MeshQuicInboundOutcome::Accepted),
            context,
        );

        let payload = service.bus_ref().drain().remove(0).payload;
        let audit: serde_json::Value = serde_json::from_str(&payload).expect("audit payload");
        assert_eq!(audit["subject_id"], hex::encode(peer.as_bytes()));
        assert_eq!(audit["path_type"], "relay");
        assert!(audit.get("remote_addr").is_none());
        assert!(audit.get("numeric_summaries").is_none());
        assert_eq!(audit["outcome"], "accepted");
    }

    #[test]
    fn mesh_operational_vocabulary_is_bounded_and_static() {
        let events = [
            MeshOperationalEvent::QuicHandlerFailed(MeshHandlerFailureClass::Internal),
            MeshOperationalEvent::QuicInboundAccepted(MeshQuicInboundOutcome::Accepted),
            MeshOperationalEvent::ControlHandlerFailed(MeshHandlerFailureClass::Capacity),
            MeshOperationalEvent::ControlAlpnRejected,
            MeshOperationalEvent::ControlConnectionAccepted,
            MeshOperationalEvent::GossipPolicyRejected(
                MeshPolicyRejectionReason::AttestationRequired,
            ),
            MeshOperationalEvent::GossipDirectPeerPromoted,
            MeshOperationalEvent::GossipIncompatibleVersionRejected,
            MeshOperationalEvent::GossipPeerRemoved(MeshPeerRemovalReason::CleanShutdown),
            MeshOperationalEvent::AutoJoinSucceeded,
            MeshOperationalEvent::AutoJoinFailed,
        ];

        for event in events {
            let code = event.code();
            assert!(code.len() <= 48, "audit code must stay bounded: {code}");
            assert!(
                code.bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte == b'_'),
                "audit code must be a static identifier: {code}"
            );
            assert!(matches!(event.level(), "info" | "warning"));
        }
    }

    #[test]
    fn peer_removal_reason_vocabulary_is_exactly_the_bounded_control_flow_causes() {
        let cases = [
            (
                MeshPeerRemovalReason::StaleDirectAndTransitive,
                "stale_direct_and_transitive",
            ),
            (
                MeshPeerRemovalReason::HeartbeatUnreachable,
                "heartbeat_unreachable",
            ),
            (
                MeshPeerRemovalReason::PeerDownProbeFailed,
                "peer_down_probe_failed",
            ),
            (
                MeshPeerRemovalReason::ClosedConnectionNoAddress,
                "closed_connection_no_address",
            ),
            (MeshPeerRemovalReason::ReconnectFailed, "reconnect_failed"),
            (
                MeshPeerRemovalReason::RecoveredGossipFailed,
                "recovered_gossip_failed",
            ),
            (MeshPeerRemovalReason::CleanShutdown, "clean_shutdown"),
            (
                MeshPeerRemovalReason::TunnelOpenFailed,
                "tunnel_open_failed",
            ),
        ];

        for (reason, expected_code) in cases {
            let code = reason.reason_code();
            assert_eq!(code, expected_code);
            assert!(code.len() <= 48, "reason code must stay bounded: {code}");
            assert!(
                code.bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte == b'_'),
                "reason code must be a static identifier: {code}"
            );
        }
    }

    #[test]
    fn gossip_peer_removed_audit_carries_only_cause_count_and_peer_path_identity() {
        let service = LoggingService::new_disabled(ServiceConfig::default());
        let peer = EndpointId::from(SecretKey::from_bytes(&[0x51; 32]).public());
        let context = mesh_peer_operational_context(
            peer,
            Some(SelectedPathObservation {
                path_type: "direct",
                rtt_ms: Some(11),
                observed_direct_remote_addr: Some(
                    "192.0.2.51:1151".parse().expect("direct peer address"),
                ),
            }),
        )
        .numeric_summary("direct_peers", 2);

        record_mesh_operational_event_with_context_and_service(
            &service,
            MeshOperationalEvent::GossipPeerRemoved(MeshPeerRemovalReason::HeartbeatUnreachable),
            context,
        );

        let payload = service.bus_ref().drain().remove(0).payload;
        let audit: serde_json::Value = serde_json::from_str(&payload).expect("removal audit JSON");
        assert_eq!(audit["reason_code"], "heartbeat_unreachable");
        assert_eq!(
            audit["numeric_summaries"],
            serde_json::json!({"direct_peers": 2})
        );
        assert_eq!(audit["subject_id"], hex::encode(peer.as_bytes()));
        assert_eq!(audit["path_type"], "direct");
        assert_eq!(audit["remote_addr"], "192.0.2.51:1151");
        for diagnostic_only_field in [
            "last_seen_age_ms",
            "last_mentioned_age_ms",
            "had_connection",
            "bridge",
            "reporter",
        ] {
            assert!(audit.get(diagnostic_only_field).is_none());
        }
    }

    #[test]
    fn reviewed_audit_outcomes_and_rejection_reasons_are_exact() {
        let cases = [
            (
                MeshOperationalEvent::GossipDirectPeerPromoted,
                "promoted",
                None,
            ),
            (
                MeshOperationalEvent::ControlAlpnRejected,
                "rejected",
                Some("alpn_unsupported"),
            ),
            (
                MeshOperationalEvent::GossipPolicyRejected(
                    MeshPolicyRejectionReason::AttestationRequired,
                ),
                "rejected",
                Some("owner_attestation_required"),
            ),
            (
                MeshOperationalEvent::GossipIncompatibleVersionRejected,
                "rejected",
                Some("protocol_version_unsupported"),
            ),
            (
                MeshOperationalEvent::GossipPeerRemoved(
                    MeshPeerRemovalReason::HeartbeatUnreachable,
                ),
                "removed",
                Some("heartbeat_unreachable"),
            ),
            (
                MeshOperationalEvent::QuicInboundAccepted(MeshQuicInboundOutcome::Readmitted),
                "readmitted",
                None,
            ),
        ];

        for (event, expected_outcome, expected_reason) in cases {
            assert_eq!(event.outcome(), expected_outcome);
            assert_eq!(event.reason_code(), expected_reason);
        }
    }

    #[test]
    fn pre_auth_alpn_rejection_stays_sparse() {
        let service = LoggingService::new_disabled(ServiceConfig::default());
        record_mesh_operational_event_with_context_and_service(
            &service,
            MeshOperationalEvent::ControlAlpnRejected,
            OperationalAuditContext::new(),
        );
        let payload = service.bus_ref().drain().remove(0).payload;
        let audit: serde_json::Value = serde_json::from_str(&payload).expect("pre-auth audit JSON");

        assert_eq!(audit["outcome"], "rejected");
        assert_eq!(audit["reason_code"], "alpn_unsupported");
        for unavailable_field in [
            "subject_kind",
            "subject_id",
            "remote_addr",
            "path_type",
            "duration_ms",
            "numeric_summaries",
        ] {
            assert!(audit.get(unavailable_field).is_none());
        }
    }

    #[test]
    fn reviewed_audit_numeric_summaries_and_handler_duration_are_exact() {
        let service = LoggingService::new_disabled(ServiceConfig::default());
        let cases = [
            (
                MeshOperationalEvent::GossipDirectPeerPromoted,
                OperationalAuditContext::new().numeric_summary("direct_peers", 4),
                serde_json::json!({"direct_peers": 4}),
                None,
            ),
            (
                MeshOperationalEvent::GossipPeerRemoved(MeshPeerRemovalReason::CleanShutdown),
                OperationalAuditContext::new().numeric_summary("direct_peers", 3),
                serde_json::json!({"direct_peers": 3}),
                None,
            ),
            (
                MeshOperationalEvent::GossipPolicyRejected(
                    MeshPolicyRejectionReason::AttestationRequired,
                ),
                OperationalAuditContext::new(),
                serde_json::json!({}),
                None,
            ),
            (
                MeshOperationalEvent::GossipIncompatibleVersionRejected,
                OperationalAuditContext::new()
                    .numeric_summary("peer_gen", 0)
                    .numeric_summary("local_gen", 1),
                serde_json::json!({"local_gen": 1, "peer_gen": 0}),
                None,
            ),
            (
                MeshOperationalEvent::QuicInboundAccepted(MeshQuicInboundOutcome::Accepted),
                OperationalAuditContext::new().numeric_summary("protocol_gen", 1),
                serde_json::json!({"protocol_gen": 1}),
                None,
            ),
            (
                MeshOperationalEvent::ControlConnectionAccepted,
                OperationalAuditContext::new().numeric_summary("protocol_gen", 1),
                serde_json::json!({"protocol_gen": 1}),
                None,
            ),
            (
                MeshOperationalEvent::ControlAlpnRejected,
                OperationalAuditContext::new(),
                serde_json::json!({}),
                None,
            ),
            (
                MeshOperationalEvent::QuicHandlerFailed(MeshHandlerFailureClass::Internal),
                OperationalAuditContext::new().duration_ms(29),
                serde_json::json!({}),
                Some(29),
            ),
            (
                MeshOperationalEvent::ControlHandlerFailed(MeshHandlerFailureClass::Internal),
                OperationalAuditContext::new().duration_ms(37),
                serde_json::json!({}),
                Some(37),
            ),
        ];

        for (event, context, expected_summaries, expected_duration) in cases {
            record_mesh_operational_event_with_context_and_service(&service, event, context);
            let payload = service.bus_ref().drain().remove(0).payload;
            let audit: serde_json::Value =
                serde_json::from_str(&payload).expect("reviewed audit JSON");
            assert_eq!(
                audit
                    .get("numeric_summaries")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({})),
                expected_summaries,
            );
            assert_eq!(
                audit.get("duration_ms").and_then(serde_json::Value::as_u64),
                expected_duration
            );
        }
    }

    #[test]
    fn auto_join_audits_remain_generic_and_context_free() {
        let service = LoggingService::new_disabled(ServiceConfig::default());

        for event in [
            MeshOperationalEvent::AutoJoinSucceeded,
            MeshOperationalEvent::AutoJoinFailed,
        ] {
            record_mesh_operational_event_with_context_and_service(
                &service,
                event,
                OperationalAuditContext::new()
                    .outcome("must_not_escape")
                    .reason_code("must_not_escape")
                    .numeric_summary("must_not_escape", 1),
            );
            let payload = service.bus_ref().drain().remove(0).payload;
            let audit: serde_json::Value =
                serde_json::from_str(&payload).expect("generic auto-join audit JSON");

            for context_field in [
                "context_version",
                "outcome",
                "reason_code",
                "numeric_summaries",
            ] {
                assert!(audit.get(context_field).is_none());
            }
        }
    }

    #[test]
    fn handler_failure_vocabulary_is_exactly_the_production_backed_classes() {
        let classes = [
            MeshHandlerFailureClass::Capacity,
            MeshHandlerFailureClass::Internal,
        ];
        assert_eq!(
            classes.map(MeshHandlerFailureClass::reason_code),
            ["capacity", "internal"]
        );
    }

    #[test]
    fn ownership_status_maps_exhaustively_to_gate_specific_policy_reasons() {
        let cases = [
            (OwnershipStatus::Verified, None),
            (
                OwnershipStatus::Unsigned,
                Some("owner_attestation_required"),
            ),
            (OwnershipStatus::Expired, Some("owner_attestation_expired")),
            (
                OwnershipStatus::InvalidSignature,
                Some("owner_attestation_invalid"),
            ),
            (
                OwnershipStatus::MismatchedNodeId,
                Some("owner_attestation_node_mismatch"),
            ),
            (OwnershipStatus::RevokedOwner, Some("owner_revoked")),
            (
                OwnershipStatus::RevokedCert,
                Some("owner_attestation_revoked"),
            ),
            (OwnershipStatus::RevokedNodeId, Some("owner_node_revoked")),
            (
                OwnershipStatus::UnsupportedProtocol,
                Some("owner_attestation_protocol_unsupported"),
            ),
            (OwnershipStatus::UntrustedOwner, Some("owner_untrusted")),
        ];

        for (status, expected_reason) in cases {
            assert_eq!(
                MeshPolicyRejectionReason::from_ownership_status(&status)
                    .map(MeshPolicyRejectionReason::reason_code),
                expected_reason,
            );
        }
    }

    #[test]
    fn handler_failure_boundaries_map_only_when_truthfully_classifiable() {
        let cases = [
            (
                MeshHandlerFailureBoundary::AcceptSetup,
                MeshHandlerFailureClass::Internal,
            ),
            (
                MeshHandlerFailureBoundary::AlpnRead,
                MeshHandlerFailureClass::Internal,
            ),
            (
                MeshHandlerFailureBoundary::Handshake,
                MeshHandlerFailureClass::Internal,
            ),
            (
                MeshHandlerFailureBoundary::CapacityPermit,
                MeshHandlerFailureClass::Capacity,
            ),
            (
                MeshHandlerFailureBoundary::StreamAccept,
                MeshHandlerFailureClass::Internal,
            ),
            (
                MeshHandlerFailureBoundary::ControlDispatch,
                MeshHandlerFailureClass::Internal,
            ),
        ];
        for (boundary, expected) in cases {
            assert_eq!(boundary.failure_class(), expected);
        }
    }

    #[test]
    fn mesh_operational_vocabulary_maps_each_variant_to_its_reviewed_code() {
        let cases = [
            (
                MeshOperationalEvent::QuicHandlerFailed(MeshHandlerFailureClass::Internal),
                "mesh_quic_handler_failed",
            ),
            (
                MeshOperationalEvent::QuicInboundAccepted(MeshQuicInboundOutcome::Accepted),
                "mesh_quic_inbound_accepted",
            ),
            (
                MeshOperationalEvent::ControlHandlerFailed(MeshHandlerFailureClass::Internal),
                "mesh_control_handler_failed",
            ),
            (
                MeshOperationalEvent::ControlAlpnRejected,
                "mesh_control_alpn_rejected",
            ),
            (
                MeshOperationalEvent::ControlConnectionAccepted,
                "mesh_control_connection_accepted",
            ),
            (
                MeshOperationalEvent::GossipPolicyRejected(
                    MeshPolicyRejectionReason::AttestationRequired,
                ),
                "gossip_policy_rejected",
            ),
            (
                MeshOperationalEvent::GossipDirectPeerPromoted,
                "gossip_direct_peer_promoted",
            ),
            (
                MeshOperationalEvent::GossipIncompatibleVersionRejected,
                "gossip_incompatible_version_rejected",
            ),
            (
                MeshOperationalEvent::GossipPeerRemoved(MeshPeerRemovalReason::CleanShutdown),
                "gossip_peer_removed",
            ),
            (
                MeshOperationalEvent::AutoJoinSucceeded,
                "mesh_auto_join_succeeded",
            ),
            (
                MeshOperationalEvent::AutoJoinFailed,
                "mesh_auto_join_failed",
            ),
        ];
        for (event, expected_code) in cases {
            assert_eq!(
                event.code(),
                expected_code,
                "reviewed vocabulary code must be exact"
            );
        }
    }

    #[test]
    fn record_mesh_operational_event_is_fail_open_without_logging_service() {
        // When the process-local logging runtime state is absent the adapter
        // must be a no-op; when a concurrent logging test installed state the
        // bounded write path must equally never panic (fail-open by contract).
        record_mesh_operational_event(MeshOperationalEvent::QuicHandlerFailed(
            MeshHandlerFailureClass::Internal,
        ));
        record_mesh_operational_event(MeshOperationalEvent::AutoJoinFailed);
        record_mesh_operational_event(MeshOperationalEvent::GossipPolicyRejected(
            MeshPolicyRejectionReason::AttestationRequired,
        ));
        record_mesh_operational_event(MeshOperationalEvent::QuicInboundAccepted(
            MeshQuicInboundOutcome::Accepted,
        ));
    }
}
