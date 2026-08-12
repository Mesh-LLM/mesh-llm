//! Bounded, metadata-only operational audit vocabulary for mesh boundaries.
//!
//! These events deliberately accept no peer, endpoint, address, token, ALPN,
//! or error values. The service receives only the static level and code below.

#[cfg(test)]
use crate::logging::LoggingService;
use crate::logging::{OperationalAuditRecord, OperationalAuditSeverity};

const OPERATIONAL_AUDIT_INFO: &str = "info";
const OPERATIONAL_AUDIT_WARNING: &str = "warning";

const OPERATIONAL_AUDIT_SOURCE: &str = "mesh";

fn operational_audit_record(code: &'static str, level: &'static str) -> OperationalAuditRecord {
    let severity = match level {
        OPERATIONAL_AUDIT_INFO => OperationalAuditSeverity::Info,
        OPERATIONAL_AUDIT_WARNING => OperationalAuditSeverity::Warning,
        _ => OperationalAuditSeverity::Error,
    };
    OperationalAuditRecord::builder(OPERATIONAL_AUDIT_SOURCE, code)
        .severity(severity)
        .build()
}

/// Static outcomes that are safe to publish through the local operational log.
///
/// Variants and codes follow the reviewed mesh audit vocabulary. `QuicInboundAccepted`
/// describes a non-Skippy inbound QUIC connection accepted after the Skippy ALPN
/// exclusion; there is no `quic_alpn_accepted` code because no explicit mesh-ALPN
/// validation branch exists in the inbound path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MeshOperationalEvent {
    QuicHandlerFailed,
    QuicInboundAccepted,
    ControlHandlerFailed,
    ControlAlpnRejected,
    ControlConnectionAccepted,
    GossipPolicyRejected,
    GossipDirectPeerPromoted,
    GossipIncompatibleVersionRejected,
    GossipPeerRemoved,
    AutoJoinSucceeded,
    AutoJoinFailed,
}

impl MeshOperationalEvent {
    const fn level(self) -> &'static str {
        match self {
            Self::QuicInboundAccepted
            | Self::ControlConnectionAccepted
            | Self::GossipDirectPeerPromoted
            | Self::GossipPeerRemoved
            | Self::AutoJoinSucceeded => OPERATIONAL_AUDIT_INFO,
            Self::QuicHandlerFailed
            | Self::ControlHandlerFailed
            | Self::ControlAlpnRejected
            | Self::GossipPolicyRejected
            | Self::GossipIncompatibleVersionRejected
            | Self::AutoJoinFailed => OPERATIONAL_AUDIT_WARNING,
        }
    }

    const fn code(self) -> &'static str {
        match self {
            Self::QuicHandlerFailed => "mesh_quic_handler_failed",
            Self::QuicInboundAccepted => "mesh_quic_inbound_accepted",
            Self::ControlHandlerFailed => "mesh_control_handler_failed",
            Self::ControlAlpnRejected => "mesh_control_alpn_rejected",
            Self::ControlConnectionAccepted => "mesh_control_connection_accepted",
            Self::GossipPolicyRejected => "gossip_policy_rejected",
            Self::GossipDirectPeerPromoted => "gossip_direct_peer_promoted",
            Self::GossipIncompatibleVersionRejected => "gossip_incompatible_version_rejected",
            Self::GossipPeerRemoved => "gossip_peer_removed",
            Self::AutoJoinSucceeded => "mesh_auto_join_succeeded",
            Self::AutoJoinFailed => "mesh_auto_join_failed",
        }
    }
}

/// Record one mesh boundary result through the process-local logging service.
/// Logging is optional and this intentionally never affects mesh serving.
pub(crate) fn record_mesh_operational_event(event: MeshOperationalEvent) {
    let Some(state) = crate::logging_runtime_state() else {
        return;
    };
    let _ = state.write_operational_audit(operational_audit_record(event.code(), event.level()));
}

#[cfg(test)]
fn record_mesh_operational_event_with_service(
    service: &LoggingService,
    event: MeshOperationalEvent,
) {
    let _ = service.write_operational_audit(operational_audit_record(event.code(), event.level()));
}

#[cfg(test)]
mod tests {
    use super::{
        MeshOperationalEvent, record_mesh_operational_event,
        record_mesh_operational_event_with_service,
    };
    use crate::logging::{LoggingService, ServiceConfig};

    fn recorded_audits(service: &LoggingService) -> Vec<serde_json::Value> {
        service
            .bus_ref()
            .drain()
            .into_iter()
            .map(|entry| {
                let audit: serde_json::Value =
                    serde_json::from_str(&entry.payload).expect("audit payload");
                serde_json::json!({
                    "kind": "audit",
                    "level": audit["severity"],
                    "message": audit["code"],
                })
            })
            .collect()
    }

    #[test]
    fn mesh_boundary_outcomes_emit_exact_static_audits_without_raw_metadata() {
        let service = LoggingService::new_disabled(ServiceConfig::default());
        let events = [
            MeshOperationalEvent::QuicInboundAccepted,
            MeshOperationalEvent::ControlAlpnRejected,
            MeshOperationalEvent::ControlHandlerFailed,
            MeshOperationalEvent::GossipDirectPeerPromoted,
            MeshOperationalEvent::GossipPolicyRejected,
            MeshOperationalEvent::AutoJoinFailed,
        ];

        for event in events {
            record_mesh_operational_event_with_service(&service, event);
        }

        let audits = recorded_audits(&service);
        assert_eq!(
            audits,
            vec![
                serde_json::json!({
                    "kind": "audit",
                    "level": "info",
                    "message": "mesh_quic_inbound_accepted",
                }),
                serde_json::json!({
                    "kind": "audit",
                    "level": "warning",
                    "message": "mesh_control_alpn_rejected",
                }),
                serde_json::json!({
                    "kind": "audit",
                    "level": "warning",
                    "message": "mesh_control_handler_failed",
                }),
                serde_json::json!({
                    "kind": "audit",
                    "level": "info",
                    "message": "gossip_direct_peer_promoted",
                }),
                serde_json::json!({
                    "kind": "audit",
                    "level": "warning",
                    "message": "gossip_policy_rejected",
                }),
                serde_json::json!({
                    "kind": "audit",
                    "level": "warning",
                    "message": "mesh_auto_join_failed",
                }),
            ]
        );

        let serialized = serde_json::to_string(&audits).expect("serialized audit payloads");
        for raw_value in [
            "node=untrusted-lab-host",
            "peer=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            "token=mesh-secret-bootstrap-token",
        ] {
            assert!(
                !serialized.contains(raw_value),
                "raw mesh metadata must not enter the audit payload"
            );
        }
    }

    #[test]
    fn mesh_operational_vocabulary_is_bounded_and_identifier_free() {
        let events = [
            MeshOperationalEvent::QuicHandlerFailed,
            MeshOperationalEvent::QuicInboundAccepted,
            MeshOperationalEvent::ControlHandlerFailed,
            MeshOperationalEvent::ControlAlpnRejected,
            MeshOperationalEvent::ControlConnectionAccepted,
            MeshOperationalEvent::GossipPolicyRejected,
            MeshOperationalEvent::GossipDirectPeerPromoted,
            MeshOperationalEvent::GossipIncompatibleVersionRejected,
            MeshOperationalEvent::GossipPeerRemoved,
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
    fn mesh_operational_vocabulary_maps_each_variant_to_its_reviewed_code() {
        let cases = [
            (
                MeshOperationalEvent::QuicHandlerFailed,
                "mesh_quic_handler_failed",
            ),
            (
                MeshOperationalEvent::QuicInboundAccepted,
                "mesh_quic_inbound_accepted",
            ),
            (
                MeshOperationalEvent::ControlHandlerFailed,
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
                MeshOperationalEvent::GossipPolicyRejected,
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
                MeshOperationalEvent::GossipPeerRemoved,
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
        record_mesh_operational_event(MeshOperationalEvent::QuicHandlerFailed);
        record_mesh_operational_event(MeshOperationalEvent::AutoJoinFailed);
        record_mesh_operational_event(MeshOperationalEvent::GossipPolicyRejected);
        record_mesh_operational_event(MeshOperationalEvent::QuicInboundAccepted);
    }
}
