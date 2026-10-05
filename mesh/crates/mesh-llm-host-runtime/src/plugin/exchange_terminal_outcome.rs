//! Resolve final client delivery separately from the admission decision.
use super::EmissionState;
use skippy_inference_api::wire_bytes::WireBytesIncomplete;

pub(super) fn resolve<'a>(emission: &'a EmissionState, fallback: &'a str) -> &'a str {
    let incomplete = emission.commitment.as_ref().and_then(|c| c.incomplete);
    match incomplete {
        Some(WireBytesIncomplete::Cancelled) => return "client_cancelled",
        Some(WireBytesIncomplete::Timeout) => return "timed_out",
        _ => {}
    }
    // The route may know this was a client disconnect even though AsyncWrite
    // reports only a generic transport error. Retain that stronger classification.
    if emission.execution_outcome.as_deref() == Some("client_cancelled") {
        return "client_cancelled";
    }
    // An upstream reader failure is explicitly recorded before a route returns
    // Dropped. Its generic cancellation fallback must not erase that cause.
    if incomplete == Some(WireBytesIncomplete::InvalidFraming)
        || emission.execution_outcome.as_deref() == Some("transport_error")
    {
        return "transport_error";
    }
    if fallback == "client_cancelled" {
        return "client_cancelled";
    }
    if matches!(
        incomplete,
        Some(WireBytesIncomplete::TransportError | WireBytesIncomplete::InvalidFraming)
    ) {
        return "transport_error";
    }
    for outcome in [emission.execution_outcome.as_deref(), Some(fallback)] {
        if let Some(outcome @ ("timed_out" | "transport_error")) = outcome {
            return outcome;
        }
    }
    // Observer copy loss changes evidence completeness, not execution.
    if emission.denied {
        "policy_denied"
    } else if emission.required_failure {
        "internal_hook_failure"
    } else {
        emission.execution_outcome.as_deref().unwrap_or(fallback)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_inference_api::wire_bytes::commit_wire_bytes;

    #[test]
    fn incomplete_client_delivery_wins_over_admission_and_backend_outcomes() {
        for denied in [false, true] {
            for (failure, expected) in [
                (WireBytesIncomplete::Cancelled, "client_cancelled"),
                (WireBytesIncomplete::Timeout, "timed_out"),
                (WireBytesIncomplete::TransportError, "transport_error"),
                (WireBytesIncomplete::InvalidFraming, "transport_error"),
            ] {
                let mut prefix = commit_wire_bytes(b"partial denial");
                prefix.incomplete = Some(failure);
                let emission = EmissionState {
                    denied,
                    required_failure: !denied,
                    execution_outcome: Some("completed".into()),
                    commitment: Some(prefix),
                    ..Default::default()
                };
                assert_eq!(resolve(&emission, "policy_denied"), expected);
            }
        }
    }

    #[test]
    fn observer_loss_preserves_success_and_successful_denial_delivery_preserves_admission() {
        let mut commitment = commit_wire_bytes(b"complete response");
        commitment.incomplete = Some(WireBytesIncomplete::ObserverUnavailable);
        let mut emission = EmissionState {
            commitment: Some(commitment),
            ..Default::default()
        };
        assert_eq!(resolve(&emission, "completed"), "completed");
        emission.denied = true;
        assert_eq!(resolve(&emission, "request_invalid"), "policy_denied");
        emission.denied = false;
        emission.required_failure = true;
        assert_eq!(resolve(&emission, "backend_error"), "internal_hook_failure");
        assert_eq!(resolve(&emission, "client_cancelled"), "client_cancelled");
        assert_eq!(resolve(&emission, "timed_out"), "timed_out");
    }

    #[test]
    fn reliable_route_disconnect_wins_over_generic_writer_error_and_admission() {
        let mut prefix = commit_wire_bytes(b"partial");
        prefix.incomplete = Some(WireBytesIncomplete::TransportError);
        let mut emission = EmissionState {
            denied: true,
            commitment: Some(prefix),
            ..Default::default()
        };
        assert_eq!(resolve(&emission, "client_cancelled"), "client_cancelled");
        emission.execution_outcome = Some("client_cancelled".into());
        assert_eq!(resolve(&emission, "policy_denied"), "client_cancelled");
        emission.execution_outcome = Some("timed_out".into());
        assert_eq!(resolve(&emission, "policy_denied"), "transport_error");
    }
    #[test]
    fn final_emitter_failure_overrides_previously_recorded_backend_timeout() {
        for recorded in ["timed_out", "transport_error", "backend_error"] {
            for (failure, expected) in [
                (WireBytesIncomplete::Cancelled, "client_cancelled"),
                (WireBytesIncomplete::Timeout, "timed_out"),
                (WireBytesIncomplete::TransportError, "transport_error"),
                (WireBytesIncomplete::InvalidFraming, "transport_error"),
            ] {
                let mut prefix = commit_wire_bytes(b"partial error");
                prefix.incomplete = Some(failure);
                let emission = EmissionState {
                    execution_outcome: Some(recorded.into()),
                    commitment: Some(prefix),
                    ..Default::default()
                };
                assert_eq!(
                    resolve(&emission, "timed_out"),
                    expected,
                    "{recorded} / {failure:?}"
                );
            }
        }
    }

    #[test]
    fn explicit_upstream_transport_failure_survives_generic_dropped_route_fallback() {
        let mut commitment = commit_wire_bytes(b"partial upstream response");
        commitment.incomplete = Some(WireBytesIncomplete::TransportError);
        let mut emission = EmissionState {
            execution_outcome: Some("transport_error".into()),
            commitment: Some(commitment),
            ..Default::default()
        };
        assert_eq!(resolve(&emission, "client_cancelled"), "transport_error");
        emission.execution_outcome = None;
        emission.commitment.as_mut().unwrap().incomplete =
            Some(WireBytesIncomplete::InvalidFraming);
        assert_eq!(resolve(&emission, "client_cancelled"), "transport_error");
    }
}
