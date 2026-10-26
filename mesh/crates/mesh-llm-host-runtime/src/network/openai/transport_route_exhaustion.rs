//! Final response classification after model-route retries are exhausted.

use super::*;

impl RouteModelState {
    fn exhausted_status(&self) -> u16 {
        // Every replica refused to serve without payment. That is the caller's
        // to settle, not an unavailable mesh (#2059).
        if self.payment_required {
            return PAYMENT_REQUIRED_STATUS;
        }
        // Keep unavailable/mixed failures as 503. A route whose every attempt
        // exceeded its deadline is a gateway timeout, after all normal retries.
        if self.attempts > 0 && self.timeout_attempts == self.attempts {
            504
        } else {
            503
        }
    }
}

pub(super) async fn finish_exhausted_route_model_request(
    node: &mesh::Node,
    tcp_stream: ClientStream,
    model: &str,
    total_targets: usize,
    state: &RouteModelState,
    route_observer: OpenAiRouteObserver<'_>,
) -> RouteDispatchOutcome {
    let status = state.exhausted_status();
    let reason = if state.payment_required {
        format!("every target for model '{model}' requires the Lightning payment protocol")
    } else {
        format!("all {} target(s) for model '{model}' failed", total_targets)
    };
    let result = send_error_observed(tcp_stream, status, &reason, route_observer).await;
    record_route_model_unavailable(node, model, state.attempts);
    tracing::warn!(
        model = model,
        attempts = state.attempts,
        route_ms = state.route_started.elapsed().as_millis(),
        "openai route_model_request exhausted targets"
    );
    response_outcome(status, result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhausted_timeouts_return_gateway_timeout_without_reclassifying_mixed_failures() {
        for (attempts, timeout_attempts, expected) in [
            (0, 0, 503),
            (1, 1, 504),
            (3, 3, 504),
            (3, 2, 503),
            (1, 0, 503),
        ] {
            let state = RouteModelState {
                route_started: Instant::now(),
                attempts,
                timeout_attempts,
                refreshed: false,
                payment_required: false,
            };
            let status = state.exhausted_status();
            assert_eq!(status, expected);
            assert!(matches!(response_outcome(status, Ok(())),
                RouteDispatchOutcome::Responded(code) if code == expected));
            assert!(matches!(
                response_outcome(status, Err(std::io::Error::other("closed"))),
                RouteDispatchOutcome::Dropped("response_write_failed")
            ));
        }
    }
}
