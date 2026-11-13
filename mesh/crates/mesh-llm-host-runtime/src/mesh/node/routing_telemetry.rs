use super::*;

/// One inference attempt as [`Node::record_inference_attempt_report`] takes it.
pub(crate) struct InferenceAttemptReport {
    pub(crate) queue_wait: std::time::Duration,
    pub(crate) attempt_time: std::time::Duration,
    /// What routing metrics score; `None` leaves the attempt out of them.
    pub(crate) metrics_outcome: Option<crate::network::metrics::AttemptOutcome>,
    /// What analytics reports.
    pub(crate) route_outcome: crate::inference_analytics::RouteOutcome,
    pub(crate) completion_tokens: Option<u64>,
}

impl Node {
    /// Record an attempt scored the same way by routing metrics and
    /// analytics. Production paths go through
    /// [`Node::record_inference_attempt_report`], which can tell a client
    /// disconnect or a failed stream apart.
    #[cfg(test)]
    pub fn record_inference_attempt(
        &self,
        model: Option<&str>,
        target: &crate::inference::election::InferenceTarget,
        queue_wait: std::time::Duration,
        attempt_time: std::time::Duration,
        outcome: crate::network::metrics::AttemptOutcome,
        completion_tokens: Option<u64>,
    ) {
        self.record_inference_attempt_report(
            model,
            target,
            InferenceAttemptReport {
                queue_wait,
                attempt_time,
                metrics_outcome: Some(outcome),
                route_outcome: outcome.into(),
                completion_tokens,
            },
        );
    }

    /// Record an attempt whose analytics outcome is finer than its routing
    /// metrics outcome, or that routing metrics do not score at all.
    pub(crate) fn record_inference_attempt_report(
        &self,
        model: Option<&str>,
        target: &crate::inference::election::InferenceTarget,
        report: InferenceAttemptReport,
    ) {
        let InferenceAttemptReport {
            queue_wait,
            attempt_time,
            metrics_outcome,
            route_outcome,
            completion_tokens,
        } = report;
        let attempt_target = match target {
            crate::inference::election::InferenceTarget::Local(port) => {
                crate::network::metrics::AttemptTarget::Local(format!("127.0.0.1:{port}"))
            }
            crate::inference::election::InferenceTarget::Remote(peer_id) => {
                crate::network::metrics::AttemptTarget::Remote(peer_id.fmt_short().to_string())
            }
            crate::inference::election::InferenceTarget::None => return,
        };
        let remote_peer = match target {
            crate::inference::election::InferenceTarget::Remote(peer_id) => Some(*peer_id),
            _ => None,
        };
        crate::inference_analytics::report_route(
            self,
            remote_peer,
            crate::inference_analytics::RouteAttempt {
                model,
                target: &attempt_target,
                queue_wait,
                attempt_time,
                outcome: route_outcome,
                completion_tokens,
            },
        );
        // A client that went away says nothing about the target, so routing
        // metrics do not score it; analytics still counts the attempt.
        let Some(outcome) = metrics_outcome else {
            return;
        };
        self.routing_metrics.record_attempt(
            model,
            attempt_target.clone(),
            queue_wait,
            attempt_time,
            outcome,
            completion_tokens,
        );
        if let Some(sink) = self.routing_telemetry_sink() {
            sink.record_route_attempt(
                model,
                &attempt_target,
                outcome,
                attempt_time.as_millis() as f64,
            );
        }
        self.publish_routing_runtime_snapshot();
    }

    pub fn record_endpoint_attempt(
        &self,
        model: Option<&str>,
        endpoint: &str,
        queue_wait: std::time::Duration,
        attempt_time: std::time::Duration,
        outcome: crate::network::metrics::AttemptOutcome,
        completion_tokens: Option<u64>,
    ) {
        let model_ref = model.map(canonical_demand_model_ref);
        let attempt_target = crate::network::metrics::AttemptTarget::Endpoint(endpoint.to_string());
        crate::inference_analytics::report_route(
            self,
            None,
            crate::inference_analytics::RouteAttempt {
                model: model_ref.as_deref(),
                target: &attempt_target,
                queue_wait,
                attempt_time,
                outcome: outcome.into(),
                completion_tokens,
            },
        );
        self.routing_metrics.record_attempt(
            model_ref.as_deref(),
            attempt_target.clone(),
            queue_wait,
            attempt_time,
            outcome,
            completion_tokens,
        );
        if let Some(sink) = self.routing_telemetry_sink() {
            sink.record_route_attempt(
                model_ref.as_deref(),
                &attempt_target,
                outcome,
                attempt_time.as_millis() as f64,
            );
        }
        self.publish_routing_runtime_snapshot();
    }

    pub fn record_routed_request(
        &self,
        model: Option<&str>,
        attempts: usize,
        outcome: crate::network::metrics::RequestOutcome,
    ) {
        let model_ref = model.map(canonical_demand_model_ref);
        self.routing_metrics
            .record_request(model_ref.as_deref(), attempts, outcome);
        if let Some(sink) = self.routing_telemetry_sink() {
            sink.record_model_request(model_ref.as_deref(), attempts, outcome);
        }
        self.publish_routing_runtime_snapshot();
    }

    pub(crate) fn record_prompt_shape(
        &self,
        model: Option<&str>,
        prompt_tokens: Option<u64>,
        completion_tokens: Option<u64>,
        outcome: crate::network::metrics::RequestOutcome,
    ) {
        let model_ref = model.map(canonical_demand_model_ref);
        if let Some(sink) = self.routing_telemetry_sink() {
            sink.record_prompt_shape(
                model_ref.as_deref(),
                prompt_tokens,
                completion_tokens,
                outcome,
            );
        }
    }
}
