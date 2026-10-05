use super::*;

impl Node {
    pub fn record_inference_attempt(
        &self,
        model: Option<&str>,
        target: &crate::inference::election::InferenceTarget,
        queue_wait: std::time::Duration,
        attempt_time: std::time::Duration,
        outcome: crate::network::metrics::AttemptOutcome,
        completion_tokens: Option<u64>,
    ) {
        let attempt_target = match target {
            crate::inference::election::InferenceTarget::Local(port) => {
                crate::network::metrics::AttemptTarget::Local(format!("127.0.0.1:{port}"))
            }
            crate::inference::election::InferenceTarget::Remote(peer_id) => {
                crate::network::metrics::AttemptTarget::Remote(peer_id.fmt_short().to_string())
            }
            crate::inference::election::InferenceTarget::None => return,
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
            sink.record_route_attempt(model, &attempt_target, outcome);
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
        self.routing_metrics.record_attempt(
            model_ref.as_deref(),
            attempt_target.clone(),
            queue_wait,
            attempt_time,
            outcome,
            completion_tokens,
        );
        if let Some(sink) = self.routing_telemetry_sink() {
            sink.record_route_attempt(model_ref.as_deref(), &attempt_target, outcome);
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
