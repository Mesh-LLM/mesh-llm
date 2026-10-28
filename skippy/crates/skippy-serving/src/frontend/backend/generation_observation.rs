use super::*;
impl StageOpenAiBackend {
    pub(super) fn observe_generation_completed(
        &self,
        output: &GeneratedText,
        demand: GenerationDemandObservation,
    ) {
        let work = GenerationAdmissionWork::new(
            usize::try_from(output.prompt_tokens).unwrap_or(usize::MAX),
            output.completion_tokens,
        );
        let completed_tokens = work.prompt_tokens.saturating_add(work.decode_tokens);
        let executed_tokens =
            u64::from(output.suffix_prefill_tokens).saturating_add(work.decode_tokens);
        let saturated = demand.waited_for_lane
            || self.generation_limit.was_saturated_since(
                demand.demand_epoch,
                demand.queued_at_start,
                self.generation_queue_depth.load(Ordering::Acquire) > 0,
            );
        let Some(decision) =
            self.generation_limit
                .observe_completed(GenerationConcurrencyObservation {
                    completed_tokens,
                    executed_tokens,
                    latency_ms: output.prompt_ms + output.predicted_ms,
                    saturated,
                    at_capacity: demand.at_capacity_at_start,
                    started_at: demand.started_at,
                })
        else {
            return;
        };
        self.emit_generation_concurrency_decision(decision);
    }

    pub(super) fn observe_generation_failed(&self) {
        if let Some(decision) = self.generation_limit.observe_failed() {
            self.emit_generation_concurrency_decision(decision);
        }
    }

    pub(super) fn emit_generation_concurrency_decision(
        &self,
        decision: GenerationConcurrencyDecision,
    ) {
        let mut attrs = lifecycle_attrs(&self.config);
        attrs.insert(
            "llama_stage.generation_concurrency_action".to_string(),
            json!(decision.action),
        );
        attrs.insert(
            "llama_stage.generation_concurrency_reason".to_string(),
            json!(decision.reason),
        );
        attrs.insert(
            "llama_stage.generation_concurrency_previous".to_string(),
            json!(decision.previous_limit),
        );
        attrs.insert(
            "llama_stage.generation_concurrency_current".to_string(),
            json!(decision.current_limit),
        );
        attrs.insert(
            "llama_stage.generation_concurrency_ceiling".to_string(),
            json!(self.generation_limit.hard_limit()),
        );
        if let Some(throughput) = decision.throughput_tokens_per_second {
            attrs.insert(
                "llama_stage.generation_throughput_tokens_per_second".to_string(),
                json!(throughput),
            );
        }
        if let Some(p95_latency_ms) = decision.p95_latency_ms {
            attrs.insert(
                "llama_stage.generation_p95_latency_ms".to_string(),
                json!(p95_latency_ms),
            );
        }
        if let Some(p95_hardware_ms_per_token) = decision.p95_hardware_ms_per_token {
            attrs.insert(
                "llama_stage.generation_p95_hardware_ms_per_token".to_string(),
                json!(p95_hardware_ms_per_token),
            );
        }
        if let Some(improvement) = decision.throughput_improvement {
            attrs.insert(
                "llama_stage.generation_throughput_improvement".to_string(),
                json!(improvement),
            );
        }
        if let Some(ratio) = decision.p95_latency_ratio {
            attrs.insert(
                "llama_stage.generation_p95_latency_ratio".to_string(),
                json!(ratio),
            );
        }
        if let Some(ratio) = decision.hardware_pressure_ratio {
            attrs.insert(
                "llama_stage.generation_hardware_pressure_ratio".to_string(),
                json!(ratio),
            );
        }
        if let Some(observed_requests) = decision.observed_requests {
            attrs.insert(
                "llama_stage.generation_observed_requests".to_string(),
                json!(observed_requests),
            );
        }
        if let Some(saturated_requests) = decision.saturated_requests {
            attrs.insert(
                "llama_stage.generation_saturated_requests".to_string(),
                json!(saturated_requests),
            );
        }
        self.telemetry
            .emit("stage.openai_generation_concurrency_adapt", attrs);
    }
}
