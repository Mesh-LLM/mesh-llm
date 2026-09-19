//! Standalone HTTP serving through the same lifecycle used by embedding applications.
use super::{ModelLoadRequest, ModelOpenEvents, OpenAiOptions};
use anyhow::{Result, bail};
use skippy_protocol::{StageConfig, StageTopology};
use skippy_server::{EmbeddedRuntimeOptions, SpeculativeDecodeConfig};
use std::{future::Future, net::SocketAddr};

/// Prepared local OpenAI serving options. Model acquisition and argument parsing belong to callers.
pub struct LocalOpenAiOptions {
    pub config: StageConfig,
    pub topology: Option<StageTopology>,
    pub speculative: SpeculativeDecodeConfig,
    pub bind_addr: SocketAddr,
    pub model_id: Option<String>,
    pub default_max_tokens: u32,
    pub generation_concurrency: Option<usize>,
    pub adaptive_generation_concurrency: bool,
    pub adaptive_generation_min_concurrency: Option<usize>,
    pub generation_queue_capacity: Option<usize>,
    pub generation_admission_timeout_secs: u64,
    pub prefill_chunk_size: usize,
    pub prefill_chunk_policy: String,
    pub prefill_chunk_schedule: Option<String>,
    pub prefill_adaptive_start: usize,
    pub prefill_adaptive_step: usize,
    pub prefill_adaptive_max: usize,
    pub prefill_adaptive_target_ms: f64,
    pub metrics_otlp_grpc: Option<String>,
    pub telemetry_queue_capacity: usize,
    pub telemetry_level: skippy_server::telemetry::TelemetryLevel,
    pub openai_guardrails: skippy_server::frontend::OpenAiGuardrailsMode,
}

impl LocalOpenAiOptions {
    fn into_request(self) -> Result<(SocketAddr, ModelLoadRequest)> {
        skippy_config::validate_config(&self.config, self.topology.as_ref())?;
        if self.config.downstream.is_some() {
            bail!(
                "serve-openai local backend requires a final/single-stage config with no downstream"
            );
        }
        if self.prefill_chunk_size == 0 {
            bail!("--prefill-chunk-size must be greater than zero");
        }
        if self.generation_concurrency == Some(0) {
            bail!("--generation-concurrency must be greater than zero");
        }
        let concurrency = self
            .generation_concurrency
            .unwrap_or_else(|| usize::try_from(self.config.lane_count).unwrap_or(usize::MAX));
        let adaptive_minimum =
            skippy_server::frontend::resolve_adaptive_generation_min_concurrency(
                self.adaptive_generation_concurrency,
                self.adaptive_generation_min_concurrency,
                concurrency,
                "--adaptive-generation-min-concurrency",
            )?;
        self.speculative.validate()?;
        let mut openai = OpenAiOptions::direct_single_stage_defaults(
            self.model_id
                .unwrap_or_else(|| self.config.model_id.clone()),
            self.default_max_tokens,
            concurrency,
            self.config.native_mtp_enabled,
        );
        openai.adaptive_generation_min_concurrency = adaptive_minimum;
        openai.generation_queue_capacity = self.generation_queue_capacity.unwrap_or_else(|| {
            skippy_server::frontend::default_generation_queue_capacity(concurrency)
        });
        openai.generation_admission_timeout_secs = self.generation_admission_timeout_secs;
        openai.prefill_chunk_size = self.prefill_chunk_size;
        openai.prefill_chunk_policy = self.prefill_chunk_policy;
        openai.prefill_chunk_schedule = self.prefill_chunk_schedule;
        openai.prefill_adaptive_start = self.prefill_adaptive_start;
        openai.prefill_adaptive_step = self.prefill_adaptive_step;
        openai.prefill_adaptive_max = self.prefill_adaptive_max;
        openai.prefill_adaptive_target_ms = self.prefill_adaptive_target_ms;
        openai.speculative = self.speculative;
        Ok((
            self.bind_addr,
            ModelLoadRequest {
                runtime: EmbeddedRuntimeOptions {
                    config: self.config,
                    topology: self.topology,
                    n_threads: None,
                    n_threads_batch: None,
                    // Preserve the previous standalone loader's launch override.
                    mtp_source: skippy_runtime::MtpSource::Disabled,
                    metrics_otlp_grpc: self.metrics_otlp_grpc,
                    telemetry_queue_capacity: self.telemetry_queue_capacity,
                    telemetry_level: self.telemetry_level,
                    operation_id: None,
                    session_lifecycle_observer: None,
                },
                openai,
                open_events: ModelOpenEvents::Disabled,
                hooks_factory: None,
                generation_observer: None,
                kv_observer: None,
                hook_policy: None,
                guardrails: Some(skippy_server::OpenAiGuardrailsConfig::for_standalone_mode(
                    self.openai_guardrails,
                )),
                guardrail_telemetry: None,
                downstream_wire_condition: skippy_server::binary_transport::WireCondition::new(
                    0.0, None,
                )?,
                serving_telemetry: None,
            },
        ))
    }
}

pub async fn serve_local_openai_with_shutdown(
    options: LocalOpenAiOptions,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<()> {
    let (bind_addr, request) = options.into_request()?;
    request
        .load()?
        .serve_http_with_shutdown(bind_addr, shutdown)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> LocalOpenAiOptions {
        LocalOpenAiOptions {
            config: serde_json::from_value(skippy_config::example_config()).unwrap(),
            topology: None,
            speculative: SpeculativeDecodeConfig::default(),
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            model_id: Some("served-model".into()),
            default_max_tokens: 37,
            generation_concurrency: None,
            adaptive_generation_concurrency: false,
            adaptive_generation_min_concurrency: None,
            generation_queue_capacity: None,
            generation_admission_timeout_secs: 0,
            prefill_chunk_size: 64,
            prefill_chunk_policy: "fixed".into(),
            prefill_chunk_schedule: None,
            prefill_adaptive_start: 64,
            prefill_adaptive_step: 64,
            prefill_adaptive_max: 512,
            prefill_adaptive_target_ms: 100.0,
            metrics_otlp_grpc: None,
            telemetry_queue_capacity: 0,
            telemetry_level: skippy_server::telemetry::TelemetryLevel::Off,
            openai_guardrails: skippy_server::frontend::OpenAiGuardrailsMode::Disabled,
        }
    }

    #[test]
    fn standalone_defaults_use_lane_capacity_and_shared_lifecycle() {
        let (_, request) = options().into_request().unwrap();
        assert_eq!(request.openai.generation_concurrency, 4);
        assert_eq!(request.openai.generation_queue_capacity, 32);
        assert_eq!(request.openai.adaptive_generation_min_concurrency, None);
        assert_eq!(request.openai.generation_admission_timeout_secs, 0);
        assert_eq!(request.openai.model_id.as_deref(), Some("served-model"));
        assert_eq!(request.openai.default_max_tokens, 37);
        assert_eq!(
            request.runtime.mtp_source,
            skippy_runtime::MtpSource::Disabled
        );
    }

    #[test]
    fn standalone_admission_overrides_survive_lifecycle_conversion() {
        let mut options = options();
        options.generation_concurrency = Some(3);
        options.adaptive_generation_concurrency = true;
        options.adaptive_generation_min_concurrency = Some(2);
        options.generation_queue_capacity = Some(7);
        options.generation_admission_timeout_secs = 11;
        let (_, request) = options.into_request().unwrap();
        assert_eq!(request.openai.generation_concurrency, 3);
        assert_eq!(request.openai.adaptive_generation_min_concurrency, Some(2));
        assert_eq!(request.openai.generation_queue_capacity, 7);
        assert_eq!(request.openai.generation_admission_timeout_secs, 11);
    }

    #[test]
    fn standalone_entrypoint_rejects_split_config_before_model_loading() {
        let mut options = options();
        options.config.model_path = Some("does-not-exist.gguf".into());
        options.config.downstream = Some(skippy_protocol::PeerConfig {
            stage_id: "stage-1".into(),
            stage_index: 1,
            endpoint: "127.0.0.1:19001".into(),
        });
        let error = options
            .into_request()
            .err()
            .expect("split configuration accepted");
        assert!(error.to_string().contains("no downstream"));
    }

    #[test]
    fn invalid_adaptive_admission_is_rejected_before_model_loading() {
        for (enabled, minimum, message) in [
            (false, 1, "requires adaptive generation concurrency"),
            (true, 0, "must be greater than zero"),
            (true, 5, "exceeds generation concurrency"),
        ] {
            let mut options = options();
            options.config.model_path = Some("does-not-exist.gguf".into());
            options.adaptive_generation_concurrency = enabled;
            options.adaptive_generation_min_concurrency = Some(minimum);
            let error = options
                .into_request()
                .err()
                .expect("invalid admission accepted");
            assert!(error.to_string().contains(message), "{error}");
        }
    }
}
